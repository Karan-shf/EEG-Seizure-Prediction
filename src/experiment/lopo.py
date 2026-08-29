"""
lopo.py
=======
Stage (experiment): the leave-one-patient-out DRIVER for ONE (alpha, span_roof).

What it wires together (all previously-built layers)
----------------------------------------------------
  raw EDF  --ChbSpdProvider-->  streamed RAW SPD windows (per channel x span)
           --dataset_builder-->  per-fold 486-D distance features (X_train/X_test)
           --balancing.balance-> cluster-centroid undersample -> Borderline-SMOTE
           --classifier.make---> Elastic-Net LR (standardized, pluggable)
           --alarms.calibrate--> Option-2 threshold on the TRAIN pool (fixed FPR/h)
           --metrics.evaluate--> per-window AUC + per-event sensitivity / FPR/h

The held-out patient is evaluated on its NATURAL (unbalanced) window stream;
balancing only ever touches the training fold. Firing Power is computed per
contiguous segment (window ordering is chronological within a segment and never
crosses an inter-file seam).

The SPD provider
----------------
`ChbSpdProvider` is the concrete `dataset_builder.SpdWindowProvider`: for each
patient it parses the summary, builds the stitched timeline (exact per-file
sample counts probed from EDF headers), the SPH/SOP label plan, and the
subsampled window plan, then streams one window at a time through
montage -> band-pass -> RSMMTN -> SPD at a FIXED alpha. Montaged+filtered file
signals are held in a tiny LRU (the dense SPD is never persisted -- exactly the
`cfg.CACHE_DENSE_SPD = False` contract).

Alpha & the fold cache
----------------------
Each alpha is a fully independent experiment, but the dataset_builder's
fold-invariant cache key (`g_patient`, `d_baseline`, `patient_anchor_means`) is
NOT alpha-aware on its own. So every fold here is built with an alpha-tagged
fingerprint (`_alpha_fingerprint`) to keep the on-disk caches for different
alphas strictly separate. Within one alpha, all span roofs reuse the same cached
artifacts (span roof only slices the feature width downstream).

Dependency note: importing this module is light. Running `run_lopo` needs mne +
scipy (provider), scikit-learn (classifier) and imbalanced-learn (balancing).
The self-test injects an in-memory SPD provider and runs the full path only when
scikit-learn + imbalanced-learn are present, else it does structural checks only.
"""
from __future__ import annotations

import importlib.util
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from src import config as cfg
from src.utils.logger import get_logger
from src.data import dataset_builder as db
from src.modeling import balancing
from src.modeling import classifier as clf
from src.evaluation import alarms as alarms_mod
from src.evaluation import metrics as metrics_mod

# Import-light pipeline pieces (mne / scipy stay lazy inside their own modules).
from src.io.summary_parser import parse_summary_file
from src.labeling.timeline import build_timeline
from src.labeling.labeler import build_label_plan
from src.preprocessing.windowing import build_windows
from src.io import edf_loader
from src.preprocessing import filters, montage
from src.features import spd, laplacian

log = get_logger(__name__)

HAVE_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAVE_IMBLEARN = importlib.util.find_spec("imblearn") is not None

DEFAULT_PATIENTS: tuple[str, ...] = tuple(f"chb{i:02d}" for i in range(1, 25))


# ===========================================================================
# Concrete SPD provider: raw EDF -> montage -> band-pass -> RSMMTN -> SPD
# ===========================================================================
class ChbSpdProvider:
    """Stream RAW SPD windows for CHB-MIT patients at a FIXED alpha.

    Implements dataset_builder.SpdWindowProvider (patient_ids / channels /
    iter_windows) plus window_meta() for downstream per-segment evaluation.
    Re-iterable: iter_windows re-streams on every call (the streaming Frechet
    mean makes several passes); per-file filtered signals are memoized in a
    small LRU so EDFs are not re-read within a pass.
    """

    def __init__(self, patients: Sequence[str], *, alpha: float,
                 raw_dir: Optional[Path] = None,
                 sop_minutes: Optional[int] = None,
                 window_seconds: Optional[float] = None,
                 overlap: Optional[float] = None,
                 apply_notch: bool = False,
                 signal_cache_size: int = 50,
                 operator: Optional[laplacian.LaplacianOperator] = None) -> None:
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._patients = tuple(patients)
        self.alpha = float(alpha)
        self.raw_dir = Path(raw_dir) if raw_dir is not None else cfg.RAW_DIR
        self.sop_minutes = int(cfg.SOP_PRIMARY_MINUTES if sop_minutes is None else sop_minutes)
        self.window_seconds = float(cfg.WINDOW_SECONDS if window_seconds is None else window_seconds)
        self.overlap = float(cfg.WINDOW_OVERLAP if overlap is None else overlap)
        self.apply_notch = bool(apply_notch)
        self._sig_cache_size = max(1, int(signal_cache_size))
        self._operator = operator
        self._plans: Dict[str, tuple] = {}
        self._sig: "OrderedDict[tuple, np.ndarray]" = OrderedDict()

    # -- protocol ----------------------------------------------------------
    def patient_ids(self) -> List[str]:
        return list(self._patients)

    def channels(self) -> tuple[str, ...]:
        return montage.CANONICAL_CHANNELS

    # -- geometry / paths --------------------------------------------------
    def _op(self) -> laplacian.LaplacianOperator:
        if self._operator is None:
            self._operator = laplacian.default_operator()
        return self._operator

    def _resolve(self, patient_id: str, name: str) -> Path:
        for cand in (self.raw_dir / patient_id / name, self.raw_dir / name):
            if cand.exists():
                return cand
        return self.raw_dir / patient_id / name

    def _probe_n_samples(self, path: Path) -> int:
        import mne  # header-only read
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        n = int(raw.n_times)
        sf = float(raw.info["sfreq"])
        if sf != cfg.FS:
            n = int(round(n * cfg.FS / sf))
        return n

    # -- per-patient plan (parsed once, memoized) --------------------------
    def _plan(self, patient_id: str) -> tuple:
        cached = self._plans.get(patient_id)
        if cached is not None:
            return cached
        summary = parse_summary_file(self._resolve(patient_id, f"{patient_id}-summary.txt"))
        file_samples = {f.name: self._probe_n_samples(self._resolve(patient_id, f.name))
                        for f in summary.files}
        tl = build_timeline(summary, fs=cfg.FS, file_samples=file_samples)
        plan = build_label_plan(summary, tl, sop_minutes=self.sop_minutes)
        ws = build_windows(plan, window_seconds=self.window_seconds, overlap=self.overlap)
        out = (summary, tl, ws)
        self._plans[patient_id] = out
        log.info("provider %s: %s", patient_id, ws.describe())
        return out

    # -- per-file montaged + filtered signal (LRU) -------------------------
    def _filtered(self, patient_id: str, file_name: str) -> np.ndarray:
        key = (patient_id, file_name)
        sig = self._sig.get(key)
        if sig is not None:
            self._sig.move_to_end(key)
            return sig
        rec = edf_loader.load_edf(self._resolve(patient_id, file_name))
        x = filters.clean(rec.data, fs=rec.sfreq, apply_notch=self.apply_notch)
        x = np.ascontiguousarray(x, dtype=float)
        self._sig[key] = x
        self._sig.move_to_end(key)
        while len(self._sig) > self._sig_cache_size:
            self._sig.popitem(last=False)
        return x

    # -- evaluation support: window metadata in iter_windows order ---------
    def window_meta(self, patient_id: str):
        """Windows (seg_index, offset, label, ...) in the SAME order iter_windows
        yields SPD -- lets the driver rebuild per-segment prob streams cheaply."""
        return list(self._plan(patient_id)[2].windows)

    # -- the stream --------------------------------------------------------
    def iter_windows(self, patient_id: str) -> Iterator[Tuple[np.ndarray, int]]:
        _, tl, ws = self._plan(patient_id)
        op = self._op()
        for w in ws.windows:
            mf = tl.files_by_name[w.file_name]
            x = self._filtered(patient_id, w.file_name)
            off = w.offset_samples - mf.seg_offset_samples
            seg = x[:, off:off + w.n_samples]
            if seg.shape[1] != w.n_samples:
                log.warning("provider %s: short window %s@%d -> skipped",
                            patient_id, w.file_name, off)
                continue
            sset = spd.spd_from_window(seg, alpha=self.alpha, operator=op,
                                       channels=self.channels(), keep_symbols=False)
            yield sset.matrices, int(w.label)


# ===========================================================================
# Driver
# ===========================================================================
@dataclass
class LopoResult:
    alpha: float
    span_roof: int
    sop_minutes: int
    classifier_name: str
    target_fpr_per_hour: float
    per_patient: List[Dict[str, object]]
    summary: Dict[str, object]
    thresholds: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "alpha": self.alpha, "span_roof": self.span_roof,
            "sop_minutes": self.sop_minutes, "classifier_name": self.classifier_name,
            "target_fpr_per_hour": self.target_fpr_per_hour,
            "summary": self.summary, "thresholds": self.thresholds,
            "per_patient": self.per_patient,
        }


def _alpha_fingerprint(alpha: float) -> str:
    """dataset_builder fingerprint tagged with alpha (keeps fold caches separate)."""
    return f"{db._fingerprint()}_a{float(alpha):.4f}"


def _split_by_segment(seg_indices, probs, labels):
    """Group a chronological prob/label stream into contiguous same-segment runs."""
    prob_segs: List[list] = []
    lab_segs: List[list] = []
    sentinel = object()
    cur = sentinel
    for s, p, l in zip(seg_indices, probs, labels):
        if s != cur:
            prob_segs.append([])
            lab_segs.append([])
            cur = s
        prob_segs[-1].append(float(p))
        lab_segs[-1].append(int(l))
    return ([np.asarray(a, dtype=float) for a in prob_segs],
            [np.asarray(a, dtype=int) for a in lab_segs])


def _segment_indices(provider, patient_id: str, n_rows: int) -> List[int]:
    """Per-row segment index in iter_windows order; falls back to one segment."""
    meta = getattr(provider, "window_meta", None)
    if meta is None:
        return [0] * n_rows
    windows = meta(patient_id)
    if len(windows) != n_rows:
        log.warning("provider %s: window_meta len %d != rows %d -> single segment",
                    patient_id, len(windows), n_rows)
        return [0] * n_rows
    return [int(w.seg_index) for w in windows]


def run_lopo(*, alpha: float, span_roof: Optional[int] = None,
             provider=None, patients: Optional[Sequence[str]] = None,
             raw_dir: Optional[Path] = None, sop_minutes: Optional[int] = None,
             classifier_name: Optional[str] = None,
             target_fpr_per_hour: Optional[float] = None,
             undersample_method: Optional[str] = None,
             oversample_method: Optional[str] = None,
             seed: Optional[int] = None,
             window_threshold: float = 0.5,
             mode: str = "exact") -> LopoResult:
    """Run one full LOPO experiment at (alpha, span_roof) and aggregate metrics.

    Provide an SPD `provider` to inject data (tests / custom loaders); otherwise
    a ChbSpdProvider is built over `patients` (default: chb01..chb24) reading
    EDFs from `raw_dir` (default: cfg.RAW_DIR).
    """
    span_roof = cfg.SPAN_MAX if span_roof is None else int(span_roof)
    sop_minutes = int(cfg.SOP_PRIMARY_MINUTES if sop_minutes is None else sop_minutes)
    cname = classifier_name or cfg.PRIMARY_CLASSIFIER
    target = float(cfg.PRIMARY_TARGET_FPR_PER_HOUR
                   if target_fpr_per_hour is None else target_fpr_per_hour)
    seed = cfg.SEED if seed is None else int(seed)
    mode = str(mode).lower()
    if mode not in ("exact", "fast"):
        raise ValueError(f"mode must be 'exact' or 'fast', got {mode!r}")

    if provider is None:
        pats = tuple(patients) if patients else DEFAULT_PATIENTS
        provider = ChbSpdProvider(pats, alpha=alpha, raw_dir=raw_dir,
                                  sop_minutes=sop_minutes)
    patient_ids = list(provider.patient_ids())
    if len(patient_ids) < 2:
        raise ValueError("LOPO needs >= 2 patients")

    stride = cfg.STRIDE_SECONDS
    fp_windows = alarms_mod.firing_power_window_count(sop_minutes=sop_minutes,
                                                      stride_seconds=stride)
    refr_seconds = cfg.SPH_SECONDS + sop_minutes * 60
    refractory_windows = alarms_mod.refractory_window_count(
        refractory_seconds=refr_seconds, stride_seconds=stride)
    fp = _alpha_fingerprint(alpha)

    bkw = {}
    if undersample_method:
        bkw["undersample_method"] = undersample_method
    if oversample_method:
        bkw["oversample_method"] = oversample_method

    per_patient: List[Dict[str, object]] = []
    thresholds: Dict[str, float] = {}
    preds_list: List[metrics_mod.PatientPredictions] = []

    log.info("run_lopo: alpha=%.2f span_roof=%d mode=%s patients=%d",
             float(alpha), span_roof, mode, len(patient_ids))
    for test_patient in patient_ids:
        fold = db.build_fold(provider, test_patient, span_roof=span_roof,
                             fingerprint=fp, fast=(mode == "fast"), alpha=alpha)
        y_train = np.asarray(fold.y_train)
        if fold.X_train.shape[0] == 0 or np.unique(y_train).size < 2:
            log.warning("skip fold %s: degenerate training set", test_patient)
            continue
        if fold.X_test.shape[0] == 0:
            log.warning("skip fold %s: no test windows", test_patient)
            continue

        # 1) balance TRAIN only, 2) fit classifier
        X_bal, y_bal = balancing.balance(fold.X_train, y_train, seed=seed, **bkw)
        model = clf.make(cname)
        model.fit(X_bal, y_bal)

        # 3) Option-2 calibration on the (unbalanced) TRAIN pool, per segment
        cal_prob_segs: List[np.ndarray] = []
        cal_lab_segs: List[np.ndarray] = []
        gids = np.asarray(fold.train_patient_ids)
        for src in fold.source_patients:
            mask = gids == src
            if not np.any(mask):
                continue
            probs_src = clf.predict_preictal_proba(model, fold.X_train[mask])
            seg_idx = _segment_indices(provider, src, int(mask.sum()))
            ps, ls = _split_by_segment(seg_idx, probs_src, y_train[mask])
            cal_prob_segs.extend(ps)
            cal_lab_segs.extend(ls)
        thr, achieved = alarms_mod.calibrate_threshold(
            cal_prob_segs, cal_lab_segs, target_fpr_per_hour=target,
            fp_windows=fp_windows, refractory_windows=refractory_windows,
            stride_seconds=stride)
        thresholds[test_patient] = float(thr)

        # 4) score the held-out patient (natural distribution), per segment
        probs_test = clf.predict_preictal_proba(model, fold.X_test)
        y_test = np.asarray(fold.y_test)
        seg_idx_t = _segment_indices(provider, test_patient, y_test.shape[0])
        ps_t, ls_t = _split_by_segment(seg_idx_t, probs_test, y_test)
        preds = metrics_mod.PatientPredictions(test_patient, ps_t, ls_t,
                                               stride_seconds=stride)
        rep = metrics_mod.evaluate_patient(
            preds, threshold=thr, fp_windows=fp_windows,
            refractory_windows=refractory_windows, window_threshold=window_threshold)
        rep["threshold"] = float(thr)
        rep["achieved_train_fpr_per_hour"] = float(achieved)
        rep["n_train"] = int(X_bal.shape[0])
        rep["n_test"] = int(fold.X_test.shape[0])
        per_patient.append(rep)
        preds_list.append(preds)
        log.info("fold %s: auc=%.3f event_sens=%.3f fpr/h=%.3f thr=%.3f",
                 test_patient, rep["auc"], rep["event"]["sensitivity"], #type: ignore
                 rep["event"]["fpr_per_hour"], thr) #type: ignore

    summary = _summarize(per_patient, preds_list)
    log.info("run_lopo alpha=%.2f m=%d: pooled_auc=%.3f mean_event_sens=%.3f fpr/h=%.3f",
             alpha, span_roof, summary["pooled_auc"],
             summary["mean_event_sensitivity"], summary["pooled_fpr_per_hour"])
    return LopoResult(float(alpha), span_roof, sop_minutes, cname, target,
                      per_patient, summary, thresholds)


def _nanmean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if x == x]  # drop NaN
    return float(np.mean(vals)) if vals else float("nan")


def _summarize(per_patient, preds_list) -> Dict[str, object]:
    if preds_list:
        all_y = np.concatenate([p.all_labels() for p in preds_list])
        all_s = np.concatenate([p.all_scores() for p in preds_list])
        pooled_auc = metrics_mod.roc_auc(all_y, all_s) if all_y.size else float("nan")
    else:
        pooled_auc = float("nan")
    tot_false = sum(r["event"]["false_alarms"] for r in per_patient)
    tot_hours = sum(r["event"]["interictal_hours"] for r in per_patient)
    n_events = sum(r["event"]["n_events"] for r in per_patient)
    n_pred = sum(r["event"]["n_predicted"] for r in per_patient)
    return {
        "n_patients": len(per_patient),
        "pooled_auc": pooled_auc,
        "mean_patient_auc": _nanmean([r["auc"] for r in per_patient]),
        "mean_event_sensitivity": _nanmean([r["event"]["sensitivity"] for r in per_patient]),
        "pooled_event_sensitivity": (n_pred / n_events) if n_events else float("nan"),
        "n_events": n_events, "n_predicted": n_pred,
        "pooled_fpr_per_hour": (tot_false / tot_hours) if tot_hours > 0 else float("nan"),
        "total_false_alarms": tot_false, "total_interictal_hours": tot_hours,
        "mean_warning_seconds": _nanmean([r["event"]["mean_warning_seconds"] for r in per_patient]),
    }


# ---------------------------------------------------------------------------
# Self-test (dual-mode: full path with sklearn+imblearn, else structural only)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    import shutil
    from src.data import cache

    print("Running lopo.py self-test ...\n")

    # --- pure helpers (always) ---
    ps, ls = _split_by_segment([0, 0, 1, 1, 1], [.1, .2, .3, .4, .5], [0, 1, 0, 1, 0])
    assert len(ps) == 2 and ps[0].shape == (2,) and ps[1].shape == (3,)
    assert list(ls[1]) == [0, 1, 0]
    assert _alpha_fingerprint(0.0) != _alpha_fingerprint(1.0)

    # --- in-memory SPD provider (no EDF / mne / scipy) ---
    rng = np.random.default_rng(cfg.SEED)
    nc, sr, dim = 3, 3, 4
    channels = tuple(f"c{i}" for i in range(nc))
    eye = np.eye(dim)

    def _spd(shift=0.0):
        A = rng.standard_normal((nc, sr, dim, dim))
        return A @ np.swapaxes(A, -1, -2) + (dim + shift) * eye

    class _Meta:
        __slots__ = ("seg_index", "label")
        def __init__(self, seg_index, label):
            self.seg_index = seg_index
            self.label = label

    class _MemProvider:
        def __init__(self, data, channels):
            self._data = data
            self._channels = tuple(channels)
        def patient_ids(self):
            return list(self._data)
        def channels(self):
            return self._channels
        def iter_windows(self, patient_id):
            for C, lab, _seg in self._data[patient_id]:
                yield np.array(C, dtype=float), int(lab)
        def window_meta(self, patient_id):
            return [_Meta(seg, lab) for _C, lab, seg in self._data[patient_id]]

    data = {}
    for p in ("A", "B", "C", "D"):
        windows = []
        windows += [(_spd(), 0, 0) for _ in range(6)]
        windows += [(_spd(1.0), 1, 0) for _ in range(3)]
        windows += [(_spd(), 0, 1) for _ in range(6)]
        windows += [(_spd(1.0), 1, 1) for _ in range(3)]
        data[p] = windows
    provider = _MemProvider(data, channels)

    assert isinstance(provider, db.SpdWindowProvider)
    assert len(provider.window_meta("A")) == 18

    cfg.CACHE_DIR = Path(tempfile.mkdtemp(prefix="sh_lopo_test_"))
    cfg.CACHE_ENABLED = True
    cache._MEM.clear()

    # build_fold is numpy-only -> testable without sklearn/imblearn
    fold = db.build_fold(provider, "A", span_roof=sr, fingerprint=_alpha_fingerprint(0.5))
    assert fold.X_test.shape[0] == len(provider.window_meta("A"))
    assert fold.X_train.shape[1] == nc * sr * cfg.N_REFERENCES

    if HAVE_SKLEARN and HAVE_IMBLEARN:
        res = run_lopo(provider=provider, alpha=0.5, span_roof=sr,
                       target_fpr_per_hour=1e9, seed=cfg.SEED)
        assert res.summary["n_patients"] == 4, res.summary
        assert set(res.thresholds) == set(provider.patient_ids())
        for r in res.per_patient:
            assert "event" in r and "window" in r
            assert 0.0 <= r["threshold"] <= 1.0 # type: ignore
            assert r["event"]["n_events"] >= 1 # type: ignore
        pa = res.summary["pooled_auc"]
        assert (pa != pa) or (0.0 <= pa <= 1.0) # type: ignore
        d = res.to_dict()
        assert d["span_roof"] == sr and "summary" in d
        print("  full integration OK (sklearn + imbalanced-learn present)")
    else:
        print("  sklearn/imbalanced-learn absent -> structural checks only")

    shutil.rmtree(cfg.CACHE_DIR, ignore_errors=True)
    print("\nOK - lopo.py self-test passed.")
