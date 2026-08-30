"""
dataset_builder.py
==================
Stage 5/6: assemble per-fold feature matrices for the LOPO experiment.

This is the wiring that turns streamed SPD windows into the flat
`feature_dim(m)` design matrices `(X_train, y_train)` / `(X_test, y_test)` for
each leave-one-patient-out fold, while respecting two hard constraints:

  1. **Never persist the dense SPD** (`cfg.CACHE_DENSE_SPD = False`). A single
     window's SPD tensor is ~42 MB at (18 x 9 x 180 x 180); we stream one window
     at a time, reduce it to its 3 reference distances, and discard it.
  2. **Cache only the fold-invariant artifacts** (`cfg.CACHE_FOLD_INVARIANT`):
     `g_patient`, `patient_anchor_means` (level-1), and `d_baseline`. These are
     computed ONCE per patient and reused across all folds; only the per-fold
     population distances (to `M_interictal` / `M_preictal`) are recomputed each
     fold, since those anchors change with the source set.

SPD source (dependency injection)
---------------------------------
The raw EDF -> filter -> window -> label -> SPD stack path is injected as a
`SpdWindowProvider`, so this module is testable without EDF data and stays
decoupled from I/O. A provider must be **re-iterable** (each `iter_windows`
call re-streams that patient's windows) because the streaming Frechet mean makes
multiple passes. In production the provider wraps `spd.spd_from_window`; the
self-test injects a small in-memory provider.

Threshold calibration (Option 2, current)
-----------------------------------------
This builder produces raw (unbalanced, unstandardized) fold matrices. The alarm
operating point is calibrated on the TRAINING pool downstream (Option 2). Each
fold also carries `train_patient_ids`, so a later upgrade to nested
source-patient-out threshold selection (Option 1) needs no changes here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol, runtime_checkable

import numpy as np

from src import config as cfg
from src.utils.logger import get_logger
from src.features import backend as bk
from src.features import alignment as al
from src.features import references as refs
from src.features import distances as dist
from src.data import cache

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Provider contract
# ---------------------------------------------------------------------------
@runtime_checkable
class SpdWindowProvider(Protocol):
    """Streams RAW SPD windows per patient. Must be re-iterable."""
    def patient_ids(self) -> list[str]: ...
    def channels(self) -> tuple[str, ...]: ...
    def iter_windows(self, patient_id: str) -> Iterator[tuple[np.ndarray, int]]:
        """Yield (C (n_channels, span_roof, dim, dim) RAW SPD, label in {0,1})."""
        ...


# ---------------------------------------------------------------------------
# Config fingerprint (invalidates stale caches)
# ---------------------------------------------------------------------------
def _fingerprint() -> str:
    parts = (
        cfg.SPD_DIM, cfg.N_CHANNELS, cfg.SPAN_MAX,
        cfg.RECENTER_METHOD, cfg.RECENTER_GRANULARITY, cfg.RECENTER_ANCHOR_STATE,
        cfg.RIEMANN_METRIC, cfg.RIEMANN_MEAN_MAX_ITER, cfg.RIEMANN_MEAN_TOL, cfg.ANCHOR_MEAN_METHOD,
        cfg.N_REFERENCES, tuple(cfg.REFERENCE_NAMES), tuple(cfg.CHANNELS),
    )
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Streaming AIRM Frechet mean (bounded memory; multi-pass Karcher)
# ---------------------------------------------------------------------------
def streaming_frechet_mean(
    stream_factory: Callable[[], Iterator[np.ndarray]],
    *, max_iter: int | None = None, tol: float | None = None,
) -> np.ndarray:
    """AIRM Frechet mean over a re-iterable stream of SPD matrices.

    Each element is (..., dim, dim) (here (n_channels, span_roof, dim, dim)).
    Holds only the running mean + one window in memory; re-streams each Karcher
    iteration. Mirrors backend.frechet_mean but never materializes the stack.
    """
    max_iter = cfg.RIEMANN_MEAN_MAX_ITER if max_iter is None else max_iter
    tol = cfg.RIEMANN_MEAN_TOL if tol is None else tol

    mean = None
    count = 0
    for C in stream_factory():
        C = np.asarray(C, dtype=float)
        mean = C.copy() if mean is None else mean + C
        count += 1
    if count == 0:
        raise ValueError("empty stream for Frechet mean")
    assert mean is not None
    mean = bk.symmetrize(mean / count)
    if count == 1:
        return mean

    iter_step = None
    for iteration in range(int(max_iter)):
        m_sqrt = bk.spd_sqrt(mean)
        m_invsqrt = bk.spd_invsqrt(mean)
        tangent = np.zeros_like(mean)
        n = 0
        for C in stream_factory():
            C = np.asarray(C, dtype=float)
            w = bk.symmetrize(m_invsqrt @ C @ m_invsqrt)
            tangent = tangent + bk.spd_log(w)
            n += 1
        tangent = tangent / n
        mean = bk.symmetrize(m_sqrt @ bk.spd_exp(tangent) @ m_sqrt)
        iter_step = float(np.sqrt(np.sum(tangent * tangent, axis=(-2, -1))).max())
        log.info("streaming_frechet_mean: iter=%d n_windows=%d step=%.3e", iteration, n, iter_step)
        if iter_step < tol:
            break
    else:
        log.warning("streaming_frechet_mean: hit max_iter=%d without converging "
                    "(final step=%.3e >= tol=%.3e)", max_iter, iter_step, tol)
    return mean


def streaming_log_euclidean_mean(
    stream_factory: Callable[[], Iterator[np.ndarray]],
) -> np.ndarray:
    """Log-Euclidean mean over a re-iterable stream of SPD matrices -- ONE PASS.

    Accumulates the running sum of each window's matrix log, divides by count,
    exponentiates once at the end. Unlike streaming_frechet_mean (AIRM Karcher,
    which re-streams -- and therefore re-loads / re-filters / re-featurizes --
    every window once per iteration, up to RIEMANN_MEAN_MAX_ITER times), this
    never re-streams. This is what fixes both the redundant rsmmtn/spd
    recomputation AND the signal LRU-cache thrashing the multi-pass design
    caused.
    """
    log_sum = None
    count = 0
    for C in stream_factory():
        C = np.asarray(C, dtype=float)
        logC = bk.spd_log(C)
        log_sum = logC.copy() if log_sum is None else log_sum + logC
        count += 1
    if count == 0:
        raise ValueError("empty stream for log-Euclidean mean")
    assert log_sum is not None
    mean_log = log_sum / count
    return bk.symmetrize(bk.spd_exp(mean_log))


# ---------------------------------------------------------------------------
# Per-patient fold-invariant artifacts (cached)
# ---------------------------------------------------------------------------
def _interictal(provider, pid):
    return (C for (C, lab) in provider.iter_windows(pid) if lab == 0)


def _preictal(provider, pid):
    return (C for (C, lab) in provider.iter_windows(pid) if lab == 1)


def patient_g(provider: SpdWindowProvider, pid: str, *, fingerprint=None) -> np.ndarray:
    """g_patient: interictal Frechet mean, per (channel x span). Fold-invariant."""
    fp = fingerprint or _fingerprint()

    # --- AIRM Karcher mean (original, multi-pass) ---
    # return cache.get_or_compute(
    #     "g_patient", pid,
    #     lambda: streaming_frechet_mean(lambda: _interictal(provider, pid)),
    #     fingerprint=fp,
    # )
    return cache.get_or_compute(
        "g_patient", pid,
        lambda: streaming_log_euclidean_mean(lambda: _interictal(provider, pid)),
        fingerprint=fp,
    )


@dataclass
class PatientBaseline:
    patient_id: str
    labels: np.ndarray          # (n_windows,)
    d_baseline: np.ndarray      # (n_windows, n_channels, span_roof)


def _pass2_artifacts(provider, pid, *, fingerprint=None):
    """The combined level-1 second pass: ONE stream over the patient's windows
    that builds BOTH d_baseline AND both (interictal/preictal) anchor means
    together, instead of the original 3 independent re-streams (d_baseline
    alone, an interictal-only filtered stream, a preictal-only filtered
    stream) each separately re-paying rsmmtn+spd+recenter. Each window's
    recentered C' is computed exactly once and reused for everything it's
    needed for -- which also means no window's front-end work is ever wasted
    computing a label it doesn't end up using (the old filtered streams paid
    the full rsmmtn+spd cost on every window BEFORE checking its label).

    Populates the SAME two on-disk cache kinds ("d_baseline",
    "patient_anchor_means") the previous independent functions used, so the
    disk layout, fingerprint behavior, and existing self-test assertions are
    all unchanged -- only the computation that fills them is merged.
    """
    fp = fingerprint or _fingerprint()

    cached_baseline = (cache.load("d_baseline", pid, fingerprint=fp)
                       if cache.has("d_baseline", pid, fingerprint=fp) else None)
    cached_means = (cache.load("patient_anchor_means", pid, fingerprint=fp)
                    if cache.has("patient_anchor_means", pid, fingerprint=fp) else None)
    if cached_baseline is not None and cached_means is not None:
        return cached_baseline, cached_means

    g_invsqrt = bk.spd_invsqrt(patient_g(provider, pid, fingerprint=fp))
    d_list, y_list = [], []
    log_sum_int = None
    log_sum_pre = None
    n_int = 0
    n_pre = 0
    for (C, lab) in provider.iter_windows(pid):
        cp = al.recenter(np.asarray(C, dtype=float), g_invsqrt=g_invsqrt)
        d_list.append(dist.baseline_distances(cp))          # (nc, sr)
        y_list.append(int(lab))
        logC = bk.spd_log(cp)                                # shared by both anchors' accumulator
        if lab == 0:
            log_sum_int = logC.copy() if log_sum_int is None else log_sum_int + logC
            n_int += 1
        else:
            log_sum_pre = logC.copy() if log_sum_pre is None else log_sum_pre + logC
            n_pre += 1

    if n_int == 0 or n_pre == 0:
        raise ValueError(f"{pid}: need >=1 interictal AND >=1 preictal window "
                          f"to build both anchors (got {n_int} interictal, "
                          f"{n_pre} preictal)")

    assert log_sum_int is not None and log_sum_pre is not None
    baseline = PatientBaseline(pid, np.asarray(y_list, dtype=int), np.stack(d_list))
    m_int = bk.symmetrize(bk.spd_exp(log_sum_int / n_int))
    m_pre = bk.symmetrize(bk.spd_exp(log_sum_pre / n_pre))
    means = refs.PatientReferenceMeans(
        patient_id=pid, channels=tuple(provider.channels()),
        interictal_mean=m_int, preictal_mean=m_pre)

    cache.save("d_baseline", pid, baseline, fingerprint=fp)
    cache.save("patient_anchor_means", pid, means, fingerprint=fp)
    return baseline, means


def patient_baseline_distances(provider, pid, *, fingerprint=None) -> PatientBaseline:
    """Per-window d_baseline = delta_R(C', I) + labels. Fold-invariant."""
    fp = fingerprint or _fingerprint()
    baseline, _means = _pass2_artifacts(provider, pid, fingerprint=fp)
    return baseline


def patient_anchor_means(provider, pid, *, fingerprint=None) -> refs.PatientReferenceMeans:
    """Level-1 interictal/preictal (Log-Euclidean) means (recentered). Fold-invariant."""
    fp = fingerprint or _fingerprint()
    _baseline, means = _pass2_artifacts(provider, pid, fingerprint=fp)
    return means


# ---------------------------------------------------------------------------
# Per-fold feature assembly
# ---------------------------------------------------------------------------
@dataclass
class FoldData:
    test_patient: str
    source_patients: tuple[str, ...]
    span_roof: int
    feature_names: list[str]
    X_train: np.ndarray
    y_train: np.ndarray
    train_patient_ids: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_patient_ids: np.ndarray

    @property
    def feature_dim(self) -> int:
        return self.X_train.shape[1]


def _patient_distance_tensor(provider, pid, references, *, fingerprint):
    """Stream one patient's windows ONCE -> (D, labels).

    D : (n_windows, n_channels, span, n_references) AIRM distances to `references`
        (in references.names order), computed at the reference's full span roof
        so any downstream span roof m is a cheap slice. The baseline column is
        read from the cached fold-invariant d_baseline; the two population
        columns are computed here against `references`. Dense SPD is discarded
        per window.
    """
    g_invsqrt = bk.spd_invsqrt(patient_g(provider, pid, fingerprint=fingerprint))
    base = patient_baseline_distances(provider, pid, fingerprint=fingerprint)
    names = references.names
    m_int = references.get("interictal")
    m_pre = references.get("preictal")

    rows = []
    idx = 0
    for (C, _lab) in provider.iter_windows(pid):
        cp = al.recenter(np.asarray(C, dtype=float), g_invsqrt=g_invsqrt)
        cols = []
        for name in names:
            if name == "baseline":
                cols.append(base.d_baseline[idx])                 # cached (nc, span)
            elif name == "interictal":
                cols.append(dist.population_distances(cp, m_int))  # (nc, span)
            elif name == "preictal":
                cols.append(dist.population_distances(cp, m_pre))  # (nc, span)
            else:
                raise ValueError(f"unknown reference name {name!r}")
        rows.append(np.stack(cols, axis=-1))                      # (nc, span, n_refs)
        idx += 1

    if rows:
        D = np.stack(rows)                                        # (nw, nc, span, n_refs)
    else:
        nc = len(references.channels)
        D = np.zeros((0, nc, references.span_roof, len(names)), dtype=float)
    return D, base.labels.copy()


def _slice_features(D: np.ndarray, span_roof: int) -> np.ndarray:
    """Slice a (nw, nc, span, n_refs) distance tensor to span roof m and flatten
    to (nw, feature_dim(m)) in canonical channel-major / span / reference order."""
    D = np.asarray(D, dtype=float)
    nw, nc, _sr, n_refs = D.shape
    return D[:, :, :span_roof, :].reshape(nw, nc * span_roof * n_refs)


def _patient_features(provider, pid, references, span_roof, *, fingerprint):
    """Stream one patient's windows -> (X (n_windows, feature_dim(m)), y).

    Thin wrapper over `_patient_distance_tensor` + `_slice_features`; the EXACT
    (per-fold anchor) path. Numerically identical to the pre-refactor version.
    """
    D, y = _patient_distance_tensor(provider, pid, references, fingerprint=fingerprint)
    return _slice_features(D, span_roof), y


# ---------------------------------------------------------------------------
# Fast (Tier-1 "purist") feature caching
# ---------------------------------------------------------------------------
# TRAIN features are computed against a GLOBAL anchor (Frechet mean over ALL
# patients' level-1 means) that does not depend on which patient is held out, so
# each patient is streamed & cached exactly once. The HELD-OUT patient's
# features use the exact source-only leave-one-out anchor (identical to exact
# mode), so nothing about the held-out patient leaks into its own features. Only
# the held-out patient is re-streamed per fold. See docs/27_fast_anchor_mode.md.
def _cohort_hash(patient_ids) -> str:
    """Stable 8-hex digest of a patient set (order-independent). Part of the
    fast-mode cache key so a DIFFERENT cohort never reuses another cohort's
    global anchor / features."""
    key = ",".join(sorted(str(p) for p in patient_ids))
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def _fast_tag(patient_ids, alpha: float) -> str:
    """Cache-key tag binding BOTH alpha and the cohort composition."""
    return f"a{float(alpha):.4f}_c{_cohort_hash(patient_ids)}"


def global_references(provider, patient_ids, *, fingerprint, tag):
    """Level-2 population anchors over ALL `patient_ids` (the fold-INVARIANT
    GLOBAL reference set used for TRAIN features in fast mode). Cached per
    (cohort, alpha) via `tag`."""
    ids = list(patient_ids)

    def compute():
        pms = [patient_anchor_means(provider, p, fingerprint=fingerprint) for p in ids]
        return refs.build_fold_references(
            pms, channels=tuple(provider.channels()), source_patient_ids=ids)

    return cache.get_or_compute("global_anchor", f"anchor_{tag}", compute,
                                fingerprint=fingerprint)


def patient_global_features(provider, pid, *, references, fingerprint, tag):
    """Per-window distance tensor for `pid` vs the GLOBAL anchor, cached per
    (patient, cohort, alpha). Streams `pid` at most once, ever."""
    def compute():
        D, y = _patient_distance_tensor(provider, pid, references, fingerprint=fingerprint)
        return {"D": D, "y": y}

    return cache.get_or_compute("global_features", f"{pid}_{tag}", compute,
                                fingerprint=fingerprint)


def patient_loo_features(provider, test_patient, *, references, fingerprint, tag):
    """Per-window distance tensor for the HELD-OUT patient vs its source-only
    leave-one-out anchor -> leakage-free TEST features. Cached per
    (patient, cohort, alpha); recomputed once per fold on a cold cache."""
    def compute():
        D, y = _patient_distance_tensor(provider, test_patient, references,
                                        fingerprint=fingerprint)
        return {"D": D, "y": y}

    return cache.get_or_compute("loo_features", f"{test_patient}_{tag}", compute,
                                fingerprint=fingerprint)


def build_fold(provider: SpdWindowProvider, test_patient: str, *,
               span_roof: int | None = None, fingerprint: str | None = None,
               fast: bool = False, alpha: float | None = None) -> FoldData:
    """Build one LOPO fold: held-out = test_patient, source = all others.

    fast=False (default, "exact"): per-fold leave-one-out level-2 anchors for
        EVERY patient (train + test) -- the original behavior; every patient is
        re-streamed each fold.
    fast=True ("purist", needs `alpha`): TRAIN features use a GLOBAL anchor over
        ALL patients, computed & cached ONCE per (cohort, alpha); the HELD-OUT
        patient's features use the exact source-only leave-one-out anchor
        (leakage-free, bit-identical to exact mode). Only the held-out patient
        is re-streamed per fold. See docs/27_fast_anchor_mode.md.
    """
    fp = fingerprint or _fingerprint()
    ids = list(provider.patient_ids())
    if test_patient not in ids:
        raise ValueError(f"unknown test patient {test_patient!r}")
    span_roof = cfg.SPAN_MAX if span_roof is None else int(span_roof)
    source = [p for p in ids if p != test_patient]
    if not source:
        raise ValueError("need >= 2 patients for a LOPO fold")
    channels = tuple(provider.channels())

    if fast:
        if alpha is None:
            raise ValueError("fast mode needs `alpha` for the feature-cache key")
        tag = _fast_tag(ids, alpha)

        # TRAIN: cached features vs the GLOBAL anchor (built over ALL patients).
        R_global = global_references(provider, ids, fingerprint=fp, tag=tag)
        Xtr, ytr, gtr = [], [], []
        for p in source:
            gf = patient_global_features(provider, p, references=R_global,
                                         fingerprint=fp, tag=tag)
            Xtr.append(_slice_features(gf["D"], span_roof))
            ytr.append(gf["y"])
            gtr.append(np.full(len(gf["y"]), p, dtype=object))

        # TEST: exact source-only leave-one-out anchor -> no test-side leakage.
        src_means = [patient_anchor_means(provider, p, fingerprint=fp) for p in source]
        R_loo = refs.build_fold_references(
            src_means, channels=channels, source_patient_ids=source)
        lf = patient_loo_features(provider, test_patient, references=R_loo,
                                  fingerprint=fp, tag=tag)
        Xte = _slice_features(lf["D"], span_roof)
        yte = lf["y"]
    else:
        # Level-1 means for source patients (cached), then per-fold level-2 anchors.
        pms = [patient_anchor_means(provider, p, fingerprint=fp) for p in source]
        references = refs.build_fold_references(
            pms, channels=channels, source_patient_ids=source)

        Xtr, ytr, gtr = [], [], []
        for p in source:
            X, y = _patient_features(provider, p, references, span_roof, fingerprint=fp)
            Xtr.append(X)
            ytr.append(y)
            gtr.append(np.full(len(y), p, dtype=object))
        Xte, yte = _patient_features(provider, test_patient, references, span_roof, fingerprint=fp)

    fnames = dist.feature_names(channels, span_roof)
    width = cfg.N_REFERENCES * span_roof * len(channels)
    if len(channels) == cfg.N_CHANNELS and width != cfg.feature_dim(span_roof):
        raise AssertionError("feature width disagrees with cfg.feature_dim")

    return FoldData(
        test_patient=test_patient, source_patients=tuple(source),
        span_roof=span_roof, feature_names=fnames,
        X_train=np.concatenate(Xtr) if Xtr else np.empty((0, width)),
        y_train=np.concatenate(ytr) if ytr else np.empty((0,), dtype=int),
        train_patient_ids=np.concatenate(gtr) if gtr else np.empty((0,), dtype=object),
        X_test=Xte, y_test=yte,
        test_patient_ids=np.full(len(yte), test_patient, dtype=object),
    )


def build_lopo(provider: SpdWindowProvider, *, span_roof: int | None = None) -> list[FoldData]:
    """Build every LOPO fold (one per patient)."""
    fp = _fingerprint()
    folds = [build_fold(provider, p, span_roof=span_roof, fingerprint=fp)
             for p in provider.patient_ids()]
    log.info("build_lopo: %d folds, span_roof=%s", len(folds),
             cfg.SPAN_MAX if span_roof is None else span_roof)
    return folds


# ---------------------------------------------------------------------------
# Self-test (NumPy path; in-memory provider, temp cache dir)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running dataset_builder.py self-test ...\n")
    import tempfile
    from pathlib import Path

    cfg.CACHE_DIR = Path(tempfile.mkdtemp(prefix="sh_ds_test_"))
    cfg.CACHE_ENABLED = True
    cache._MEM.clear()

    rng = np.random.default_rng(cfg.SEED)
    nc, sr, dim = 3, 3, 4
    n_refs = cfg.N_REFERENCES
    channels = tuple(f"ch{i}" for i in range(nc))

    def spd(shift=0.0):
        A = rng.standard_normal((nc, sr, dim, dim))
        return A @ np.swapaxes(A, -1, -2) + (dim + shift) * np.eye(dim)

    class MemProvider:
        def __init__(self, data, channels):
            self._data = data
            self._channels = tuple(channels)
        def patient_ids(self):
            return list(self._data.keys())
        def channels(self):
            return self._channels
        def iter_windows(self, patient_id):
            for C, lab in self._data[patient_id]:
                yield np.array(C, dtype=float), int(lab)

    data = {}
    for p in ("A", "B", "C", "D"):
        ws = [(spd(), 0) for _ in range(4)] + [(spd(shift=1.0), 1) for _ in range(3)]
        data[p] = ws
    provider = MemProvider(data, channels)

    # --- streaming Frechet mean == batched backend on a small stack ---
    stack = np.stack([spd() for _ in range(5)])
    sm = streaming_frechet_mean(lambda: iter(stack))
    bm = bk.frechet_mean(stack, axis=0)
    assert np.allclose(sm, bm, atol=1e-6), np.abs(sm - bm).max()

    fp = _fingerprint()

    # --- build one fold ---
    fold = build_fold(provider, "A", span_roof=sr)
    assert fold.test_patient == "A" and "A" not in fold.source_patients
    assert fold.source_patients == ("B", "C", "D")
    width = nc * sr * n_refs
    assert fold.X_train.shape[1] == width == len(fold.feature_names)
    assert fold.X_test.shape == (7, width)
    assert fold.X_train.shape[0] == 3 * 7
    assert set(np.unique(fold.y_train)).issubset({0, 1})
    assert set(np.unique(fold.y_test)).issubset({0, 1})
    assert np.all(np.isfinite(fold.X_train)) and np.all(np.isfinite(fold.X_test))
    assert len(fold.train_patient_ids) == fold.X_train.shape[0]

    # --- fold-invariant artifacts got cached ---
    assert cache.has("g_patient", "B", fingerprint=fp)
    assert cache.has("patient_anchor_means", "B", fingerprint=fp)
    assert cache.has("d_baseline", "A", fingerprint=fp)

    # --- baseline column matches the cached d_baseline (reference index 0) ---
    base = patient_baseline_distances(provider, "A", fingerprint=fp)
    bi = cfg.REFERENCE_NAMES.index("baseline")
    D0 = fold.X_test[0].reshape(nc, sr, n_refs)
    assert np.allclose(D0[:, :, bi], base.d_baseline[0], atol=1e-8)

    # --- determinism ---
    fold2 = build_fold(provider, "A", span_roof=sr)
    assert np.allclose(fold.X_train, fold2.X_train)
    assert np.allclose(fold.X_test, fold2.X_test)

    # --- span roof narrows the width ---
    fold_m1 = build_fold(provider, "A", span_roof=1)
    assert fold_m1.X_train.shape[1] == nc * 1 * n_refs
    assert len(fold_m1.feature_names) == nc * 1 * n_refs

    # --- full LOPO: one fold per patient ---
    folds = build_lopo(provider, span_roof=1)
    assert len(folds) == len(provider.patient_ids())
    assert {f.test_patient for f in folds} == set(provider.patient_ids())

    # --- fast (Tier-1 "purist") mode ---
    # Held-out TEST features must be IDENTICAL to exact mode (same source-only
    # leave-one-out anchor); TRAIN features differ (global vs per-fold anchor).
    fold_exact = build_fold(provider, "A", span_roof=sr, fast=False)
    fold_fast = build_fold(provider, "A", span_roof=sr, fast=True, alpha=0.5)
    assert fold_fast.X_test.shape == fold_exact.X_test.shape
    assert np.allclose(fold_fast.X_test, fold_exact.X_test, atol=1e-9), \
        "fast-mode held-out features must match the exact leave-one-out anchor"
    assert fold_fast.X_train.shape == fold_exact.X_train.shape
    assert np.array_equal(fold_fast.y_test, fold_exact.y_test)
    assert np.all(np.isfinite(fold_fast.X_train)) and np.all(np.isfinite(fold_fast.X_test))
    tag05 = _fast_tag(provider.patient_ids(), 0.5)
    assert cache.has("global_anchor", f"anchor_{tag05}", fingerprint=fp)
    assert cache.has("global_features", f"B_{tag05}", fingerprint=fp)
    assert cache.has("loo_features", f"A_{tag05}", fingerprint=fp)
    # alpha is part of the cache key -> no cross-alpha collision
    assert _fast_tag(provider.patient_ids(), 0.5) != _fast_tag(provider.patient_ids(), 1.0)
    build_fold(provider, "A", span_roof=sr, fast=True, alpha=1.0)
    assert cache.has("global_features", f"B_{_fast_tag(provider.patient_ids(), 1.0)}", fingerprint=fp)
    # fast train narrows with span roof just like exact
    fold_fast_m1 = build_fold(provider, "A", span_roof=1, fast=True, alpha=0.5)
    assert fold_fast_m1.X_train.shape[1] == nc * 1 * n_refs

    import shutil
    shutil.rmtree(cfg.CACHE_DIR, ignore_errors=True)
    print("OK - dataset_builder.py self-test passed.")
