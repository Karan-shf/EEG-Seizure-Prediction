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
        cfg.RIEMANN_METRIC, cfg.RIEMANN_MEAN_MAX_ITER, cfg.RIEMANN_MEAN_TOL,
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

    for _ in range(int(max_iter)):
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
        step = float(np.sqrt(np.sum(tangent * tangent, axis=(-2, -1))).max())
        if step < tol:
            break
    return mean


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
    return cache.get_or_compute(
        "g_patient", pid,
        lambda: streaming_frechet_mean(lambda: _interictal(provider, pid)),
        fingerprint=fp,
    )


@dataclass
class PatientBaseline:
    patient_id: str
    labels: np.ndarray          # (n_windows,)
    d_baseline: np.ndarray      # (n_windows, n_channels, span_roof)


def patient_baseline_distances(provider, pid, *, fingerprint=None) -> PatientBaseline:
    """Per-window d_baseline = delta_R(C', I) + labels. Fold-invariant."""
    fp = fingerprint or _fingerprint()

    def compute():
        g_invsqrt = bk.spd_invsqrt(patient_g(provider, pid, fingerprint=fp))
        d_list, y_list = [], []
        for (C, lab) in provider.iter_windows(pid):
            cp = al.recenter(np.asarray(C, dtype=float), g_invsqrt=g_invsqrt)
            d_list.append(dist.baseline_distances(cp))    # (nc, sr)
            y_list.append(int(lab))
        return PatientBaseline(pid, np.asarray(y_list, dtype=int), np.stack(d_list))

    return cache.get_or_compute("d_baseline", pid, compute, fingerprint=fp)


def patient_anchor_means(provider, pid, *, fingerprint=None) -> refs.PatientReferenceMeans:
    """Level-1 interictal/preictal Frechet means (recentered). Fold-invariant."""
    fp = fingerprint or _fingerprint()

    def compute():
        g_invsqrt = bk.spd_invsqrt(patient_g(provider, pid, fingerprint=fp))
        m_int = streaming_frechet_mean(
            lambda: (al.recenter(np.asarray(C, dtype=float), g_invsqrt=g_invsqrt)
                     for C in _interictal(provider, pid)))
        m_pre = streaming_frechet_mean(
            lambda: (al.recenter(np.asarray(C, dtype=float), g_invsqrt=g_invsqrt)
                     for C in _preictal(provider, pid)))
        return refs.PatientReferenceMeans(
            patient_id=pid, channels=tuple(provider.channels()),
            interictal_mean=m_int, preictal_mean=m_pre)

    return cache.get_or_compute("patient_anchor_means", pid, compute, fingerprint=fp)


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


def _patient_features(provider, pid, references, span_roof, *, fingerprint):
    """Stream one patient's windows -> (X (n_windows, feature_dim), y).

    Baseline column is read from the cached fold-invariant d_baseline; the two
    population columns are recomputed here (per fold) by streaming + recentering.
    Dense SPD is discarded per window.
    """
    g_invsqrt = bk.spd_invsqrt(patient_g(provider, pid, fingerprint=fingerprint))
    base = patient_baseline_distances(provider, pid, fingerprint=fingerprint)
    m = span_roof
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
                cols.append(base.d_baseline[idx])                 # cached (nc, sr)
            elif name == "interictal":
                cols.append(dist.population_distances(cp, m_int))  # (nc, sr)
            elif name == "preictal":
                cols.append(dist.population_distances(cp, m_pre))  # (nc, sr)
            else:
                raise ValueError(f"unknown reference name {name!r}")
        D = np.stack(cols, axis=-1)[:, :m, :]                     # (nc, m, n_refs)
        rows.append(D.reshape(-1))
        idx += 1

    X = np.asarray(rows, dtype=float) if rows else np.empty((0, 0))
    return X, base.labels.copy()


def build_fold(provider: SpdWindowProvider, test_patient: str, *,
               span_roof: int | None = None, fingerprint: str | None = None) -> FoldData:
    """Build one LOPO fold: held-out = test_patient, source = all others."""
    fp = fingerprint or _fingerprint()
    ids = list(provider.patient_ids())
    if test_patient not in ids:
        raise ValueError(f"unknown test patient {test_patient!r}")
    span_roof = cfg.SPAN_MAX if span_roof is None else int(span_roof)
    source = [p for p in ids if p != test_patient]
    if not source:
        raise ValueError("need >= 2 patients for a LOPO fold")

    # Level-1 means for source patients (cached), then per-fold level-2 anchors.
    pms = [patient_anchor_means(provider, p, fingerprint=fp) for p in source]
    references = refs.build_fold_references(
        pms, channels=tuple(provider.channels()), source_patient_ids=source)

    Xtr, ytr, gtr = [], [], []
    for p in source:
        X, y = _patient_features(provider, p, references, span_roof, fingerprint=fp)
        Xtr.append(X)
        ytr.append(y)
        gtr.append(np.full(len(y), p, dtype=object))
    Xte, yte = _patient_features(provider, test_patient, references, span_roof, fingerprint=fp)

    channels = tuple(provider.channels())
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

    import shutil
    shutil.rmtree(cfg.CACHE_DIR, ignore_errors=True)
    print("OK - dataset_builder.py self-test passed.")
