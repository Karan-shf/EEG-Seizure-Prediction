"""
references.py
=============
Stage 4 / Req E: the population reference anchors.

Each window's feature vector is its AIRM distance to THREE references (per
channel x span):

    baseline   = I                     (identity; distance = deviation from baseline)
    interictal = M_interictal          (population interictal mean)
    preictal   = M_preictal            (population preictal mean)

The two population anchors are built with a TWO-LEVEL Frechet mean
(ANCHOR_ESTIMATOR = "two_level"):

  Level 1 (per patient, fold-INVARIANT, cached): for each patient p, the AIRM
           Frechet mean of that patient's RECENTERED interictal C' matrices ->
           M_interictal^(p); likewise M_preictal^(p). One matrix per
           (channel x span). These never change across folds, so they are
           cached (CACHE_FOLD_INVARIANT contains "patient_anchor_means").

  Level 2 (per fold): the AIRM Frechet mean ACROSS the source patients' level-1
           means, with EQUAL WEIGHT per patient (so a patient with many windows
           does not dominate). Recomputed each fold because the source set
           changes (ANCHOR_SCOPE = "train_fold_only": the held-out patient is
           excluded).

Everything here operates on RECENTERED matrices (post alignment.recenter), so the
anchors live in the same patient-neutral frame as the windows they will be
compared against. The heavy lifting (Frechet mean) is backend.frechet_mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import config as cfg
from src.utils.logger import get_logger
from src.features import backend as bk

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Level 1: per-patient means (fold-invariant, cached)
# ---------------------------------------------------------------------------
@dataclass
class PatientReferenceMeans:
    """Per-(channel x span) interictal/preictal Frechet means for one patient."""
    patient_id: str
    channels: tuple[str, ...]
    interictal_mean: np.ndarray     # (n_channels, span_roof, dim, dim)
    preictal_mean: np.ndarray       # (n_channels, span_roof, dim, dim)

    @property
    def span_roof(self) -> int:
        return self.interictal_mean.shape[1]

    @property
    def dim(self) -> int:
        return self.interictal_mean.shape[-1]


def patient_reference_means(
    patient_id: str,
    interictal_cp: np.ndarray,
    preictal_cp: np.ndarray,
    *,
    channels: tuple[str, ...],
    max_iter: int | None = None,
    tol: float | None = None,
) -> PatientReferenceMeans:
    """Level-1 anchors for one patient from that patient's RECENTERED C' stacks.

    interictal_cp : (n_interictal_windows, n_channels, span_roof, dim, dim)
    preictal_cp   : (n_preictal_windows,   n_channels, span_roof, dim, dim)
    Both must already be recentered (alignment.recenter). Returns the per-
    (channel x span) AIRM Frechet mean of each class. Fold-invariant -> cache it.
    """
    inter = np.asarray(interictal_cp, dtype=float)
    pre = np.asarray(preictal_cp, dtype=float)
    if inter.ndim != 5 or pre.ndim != 5:
        raise ValueError(
            "expected (n_windows, n_channels, span_roof, dim, dim) for both "
            f"classes, got interictal {inter.shape}, preictal {pre.shape}"
        )
    if inter.shape[1:] != pre.shape[1:]:
        raise ValueError(
            f"interictal/preictal (channel, span, dim) mismatch: "
            f"{inter.shape[1:]} vs {pre.shape[1:]}"
        )
    m_int = bk.frechet_mean(inter, axis=0, max_iter=max_iter, tol=tol)
    m_pre = bk.frechet_mean(pre, axis=0, max_iter=max_iter, tol=tol)
    log.info(
        "patient_reference_means[%s]: interictal from %d, preictal from %d windows",
        patient_id, inter.shape[0], pre.shape[0],
    )
    return PatientReferenceMeans(
        patient_id=patient_id, channels=tuple(channels),
        interictal_mean=m_int, preictal_mean=m_pre,
    )


# ---------------------------------------------------------------------------
# Level 2: per-fold population references (equal weight per patient)
# ---------------------------------------------------------------------------
@dataclass
class ReferenceSet:
    """The 3 references for one fold, per (channel x span)."""
    names: tuple[str, ...]          # == cfg.REFERENCE_NAMES
    channels: tuple[str, ...]
    source_patient_ids: tuple[str, ...]
    matrices: np.ndarray            # (n_references, n_channels, span_roof, dim, dim)

    @property
    def span_roof(self) -> int:
        return self.matrices.shape[2]

    @property
    def dim(self) -> int:
        return self.matrices.shape[-1]

    def get(self, name: str) -> np.ndarray:
        return self.matrices[self.names.index(name)]


def build_fold_references(
    patient_means: list[PatientReferenceMeans],
    *,
    channels: tuple[str, ...] | None = None,
    source_patient_ids: list[str] | None = None,
    max_iter: int | None = None,
    tol: float | None = None,
) -> ReferenceSet:
    """Level-2 population references from cached per-patient level-1 means.

    Uses only `source_patient_ids` (train fold; the held-out patient is left
    out). The interictal / preictal anchors are the EQUAL-WEIGHT AIRM Frechet
    mean across the source patients' level-1 means. The baseline anchor is I.
    """
    if not patient_means:
        raise ValueError("need at least one patient's means")
    if source_patient_ids is not None:
        wanted = set(source_patient_ids)
        srcs = [pm for pm in patient_means if pm.patient_id in wanted]
        if not srcs:
            raise ValueError("no patient_means match source_patient_ids")
    else:
        srcs = list(patient_means)

    channels = tuple(channels) if channels is not None else srcs[0].channels
    shape = srcs[0].interictal_mean.shape
    for pm in srcs:
        if pm.interictal_mean.shape != shape or pm.preictal_mean.shape != shape:
            raise ValueError("all patient means must share (nc, sr, dim, dim)")

    inter_stack = np.stack([pm.interictal_mean for pm in srcs])   # (P, nc, sr, d, d)
    pre_stack = np.stack([pm.preictal_mean for pm in srcs])
    # Equal weight per patient (weights=None) -> level-2 Frechet mean.
    m_interictal = bk.frechet_mean(inter_stack, axis=0, max_iter=max_iter, tol=tol)
    m_preictal = bk.frechet_mean(pre_stack, axis=0, max_iter=max_iter, tol=tol)

    nc, sr, d, _ = shape
    eye = np.broadcast_to(np.eye(d), (nc, sr, d, d)).copy()
    by_name = {"baseline": eye, "interictal": m_interictal, "preictal": m_preictal}
    missing = [n for n in cfg.REFERENCE_NAMES if n not in by_name]
    if missing:
        raise ValueError(f"unhandled reference name(s): {missing}")
    matrices = np.stack([by_name[n] for n in cfg.REFERENCE_NAMES])  # (n_refs, nc, sr, d, d)

    log.info(
        "build_fold_references: %d source patients -> %d references %s (nc=%d, sr=%d, dim=%d)",
        len(srcs), len(cfg.REFERENCE_NAMES), cfg.REFERENCE_NAMES, nc, sr, d,
    )
    return ReferenceSet(
        names=tuple(cfg.REFERENCE_NAMES), channels=channels,
        source_patient_ids=tuple(pm.patient_id for pm in srcs), matrices=matrices,
    )


# ---------------------------------------------------------------------------
# Self-test (NumPy path; no torch / scipy / pyriemann required)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running references.py self-test ...\n")
    rng = np.random.default_rng(cfg.SEED)
    n = 5
    nc, sr = 4, 3
    channels = tuple(f"ch{i}" for i in range(nc))

    def rand_spd(lead=()):
        A = rng.standard_normal(lead + (n, n))
        return A @ np.swapaxes(A, -1, -2) + n * np.eye(n)

    eye = np.eye(n)

    # --- level 1: mean of identical windows equals that window ---
    C0 = rand_spd((nc, sr))
    inter = np.stack([C0] * 6)
    pre = np.stack([rand_spd((nc, sr)) for _ in range(4)])
    pm = patient_reference_means("p0", inter, pre, channels=channels)
    assert pm.interictal_mean.shape == (nc, sr, n, n)
    assert np.allclose(pm.interictal_mean, C0, atol=1e-8)
    assert pm.span_roof == sr and pm.dim == n

    # --- level 2: baseline is I; anchors SPD; correct shape/order ---
    pms = []
    for pid in ("p0", "p1", "p2"):
        inter_p = np.stack([rand_spd((nc, sr)) for _ in range(5)])
        pre_p = np.stack([rand_spd((nc, sr)) for _ in range(5)])
        pms.append(patient_reference_means(pid, inter_p, pre_p, channels=channels))
    R = build_fold_references(pms)
    assert R.matrices.shape == (cfg.N_REFERENCES, nc, sr, n, n)
    assert R.names == cfg.REFERENCE_NAMES
    assert np.allclose(R.get("baseline"), eye), "baseline reference must be I"
    assert np.linalg.eigvalsh(R.get("interictal")).min() > 0.0
    assert np.linalg.eigvalsh(R.get("preictal")).min() > 0.0

    # --- source filtering: single source patient -> anchors equal its means ---
    R1 = build_fold_references(pms, source_patient_ids=["p1"])
    p1 = next(pm for pm in pms if pm.patient_id == "p1")
    assert R1.source_patient_ids == ("p1",)
    assert np.allclose(R1.get("interictal"), p1.interictal_mean, atol=1e-8)
    assert np.allclose(R1.get("preictal"), p1.preictal_mean, atol=1e-8)

    # --- two-level equal weight: diagonal means -> geometric mean across patients ---
    d_vals = rng.uniform(0.5, 3.0, size=(3, nc, sr, n))
    diag_pms = []
    for k, pid in enumerate(("a", "b", "c")):
        M = np.zeros((nc, sr, n, n))
        for i in range(nc):
            for j in range(sr):
                M[i, j] = np.diag(d_vals[k, i, j])
        diag_pms.append(PatientReferenceMeans(pid, channels, M, M))
    Rd = build_fold_references(diag_pms)
    expected_diag = np.exp(np.mean(np.log(d_vals), axis=0))         # (nc, sr, n)
    got = np.diagonal(Rd.get("interictal"), axis1=-2, axis2=-1)      # (nc, sr, n)
    assert np.allclose(got, expected_diag, atol=1e-7)

    print("OK - references.py self-test passed.")
