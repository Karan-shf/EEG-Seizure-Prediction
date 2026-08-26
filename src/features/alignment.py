"""
alignment.py
============
Stage 4 / Req E: Riemannian recentering (the RCT analogue at 180 x 180).

Cross-patient comparison only makes sense if every patient's SPD matrices are
expressed relative to that patient's own baseline. We whiten each SPD matrix by
a baseline anchor G on the SPD manifold:

    C' = G^-1/2 . C . G^-1/2

This is a congruence transform that maps G -> I: it moves the patient's baseline
to the identity so that C' encodes deviation-from-baseline in a patient-neutral
frame. Because AIRM is affine-invariant, recentering does NOT distort geodesic
distances between two matrices sharing the same G -- it only re-references them.

Granularity (config)
--------------------
* RECENTER_METHOD        = "riemannian_recenter"
* RECENTER_GRANULARITY   = "channel_span"  -> one baseline G per (channel, span)
* RECENTER_ANCHOR_STATE  = "interictal"    -> G is label-free calibration data

So G has shape (n_channels, span_roof, dim, dim): the patient's interictal
Frechet mean, computed per (channel x span). G_patient is a fold-invariant,
per-patient quantity (cached across LOPO folds) and is formed for the held-out
patient too, from that patient's own interictal windows -- a deliberate design
choice with a deployment caveat (needs a baseline-calibration recording).

The Frechet mean and the SPD matrix functions live in backend.py; this module is
just the baseline estimator + the recentering transform.
"""

from __future__ import annotations

import numpy as np

from src import config as cfg
from src.utils.logger import get_logger
from src.features import backend as bk

log = get_logger(__name__)


def patient_baseline(interictal_spd: np.ndarray, *, max_iter: int | None = None, tol: float | None = None) -> np.ndarray:
    """Per-(channel x span) baseline anchor G from a patient's interictal SPD set.

    interictal_spd : (n_windows, n_channels, span_roof, dim, dim)
    returns G      : (n_channels, span_roof, dim, dim)  -- the AIRM Frechet mean
                     over the window axis, one anchor per (channel, span).
    """
    X = np.asarray(interictal_spd, dtype=float)
    if X.ndim != 5:
        raise ValueError(
            "expected (n_windows, n_channels, span_roof, dim, dim), got "
            f"{X.shape}"
        )
    if X.shape[0] < 1:
        raise ValueError("need at least one interictal window to form a baseline")
    G = bk.frechet_mean(X, axis=0, max_iter=max_iter, tol=tol)
    log.info(
        "patient_baseline: G from %d interictal windows -> %s (method=%s, %s)",
        X.shape[0], G.shape, cfg.RECENTER_METHOD, cfg.RECENTER_GRANULARITY,
    )
    return G


def baseline_invsqrt(G: np.ndarray) -> np.ndarray:
    """Precompute G^-1/2 once so it can be reused across many windows/labels."""
    return bk.spd_invsqrt(G)


def recenter(C: np.ndarray, G: np.ndarray | None = None, *, g_invsqrt: np.ndarray | None = None) -> np.ndarray:
    """Recenter SPD matrices to their baseline: C' = G^-1/2 . C . G^-1/2.

    C : (..., n_channels, span_roof, dim, dim) -- any number of leading dims
        (e.g. a single window (nc, sr, n, n) or a stack (nw, nc, sr, n, n)).
    G / g_invsqrt : the per-(channel x span) baseline (nc, sr, n, n). Pass
        g_invsqrt (from baseline_invsqrt) to avoid recomputing the inverse sqrt.
    The baseline dims broadcast against any extra leading (window) dims of C.
    """
    if g_invsqrt is None:
        if G is None:
            raise ValueError("provide either G or a precomputed g_invsqrt")
        g_invsqrt = baseline_invsqrt(G)
    Cp = g_invsqrt @ np.asarray(C, dtype=float) @ g_invsqrt
    return 0.5 * (Cp + np.swapaxes(Cp, -1, -2))


# ---------------------------------------------------------------------------
# Self-test (NumPy path; no torch / scipy / pyriemann required)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running alignment.py self-test ...\n")
    rng = np.random.default_rng(cfg.SEED)
    n = 5
    nc, sr = 4, 3

    def rand_spd(lead=()):
        A = rng.standard_normal(lead + (n, n))
        return A @ np.swapaxes(A, -1, -2) + n * np.eye(n)

    eye = np.eye(n)

    # --- recenter(G, G) == I, per (channel x span) ---
    G = rand_spd((nc, sr))
    I_check = recenter(G, G)
    assert I_check.shape == (nc, sr, n, n)
    assert np.allclose(I_check, eye, atol=1e-8), "recentering the baseline must give I"

    # --- recenter matches the explicit congruence and stays SPD ---
    C = rand_spd((nc, sr))
    Gih = baseline_invsqrt(G)
    Cp = recenter(C, g_invsqrt=Gih)
    assert np.allclose(Cp, Gih @ C @ Gih, atol=1e-10)
    assert np.allclose(Cp, np.swapaxes(Cp, -1, -2), atol=1e-10)
    assert np.linalg.eigvalsh(Cp).min() > 0.0

    # --- broadcasting over a window axis ---
    Cw = rand_spd((7, nc, sr))                       # (nw, nc, sr, n, n)
    Cwp = recenter(Cw, g_invsqrt=Gih)
    assert Cwp.shape == (7, nc, sr, n, n)
    assert np.allclose(Cwp[0], recenter(Cw[0], g_invsqrt=Gih), atol=1e-10)

    # --- affine invariance: recentering preserves AIRM distances ---
    C1 = rand_spd((nc, sr))
    C2 = rand_spd((nc, sr))
    d_before = bk.airm_distance(C1, C2)
    d_after = bk.airm_distance(recenter(C1, g_invsqrt=Gih), recenter(C2, g_invsqrt=Gih))
    assert np.allclose(d_before, d_after, atol=1e-7), "recenter must preserve AIRM distance"

    # --- patient_baseline from an interictal stack ---
    stack = rand_spd((12, nc, sr))                   # 12 interictal windows
    Gp = patient_baseline(stack)
    assert Gp.shape == (nc, sr, n, n)
    assert np.linalg.eigvalsh(Gp).min() > 0.0
    assert np.allclose(recenter(Gp, Gp), eye, atol=1e-8)

    # --- integration with spd/rsmmtn ---
    from src.features.spd import spd_from_window
    nchan = cfg.N_CHANNELS
    windows = [spd_from_window(rng.standard_normal((nchan, 400)), alpha=0.5,
                               n_angular=6, n_radial=4, span_max=3).matrices
               for _ in range(5)]
    stack2 = np.stack(windows)                       # (5, 18, 3, 24, 24)
    G2 = patient_baseline(stack2)
    assert G2.shape == (nchan, 3, 24, 24)
    rec = recenter(stack2, G=G2)
    assert rec.shape == (5, nchan, 3, 24, 24)
    assert np.linalg.eigvalsh(rec).min() > 0.0

    print("OK - alignment.py self-test passed.")
