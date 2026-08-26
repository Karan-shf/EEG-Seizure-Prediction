"""
spd.py
======
Stage 4 / Req C+E: the SPD-manifold lift.

Each cumulative multi-span transition network A^i from rsmmtn.py (a row-
normalized n_symbols x n_symbols matrix) is turned into a Symmetric Positive-
Definite matrix:

    C^i = w * I + A^i (A^i)^T          (w = identity_weight, 1.0 by design)

Why this construction
---------------------
* A^i (A^i)^T is symmetric positive-SEMI-definite (a Gram matrix), but it can be
  rank-deficient / singular -- a sparsely populated 180x180 transition network
  has many empty rows, so many zero eigenvalues.
* Adding I makes every eigenvalue >= 1: the result is strictly SPD and
  well-conditioned regardless of how sparse the network is. That is exactly what
  the downstream affine-invariant Riemannian metric (AIRM) needs -- it takes
  matrix logs / inverses, which require strictly positive eigenvalues.

So C^i lives on the SPD(n_symbols) manifold. The next stages recenter it per
(channel x span) to that patient's baseline (alignment.py), build population
Frechet-mean anchors (references.py), and reduce it to AIRM distances
(distances.py). This module is the single source of truth for the C = I + A A^T
construction.

Shape / cost
------------
For one window at one alpha the SPD tensor is
    (n_channels, span_roof, n_symbols, n_symbols)
identical in size to the adjacency tensor (~42 MB at 18 x 9 x 180 x 180). Like
the networks, these are STREAMED per window and never persisted; distances.py
collapses them to feature_dim(m) floats.

NumPy only and fully batched (operates on the trailing two axes), so the same
code path runs unchanged once backend.py routes matmul/eigh through a GPU
tensor library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import config as cfg
from src.utils.logger import get_logger
from src.features.rsmmtn import TransitionNetworkSet, build_transition_networks

log = get_logger(__name__)

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Core construction
# ---------------------------------------------------------------------------
def build_spd(A: np.ndarray, *, identity_weight: float = 1.0) -> np.ndarray:
    """Lift adjacency matrices to SPD: C = identity_weight * I + A A^T.

    A: (..., n, n), batched over any leading dims (the trailing two axes are the
    matrix). Returns C with the same shape. The output is exactly symmetric
    (symmetrized to kill matmul round-off) and SPD when identity_weight > 0.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim < 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError(f"expected square trailing dims (..., n, n), got {A.shape}")
    if identity_weight <= 0.0:
        raise ValueError(f"identity_weight must be > 0 for SPD, got {identity_weight}")
    n = A.shape[-1]
    gram = np.matmul(A, np.swapaxes(A, -1, -2))          # A A^T, PSD
    gram = 0.5 * (gram + np.swapaxes(gram, -1, -2))       # exact symmetry
    return gram + identity_weight * np.eye(n)


def is_spd(C: np.ndarray, *, tol: float = 1e-10) -> bool:
    """True iff C is symmetric with all eigenvalues > tol (single matrix)."""
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"expected a single (n, n) matrix, got {C.shape}")
    if not np.allclose(C, C.T, atol=1e-12):
        return False
    return bool(np.linalg.eigvalsh(C).min() > tol)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class SpdSet:
    """Per-(channel x span) SPD matrices for one window at one alpha."""
    channels: tuple[str, ...]
    alpha: float
    span_roof: int
    identity_weight: float
    matrices: np.ndarray            # (n_channels, span_roof, dim, dim)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def dim(self) -> int:
        return self.matrices.shape[-1]

    def span_slice(self, m: int) -> np.ndarray:
        """SPD matrices for span roof m: (n_channels, m, dim, dim)."""
        if not 1 <= m <= self.span_roof:
            raise ValueError(f"span roof m={m} outside [1, {self.span_roof}]")
        return self.matrices[:, :m]

    def cumulative(self, m: int) -> np.ndarray:
        """The single SPD matrix C^m: (n_channels, dim, dim)."""
        if not 1 <= m <= self.span_roof:
            raise ValueError(f"span roof m={m} outside [1, {self.span_roof}]")
        return self.matrices[:, m - 1]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def spd_from_networks(
    net: TransitionNetworkSet, *, identity_weight: float = 1.0
) -> SpdSet:
    """Lift every cumulative network in a TransitionNetworkSet to an SPD matrix."""
    matrices = build_spd(net.adjacency, identity_weight=identity_weight)
    log.info(
        "spd: alpha=%.2f -> %d channels x %d spans of %dx%d SPD (%.1f MB)",
        net.alpha, net.n_channels, net.span_roof, net.n_symbols, net.n_symbols,
        matrices.nbytes / 1e6,
    )
    return SpdSet(
        channels=net.channels,
        alpha=net.alpha,
        span_roof=net.span_roof,
        identity_weight=float(identity_weight),
        matrices=matrices,
    )


def spd_from_window(
    X: np.ndarray, *, alpha: float, identity_weight: float = 1.0, **kwargs
) -> SpdSet:
    """Convenience: window -> transition networks -> SPD matrices in one call.

    Extra kwargs are forwarded to rsmmtn.build_transition_networks (operator,
    span_max, n_angular, n_radial, channels, standardize, keep_symbols).
    """
    net = build_transition_networks(X, alpha=alpha, **kwargs)
    return spd_from_networks(net, identity_weight=identity_weight)


# ---------------------------------------------------------------------------
# Self-test (NumPy only; no SciPy / MNE / EDF)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running spd.py self-test ...\n")
    rng = np.random.default_rng(cfg.SEED)

    def _row_norm(M):
        return M / M.sum(axis=-1, keepdims=True)

    # --- single matrix: C = I + A A^T, symmetric, SPD, min eigenvalue >= 1 ---
    A = _row_norm(rng.random((5, 5)))
    C = build_spd(A)
    assert np.allclose(C, np.eye(5) + A @ A.T)
    assert np.allclose(C, C.T)
    w = np.linalg.eigvalsh(C)
    assert w.min() >= 1.0 - _EPS, "min eigenvalue must be >= identity_weight"
    assert is_spd(C)

    # --- identity_weight scales the floor ---
    C2 = build_spd(A, identity_weight=2.0)
    assert np.allclose(C2, 2.0 * np.eye(5) + A @ A.T)
    assert np.linalg.eigvalsh(C2).min() >= 2.0 - _EPS
    try:
        build_spd(A, identity_weight=0.0)
        raise AssertionError("identity_weight=0 must raise")
    except ValueError:
        pass

    # --- batched matches per-matrix ---
    Ab = _row_norm(rng.random((3, 4, 6, 6)))
    Cb = build_spd(Ab)
    assert Cb.shape == (3, 4, 6, 6)
    for i in range(3):
        for j in range(4):
            assert np.allclose(Cb[i, j], np.eye(6) + Ab[i, j] @ Ab[i, j].T)
    assert np.allclose(Cb, np.swapaxes(Cb, -1, -2)), "batched output must be symmetric"

    # --- integration with rsmmtn (small grid) ---
    n = cfg.N_CHANNELS
    T = 400
    X = rng.standard_normal((n, T))
    net = build_transition_networks(X, alpha=0.5, n_angular=6, n_radial=4, span_max=3)
    sset = spd_from_networks(net)
    assert sset.matrices.shape == (n, 3, 24, 24)
    assert sset.alpha == 0.5 and sset.span_roof == 3 and sset.dim == 24
    assert np.allclose(sset.matrices, np.swapaxes(sset.matrices, -1, -2))
    assert np.linalg.eigvalsh(sset.matrices).min() > 0.0, "every C must be SPD"
    assert sset.span_slice(2).shape == (n, 2, 24, 24)
    assert sset.cumulative(3).shape == (n, 24, 24)

    # --- spd_from_window convenience path agrees ---
    sset2 = spd_from_window(X, alpha=0.5, n_angular=6, n_radial=4, span_max=3)
    assert np.array_equal(sset.matrices, sset2.matrices)

    # --- determinism ---
    assert np.array_equal(build_spd(Ab), build_spd(Ab))

    print("OK - spd.py self-test passed.")
