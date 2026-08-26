"""
laplacian.py
============
Stage 3 / Req D (spatial term): the weighted graph-Laplacian spatial-contrast
operator L_c(t) that gets alpha-blended with the temporal term before RSMMTN.

    L_c(t) = X_c(t) - ( sum_j w_cj * X_j(t) ) / ( sum_j w_cj )
    w_cj   = 1 / dist(c, j)   restricted to the k nearest neighbours of c

This is the classic Hjorth nearest-neighbour Laplacian over the sparse
double-banana montage -- a LOCAL spatial-contrast operator, NOT spherical-
spline CSD (which needs >= ~64 channels and extrapolates badly at the array
edges where foci localize). Honest caveat, per config: over 18 bipolar
channels this is a spatial-contrast filter, not a textbook monopolar CSD.

Geometry
--------
Each bipolar channel (e.g. "FP1-F7") is located at the MIDPOINT of its two
scalp electrodes. Electrode positions come from an idealized standard_1020
sphere (unit radius). The midpoint of the two electrode unit-vectors is
re-projected onto the sphere, and dist(c, j) is the GEODESIC (great-circle arc
length) between channel midpoints.

Scale-invariance: because the neighbour weights are normalized (we divide by
sum_j w_cj), multiplying every distance by a constant leaves L unchanged -- so
the absolute sphere radius is irrelevant and only relative geometry matters.

The operator is linear and time-invariant, so it collapses to a single
(n_channels x n_channels) matrix M = I - W_hat (W_hat = row-normalized neighbour
weights). Then L = M @ X for a montage-ordered signal X of shape
(n_channels, n_samples). Row sums of M are exactly 0 (Laplacian property):
a spatially uniform signal maps to 0.

Uses NumPy only. No SciPy / MNE / EDF. Electrode coordinates are hard-coded so
the operator is reproducible without MNE installed; MONTAGE_NAME documents the
reference layout (mne.channels.make_standard_montage("standard_1020")).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src import config as cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

_EPS = 1e-9

# ---------------------------------------------------------------------------
# Idealized standard_1020 electrode positions.
#
# Each electrode is given as (colatitude, azimuth) in DEGREES on a unit sphere:
#   colatitude = angle away from the vertex Cz (0 deg at Cz, 90 deg at the
#                nasion/inion/pre-auricular equator),
#   azimuth    = angle in the horizontal plane measured from the +x axis
#                (right ear) toward the +y axis (nasion / front):
#                  0 deg -> right, 90 deg -> front, 180 deg -> left, 270 -> back.
# Cartesian: x = sin(col)cos(az), y = sin(col)sin(az), z = cos(col)
#            (x = right, y = front, z = up). Left/right and front/back pairs are
# mirror-symmetric. These idealized angles reproduce the standard_1020 neighbour
# topology; exact millimetre positions are unnecessary because only the
# k-nearest-neighbour ordering and relative (normalized) weights are used.
# ---------------------------------------------------------------------------
_ELECTRODES_DEG: dict[str, tuple[float, float]] = {
    "FP1": (72.0, 108.0), "FP2": (72.0, 72.0),
    "F7":  (72.0, 144.0), "F8":  (72.0, 36.0),
    "F3":  (48.0, 122.0), "F4":  (48.0, 58.0),
    "FZ":  (36.0, 90.0),
    "T7":  (90.0, 180.0), "T8":  (90.0, 0.0),
    "C3":  (45.0, 180.0), "C4":  (45.0, 0.0),
    "CZ":  (0.0, 0.0),
    "P7":  (72.0, 216.0), "P8":  (72.0, 324.0),
    "P3":  (48.0, 238.0), "P4":  (48.0, 302.0),
    "PZ":  (36.0, 270.0),
    "O1":  (72.0, 252.0), "O2":  (72.0, 288.0),
}


def _unit_vector(colat_deg: float, az_deg: float) -> np.ndarray:
    col = math.radians(colat_deg)
    az = math.radians(az_deg)
    return np.array(
        [math.sin(col) * math.cos(az), math.sin(col) * math.sin(az), math.cos(col)],
        dtype=float,
    )


def electrode_unit_vectors() -> dict[str, np.ndarray]:
    """Return {electrode_name: unit 3-vector} for the standard_1020 layout."""
    return {name: _unit_vector(c, a) for name, (c, a) in _ELECTRODES_DEG.items()}


def channel_midpoints(channels: tuple[str, ...] = cfg.CHANNELS) -> np.ndarray:
    """Unit-sphere midpoint of each bipolar channel, shape (n_channels, 3)."""
    ev = electrode_unit_vectors()
    mids = np.empty((len(channels), 3), dtype=float)
    for i, ch in enumerate(channels):
        try:
            a, b = ch.split("-")
        except ValueError as exc:
            raise ValueError(f"channel {ch!r} is not a bipolar 'A-B' pair") from exc
        for e in (a, b):
            if e not in ev:
                raise KeyError(f"electrode {e!r} (channel {ch!r}) not in standard_1020 table")
        m = ev[a] + ev[b]
        norm = np.linalg.norm(m)
        mids[i] = m / norm if norm > _EPS else m
    return mids


def _distance_matrix(mids: np.ndarray, metric: str) -> np.ndarray:
    if metric == "geodesic":
        gram = np.clip(mids @ mids.T, -1.0, 1.0)
        D = np.arccos(gram)
        np.fill_diagonal(D, 0.0)  # arccos(1) is numerically noisy; self-distance is exactly 0
        return D
    if metric == "euclidean":
        diff = mids[:, None, :] - mids[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))
    raise ValueError(f"unknown distance metric {metric!r}")


@dataclass
class LaplacianOperator:
    """Precomputed linear spatial operator: L = matrix @ X (channels first)."""
    channels: tuple[str, ...]
    k: int
    weighting: str
    distance: str
    matrix: np.ndarray                                  # (n, n), row sums == 0
    neighbors: dict[str, list[tuple[str, float]]] = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def apply(self, X: np.ndarray, *, channel_axis: int = 0) -> np.ndarray:
        """Apply the Laplacian along channel_axis. X[channel_axis] must be n_channels."""
        X = np.asarray(X, dtype=float)
        if X.shape[channel_axis] != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels on axis {channel_axis}, "
                f"got shape {X.shape}"
            )
        Xm = np.moveaxis(X, channel_axis, 0)
        Lm = self.matrix @ Xm
        return np.moveaxis(Lm, 0, channel_axis)


def build_laplacian_operator(
    channels: tuple[str, ...] = cfg.CHANNELS,
    *,
    k: int = cfg.LAPLACIAN_K,
    weighting: str = cfg.LAPLACIAN_WEIGHTING,
    distance: str = cfg.LAPLACIAN_DISTANCE,
) -> LaplacianOperator:
    """Build the (n_channels x n_channels) weighted-Laplacian operator matrix."""
    n = len(channels)
    if not 1 <= k < n:
        raise ValueError(f"k={k} must be in [1, {n - 1}]")
    mids = channel_midpoints(channels)
    D = _distance_matrix(mids, distance)

    W = np.zeros((n, n), dtype=float)
    neighbors: dict[str, list[tuple[str, float]]] = {}
    for c in range(n):
        order = [j for j in np.argsort(D[c], kind="stable") if j != c][:k]
        d = D[c, order]
        if weighting == "inverse_distance":
            w = 1.0 / np.maximum(d, _EPS)
        elif weighting == "uniform":
            w = np.ones(len(order), dtype=float)
        else:
            raise ValueError(f"unknown weighting {weighting!r}")
        w = w / w.sum()
        for j, wj in zip(order, w):
            W[c, j] = wj
        neighbors[channels[c]] = [(channels[j], float(W[c, j])) for j in order]

    matrix = np.eye(n) - W
    log.info(
        "laplacian operator: %d channels, k=%d, %s weights, %s distance",
        n, k, weighting, distance,
    )
    return LaplacianOperator(
        channels=tuple(channels),
        k=k,
        weighting=weighting,
        distance=distance,
        matrix=matrix,
        neighbors=neighbors,
    )


# Cache the default operator (montage geometry never changes within a run).
_DEFAULT_OP: LaplacianOperator | None = None


def default_operator() -> LaplacianOperator:
    global _DEFAULT_OP
    if _DEFAULT_OP is None:
        _DEFAULT_OP = build_laplacian_operator()
    return _DEFAULT_OP


def surface_laplacian(X: np.ndarray, *, channel_axis: int = 0, **kwargs) -> np.ndarray:
    """Convenience: apply the (default or kwarg-built) Laplacian to a signal.

    L_c(t) for montage-ordered X. With no kwargs, uses the cached default
    operator (k=cfg.LAPLACIAN_K, inverse-distance, geodesic).
    """
    op = default_operator() if not kwargs else build_laplacian_operator(**kwargs)
    return op.apply(X, channel_axis=channel_axis)


# ---------------------------------------------------------------------------
# Self-test (NumPy only; no SciPy / MNE / EDF)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running laplacian.py self-test ...\n")
    rng = np.random.default_rng(cfg.SEED)
    n = cfg.N_CHANNELS

    # --- geometry sanity ---
    mids = channel_midpoints()
    assert mids.shape == (n, 3)
    assert np.allclose(np.linalg.norm(mids, axis=1), 1.0), "midpoints must be unit-norm"
    D = _distance_matrix(mids, "geodesic")
    assert np.allclose(D, D.T) and np.allclose(np.diag(D), 0.0)
    assert D.max() <= math.pi + 1e-6 and (D[~np.eye(n, dtype=bool)] > 0).all()

    # --- operator structure ---
    op = build_laplacian_operator()
    M = op.matrix
    assert M.shape == (n, n)
    assert np.allclose(M.sum(axis=1), 0.0), "row sums must be 0 (Laplacian property)"
    assert np.allclose(np.diag(M), 1.0), "diagonal must be 1"
    for c in range(n):
        off = M[c].copy()
        off[c] = 0.0
        assert np.count_nonzero(off) == cfg.LAPLACIAN_K, "exactly k neighbours per row"
        assert (off <= _EPS).all(), "neighbour entries must be negative (=-w_hat)"
    # nearer neighbour -> larger (more negative) weight
    for ch, nbrs in op.neighbors.items():
        ws = [w for _, w in nbrs]
        assert ws == sorted(ws, reverse=True), f"{ch}: weights not distance-ordered"

    # --- neighbour plausibility: FP1-F7 sits in the left-frontal cluster ---
    fp1f7 = {name for name, _ in op.neighbors["FP1-F7"]}
    assert fp1f7 & {"FP1-F3", "F7-T7"}, f"unexpected FP1-F7 neighbours: {fp1f7}"

    # --- action on signals ---
    T = 500
    # spatially uniform signal -> Laplacian is ~0
    uniform = np.tile(rng.standard_normal(T), (n, 1))
    assert np.allclose(op.apply(uniform), 0.0, atol=1e-9), "uniform field must map to 0"
    # generic signal: right shape, and channel_axis handling is consistent
    X = rng.standard_normal((n, T))
    L0 = op.apply(X, channel_axis=0)
    assert L0.shape == (n, T)
    L1 = op.apply(X.T, channel_axis=1)
    assert np.allclose(L0, L1.T), "channel_axis handling inconsistent"
    assert np.allclose(surface_laplacian(X), L0), "convenience wrapper mismatch"

    # --- k sanity grid (train/val-only) ---
    for k in cfg.LAPLACIAN_K_SANITY:
        opk = build_laplacian_operator(k=k)
        assert np.allclose(opk.matrix.sum(axis=1), 0.0)
        for c in range(n):
            off = opk.matrix[c].copy()
            off[c] = 0.0
            assert np.count_nonzero(off) == k

    # --- determinism ---
    assert np.allclose(build_laplacian_operator().matrix, M)

    print(op.neighbors["FP1-F7"])
    print("\nOK - laplacian.py self-test passed.")
