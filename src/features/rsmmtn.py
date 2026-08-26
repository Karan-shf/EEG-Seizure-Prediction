"""
rsmmtn.py
=========
Stage 4 / Req C+D: the RSMMTN spatio-temporal feature FRONT-END -- the "twist".

Given one montage-ordered window X of shape (n_channels, n_samples), this module
turns each channel's time series into a cumulative multi-span symbolic
TRANSITION NETWORK: a stack of row-normalized (n_symbols x n_symbols) adjacency
matrices A^1..A^m, one per cumulative span roof. Those adjacencies are what the
next stage (spd.py) lifts to SPD matrices C^i = I + A^i (A^i)^T.

The pipeline per channel c
--------------------------
1. Phase-portrait coordinates (the "twist"):
       x(t) = X_c(t)                              # the signal itself
       y(t) = alpha * d2X_c(t) + (1 - alpha) * L_c(t)
   where d2X_c(t) = X_c(t+1) - 2 X_c(t) + X_c(t-1)  is the 2nd-order temporal
   difference (discrete curvature / acceleration of the trace) and L_c(t) is the
   surface-Laplacian spatial-contrast term from laplacian.py. alpha in [0, 1]
   blends a purely TEMPORAL portrait (alpha = 1, the original paper) with a
   purely SPATIAL one (alpha = 0). Each alpha is a fully independent experiment.

   d2X is only defined on the interior t = 1 .. T-2, so x, L and d2X are all
   aligned to that interior range (length T-2) before blending.

2. Symbolization -> polar sector-ring grid:
   x and y are z-scored per channel (so the two physically different axes are
   comparable and the grid is scale-invariant), then each point (x, y) is mapped
   to a polar cell:
       angle  = atan(y / x)  folded to (-90, 90]   -> N_ANGULAR_BINS sectors
       radius = sqrt(x^2 + y^2), 0 .. r_max         -> N_RADIAL_BINS rings
                (r_max = the window's own max radius; dr = r_max / N_RADIAL_BINS)
   symbol = angular_bin * N_RADIAL_BINS + radial_bin  in [0, N_SYMBOLS).
   With the config defaults (18 x 10) there are 180 symbols -> 180 x 180 network.

3. Cumulative multi-span transition counts:
   For each span k = 1 .. SPAN_MAX, count transitions symbol(t) -> symbol(t+k).
   The span roof m keeps the CUMULATIVE network over spans {1..m}:
       A^m_counts = sum_{k=1}^{m} T_k
   (NOT a single step-m matrix -- the roofs in SPAN_ROOF_GRID are nested inside
   these cumulative sets, so we compute up to SPAN_MAX once and slice). Each A^m
   is finally row-normalized into a transition-probability matrix.

Output
------
A TransitionNetworkSet whose `adjacency` is
    (n_channels, SPAN_MAX, n_symbols, n_symbols)  float64, row-normalized.
At the config defaults that is 18 x 9 x 180 x 180 ~= 42 MB per window per alpha,
which is exactly why these are STREAMED and never persisted (see the LOPO
caching plan in config sections 10): spd.py -> distances.py collapse each window
to just N_REFERENCES x m x N_CHANNELS = feature_dim(m) floats (486 at m = 9).

Everything that is part of the experimental design is imported from config:
N_ANGULAR_BINS, N_RADIAL_BINS, N_SYMBOLS, SPAN_MAX, ALPHA_GRID, CHANNELS. The
only implementation choice not (yet) in config is the per-axis z-scoring of the
symbol coordinates (`standardize=True`); flip it or promote it to a config
constant if you want to sweep it.

NumPy only. The SPD lift (C = I + A A^T) deliberately lives in spd.py so there
is a single source of truth for the manifold construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import config as cfg
from src.utils.logger import get_logger
from src.features.laplacian import LaplacianOperator, default_operator

log = get_logger(__name__)

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Temporal term
# ---------------------------------------------------------------------------
def second_time_difference(X: np.ndarray) -> np.ndarray:
    """Discrete 2nd-order temporal difference along the last (time) axis.

    d2X[:, t] = X[:, t+2] - 2 X[:, t+1] + X[:, t]  (centered on the interior).
    Input (n_channels, n_samples) -> output (n_channels, n_samples - 2), aligned
    to interior times t = 1 .. T-2.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected 2-D (n_channels, n_samples), got {X.shape}")
    if X.shape[1] < 3:
        raise ValueError(f"need >= 3 samples for a 2nd difference, got {X.shape[1]}")
    return X[:, 2:] - 2.0 * X[:, 1:-1] + X[:, :-2]


# ---------------------------------------------------------------------------
# Symbolization (polar sector-ring grid)
# ---------------------------------------------------------------------------
def _standardize_rows(A: np.ndarray) -> np.ndarray:
    """Z-score each row independently; flat rows (std ~ 0) collapse to zeros."""
    A = np.asarray(A, dtype=float)
    mean = A.mean(axis=1, keepdims=True)
    std = A.std(axis=1, keepdims=True)
    out = (A - mean) / np.where(std > _EPS, std, 1.0)
    flat = (std <= _EPS)[:, 0]
    if flat.any():
        out[flat] = 0.0
    return out


def symbolize(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_angular: int = cfg.N_ANGULAR_BINS,
    n_radial: int = cfg.N_RADIAL_BINS,
    standardize: bool = True,
) -> np.ndarray:
    """Map paired (x, y) trajectories to polar sector-ring symbols.

    x, y are (n_channels, L). Returns int16 symbols (n_channels, L) in
    [0, n_angular * n_radial).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x/y shape mismatch: {x.shape} vs {y.shape}")
    if x.ndim != 2:
        raise ValueError(f"expected 2-D (n_channels, L), got {x.shape}")
    if standardize:
        x = _standardize_rows(x)
        y = _standardize_rows(y)

    # Angle folded to (-90, 90]  ==  atan(y / x)  (line orientation, mod 180 deg).
    ang = np.degrees(np.arctan2(y, x))
    ang = (ang + 90.0) % 180.0 - 90.0
    a_width = 180.0 / n_angular
    a_bin = np.clip(np.floor((ang + 90.0) / a_width).astype(int), 0, n_angular - 1)

    # Radius binned linearly over each row's own [0, r_max].
    r = np.sqrt(x * x + y * y)
    r_max = r.max(axis=1, keepdims=True)
    dr = r_max / n_radial
    r_bin = np.floor(np.divide(r, dr, out=np.zeros_like(r), where=dr > _EPS))
    r_bin = np.clip(r_bin.astype(int), 0, n_radial - 1)

    return (a_bin * n_radial + r_bin).astype(np.int16)


# ---------------------------------------------------------------------------
# Cumulative multi-span transition tensor
# ---------------------------------------------------------------------------
def transition_tensor(
    symbols: np.ndarray,
    *,
    span_max: int = cfg.SPAN_MAX,
    n_symbols: int = cfg.N_SYMBOLS,
    row_normalize: bool = True,
) -> np.ndarray:
    """Cumulative multi-span transition matrices from symbol sequences.

    symbols: (n_channels, L) or (L,). Returns
    (n_channels, span_max, n_symbols, n_symbols); slot i-1 is the CUMULATIVE
    network over spans {1..i}. Row-normalized to transition probabilities unless
    row_normalize=False (raw counts, useful for testing).
    """
    symbols = np.asarray(symbols)
    if symbols.ndim == 1:
        symbols = symbols[None, :]
    if symbols.ndim != 2:
        raise ValueError(f"expected (n_channels, L) symbols, got {symbols.shape}")
    n_ch, L = symbols.shape
    if span_max < 1:
        raise ValueError(f"span_max must be >= 1, got {span_max}")
    if L <= span_max:
        raise ValueError(f"need L > span_max ({span_max}) transitions, got L={L}")
    if symbols.min() < 0 or symbols.max() >= n_symbols:
        raise ValueError("symbol out of range [0, n_symbols)")

    out = np.zeros((n_ch, span_max, n_symbols, n_symbols), dtype=float)
    for c in range(n_ch):
        s = symbols[c]
        cum = np.zeros((n_symbols, n_symbols), dtype=float)
        for k in range(1, span_max + 1):
            np.add.at(cum, (s[:-k], s[k:]), 1.0)   # T_k accumulated in place
            out[c, k - 1] = cum                    # cumulative over spans {1..k}

    if row_normalize:
        rows = out.sum(axis=-1, keepdims=True)
        np.divide(out, rows, out=out, where=rows > _EPS)
    return out


# ---------------------------------------------------------------------------
# Public result container
# ---------------------------------------------------------------------------
@dataclass
class TransitionNetworkSet:
    """Cumulative multi-span transition networks for one window at one alpha."""
    channels: tuple[str, ...]
    alpha: float
    span_roof: int                 # highest span computed (== SPAN_MAX)
    n_angular: int
    n_radial: int
    adjacency: np.ndarray          # (n_channels, span_roof, n_symbols, n_symbols)
    symbols: np.ndarray | None = None  # (n_channels, L) int16, optional

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_symbols(self) -> int:
        return self.n_angular * self.n_radial

    def span_slice(self, m: int) -> np.ndarray:
        """Cumulative networks for span roof m: (n_channels, m, n_sym, n_sym)."""
        if not 1 <= m <= self.span_roof:
            raise ValueError(f"span roof m={m} outside [1, {self.span_roof}]")
        return self.adjacency[:, :m]

    def cumulative(self, m: int) -> np.ndarray:
        """The single cumulative network A^m: (n_channels, n_sym, n_sym)."""
        if not 1 <= m <= self.span_roof:
            raise ValueError(f"span roof m={m} outside [1, {self.span_roof}]")
        return self.adjacency[:, m - 1]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_transition_networks(
    X: np.ndarray,
    *,
    alpha: float,
    operator: LaplacianOperator | None = None,
    span_max: int = cfg.SPAN_MAX,
    n_angular: int = cfg.N_ANGULAR_BINS,
    n_radial: int = cfg.N_RADIAL_BINS,
    channels: tuple[str, ...] = cfg.CHANNELS,
    standardize: bool = True,
    keep_symbols: bool = True,
) -> TransitionNetworkSet:
    """Build the cumulative multi-span transition networks for one window.

    X: (n_channels, n_samples) montage-ordered window. alpha in [0, 1] sets the
    temporal/spatial blend. When alpha == 1 the Laplacian is never evaluated;
    when alpha == 0 the temporal difference is never evaluated.
    """
    X = np.asarray(X, dtype=float)
    n_ch = len(channels)
    if X.ndim != 2 or X.shape[0] != n_ch:
        raise ValueError(f"expected X of shape ({n_ch}, n_samples), got {X.shape}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if X.shape[1] < 3:
        raise ValueError(f"need >= 3 samples, got {X.shape[1]}")

    x_traj = X[:, 1:-1]                      # interior-aligned signal term
    y = np.zeros_like(x_traj)
    if alpha > 0.0:
        y = y + alpha * second_time_difference(X)
    if alpha < 1.0:
        op = operator if operator is not None else default_operator()
        y = y + (1.0 - alpha) * op.apply(X, channel_axis=0)[:, 1:-1]

    symbols = symbolize(
        x_traj, y, n_angular=n_angular, n_radial=n_radial, standardize=standardize
    )
    n_symbols = n_angular * n_radial
    adjacency = transition_tensor(
        symbols, span_max=span_max, n_symbols=n_symbols, row_normalize=True
    )
    log.info(
        "rsmmtn: alpha=%.2f -> %d channels x %d spans x %d symbols (%.1f MB)",
        alpha, n_ch, span_max, n_symbols, adjacency.nbytes / 1e6,
    )
    return TransitionNetworkSet(
        channels=tuple(channels),
        alpha=float(alpha),
        span_roof=span_max,
        n_angular=n_angular,
        n_radial=n_radial,
        adjacency=adjacency,
        symbols=symbols if keep_symbols else None,
    )


def build_alpha_arms(
    X: np.ndarray,
    *,
    alphas: tuple[float, ...] = cfg.ALPHA_GRID,
    operator: LaplacianOperator | None = None,
    **kwargs,
) -> dict[float, TransitionNetworkSet]:
    """Convenience: build one TransitionNetworkSet per alpha arm.

    Note each arm is ~42 MB at config defaults -- callers that stream should
    prefer build_transition_networks per alpha rather than holding all arms.
    """
    op = operator if operator is not None else default_operator()
    return {
        float(a): build_transition_networks(X, alpha=a, operator=op, **kwargs)
        for a in alphas
    }


# ---------------------------------------------------------------------------
# Self-test (NumPy only; no SciPy / MNE / EDF)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running rsmmtn.py self-test ...\n")
    rng = np.random.default_rng(cfg.SEED)
    n = cfg.N_CHANNELS
    T = 800
    t = np.arange(T) / cfg.FS
    base = np.sin(2 * np.pi * 10.0 * t)
    X = np.stack([
        (1.0 + 0.1 * i) * np.roll(base, 3 * i) + 0.3 * rng.standard_normal(T)
        for i in range(n)
    ])

    # --- 2nd temporal difference ---
    d2 = second_time_difference(X)
    assert d2.shape == (n, T - 2)
    ramp = np.tile(np.arange(T, dtype=float), (n, 1))          # linear -> 0 curvature
    assert np.allclose(second_time_difference(ramp), 0.0)

    # --- symbolization ranges ---
    x_traj = X[:, 1:-1]
    sym = symbolize(x_traj, d2)
    assert sym.shape == (n, T - 2)
    assert sym.dtype == np.int16
    assert sym.min() >= 0 and sym.max() < cfg.N_SYMBOLS

    # --- full-config build: shape + row-normalization ---
    net = build_transition_networks(X, alpha=0.5)
    assert net.adjacency.shape == (n, cfg.SPAN_MAX, cfg.N_SYMBOLS, cfg.N_SYMBOLS)
    rows = net.adjacency.sum(axis=-1)
    nz = rows > _EPS
    assert np.allclose(rows[nz], 1.0), "non-empty rows must sum to 1"
    assert net.adjacency.min() >= 0.0

    # --- cumulative counts are non-decreasing; span-1 has L-1 transitions ---
    counts = transition_tensor(sym, row_normalize=False)
    tot = counts.sum(axis=(-1, -2))                            # (n, span_max)
    assert np.all(np.diff(tot, axis=1) >= -_EPS), "cumulative counts must grow"
    assert np.allclose(tot[:, 0], (T - 2) - 1), "span-1 == L-1 transitions"
    # exact cumulative accounting: A^m total == sum_{k=1}^m (L - k)
    L = T - 2
    expected = np.cumsum([L - k for k in range(1, cfg.SPAN_MAX + 1)])
    assert np.allclose(tot[0], expected)

    # --- alpha endpoints: temporal-only ignores the Laplacian ---
    net_t = build_transition_networks(X, alpha=1.0)
    assert net_t.symbols is not None
    assert np.array_equal(net_t.symbols, symbolize(x_traj, second_time_difference(X)))
    net_s = build_transition_networks(X, alpha=0.0)
    assert net_s.symbols is not None
    L_spatial = default_operator().apply(X)[:, 1:-1]
    assert np.array_equal(net_s.symbols, symbolize(x_traj, L_spatial))
    assert not np.array_equal(net_t.symbols, net_s.symbols), "blend must matter"

    # --- span_slice matches the reported roofs / feature accounting ---
    for m in cfg.SPAN_ROOF_GRID:
        sl = net.span_slice(m)
        assert sl.shape == (n, m, cfg.N_SYMBOLS, cfg.N_SYMBOLS)
        # distances.py will emit N_REFERENCES per (channel, span):
        assert cfg.N_REFERENCES * m * n == cfg.feature_dim(m)

    # --- determinism ---
    assert np.array_equal(net.adjacency, build_transition_networks(X, alpha=0.5).adjacency)

    # --- small custom grid stays consistent ---
    small = build_transition_networks(X, alpha=1.0, n_angular=4, n_radial=3, span_max=3)
    assert small.adjacency.shape == (n, 3, 12, 12)
    assert small.n_symbols == 12

    print("OK - rsmmtn.py self-test passed.")
