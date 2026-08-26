"""
distances.py
============
Stage 4 / Req E: the feature reduction -- from SPD matrices to the flat vector.

For each RECENTERED window C' (one SPD matrix per channel x span) we measure its
AIRM geodesic distance to the three references from references.py, per
(channel x span):

    d_baseline   = delta_R(C', I)            = || log(eig(C')) ||_2
    d_interictal = delta_R(C', M_interictal)
    d_preictal   = delta_R(C', M_preictal)

Stacking over (channel, span, reference) and slicing to a span roof m gives the
feature vector of length

    feature_dim(m) = N_REFERENCES (3) x m x N_CHANNELS (18)   -> 486 at m = 9.

This is the collapse that lets us stream the (18 x span x 180 x 180) SPD tensor
and keep only feature_dim(m) floats per window -- the dense SPD is never stored.

Flatten order (canonical, must stay consistent across all windows)
------------------------------------------------------------------
channel-major, then span (1..m), then reference:
    index(c, s, r) = (c * m + s) * N_REFERENCES + r
feature_names() returns labels in exactly this order.

Fold-invariance seam
--------------------
d_baseline depends only on C' (hence on the fold-invariant G_patient), so the
baseline column is fold-invariant and cacheable. d_interictal / d_preictal use
the per-fold population anchors and are recomputed each fold. baseline_distances
and population_distances are exposed separately so dataset_builder can cache the
former and recompute only the latter.
"""

from __future__ import annotations

import numpy as np

from src import config as cfg
from src.utils.logger import get_logger
from src.features import backend as bk
from src.features.references import ReferenceSet

log = get_logger(__name__)

_EPS = 1e-12


def baseline_distances(cp: np.ndarray) -> np.ndarray:
    """delta_R(C', I) = sqrt(sum(log(eig C')^2)), batched over (..., nc, sr).

    cp : (..., dim, dim) recentered SPD; returns (...,) with the two matrix axes
    reduced. Fold-invariant (cacheable).
    """
    w = bk.eigvalsh(cp)
    logw = np.log(np.clip(w, _EPS, None))
    return np.sqrt(np.sum(logw * logw, axis=-1))


def population_distances(cp: np.ndarray, M: np.ndarray) -> np.ndarray:
    """delta_R(C', M) to a single population reference M (broadcast over windows)."""
    return bk.airm_distance(cp, M)


def window_distances(cp: np.ndarray, references: ReferenceSet) -> np.ndarray:
    """Distances to all references, in references.names order.

    cp : (..., nc, sr, dim, dim) recentered SPD (single window (nc, sr, d, d) or
         a stack (nw, nc, sr, d, d)).
    returns : (..., nc, sr, n_references)
    """
    cols = []
    for name in references.names:
        if name == "baseline":
            cols.append(baseline_distances(cp))
        else:
            cols.append(population_distances(cp, references.get(name)))
    return np.stack(cols, axis=-1)


def _roof(references: ReferenceSet, span_roof: int | None) -> int:
    m = references.span_roof if span_roof is None else span_roof
    if not 1 <= m <= references.span_roof:
        raise ValueError(f"span roof m={m} outside [1, {references.span_roof}]")
    return m


def feature_vector(cp: np.ndarray, references: ReferenceSet, *, span_roof: int | None = None) -> np.ndarray:
    """Flatten distances for span roof m into feature_dim(m) floats.

    cp : (nc, sr, dim, dim) single window, or (nw, nc, sr, dim, dim) stack.
    returns : (feature_dim,) or (nw, feature_dim), channel-major/span/reference.
    """
    m = _roof(references, span_roof)
    D = window_distances(cp, references)             # (..., nc, sr, n_refs)
    D = D[..., :m, :]                                 # slice to roof m
    lead = D.shape[:-3]
    nc, _, n_refs = D.shape[-3:]
    return D.reshape(*lead, nc * m * n_refs)


def feature_matrix(cp_stack: np.ndarray, references: ReferenceSet, *, span_roof: int | None = None) -> np.ndarray:
    """Feature vectors for a stack of windows: (nw, feature_dim)."""
    cp_stack = np.asarray(cp_stack, dtype=float)
    if cp_stack.ndim != 5:
        raise ValueError(
            f"expected (nw, nc, sr, dim, dim), got {cp_stack.shape}"
        )
    return feature_vector(cp_stack, references, span_roof=span_roof)


def feature_names(channels: tuple[str, ...], span_roof: int, names: tuple[str, ...] | None = None) -> list[str]:
    """Labels aligned to the flatten order: channel-major, span, reference."""
    names = tuple(names) if names is not None else tuple(cfg.REFERENCE_NAMES)
    out = []
    for c in channels:
        for s in range(1, span_roof + 1):
            for r in names:
                out.append(f"{c}|span{s}|{r}")
    return out


# ---------------------------------------------------------------------------
# Self-test (NumPy path; no torch / scipy / pyriemann required)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running distances.py self-test ...\n")
    from src.features import references as refs

    rng = np.random.default_rng(cfg.SEED)
    n = 5
    nc, sr = 4, 3
    channels = tuple(f"ch{i}" for i in range(nc))

    def rand_spd(lead=()):
        A = rng.standard_normal(lead + (n, n))
        return A @ np.swapaxes(A, -1, -2) + n * np.eye(n)

    # --- baseline distance: d(I, I) = 0; matches AIRM to identity ---
    eye_cs = np.broadcast_to(np.eye(n), (nc, sr, n, n)).copy()
    assert np.allclose(baseline_distances(eye_cs), 0.0, atol=1e-9)
    cp = rand_spd((nc, sr))
    assert np.allclose(baseline_distances(cp),
                       bk.airm_distance(cp, eye_cs), atol=1e-8)

    # --- build a reference set and check window_distances ---
    pms = []
    for pid in ("p0", "p1"):
        inter_p = np.stack([rand_spd((nc, sr)) for _ in range(5)])
        pre_p = np.stack([rand_spd((nc, sr)) for _ in range(5)])
        pms.append(refs.patient_reference_means(pid, inter_p, pre_p, channels=channels))
    R = refs.build_fold_references(pms)

    D = window_distances(cp, R)
    assert D.shape == (nc, sr, cfg.N_REFERENCES)
    assert np.all(D >= 0.0)
    # column order matches names
    assert np.allclose(D[..., R.names.index("baseline")], baseline_distances(cp), atol=1e-9)
    assert np.allclose(D[..., R.names.index("preictal")],
                       bk.airm_distance(cp, R.get("preictal")), atol=1e-8)

    # --- feature vector length + flatten order ---
    m = 2
    fv = feature_vector(cp, R, span_roof=m)
    assert fv.shape == (nc * m * cfg.N_REFERENCES,)
    # reconstruct expected order (channel, span, reference)
    expected = D[:, :m, :].reshape(-1)
    assert np.allclose(fv, expected, atol=1e-12)
    # spot-check an explicit index
    c_i, s_i, r_i = 2, 1, cfg.REFERENCE_NAMES.index("interictal")
    idx = (c_i * m + s_i) * cfg.N_REFERENCES + r_i
    assert np.allclose(fv[idx], D[c_i, s_i, r_i], atol=1e-12)

    # --- feature_names align with the vector ---
    fn = feature_names(channels, m)
    assert len(fn) == fv.shape[0]
    assert fn[idx] == f"ch2|span2|interictal"
    assert len(set(fn)) == len(fn)

    # --- batched feature_matrix + agreement with cfg.feature_dim (real nc) ---
    from src.features.spd import spd_from_window
    from src.features import alignment as al
    nchan = cfg.N_CHANNELS
    stack = np.stack([
        spd_from_window(rng.standard_normal((nchan, 400)), alpha=0.5,
                        n_angular=6, n_radial=4, span_max=3).matrices
        for _ in range(6)
    ])                                            # (6, 18, 3, 24, 24)
    G = al.patient_baseline(stack)
    cpw = al.recenter(stack, G=G)
    ch18 = tuple(cfg.CHANNELS)
    big_pms = [
        refs.patient_reference_means("q0", cpw[:3], cpw[:3], channels=ch18),
        refs.patient_reference_means("q1", cpw[3:], cpw[3:], channels=ch18),
    ]
    Rbig = refs.build_fold_references(big_pms)
    feats = feature_matrix(cpw, Rbig, span_roof=3)
    assert feats.shape == (6, cfg.feature_dim(3)), (feats.shape, cfg.feature_dim(3))
    assert feats.shape[1] == cfg.N_REFERENCES * 3 * nchan == 162
    assert np.all(np.isfinite(feats))

    print("OK - distances.py self-test passed.")
