"""
backend.py
==========
Stage 4 / Req E: the Riemannian compute shim.

Everything downstream of spd.py (alignment, references, distances) needs the
same small set of SPD-manifold primitives -- symmetric matrix functions
(sqrt / inv-sqrt / inv / log / exp / pow), the AIRM geodesic distance, and the
Frechet (Karcher) mean. This module is the single place those live, so the rest
of the pipeline never touches raw eigendecompositions.

Device / precision
------------------
Controlled entirely by config (nothing hard-coded):
* COMPUTE_BACKEND = "auto" resolves the best torch device in the order the user
  asked for -- cuda -> mps -> cpu -- via the canonical line
      torch.device('cuda' if cuda else 'mps' if mps else 'cpu')
  It can also be forced to "cuda" / "mps" / "cpu" / "numpy".
* COMPUTE_DTYPE = "float64": all manifold math runs in double precision.
* EIGH_BATCH_SIZE chunks large batched eigh calls so GPU VRAM stays bounded.

Important caveat (Apple Silicon): PyTorch's MPS backend does NOT support float64
nor `linalg.eigh`. Because we deliberately chose float64 for the AIRM / Frechet
math, an auto-resolved `mps` device is transparently downgraded to CPU for the
linear-algebra core (CUDA keeps full float64 on-GPU). This trades a bit of Mac
speed for the numerical correctness the manifold math requires.

Graceful fallback
-----------------
torch is imported lazily. If it is missing (as in the offline sandbox) or the
effective device is CPU, every primitive runs on a NumPy path that is numerically
identical. The dependency-light self-test therefore runs anywhere; the GPU path
is exercised on the user's machine once torch is installed.

Public surface (all take/return NumPy float64 arrays, batched over leading dims)
-------------------------------------------------------------------------------
    resolve_device(), effective_device(), torch_available(), backend_info()
    symmetrize(C)                         exact symmetry
    spd_sqrt / spd_invsqrt / spd_inv / spd_log / spd_exp / spd_pow(C, p)
    eigvalsh(C)                           ascending eigenvalues
    airm_distance(P, Q)                   delta_R = || log(eig(P^-1 Q)) ||_2
    frechet_mean(mats, *, axis=0, weights=None)   AIRM Karcher mean
    logdet_spd(C)                         log det(C) via Cholesky (batched)
    jbld_divergence(P, Q) / jbld_distance(P, Q)   Stein/JBLD divergence and its
                                           sqrt (a genuine metric); affine-
                                           invariant like AIRM, cheaper (no
                                           eigendecomposition -- see the block
                                           below airm_distance for rationale)
"""

from __future__ import annotations

import numpy as np

from src import config as cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

_EPS = 1e-12                     # eigenvalue floor: guards log/inverse of round-off zeros


# ---------------------------------------------------------------------------
# Lazy torch + device resolution
# ---------------------------------------------------------------------------
_TORCH = None
_TORCH_CHECKED = False


def _torch():
    global _TORCH, _TORCH_CHECKED
    if not _TORCH_CHECKED:
        _TORCH_CHECKED = True
        try:
            import torch  # type: ignore
            _TORCH = torch
        except Exception:            # not installed / import failure -> NumPy path
            _TORCH = None
    return _TORCH


def torch_available() -> bool:
    return _torch() is not None


_AUTO_DEVICE = None


def _auto_device() -> str:
    """Best available torch device: cuda -> mps -> cpu (the user's canonical line)."""
    global _AUTO_DEVICE
    if _AUTO_DEVICE is not None:
        return _AUTO_DEVICE
    torch = _torch()
    if torch is None:
        _AUTO_DEVICE = "cpu"
    else:
        mps = getattr(torch.backends, "mps", None)
        if torch.cuda.is_available():
            _AUTO_DEVICE = "cuda"
        elif mps is not None and mps.is_available():
            _AUTO_DEVICE = "mps"
        else:
            _AUTO_DEVICE = "cpu"
    return _AUTO_DEVICE


def resolve_device() -> str:
    """The device implied by COMPUTE_BACKEND, before the float64/MPS downgrade."""
    backend = cfg.COMPUTE_BACKEND
    if backend == "numpy":
        return "cpu"
    if backend in ("cpu", "cuda", "mps"):
        return backend
    return _auto_device()


def _double() -> bool:
    return cfg.COMPUTE_DTYPE == "float64"


def effective_device() -> str:
    """Device actually used for linalg, after the MPS+float64 -> CPU downgrade."""
    dev = resolve_device()
    if dev == "mps" and _double():
        return "cpu"                # MPS lacks float64 / eigh
    return dev


def _use_torch() -> bool:
    """Use the torch engine only when it actually buys GPU acceleration."""
    if not torch_available() or cfg.COMPUTE_BACKEND == "numpy":
        return False
    return effective_device() in ("cuda", "mps")


def backend_info() -> dict:
    return {
        "torch_available": torch_available(),
        "compute_backend": cfg.COMPUTE_BACKEND,
        "resolve_device": resolve_device(),
        "effective_device": effective_device(),
        "dtype": cfg.COMPUTE_DTYPE,
        "engine": "torch" if _use_torch() else "numpy",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64 if _double() else np.float32)


def symmetrize(C: np.ndarray) -> np.ndarray:
    C = _np(C)
    return 0.5 * (C + np.swapaxes(C, -1, -2))


_CLIP_KINDS = {"sqrt", "invsqrt", "inv", "log", "pow"}


def _apply_scalar_np(w, kind, p):
    if kind in _CLIP_KINDS:
        w = np.clip(w, _EPS, None)
    if kind == "sqrt":
        return np.sqrt(w)
    if kind == "invsqrt":
        return 1.0 / np.sqrt(w)
    if kind == "inv":
        return 1.0 / w
    if kind == "log":
        return np.log(w)
    if kind == "exp":
        return np.exp(w)
    if kind == "pow":
        return np.power(w, p)
    raise ValueError(f"unknown matrix function {kind!r}")


def _funm_np(C, kind, p=None):
    Csym = symmetrize(C)
    w, V = np.linalg.eigh(Csym)
    fw = _apply_scalar_np(w, kind, p)
    out = (V * fw[..., None, :]) @ np.swapaxes(V, -1, -2)
    return 0.5 * (out + np.swapaxes(out, -1, -2))


def _funm_torch(C, kind, p=None):
    torch = _torch()
    assert torch is not None
    dev = effective_device()
    dtype = torch.float64 if _double() else torch.float32
    Ct = torch.as_tensor(np.asarray(C), dtype=dtype, device=dev)
    Ct = 0.5 * (Ct + Ct.transpose(-1, -2))
    lead = Ct.shape[:-2]
    n = Ct.shape[-1]
    flat = Ct.reshape(-1, n, n)
    batch = int(cfg.EIGH_BATCH_SIZE)
    outs = []
    for i in range(0, flat.shape[0], batch):
        chunk = flat[i:i + batch]
        w, V = torch.linalg.eigh(chunk)
        if kind in _CLIP_KINDS:
            w = torch.clamp(w, min=_EPS)
        if kind == "sqrt":
            fw = torch.sqrt(w)
        elif kind == "invsqrt":
            fw = torch.rsqrt(w)
        elif kind == "inv":
            fw = 1.0 / w
        elif kind == "log":
            fw = torch.log(w)
        elif kind == "exp":
            fw = torch.exp(w)
        elif kind == "pow":
            assert p is not None, "p must be provided when kind='pow'"
            fw = torch.pow(w, p)
        else:
            raise ValueError(f"unknown matrix function {kind!r}")
        rec = (V * fw.unsqueeze(-2)) @ V.transpose(-1, -2)
        outs.append(0.5 * (rec + rec.transpose(-1, -2)))
    out = torch.cat(outs, dim=0).reshape(*lead, n, n)
    return out.detach().to("cpu", dtype=torch.float64).numpy()


def _funm(C, kind, p=None):
    if _use_torch():
        return _funm_torch(C, kind, p)
    return _funm_np(C, kind, p)


# ---------------------------------------------------------------------------
# Public matrix functions
# ---------------------------------------------------------------------------
def spd_sqrt(C):    return _funm(C, "sqrt")
def spd_invsqrt(C): return _funm(C, "invsqrt")
def spd_inv(C):     return _funm(C, "inv")
def spd_log(C):     return _funm(C, "log")
def spd_exp(S):     return _funm(S, "exp")
def spd_pow(C, p):  return _funm(C, "pow", p)


def eigvalsh(C) -> np.ndarray:
    """Ascending eigenvalues of symmetric matrices, batched: (..., n) ."""
    if _use_torch():
        torch = _torch()
        assert torch is not None
        dev = effective_device()
        dtype = torch.float64 if _double() else torch.float32
        Ct = torch.as_tensor(np.asarray(C), dtype=dtype, device=dev)
        Ct = 0.5 * (Ct + Ct.transpose(-1, -2))
        lead = Ct.shape[:-2]
        n = Ct.shape[-1]
        flat = Ct.reshape(-1, n, n)
        batch = int(cfg.EIGH_BATCH_SIZE)
        outs = [torch.linalg.eigvalsh(flat[i:i + batch])
                for i in range(0, flat.shape[0], batch)]
        w = torch.cat(outs, dim=0).reshape(*lead, n)
        return w.detach().to("cpu", dtype=torch.float64).numpy()
    return np.linalg.eigvalsh(symmetrize(C))


def airm_distance(P, Q) -> np.ndarray:
    """AIRM geodesic distance delta_R(P, Q) = || log(eig(P^-1 Q)) ||_2, batched.

    Computed from the SPD-similarity form M = P^-1/2 Q P^-1/2 (eigenvalues equal
    the generalized eigenvalues of (Q, P)), which is symmetric and numerically
    stable. P and Q broadcast over leading dims.
    """
    P_ih = spd_invsqrt(P)
    M = P_ih @ _np(Q) @ P_ih
    w = eigvalsh(M)
    logw = np.log(np.clip(w, _EPS, None))
    return np.sqrt(np.sum(logw * logw, axis=-1))


# ---------------------------------------------------------------------------
# Jensen-Bregman LogDet Divergence (JBLD / "Stein divergence") -- an optional,
# cheaper substitute for AIRM (Sra, "A new metric on the manifold of kernel
# matrices...", 2012). This is a Tier-2 performance change: the benchmark
# showed AIRM's `distances` step (2 eigh calls x 2 references + 1 more for
# baseline = up to 5 batched eigh calls per window) dominating wall-clock,
# far outweighing rsmmtn+spd+recenter combined.
#
# JBLD needs only a Cholesky factorization per matrix -- no eigenvectors are
# reconstructed, just the triangular factor's diagonal -- which is
# algorithmically cheaper than a full symmetric eigendecomposition at the same
# matrix size. It is ALSO affine-invariant, exactly like AIRM:
#     JBLD(A P A^T, A Q A^T) == JBLD(P, Q)   for any invertible A
# so the existing G^-1/2 . C . G^-1/2 recentering in alignment.py remains
# mathematically valid, unchanged, under this metric -- see the "affine
# invariance" self-test assertion below.
#
# This block is purely ADDITIVE: airm_distance / frechet_mean above are
# untouched (anchor construction still uses the AIRM Karcher mean -- it is
# cheap in aggregate, built only a handful of times, and was not the measured
# bottleneck). Only the per-window DISTANCE measurement in distances.py is
# switched to call jbld_distance() instead of airm_distance() -- see that
# module for the (commented, not deleted) AIRM call sites.
# ---------------------------------------------------------------------------
def _logdet_spd_np(C) -> np.ndarray:
    C = symmetrize(C)
    n = C.shape[-1]
    # tiny diagonal jitter: Cholesky (unlike eigh + clip) hard-fails on a
    # matrix that is symmetric but not numerically positive-definite, which
    # roundoff (especially in float32) can produce after a congruence
    # transform. This costs nothing and matches the same _EPS floor the
    # eigh-based path already clips to.
    L = np.linalg.cholesky(C + _EPS * np.eye(n))
    diag = np.diagonal(L, axis1=-2, axis2=-1)
    return 2.0 * np.sum(np.log(np.clip(diag, _EPS, None)), axis=-1)


def _logdet_spd_torch(C) -> np.ndarray:
    torch = _torch()
    assert torch is not None
    dev = effective_device()
    dtype = torch.float64 if _double() else torch.float32
    Ct = torch.as_tensor(np.asarray(C), dtype=dtype, device=dev)
    Ct = 0.5 * (Ct + Ct.transpose(-1, -2))
    lead = Ct.shape[:-2]
    n = Ct.shape[-1]
    flat = Ct.reshape(-1, n, n)
    jitter = torch.eye(n, dtype=dtype, device=dev) * _EPS
    batch = int(cfg.EIGH_BATCH_SIZE)
    outs = []
    for i in range(0, flat.shape[0], batch):
        chunk = flat[i:i + batch] + jitter
        L = torch.linalg.cholesky(chunk)
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        outs.append(2.0 * torch.sum(torch.log(torch.clamp(diag, min=_EPS)), dim=-1))
    out = torch.cat(outs, dim=0).reshape(tuple(lead))
    return out.detach().to("cpu", dtype=torch.float64).numpy()


def logdet_spd(C) -> np.ndarray:
    """log det(C) for batched SPD matrices, via Cholesky (batched over leading
    dims, mirroring eigvalsh). Exposed publicly so callers can compute a
    window's or a reference's log-det ONCE and reuse it across multiple JBLD
    comparisons instead of recomputing it per (window, reference) pair -- a
    further optimization available on top of the base metric swap; see the
    note in distances.py.
    """
    if _use_torch():
        return _logdet_spd_torch(C)
    return _logdet_spd_np(C)


def jbld_divergence(P, Q) -> np.ndarray:
    """Jensen-Bregman LogDet Divergence (symmetric Stein divergence), Sra 2012:
        JBLD(P, Q) = log det((P+Q)/2) - 0.5 * (log det P + log det Q)
    Non-negative for SPD P, Q; zero iff P == Q. NOT itself a metric (fails the
    triangle inequality in general) -- see jbld_distance() for the metric
    version. P, Q broadcast over leading dims, mirroring airm_distance.
    """
    P = _np(P)
    Q = _np(Q)
    avg = symmetrize(0.5 * (P + Q))
    return logdet_spd(avg) - 0.5 * (logdet_spd(P) + logdet_spd(Q))


def jbld_distance(P, Q) -> np.ndarray:
    """sqrt(JBLD(P, Q)). Sra (2012) shows the square root of the S-divergence
    satisfies the triangle inequality (a genuine metric), unlike the raw
    divergence -- so this, not jbld_divergence(), is the drop-in replacement
    for airm_distance() as a `deltaR`-style feature.
    """
    div = jbld_divergence(P, Q)
    return np.sqrt(np.clip(div, 0.0, None))


def jbld_divergence_from_logdets(P, Q, logdet_p, logdet_q) -> np.ndarray:
    """JBLD divergence when log det(P) and log det(Q) are ALREADY KNOWN.

    Only the (P,Q)-dependent log det((P+Q)/2) term is computed here -- the
    two single-matrix logdets are NOT recomputed, unlike jbld_divergence()
    which recomputes everything from scratch on every call. This matters a
    lot when the SAME P is compared against many Q's (or vice versa): e.g.
    dataset_builder's all-fold batched distance tensor compares every window
    against a fixed batch of N fold references -- logdet(Q) for that batch
    never changes across windows (compute it ONCE per batch, not once per
    window), and logdet(P) for one window never changes across the
    interictal/preictal comparisons made for it (compute it ONCE per
    window, not once per reference). Naively recomputing both was found to
    be the dominant real-world cost of the batched design -- for a batch of
    8 folds, thousands of redundant 180x180 Cholesky factorizations per
    window that never needed to change.
    """
    P = _np(P)
    Q = _np(Q)
    avg = symmetrize(0.5 * (P + Q))
    return logdet_spd(avg) - 0.5 * (logdet_p + logdet_q)


def jbld_distance_from_logdets(P, Q, logdet_p, logdet_q) -> np.ndarray:
    """sqrt(jbld_divergence_from_logdets(...)) -- see that function's
    docstring. Drop-in replacement for jbld_distance() wherever logdet(P)
    and/or logdet(Q) are already known and reusable across several calls.
    """
    div = jbld_divergence_from_logdets(P, Q, logdet_p, logdet_q)
    return np.sqrt(np.clip(div, 0.0, None))


# ---------------------------------------------------------------------------
# Frechet (Karcher) mean under AIRM
# ---------------------------------------------------------------------------
def frechet_mean(mats, *, axis: int = 0, weights=None, max_iter: int | None = None , tol: float | None = None) -> np.ndarray:
    """Weighted AIRM Frechet (Karcher) mean of SPD matrices along `axis`.

    mats : (..., K, ..., n, n) SPD matrices; the mean is taken over `axis`, which
           is treated as the sample axis. All other leading dims (e.g. channel x
           span) are batched independently. Returns shape with `axis` removed.
    weights : optional length-K non-negative weights (normalized internally);
              None -> equal weights. Used for the two-level (equal-per-patient)
              population anchors in references.py.
    """
    max_iter = cfg.RIEMANN_MEAN_MAX_ITER if max_iter is None else max_iter
    tol = cfg.RIEMANN_MEAN_TOL if tol is None else tol

    X = np.moveaxis(_np(mats), axis, 0)          # (K, ..., n, n)
    K = X.shape[0]
    if K == 0:
        raise ValueError("frechet_mean requires at least one matrix")

    if weights is None:
        w = np.full(K, 1.0 / K, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (K,):
            raise ValueError(f"weights must have shape ({K},), got {w.shape}")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        s = w.sum()
        if s <= 0:
            raise ValueError("weights must sum to a positive value")
        w = w / s
    w_b = w.reshape((K,) + (1,) * (X.ndim - 1))   # broadcast over trailing dims

    if K == 1:
        return symmetrize(X[0])

    # Initialize at the (SPD) weighted arithmetic mean.
    M = symmetrize(np.sum(w_b * X, axis=0))
    for _ in range(int(max_iter)):
        M_half = spd_sqrt(M)
        M_ih = spd_invsqrt(M)
        proj = M_ih[None] @ X @ M_ih[None]        # whiten each sample: (K, ..., n, n)
        tang = spd_log(proj)                       # tangent vectors at I
        Tbar = np.sum(w_b * tang, axis=0)          # weighted mean tangent
        M = symmetrize(M_half @ spd_exp(Tbar) @ M_half)
        step = np.sqrt(np.sum(Tbar * Tbar, axis=(-1, -2)))
        if float(np.max(step)) < tol:
            break
    return M

def log_euclidean_mean(mats, *, axis: int = 0, weights=None,
                        max_iter: int | None = None, tol: float | None = None) -> np.ndarray:
    """Log-Euclidean mean of SPD matrices along `axis`: exp(weighted average of
    log(C_i)). NOT the AIRM Frechet/Karcher mean above -- a different (flat,
    non-curved) mean on the SPD manifold (Arsigny et al. 2006), used here for
    exactly the reason they proposed it: closed-form, needs exactly ONE pass
    over the data (each matrix's log computed once, averaged, exponentiated
    once) instead of frechet_mean's iterative multi-pass Karcher refinement.

    `max_iter` / `tol` are accepted and ignored -- there is no iteration to
    control -- purely so this is a drop-in replacement at call sites that
    still pass them.
    """
    X = np.moveaxis(_np(mats), axis, 0)
    K = X.shape[0]
    if K == 0:
        raise ValueError("log_euclidean_mean requires at least one matrix")

    if weights is None:
        w = np.full(K, 1.0 / K, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (K,):
            raise ValueError(f"weights must have shape ({K},), got {w.shape}")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        s = w.sum()
        if s <= 0:
            raise ValueError("weights must sum to a positive value")
        w = w / s
    w_b = w.reshape((K,) + (1,) * (X.ndim - 1))

    if K == 1:
        return symmetrize(X[0])

    logs = spd_log(X)                       # batched: log of all K at once
    mean_log = np.sum(w_b * logs, axis=0)
    return symmetrize(spd_exp(mean_log))


# ---------------------------------------------------------------------------
# Self-test (NumPy path; no torch / scipy / pyriemann required)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running backend.py self-test ...\n")
    print("backend_info:", backend_info())
    rng = np.random.default_rng(cfg.SEED)
    n = 6

    def rand_spd(lead=()):
        A = rng.standard_normal(lead + (n, n))
        return A @ np.swapaxes(A, -1, -2) + n * np.eye(n)

    eye = np.eye(n)
    C = rand_spd((4,))                      # batched SPD

    # --- matrix functions ---
    S = spd_sqrt(C)
    assert np.allclose(S @ S, C, atol=1e-8)
    Ih = spd_invsqrt(C)
    assert np.allclose(Ih @ C @ Ih, eye, atol=1e-8)
    assert np.allclose(spd_inv(C) @ C, eye, atol=1e-8)
    assert np.allclose(spd_exp(spd_log(C)), C, atol=1e-8)
    assert np.allclose(spd_pow(C, 0.5), spd_sqrt(C), atol=1e-8)
    assert np.allclose(spd_pow(C, -0.5), spd_invsqrt(C), atol=1e-8)

    # --- AIRM distance: zero, symmetry, affine invariance, reference ---
    P, Q = rand_spd(), rand_spd()
    assert abs(float(airm_distance(P, P))) < 1e-9
    assert np.allclose(airm_distance(P, Q), airm_distance(Q, P), atol=1e-9)

    W = rng.standard_normal((n, n))
    while abs(np.linalg.det(W)) < 1e-2:
        W = rng.standard_normal((n, n))
    d_plain = airm_distance(P, Q)
    d_congr = airm_distance(W @ P @ W.T, W @ Q @ W.T)
    assert np.allclose(d_plain, d_congr, atol=1e-7), "AIRM must be affine-invariant"

    ref_eigs = np.linalg.eigvals(np.linalg.solve(P, Q)).real
    d_ref = np.sqrt(np.sum(np.log(np.clip(ref_eigs, _EPS, None)) ** 2))
    assert np.allclose(d_plain, d_ref, atol=1e-6)

    # --- JBLD (Stein divergence): zero, symmetry, affine invariance, reference ---
    assert abs(float(jbld_distance(P, P))) < 1e-9
    assert np.allclose(jbld_distance(P, Q), jbld_distance(Q, P), atol=1e-9)

    d_plain_jbld = jbld_distance(P, Q)
    d_congr_jbld = jbld_distance(W @ P @ W.T, W @ Q @ W.T)
    assert np.allclose(d_plain_jbld, d_congr_jbld, atol=1e-6), "JBLD must be affine-invariant"

    logdet_p_ref = float(np.linalg.slogdet(P)[1])
    logdet_q_ref = float(np.linalg.slogdet(Q)[1])
    logdet_avg_ref = float(np.linalg.slogdet(0.5 * (P + Q))[1])
    ref_jbld = logdet_avg_ref - 0.5 * (logdet_p_ref + logdet_q_ref)
    assert ref_jbld >= -1e-9, "JBLD divergence must be non-negative for SPD inputs"
    assert np.allclose(jbld_divergence(P, Q), ref_jbld, atol=1e-6)
    assert np.allclose(jbld_distance(P, Q), np.sqrt(max(ref_jbld, 0.0)), atol=1e-6)

    # --- jbld_*_from_logdets must match the from-scratch versions exactly,
    # given the correct precomputed logdets ---
    logdet_P = logdet_spd(P)
    logdet_Q = logdet_spd(Q)
    assert np.allclose(jbld_divergence_from_logdets(P, Q, logdet_P, logdet_Q),
                       jbld_divergence(P, Q), atol=1e-9)
    assert np.allclose(jbld_distance_from_logdets(P, Q, logdet_P, logdet_Q),
                       jbld_distance(P, Q), atol=1e-9)

    # batched logdet_spd matches np.linalg.slogdet
    Cb = rand_spd((4,))
    sign_ref, logdet_ref = np.linalg.slogdet(Cb)
    assert np.all(sign_ref > 0), "synthetic SPD batch must have positive determinant"
    assert np.allclose(logdet_spd(Cb), logdet_ref, atol=1e-6)

    # --- Frechet mean ---
    C1 = rand_spd()
    assert np.allclose(frechet_mean(np.stack([C1] * 5)), C1, atol=1e-8)

    K = 7
    d = rng.uniform(0.5, 3.0, size=(K, n))
    Xd = np.stack([np.diag(d[k]) for k in range(K)])
    fm = frechet_mean(Xd, axis=0)
    expected = np.diag(np.exp(np.mean(np.log(d), axis=0)))   # geometric mean on the diagonal
    assert np.allclose(fm, expected, atol=1e-7)

    # --- log_euclidean_mean: closed-form, matches AIRM Karcher EXACTLY for
    # commuting (diagonal) matrices -- both reduce to the elementwise
    # geometric mean when eigenvectors are shared, a nice free sanity check ---
    lem = log_euclidean_mean(Xd, axis=0)
    assert np.allclose(lem, expected, atol=1e-7)
    assert np.allclose(lem, fm, atol=1e-7), "must coincide with AIRM mean on commuting matrices"

    # batched (channel x span) Frechet mean stays SPD
    Xb = rand_spd((K, 3, 2))                # (K, nc, sr, n, n)
    Mb = frechet_mean(Xb, axis=0)
    assert Mb.shape == (3, 2, n, n)
    assert np.linalg.eigvalsh(Mb).min() > 0.0

    # weighted mean with equal weights == unweighted
    assert np.allclose(frechet_mean(Xd, weights=np.ones(K)), fm, atol=1e-9)

    # determinism
    assert np.array_equal(spd_log(C), spd_log(C))

    print("OK - backend.py self-test passed.")