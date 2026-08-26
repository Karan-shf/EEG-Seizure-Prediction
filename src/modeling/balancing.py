"""
balancing.py
============
Stage F (modeling): class-imbalance handling for the 486-D distance features.

TRAIN-FOLD ONLY. The held-out LOPO patient is NEVER balanced -- it is scored on
its natural class distribution. Anchors and raw distance features must already
be computed on real, unbalanced data before this runs (balancing on synthetic /
pre-thinned data would corrupt the anchors' meaning).

Strict two-stage order (cfg section 7):
    1. cluster-centroid UNDER-sample interictal -> 2:1 (interictal:preictal)
    2. Borderline-SMOTE  OVER-sample  preictal   -> 1:1
    3. (caller then fits the classifier)

Everything is a config-driven REGISTRY so the method and ratios can be swapped
(or brand-new samplers registered) without touching any call site.
imbalanced-learn is imported lazily inside each factory so importing this
module stays dependency-light.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from src import config as cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

Array = np.ndarray


# ---------------------------------------------------------------------------
# Sampler registries   name -> factory(sampling_strategy, seed, **kw)
# ---------------------------------------------------------------------------
_UNDERSAMPLERS: Dict[str, Callable[..., Any]] = {}
_OVERSAMPLERS: Dict[str, Callable[..., Any]] = {}


def register_undersampler(name: str, factory: Callable[..., Any], *,
                          overwrite: bool = False) -> None:
    key = str(name).lower()
    if key in _UNDERSAMPLERS and not overwrite:
        raise ValueError(f"undersampler {key!r} already registered")
    _UNDERSAMPLERS[key] = factory


def register_oversampler(name: str, factory: Callable[..., Any], *,
                         overwrite: bool = False) -> None:
    key = str(name).lower()
    if key in _OVERSAMPLERS and not overwrite:
        raise ValueError(f"oversampler {key!r} already registered")
    _OVERSAMPLERS[key] = factory


def available_undersamplers() -> Tuple[str, ...]:
    return tuple(sorted(_UNDERSAMPLERS))


def available_oversamplers() -> Tuple[str, ...]:
    return tuple(sorted(_OVERSAMPLERS))


# --- built-in undersamplers (lazy imblearn imports) ---
def _cluster_centroids(sampling_strategy, seed, **kw):
    from imblearn.under_sampling import ClusterCentroids
    return ClusterCentroids(sampling_strategy=sampling_strategy,
                            random_state=seed, **kw)


def _random_under(sampling_strategy, seed, **kw):
    from imblearn.under_sampling import RandomUnderSampler
    return RandomUnderSampler(sampling_strategy=sampling_strategy,
                              random_state=seed, **kw)


def _nearmiss(sampling_strategy, seed, **kw):
    from imblearn.under_sampling import NearMiss
    return NearMiss(sampling_strategy=sampling_strategy, **kw)


register_undersampler("cluster_centroids", _cluster_centroids)
register_undersampler("random", _random_under)
register_undersampler("nearmiss", _nearmiss)


# --- built-in oversamplers (lazy imblearn imports) ---
def _borderline_smote(sampling_strategy, seed, *, k_neighbors=5, **kw):
    from imblearn.over_sampling import BorderlineSMOTE
    return BorderlineSMOTE(sampling_strategy=sampling_strategy,
                           random_state=seed, k_neighbors=k_neighbors, **kw)


def _smote(sampling_strategy, seed, *, k_neighbors=5, **kw):
    from imblearn.over_sampling import SMOTE
    return SMOTE(sampling_strategy=sampling_strategy, random_state=seed,
                 k_neighbors=k_neighbors, **kw)


def _adasyn(sampling_strategy, seed, *, n_neighbors=5, **kw):
    from imblearn.over_sampling import ADASYN
    return ADASYN(sampling_strategy=sampling_strategy, random_state=seed,
                  n_neighbors=n_neighbors, **kw)


def _random_over(sampling_strategy, seed, **kw):
    from imblearn.over_sampling import RandomOverSampler
    return RandomOverSampler(sampling_strategy=sampling_strategy,
                             random_state=seed, **kw)


register_oversampler("borderline_smote", _borderline_smote)
register_oversampler("smote", _smote)
register_oversampler("adasyn", _adasyn)
register_oversampler("random", _random_over)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def class_counts(y) -> Dict[int, int]:
    """Return {label: count} for a 1-D label array."""
    y = np.asarray(y).ravel()
    vals, cnts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}


def make_undersampler(method: Optional[str] = None, *, ratio: Optional[float] = None,
                      seed: Optional[int] = None, **kw):
    """Build an undersampler. `ratio` is the majority:minority target (2.0 -> 2:1)."""
    method = (method or cfg.UNDERSAMPLE_METHOD).lower()
    ratio = float(cfg.UNDERSAMPLE_INTERICTAL_TO_PREICTAL_RATIO if ratio is None else ratio)
    seed = cfg.SEED if seed is None else seed
    if method not in _UNDERSAMPLERS:
        raise KeyError(f"unknown undersampler {method!r}; have {available_undersamplers()}")
    # imblearn float sampling_strategy = n_minority / n_majority AFTER = 1 / (maj:min)
    ss = 1.0 / ratio
    return _UNDERSAMPLERS[method](ss, seed, **kw)


def make_oversampler(method: Optional[str] = None, *, ratio: Optional[float] = None,
                     seed: Optional[int] = None, **kw):
    """Build an oversampler. `ratio` is the minority:majority target (1.0 -> 1:1)."""
    method = (method or cfg.OVERSAMPLE_METHOD).lower()
    ratio = float(cfg.OVERSAMPLE_TARGET_RATIO if ratio is None else ratio)
    seed = cfg.SEED if seed is None else seed
    if method not in _OVERSAMPLERS:
        raise KeyError(f"unknown oversampler {method!r}; have {available_oversamplers()}")
    # imblearn float sampling_strategy = n_minority / n_majority AFTER
    return _OVERSAMPLERS[method](ratio, seed, **kw)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def balance(X, y, *, undersample_method: Optional[str] = None,
            undersample_ratio: Optional[float] = None,
            oversample_method: Optional[str] = None,
            oversample_ratio: Optional[float] = None,
            seed: Optional[int] = None,
            minority_label: Optional[int] = None,
            verbose: bool = True) -> Tuple[Array, Array]:
    """Two-stage train-fold resampling: undersample majority -> oversample minority.

    Returns (X_res, y_res). No-op (returns inputs) when cfg.USE_RESAMPLING is
    False or only one class is present. Each stage is skipped safely when it
    would have nothing to do or there are too few minority samples for SMOTE.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel().astype(int)

    if not bool(getattr(cfg, "USE_RESAMPLING", True)):
        if verbose:
            log.info("balance: USE_RESAMPLING is False; returning inputs unchanged")
        return X, y

    seed = cfg.SEED if seed is None else seed
    minority_label = cfg.LABEL_PREICTAL if minority_label is None else int(minority_label)
    u_ratio = float(cfg.UNDERSAMPLE_INTERICTAL_TO_PREICTAL_RATIO
                    if undersample_ratio is None else undersample_ratio)
    o_ratio = float(cfg.OVERSAMPLE_TARGET_RATIO
                    if oversample_ratio is None else oversample_ratio)

    counts = class_counts(y)
    if len(counts) < 2:
        log.warning("balance: only one class present (%s); skipping resampling", counts)
        return X, y

    n_min = counts.get(minority_label, 0)
    n_maj = max((c for lab, c in counts.items() if lab != minority_label), default=0)
    if verbose:
        log.info("balance: input counts=%s (minority=%s)", counts, minority_label)

    # ---- Stage 1: cluster-centroid undersample majority -> u_ratio:1 ----
    if n_min > 0 and n_maj > u_ratio * n_min:
        try:
            us = make_undersampler(undersample_method, ratio=u_ratio, seed=seed)
            X, y = us.fit_resample(X, y)
            if verbose:
                log.info("balance: after undersample=%s", class_counts(y))
        except Exception as exc:  # pragma: no cover - env/degenerate guard
            log.warning("balance: undersample skipped (%s)", exc)
    elif verbose:
        log.info("balance: undersample skipped (maj %d <= %.2f * min %d)",
                 n_maj, u_ratio, n_min)

    # ---- Stage 2: Borderline-SMOTE oversample minority -> o_ratio:1 ----
    counts = class_counts(y)
    n_min = counts.get(minority_label, 0)
    n_maj = max((c for lab, c in counts.items() if lab != minority_label), default=0)
    if n_min >= 2 and n_min < o_ratio * n_maj:
        k = int(min(5, n_min - 1))
        try:
            ovs = make_oversampler(oversample_method, ratio=o_ratio, seed=seed,
                                   k_neighbors=k)
            X, y = ovs.fit_resample(X, y)
            if verbose:
                log.info("balance: after oversample=%s", class_counts(y))
        except TypeError:
            # sampler without a k_neighbors kwarg (e.g. random) -> retry plain
            ovs = make_oversampler(oversample_method, ratio=o_ratio, seed=seed)
            X, y = ovs.fit_resample(X, y)
            if verbose:
                log.info("balance: after oversample=%s", class_counts(y))
        except Exception as exc:  # pragma: no cover
            log.warning("balance: oversample skipped (%s)", exc)
    elif verbose:
        log.info("balance: oversample skipped (min=%d, maj=%d)", n_min, n_maj)

    return np.asarray(X, dtype=float), np.asarray(y).ravel().astype(int)


# ---------------------------------------------------------------------------
# self-test (author runs it; not run automatically)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running balancing.py self-test ...")
    rng = np.random.default_rng(cfg.SEED)
    # Overlapping 2-class data so BorderlineSMOTE has real "in danger" samples.
    # (Cleanly separated classes give BorderlineSMOTE zero borderline points,
    # so it would synthesize nothing -- which is expected, not a bug.)
    n_feat = 20
    n_maj, n_min = 300, 60
    X = np.vstack([rng.normal(0.0, 1.0, (n_maj, n_feat)),
                   rng.normal(0.25, 1.0, (n_min, n_feat))])
    y = np.array([cfg.LABEL_INTERICTAL] * n_maj + [cfg.LABEL_PREICTAL] * n_min)

    print("  registries:", available_undersamplers(), available_oversamplers())
    assert "cluster_centroids" in available_undersamplers()
    assert "borderline_smote" in available_oversamplers()

    print("  input counts:", class_counts(y))

    # (1) exact ratio math, verified with plain SMOTE (always fills to target):
    #     300:60 -> undersample majority to 2*60=120 -> SMOTE minority to 120.
    Xr, yr = balance(X, y, oversample_method="smote", verbose=True)
    out = class_counts(yr)
    print("  smote output counts:", out)
    assert out[cfg.LABEL_INTERICTAL] == 120, out
    assert out[cfg.LABEL_PREICTAL] == 120, out
    assert Xr.shape[1] == n_feat

    # (2) default borderline_smote path runs and never SHRINKS the minority.
    #     It may stop short of a perfect 1:1 when few points are "in danger";
    #     that is correct BorderlineSMOTE behavior, not a failure.
    Xb, yb = balance(X, y, verbose=True)
    outb = class_counts(yb)
    print("  borderline output counts:", outb)
    assert outb[cfg.LABEL_INTERICTAL] == 120, outb
    assert 60 <= outb[cfg.LABEL_PREICTAL] <= 120, outb

    # no-op path when a single class is present
    Xs, ys = balance(X[:10], np.zeros(10, dtype=int), verbose=False)
    assert len(np.unique(ys)) == 1

    print("balancing.py self-test OK")
