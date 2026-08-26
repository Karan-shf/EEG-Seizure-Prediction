"""
classifier.py
=============
Stage F (modeling): a pluggable classifier REGISTRY for the 486-D distance
features.

The headline model is Elastic-Net Logistic Regression (cfg.LR_PARAMS), but this
module is deliberately a registry: every classifier is a named factory, so KNN,
SVM, Random Forest, MLP, ... can be added and benchmarked against LR later
WITHOUT changing any call site. Downstream code only ever calls `make(name)`
and gets an unfitted sklearn estimator with a uniform fit / predict /
predict_proba API.

Input contract: tabular X of shape (n_windows, feature_dim) -- the standardized
distance features. The manifold-native benchmarks (mdm / fgmdm) instead consume
raw SPD matrices, so they live behind a SEPARATE contract (see MANIFOLD_NATIVE)
and are intentionally NOT part of this vector registry.

scikit-learn is imported lazily inside each factory so importing this module
stays dependency-light.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

from src import config as cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


# Manifold-native benchmarks operate on SPD matrices, not the 486-D vectors;
# they belong to a future modeling/manifold.py and are kept out of this
# (tabular) registry on purpose.
MANIFOLD_NATIVE: Tuple[str, ...] = ("mdm", "fgmdm")


@dataclass(frozen=True)
class ClassifierSpec:
    name: str
    factory: Callable[..., Any]   # (**overrides) -> unfitted sklearn estimator
    scale: bool = True            # prepend StandardScaler when cfg allows
    description: str = ""


_REGISTRY: Dict[str, ClassifierSpec] = {}


def register(name: str, factory: Callable[..., Any], *, scale: bool = True,
             description: str = "", overwrite: bool = False) -> None:
    """Register a classifier factory under `name`. Set overwrite=True to replace."""
    key = str(name).lower()
    if key in _REGISTRY and not overwrite:
        raise ValueError(f"classifier {key!r} already registered")
    _REGISTRY[key] = ClassifierSpec(key, factory, scale, description)


def available() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def is_registered(name: str) -> bool:
    return str(name).lower() in _REGISTRY


def describe() -> Dict[str, str]:
    return {k: v.description for k, v in sorted(_REGISTRY.items())}


# ---------------------------------------------------------------------------
# built-in factories (lazy sklearn imports)
# ---------------------------------------------------------------------------
def _lr(**overrides):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(**{**cfg.LR_PARAMS, **overrides})


def _lda(**overrides):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis(**overrides)


def _svm_rbf(**overrides):
    from sklearn.svm import SVC
    return SVC(**{**cfg.SVM_PARAMS, **overrides})


def _svm_linear(**overrides):
    from sklearn.svm import SVC
    params = {**cfg.SVM_PARAMS, "kernel": "linear"}
    params.update(overrides)
    return SVC(**params)


def _knn(**overrides):
    from sklearn.neighbors import KNeighborsClassifier
    params = {"n_neighbors": 15, "weights": "distance"}
    params.update(overrides)
    return KNeighborsClassifier(**params)


def _random_forest(**overrides):
    from sklearn.ensemble import RandomForestClassifier
    params = {"n_estimators": 400, "class_weight": None,
              "random_state": cfg.SEED, "n_jobs": -1}
    params.update(overrides)
    return RandomForestClassifier(**params)


def _mlp(**overrides):
    from sklearn.neural_network import MLPClassifier
    params = {"hidden_layer_sizes": (128, 64), "activation": "relu",
              "alpha": 1e-4, "max_iter": 500, "random_state": cfg.SEED}
    params.update(overrides)
    return MLPClassifier(**params)


register("lr", _lr, scale=True, description="Elastic-Net Logistic Regression (headline)")
register("lda", _lda, scale=True, description="Linear Discriminant Analysis")
register("svm_rbf", _svm_rbf, scale=True, description="RBF-kernel SVM")
register("svm_linear", _svm_linear, scale=True, description="linear-kernel SVM")
register("knn", _knn, scale=True, description="k-Nearest Neighbours")
register("rf", _random_forest, scale=False, description="Random Forest")
register("mlp", _mlp, scale=True, description="Multi-Layer Perceptron")


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def make(name: Optional[str] = None, *, standardize: Optional[bool] = None,
         **overrides):
    """Build an unfitted estimator by registry name (default: PRIMARY_CLASSIFIER).

    Returns an sklearn Pipeline with a uniform fit / predict / predict_proba
    API. A StandardScaler is prepended when the classifier is scale-sensitive
    AND cfg.STANDARDIZE_FEATURES is on; override per call with standardize=.
    Extra kwargs pass straight through to the underlying estimator.
    """
    from sklearn.pipeline import Pipeline

    key = (name or cfg.PRIMARY_CLASSIFIER).lower()
    if key in MANIFOLD_NATIVE:
        raise NotImplementedError(
            f"{key!r} is a manifold-native benchmark (SPD input); it is not part "
            f"of the vector classifier registry. See modeling/manifold.py (future).")
    spec = _REGISTRY.get(key)
    if spec is None:
        raise KeyError(f"unknown classifier {key!r}; available={available()}")

    est = spec.factory(**overrides)
    if standardize is None:
        do_scale = spec.scale and bool(cfg.STANDARDIZE_FEATURES)
    else:
        do_scale = bool(standardize)

    steps = []
    if do_scale:
        from sklearn.preprocessing import StandardScaler
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", est))
    return Pipeline(steps)


def predict_preictal_proba(model, X, *, preictal_label: Optional[int] = None) -> np.ndarray:
    """Return P(preictal) as a 1-D array, robust to class ordering and to
    estimators that expose only decision_function."""
    preictal_label = cfg.LABEL_PREICTAL if preictal_label is None else int(preictal_label)
    X = np.asarray(X, dtype=float)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        classes = list(getattr(model, "classes_",
                               [cfg.LABEL_INTERICTAL, cfg.LABEL_PREICTAL]))
        idx = classes.index(preictal_label) if preictal_label in classes else proba.shape[1] - 1
        return proba[:, idx]
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-s))   # squash to (0,1); monotonic in score
    return np.asarray(model.predict(X), dtype=float)


# ---------------------------------------------------------------------------
# self-test (author runs it; not run automatically)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running classifier.py self-test ...")
    rng = np.random.default_rng(cfg.SEED)
    n, d = 300, 40
    half = n // 2
    X = np.vstack([rng.normal(0.0, 1.0, (half, d)),
                   rng.normal(0.8, 1.0, (half, d))])
    y = np.array([cfg.LABEL_INTERICTAL] * half + [cfg.LABEL_PREICTAL] * half)

    print("  registry:", available())
    assert "lr" in available()

    default = make()  # PRIMARY_CLASSIFIER
    assert default.named_steps["clf"].__class__.__name__ == "LogisticRegression"

    for name in available():
        clf = make(name)
        clf.fit(X, y)
        p = predict_preictal_proba(clf, X)
        assert p.shape == (n,), (name, p.shape)
        assert np.all(p >= 0.0) and np.all(p <= 1.0), f"{name}: proba out of range"

    # manifold-native names are rejected by the vector registry
    try:
        make("mdm")
        raise AssertionError("mdm should not build in the vector registry")
    except NotImplementedError:
        pass

    # extensibility: register a new classifier without touching call sites
    def _dummy(**kw):
        from sklearn.dummy import DummyClassifier
        return DummyClassifier(strategy="stratified", random_state=cfg.SEED)
    register("dummy", _dummy, scale=False, description="stratified dummy")
    assert "dummy" in available()
    make("dummy").fit(X, y)

    print("classifier.py self-test OK")
