"""
config.py
=========
Single source of truth for the SeizureHorizon rebuild.

Every constant that any other module needs -- file paths, sampling rate, the
canonical channel montage, filter-bank definition, windowing geometry, the
SPH/SOP labeling scheme, feature dimensions, classifier hyper-parameters and
evaluation settings -- is defined here and ONLY here.

Design rule: no other file may hard-code a magic number that belongs to the
pipeline definition. If a value is part of the experimental design (A-I in the
design doc), it lives in this file so a single edit propagates everywhere and
so every experiment is fully described by this one module.
"""

from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. Project paths
# ---------------------------------------------------------------------------
# config.py lives in <project>/src/, so the project root is two levels up.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw" / "chb-mit"    # original EDF + *-summary.txt
PROCESSED_DIR: Path = DATA_DIR / "processed"    # cached feature matrices

EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
LOG_DIR: Path = EXPERIMENTS_DIR / "logs"
CHECKPOINT_DIR: Path = EXPERIMENTS_DIR / "checkpoints"
RESULTS_DIR: Path = EXPERIMENTS_DIR / "results"

ALL_DIRS = [DATA_DIR, RAW_DIR, PROCESSED_DIR, EXPERIMENTS_DIR,
            LOG_DIR, CHECKPOINT_DIR, RESULTS_DIR]


def ensure_dirs() -> None:
    """Create every project directory if it does not already exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42


# ---------------------------------------------------------------------------
# 2. Signal / acquisition  [Req A]
# ---------------------------------------------------------------------------
FS: int = 256  # CHB-MIT sampling rate (Hz), identical for every recording.

# Canonical 18 bipolar channels, in FIXED order. Every EDF is reduced to
# exactly this montage (see preprocessing/montage.py). Order matters because
# the covariance matrix rows/cols are indexed by this list.
CHANNELS: tuple[str, ...] = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
)
N_CHANNELS: int = len(CHANNELS)  # 18


# ---------------------------------------------------------------------------
# 3. Filter bank  [Req D]
# ---------------------------------------------------------------------------
NOTCH_FREQ: float = 60.0     # US mains hum (CHB-MIT recorded in Boston)
NOTCH_QUALITY: float = 30.0  # Q factor for the IIR notch
FILTER_ORDER: int = 4        # Butterworth order (applied via filtfilt -> zero phase)

# name -> (low_hz, high_hz)
BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
N_BANDS: int = len(BANDS)  # 5


# ---------------------------------------------------------------------------
# 4. Windowing  [Req B]
# ---------------------------------------------------------------------------
WINDOW_SECONDS: float = 4.0
WINDOW_OVERLAP: float = 0.40  # 40% overlap between consecutive windows

WINDOW_SAMPLES: int = int(round(WINDOW_SECONDS * FS))                      # 1024
STRIDE_SAMPLES: int = int(round(WINDOW_SAMPLES * (1.0 - WINDOW_OVERLAP)))  # 614
STRIDE_SECONDS: float = STRIDE_SAMPLES / FS                                # ~2.398 s


# ---------------------------------------------------------------------------
# 5. Covariance / features  [Req C, D, E]
# ---------------------------------------------------------------------------
COV_ESTIMATOR: str = "oas"           # Oracle Approximating Shrinkage (pyriemann)
COV_FALLBACK_ESTIMATOR: str = "sch"  # Schaefer-Strimmer, if OAS ill-conditioned

# Tangent-space dimensionality. A symmetric n x n matrix has n(n+1)/2 unique
# entries, which is the length of its tangent-space (upper-triangular) vector.
TANGENT_DIM_PER_BAND: int = N_CHANNELS * (N_CHANNELS + 1) // 2  # 171
FEATURE_DIM: int = TANGENT_DIM_PER_BAND * N_BANDS              # 855

# Riemannian alignment (Req E): recenter each patient's covariances to the
# identity using a calibration anchor, fit on TRAIN patients only (no leakage).
ALIGNMENT_METHOD: str = "rct"  # Riemannian re-centering (pyriemann transfer.TLCenter)


# ---------------------------------------------------------------------------
# 6. Labeling: SPH / SOP timeline  [Req G]
# ---------------------------------------------------------------------------
SPH_MINUTES: int = 5                 # Seizure Prediction Horizon (fixed)
SPH_SECONDS: int = SPH_MINUTES * 60  # 300

SOP_GRID_MINUTES: tuple[int, ...] = (15, 30, 45, 60)  # Seizure Occurrence Period sweep
SOP_PRIMARY_MINUTES: int = 30         # headline operating point
SOP_ASPIRATIONAL_MINUTES: int = 60    # the "1 hour ahead" stretch goal

# Data hygiene (Req G): drop noisy / non-independent segments.
POSTICTAL_SECONDS: int = 3 * 3600     # exclude 3 h after each seizure ends
# A seizure is a usable "lead" seizure only if preceded by at least this much
# seizure-free EEG (otherwise it is part of a cluster / not independent).
LEAD_SEIZURE_MIN_PRECEDING_SECONDS: int = 4 * 3600
# Interictal windows must sit at least this far from ANY seizure.
INTERICTAL_GUARD_SECONDS: int = 4 * 3600

# Class labels (2-class problem, Req G)
LABEL_INTERICTAL: int = 0
LABEL_PREICTAL: int = 1


# ---------------------------------------------------------------------------
# 7. Classifier  [Req F]
# ---------------------------------------------------------------------------
PRIMARY_CLASSIFIER: str = "lr"
# Benchmarks evaluated on TRAIN folds only (never used to pick the final model
# on test patients).
BENCHMARK_CLASSIFIERS: tuple[str, ...] = ("lr", "lda", "svm_rbf", "mdm", "fgmdm")

LR_PARAMS: dict = {
    "C": 1.0,
    "class_weight": "balanced",  # Req F: interictal >> preictal imbalance
    "max_iter": 1000,
    "solver": "liblinear",
    "random_state": SEED,
}
SVM_PARAMS: dict = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "class_weight": "balanced",
    "probability": True,
    "random_state": SEED,
}

# Optional SMOTE oversampling (Req F: default OFF; class_weight is primary).
USE_SMOTE: bool = False

# Per-branch standardization before (future) fusion / classification.
STANDARDIZE_FEATURES: bool = True


# ---------------------------------------------------------------------------
# 8. Evaluation  [Req I]
# ---------------------------------------------------------------------------
CV_SCHEME: str = "lopo"  # Leave-One-Patient-Out

# Operating points for "Sensitivity @ fixed FPR/h" reporting.
TARGET_FPR_PER_HOUR: tuple[float, ...] = (0.1, 0.15, 0.2, 0.5)
PRIMARY_TARGET_FPR_PER_HOUR: float = 0.15

# Firing Power alarm logic (Req I): smooth the per-window preictal probability,
# raise an alarm when it crosses the threshold, then stay silent (refractory).
FIRING_POWER_THRESHOLD: float = 0.5  # default; tuned on train/val, never on test
# Refractory + firing-power window are SOP-dependent and set at runtime in
# evaluation/alarms.py; these are defaults / fallbacks.
REFRACTORY_SECONDS: int = SPH_SECONDS + SOP_PRIMARY_MINUTES * 60

# How the decision threshold is chosen: on TRAIN/VAL only, to hit a target FPR.
THRESHOLD_SELECTION: str = "train_val_fixed_fpr"


# ---------------------------------------------------------------------------
# 9. Convenience summary
# ---------------------------------------------------------------------------
def summary() -> str:
    """Return a human-readable dump of the most important derived settings."""
    return "\n".join([
        "SeizureHorizon configuration",
        "============================",
        f"Sampling rate            : {FS} Hz",
        f"Channels                 : {N_CHANNELS} ({', '.join(CHANNELS[:4])}, ...)",
        f"Filter bands             : {N_BANDS} -> {list(BANDS.keys())}",
        f"Window                   : {WINDOW_SECONDS}s = {WINDOW_SAMPLES} samples",
        f"Stride                   : {STRIDE_SECONDS:.3f}s = {STRIDE_SAMPLES} samples "
        f"({int(WINDOW_OVERLAP*100)}% overlap)",
        f"Tangent dim / band       : {TANGENT_DIM_PER_BAND}",
        f"Feature dim (5 bands)    : {FEATURE_DIM}",
        f"SPH                      : {SPH_MINUTES} min",
        f"SOP grid                 : {SOP_GRID_MINUTES} min (primary {SOP_PRIMARY_MINUTES})",
        f"Postictal exclusion      : {POSTICTAL_SECONDS // 3600} h",
        f"Primary classifier       : {PRIMARY_CLASSIFIER}",
        f"CV scheme                : {CV_SCHEME}",
    ])


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running config.py self-test ...\n")

    # --- geometry / derived constants ---
    assert FS == 256
    assert N_CHANNELS == 18, f"expected 18 channels, got {N_CHANNELS}"
    assert len(set(CHANNELS)) == N_CHANNELS, "duplicate channel in montage!"
    assert WINDOW_SAMPLES == 1024, f"window samples wrong: {WINDOW_SAMPLES}"
    assert STRIDE_SAMPLES == 614, f"stride samples wrong: {STRIDE_SAMPLES}"
    assert N_BANDS == 5
    assert TANGENT_DIM_PER_BAND == 171
    assert FEATURE_DIM == 855, f"feature dim wrong: {FEATURE_DIM}"

    # --- filter bank sanity: ascending, non-overlapping, below Nyquist ---
    nyq = FS / 2
    prev_high = 0.0
    for name, (lo, hi) in BANDS.items():
        assert 0 < lo < hi < nyq, f"band {name} out of range"
        assert lo >= prev_high, f"band {name} overlaps previous band"
        prev_high = hi

    # --- labeling sanity ---
    assert SPH_SECONDS == 300
    assert tuple(sorted(SOP_GRID_MINUTES)) == SOP_GRID_MINUTES, "SOP grid must be sorted"
    assert SOP_PRIMARY_MINUTES in SOP_GRID_MINUTES
    assert SOP_ASPIRATIONAL_MINUTES in SOP_GRID_MINUTES
    assert LABEL_INTERICTAL != LABEL_PREICTAL

    # --- evaluation sanity ---
    assert PRIMARY_TARGET_FPR_PER_HOUR in TARGET_FPR_PER_HOUR
    assert PRIMARY_CLASSIFIER in BENCHMARK_CLASSIFIERS

    # --- paths are Path objects and dirs can be created ---
    assert isinstance(PROJECT_ROOT, Path)
    ensure_dirs()
    for d in ALL_DIRS:
        assert d.exists(), f"directory not created: {d}"

    print(summary())
    print("\nAll config.py self-tests passed.")