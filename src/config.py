"""
config.py
=========
Single source of truth for the SeizureHorizon rebuild.

Every constant that any other module needs -- file paths, sampling rate, the
canonical channel montage, the single cleanup filter, windowing geometry, the
RSMMTN spatio-temporal feature definition, per-(channel x span) Riemannian
recentering, the Distance-to-References feature reduction, the SPH/SOP labeling
+ recording-continuity scheme, class-balancing, classifier hyper-parameters and
evaluation settings -- is defined here and ONLY here.

Design rule: no other file may hard-code a magic number that belongs to the
pipeline definition. If a value is part of the experimental design (A-I in the
design doc), it lives in this file so a single edit propagates everywhere and
so every experiment is fully described by this one module.

Pivot note (19 Aug 2026): the Riemannian covariance / 5-band filter-bank stream
is retired. Features are alpha-blended multi-span RSMMTN transition networks --
one SPD matrix per (channel, span) -- recentered per (channel x span) per
patient.

Pivot note v2 (25 Aug 2026): the tangent-vectorize -> in-fold PCA path is
RETIRED (it produced a dense ~2.6M-wide per-window vector: memory-infeasible).
Each recentered SPD matrix is instead reduced to 3 geodesic DISTANCES to
physically meaningful anchors (own baseline, pooled interictal mean, pooled
preictal mean) -> 3 x span x 18 = 486 features/window at m=9. Class imbalance
is handled by cluster-centroid undersampling -> Borderline-SMOTE, and the
classifier is Elastic-Net Logistic Regression. See design doc sections 3-5 and
the RSMMTN Pipeline v2 note.
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
# exactly this montage (see preprocessing/montage.py). Order matters: post-pivot
# each channel becomes its OWN RSMMTN SPD matrix, and the per-(channel, span)
# distance features are laid out in THIS order -- so feature block k always
# refers to the same electrode pair across patients. (There is no longer a
# single cross-channel covariance matrix.)
CHANNELS: tuple[str, ...] = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
)
N_CHANNELS: int = len(CHANNELS)  # 18


# ---------------------------------------------------------------------------
# 3. Filtering -- single broadband cleanup  [Req D]
# ---------------------------------------------------------------------------
# NOT a feature step: one wide band-pass that only removes DC drift and
# HF/EMG noise. The 5-band filter bank is RETIRED (it fought RSMMTN's
# parameter-free broadband design and multiplied cost with no validated gain).
BANDPASS_LOW_HZ: float = 0.5
BANDPASS_HIGH_HZ: float = 45.0     # gamma-edge cap preserved
FILTER_ORDER: int = 4              # Butterworth, applied via filtfilt -> zero phase

# 60 Hz US-mains notch: specified for reproducibility but INERT, because 60 Hz
# already sits in the stop-band of the 0.5-45 Hz cleanup filter.
NOTCH_FREQ_HZ: float = 60.0
NOTCH_QUALITY: float = 30.0        # Q factor for the IIR notch
NOTCH_ENABLED: bool = True         # kept for reproducibility; inert given the passband


# ---------------------------------------------------------------------------
# 4. Windowing  [Req B]
# ---------------------------------------------------------------------------
# RSMMTN is ORDER-SENSITIVE (it counts symbol transitions), so windows are
# genuine time-ordered segments built per contiguous recording span and must
# NEVER cross an inter-file gap (see section 6 continuity).
WINDOW_SECONDS: float = 6.0                                   # default
WINDOW_SECONDS_GRID: tuple[float, ...] = (4.0, 6.0, 8.0, 10.0)  # train/val-only sweep
WINDOW_OVERLAP: float = 0.40                                  # 40% overlap


def window_samples(seconds: float = WINDOW_SECONDS) -> int:
    """Number of samples in a window of the given length."""
    return int(round(seconds * FS))


def stride_samples(seconds: float = WINDOW_SECONDS,
                   overlap: float = WINDOW_OVERLAP) -> int:
    """Hop size in samples for the given window length / overlap."""
    return int(round(window_samples(seconds) * (1.0 - overlap)))


WINDOW_SAMPLES: int = window_samples()                 # 1536 @ 6 s
STRIDE_SAMPLES: int = stride_samples()                 # 922
STRIDE_SECONDS: float = STRIDE_SAMPLES / FS            # ~3.602 s


# ---------------------------------------------------------------------------
# 5. RSMMTN spatio-temporal features  [Req C, D, E]
# ---------------------------------------------------------------------------
# --- Symbol space (RSMMTN) [Req C] ---
# Scatter x(t)=X_c(t), y(t)=blend; polar (r, angle) discretized into symbols.
N_ANGULAR_BINS: int = 18          # angle in [-90, 90] deg
N_RADIAL_BINS: int = 10           # radius in [0, r_max], dr = r_max / 10
N_SYMBOLS: int = N_ANGULAR_BINS * N_RADIAL_BINS   # 180  -> SPD is 180 x 180
SPD_DIM: int = N_SYMBOLS

# SPD construction:  C^i = I + A^i (A^i)^T   with A^i the row-normalized
# transition (adjacency) matrix for span i. The + I guarantees the matrix is
# SPD and well-conditioned even when the 180x180 network is sparsely populated.

# --- alpha spatio-temporal blend [Req D] ---
# y(t) = alpha * d2X_c(t) + (1 - alpha) * L_c(t) ; x(t) = X_c(t).
# alpha = 1 -> temporal-only (paper), alpha = 0 -> spatial-only, else blend.
# Each alpha is a FULLY INDEPENDENT experiment (own recentering / anchors /
# distances / LR / LOPO). alpha arms are NEVER concatenated across alpha.
ALPHA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

# --- multi-span cumulative roof [Req D] ---
# span roof m => the network cumulated over spans {1, 2, ..., m}: m adjacency
# matrices (NOT a single step-m matrix). The grid lists roofs; skipped evens
# live inside the cumulative sets. Compute all spans up to SPAN_MAX once.
SPAN_ROOF_GRID: tuple[int, ...] = (1, 3, 5, 7, 9)
SPAN_MAX: int = max(SPAN_ROOF_GRID)               # 9

# --- spatial term: weighted graph Laplacian (Hjorth-style k-NN) [Req D] ---
# L_c(t) = X_c(t) - sum_j w_cj * X_j(t) / sum_j w_cj ,  w_cj = 1 / dist(c, j),
# restricted to the k nearest neighbours. This is the classic Hjorth
# nearest-neighbour Laplacian -- the established sparse-montage method -- NOT
# spherical-spline CSD (which needs ~>=64 channels and extrapolates badly at
# the 18-channel array edges where foci localize). Distances are geodesic (arc
# length) on the standard_1020 sphere between the MIDPOINTS of each bipolar
# pair's two electrodes; honest caveat: this is a local spatial-contrast
# operator over the double-banana montage, not a textbook monopolar CSD.
LAPLACIAN_K: int = 4              # ~ Hjorth's original orthogonal-neighbour stencil
LAPLACIAN_WEIGHTING: str = "inverse_distance"
LAPLACIAN_DISTANCE: str = "geodesic"      # arc length on the unit sphere
MONTAGE_NAME: str = "standard_1020"       # mne.channels.make_standard_montage
# k kept out of the headline (alpha, span, window) grid; sanity-check only:
LAPLACIAN_K_SANITY: tuple[int, ...] = (4, 6)

# --- per-(channel x span) Riemannian recentering [Req E] ---
# G_{patient,c,i} = RiemannianMean(C_i over that patient's INTERICTAL epochs
# only, label-free); C' = G^-1/2 . C . G^-1/2. Calibration epochs are held
# DISJOINT from scored epochs. Honesty caveat: this is unsupervised domain
# adaptation / calibration, NOT strict zero-shot. Post-recenter, the window's
# own baseline sits at the identity I, so d_baseline = deltaR(C', I) measures
# how far the window deviates from that patient's normal.
RECENTER_METHOD: str = "riemannian_recenter"   # RCT-analogue, now at 180 x 180
RECENTER_GRANULARITY: str = "channel_span"     # one anchor per (channel, span)
RECENTER_ANCHOR_STATE: str = "interictal"      # label-free calibration only

# --- Distance-to-References features [Req C/E, v2] ---
# v2 pivot (25 Aug 2026): the tangent-vectorize -> in-fold PCA path is RETIRED
# (a dense ~2.6M-wide per-window vector; see design doc section 3). Instead,
# each window's recentered SPD matrix C' (per channel, per span) is reduced to
# a SMALL set of geodesic distances to physically meaningful anchors:
#   d_baseline   = deltaR(C', I)                 # deviation from own baseline
#   d_interictal = deltaR(C', M_interictal_ci)   # to pooled source interictal mean
#   d_preictal   = deltaR(C', M_preictal_ci)     # to pooled source preictal mean
# deltaR = affine-invariant Riemannian metric (AIRM):
#   deltaR(P, Q) = || log( eigvals( P^-1 Q ) ) ||_2
N_REFERENCES: int = 3             # baseline (I), interictal mean, preictal mean
REFERENCE_NAMES: tuple[str, ...] = ("baseline", "interictal", "preictal")
RIEMANN_METRIC: str = "airm"      # affine-invariant Riemannian metric

# The two population anchors (M_interictal, M_preictal) are Riemannian (Frechet)
# means, computed per (channel x span) from TRAINING-fold (source) patients only
# -- pooled AFTER each source patient's windows are recentered to their own
# G_patient -- and recomputed fresh inside every LOPO fold. G_patient itself is
# patient-local (that patient's own interictal windows) and may be formed for
# the held-out patient too (uses only their own unlabeled baseline).
RIEMANN_MEAN_MAX_ITER: int = 50
RIEMANN_MEAN_TOL: float = 1e-6
ANCHOR_SCOPE: str = "train_fold_only"    # M_interictal / M_preictal: source patients only


def feature_dim(span_roof: int) -> int:
    """Distance-to-reference feature width for a cumulative span roof m.

    = N_REFERENCES (3) x span_roof (m) x N_CHANNELS (18).  486 at m = 9.
    """
    return N_REFERENCES * span_roof * N_CHANNELS


# ---------------------------------------------------------------------------
# 6. Labeling: SPH / SOP timeline + recording continuity  [Req G]
# ---------------------------------------------------------------------------
SPH_MINUTES: int = 5                 # Seizure Prediction Horizon (fixed)
SPH_SECONDS: int = SPH_MINUTES * 60  # 300

SOP_GRID_MINUTES: tuple[int, ...] = (15, 30, 45, 60)  # Seizure Occurrence Period sweep
SOP_PRIMARY_MINUTES: int = 30         # headline operating point
SOP_ASPIRATIONAL_MINUTES: int = 60    # the "1 hour ahead" stretch goal

# Data hygiene (Req G, v2): symmetric seizure-exclusion buffer.
# v2 adopts Truong et al. 2018's convention: interictal windows must sit at
# least 4 h from ANY seizure in BOTH directions (pre and post). This replaces
# the previous asymmetric 4 h pre-guard / 3 h postictal split with a single
# symmetric +/- 4 h buffer.
SEIZURE_EXCLUSION_SECONDS: int = 4 * 3600   # +/- 4 h symmetric (Truong 2018)
POSTICTAL_SECONDS: int = SEIZURE_EXCLUSION_SECONDS       # 4 h after each seizure ends
INTERICTAL_GUARD_SECONDS: int = SEIZURE_EXCLUSION_SECONDS  # 4 h before any seizure
# A seizure is a usable "lead" seizure only if preceded by at least this much
# seizure-free EEG (otherwise it is part of a cluster / not independent).
LEAD_SEIZURE_MIN_PRECEDING_SECONDS: int = 4 * 3600

# --- Interictal candidate-pool subsampling [Req F, v2] ---
# CHB-MIT has ~40 h/patient; computing the full matrix pipeline for every
# possible interictal window is intractable and unnecessary (and not what
# published CHB-MIT work does). Instead, subsample a GENEROUS pool from the
# "safe" interictal zone (outside every seizure's +/- 4 h exclusion) -- large
# enough for stable anchor estimation and to give the balancing step real
# diversity. Do NOT collapse to a class ratio here; balancing happens later,
# in-fold, on the small 486-D distance features.
INTERICTAL_POOL_MULTIPLIER: float = 5.0   # pool ~= 5 x the preictal window count
INTERICTAL_POOL_STRATIFY: str = "recording_file"  # spread across files / time-of-day
INTERICTAL_POOL_SEED: int = SEED

# --- Recording continuity (Req G, critical) ---
# CHB-MIT gives each patient as one continuous recording chopped into ~1 h EDFs
# with small (occasionally large) inter-file gaps; seizure onsets in the
# summary are offsets RELATIVE to their own file. Labels are computed on a
# per-patient GLOBAL timeline stitched from File Start/End clocks
# (summary_parser handles midnight-wrap). Because RSMMTN is order-sensitive, a
# window / transition must NEVER cross an inter-file gap.
#
# INTER_FILE_GAP_TOLERANCE_SECONDS is the MERGE threshold used when building a
# patient's CONTIGUOUS SEGMENTS from consecutive EDFs:
#   * gap <= tolerance  -> the two files join into ONE contiguous segment
#                          (samples concatenated / treated as continuous; only
#                          a few windows may straddle the sub-tolerance seam).
#   * gap  > tolerance  -> HARD BREAK: a new segment starts and no window or
#                          transition ever spans it.
# It NEVER drops a file -- every file is always used; the tolerance only moves
# where segment boundaries fall. 10 s absorbs the routine few-second gaps
# between CHB-MIT files (summary clocks have 1 s resolution) while still
# treating genuine multi-minute / hour pauses as hard breaks.
INTER_FILE_GAP_TOLERANCE_SECONDS: float = 10.0
# Patients whose summaries lack wall clocks cannot get an absolute timeline;
# fall back to concatenation by EDF sample counts, gaps = unknown hard breaks.
PATIENTS_WITHOUT_CLOCKS: tuple[str, ...] = ("chb24",)

# Class labels (2-class problem, Req G)
LABEL_INTERICTAL: int = 0
LABEL_PREICTAL: int = 1


# ---------------------------------------------------------------------------
# 7. Classifier + class balancing  [Req F]
# ---------------------------------------------------------------------------
PRIMARY_CLASSIFIER: str = "lr"
# lr / lda / svm_rbf act on the 486-D distance features; mdm / fgmdm are
# manifold-native sanity checks that operate directly on the SPD matrices. All
# benchmarks are evaluated on TRAIN folds only (never used to pick the final
# model on test patients).
BENCHMARK_CLASSIFIERS: tuple[str, ...] = ("lr", "lda", "svm_rbf", "mdm", "fgmdm")

# v2: Elastic-Net-regularized Logistic Regression on the standardized 486-D
# distance features. Elastic-Net (L1 + L2) keeps L1's sparse feature selection
# -- an interpretable readout of which (channel, span, reference) combos matter
# for seizure localization -- while L2's grouping effect stabilizes that
# selection across LOPO folds given the correlated distance features. Requires
# the saga solver. Class balance is handled UPSTREAM by resampling (to 1:1),
# so no class_weight is applied here.
LR_PARAMS: dict = {
    "penalty": "elasticnet",
    "solver": "saga",          # only saga supports elasticnet
    "l1_ratio": 0.5,           # default; tuned train/val-only over L1_RATIO_GRID
    "C": 1.0,                  # default; tuned train/val-only over C_GRID
    "class_weight": None,      # balancing done via resampling, not weights
    "max_iter": 5000,          # saga needs more iterations to converge
    "random_state": SEED,
}
L1_RATIO_GRID: tuple[float, ...] = (0.2, 0.5, 0.8)   # train/val-only tuning
C_GRID: tuple[float, ...] = (0.1, 1.0, 10.0)         # train/val-only tuning

SVM_PARAMS: dict = {
    "C": 1.0,
    "kernel": "rbf",
    "gamma": "scale",
    "class_weight": "balanced",
    "probability": True,
    "random_state": SEED,
}

# --- Class imbalance handling [Req F, v2] ---
# STRICT ORDER (train fold only; never touches the held-out patient):
#   1. real, imbalanced 486-D features
#   2. cluster-centroid UNDER-sample interictal -> 2:1 (interictal:preictal)
#   3. Borderline-SMOTE OVER-sample preictal    -> 1:1
#   4. fit the classifier
# Anchors + raw features MUST be computed on real, unbalanced data first;
# balancing on synthetic / pre-thinned data would corrupt the anchors' meaning.
# The held-out patient is NEVER balanced (evaluated on its natural distribution).
USE_RESAMPLING: bool = True
UNDERSAMPLE_METHOD: str = "cluster_centroids"
# interictal:preictal ratio AFTER cluster-centroid undersampling (2:1).
UNDERSAMPLE_INTERICTAL_TO_PREICTAL_RATIO: float = 2.0
OVERSAMPLE_METHOD: str = "borderline_smote"
# interictal:preictal ratio AFTER Borderline-SMOTE (1:1).
OVERSAMPLE_TARGET_RATIO: float = 1.0
BALANCING_SCOPE: str = "train_fold_only"

# Per-alpha standardization (z-score) of the 486-D features. Fit IN-FOLD on
# training data only, applied to the held-out patient.
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
        "SeizureHorizon configuration (RSMMTN spatio-temporal, v2)",
        "=========================================================",
        f"Sampling rate            : {FS} Hz",
        f"Channels                 : {N_CHANNELS} ({', '.join(CHANNELS[:4])}, ...)",
        f"Cleanup band-pass        : {BANDPASS_LOW_HZ}-{BANDPASS_HIGH_HZ} Hz "
        f"(order {FILTER_ORDER}, zero-phase; 60 Hz notch inert)",
        f"Window (default)         : {WINDOW_SECONDS}s = {WINDOW_SAMPLES} samples",
        f"Window grid              : {WINDOW_SECONDS_GRID} s",
        f"Stride                   : {STRIDE_SECONDS:.3f}s = {STRIDE_SAMPLES} samples "
        f"({int(WINDOW_OVERLAP*100)}% overlap)",
        f"RSMMTN symbols           : {N_ANGULAR_BINS} x {N_RADIAL_BINS} = {N_SYMBOLS} "
        f"(SPD {SPD_DIM}x{SPD_DIM})",
        f"alpha grid               : {ALPHA_GRID}",
        f"Span roof grid           : {SPAN_ROOF_GRID} (max {SPAN_MAX})",
        f"Spatial Laplacian        : weighted graph, k={LAPLACIAN_K} ({LAPLACIAN_DISTANCE})",
        f"Recentering              : {RECENTER_GRANULARITY} ({RECENTER_ANCHOR_STATE} anchor)",
        f"References / window      : {N_REFERENCES} {REFERENCE_NAMES}",
        f"Feature dim @m={SPAN_MAX}        : {feature_dim(SPAN_MAX):,} "
        f"(= {N_REFERENCES} x {SPAN_MAX} x {N_CHANNELS})",
        f"SPH                      : {SPH_MINUTES} min",
        f"SOP grid                 : {SOP_GRID_MINUTES} min (primary {SOP_PRIMARY_MINUTES})",
        f"Seizure exclusion        : +/- {SEIZURE_EXCLUSION_SECONDS // 3600} h (symmetric)",
        f"Interictal pool          : {INTERICTAL_POOL_MULTIPLIER}x preictal count",
        f"Balancing                : {UNDERSAMPLE_METHOD} -> "
        f"{UNDERSAMPLE_INTERICTAL_TO_PREICTAL_RATIO:.0f}:1 -> {OVERSAMPLE_METHOD} -> 1:1",
        f"Classifier               : {PRIMARY_CLASSIFIER} "
        f"({LR_PARAMS['penalty']}, solver={LR_PARAMS['solver']})",
        f"CV scheme                : {CV_SCHEME}",
    ])


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running config.py self-test ...\n")

    # --- signal / geometry ---
    assert FS == 256
    assert N_CHANNELS == 18, f"expected 18 channels, got {N_CHANNELS}"
    assert len(set(CHANNELS)) == N_CHANNELS, "duplicate channel in montage!"

    # --- single cleanup filter (no band split) ---
    nyq = FS / 2
    assert 0 < BANDPASS_LOW_HZ < BANDPASS_HIGH_HZ < nyq, "bad cleanup band-pass"
    assert BANDPASS_HIGH_HZ <= NOTCH_FREQ_HZ, "notch must sit outside passband to be inert"

    # --- windowing ---
    assert WINDOW_SAMPLES == 1536, f"window samples wrong: {WINDOW_SAMPLES}"
    assert STRIDE_SAMPLES == 922, f"stride samples wrong: {STRIDE_SAMPLES}"
    assert WINDOW_SECONDS in WINDOW_SECONDS_GRID
    assert window_samples(4.0) == 1024, "4 s reference should be 1024 samples"

    # --- RSMMTN symbol / SPD geometry ---
    assert N_SYMBOLS == 180
    assert N_ANGULAR_BINS * N_RADIAL_BINS == N_SYMBOLS
    assert SPD_DIM == 180

    # --- alpha grid: independent runs spanning spatial(0)..temporal(1) ---
    assert ALPHA_GRID[0] == 0.0 and ALPHA_GRID[-1] == 1.0
    assert all(0.0 <= a <= 1.0 for a in ALPHA_GRID)
    assert tuple(sorted(ALPHA_GRID)) == ALPHA_GRID, "alpha grid must be sorted"

    # --- span roofs: cumulative, ascending ---
    assert tuple(sorted(SPAN_ROOF_GRID)) == SPAN_ROOF_GRID, "span roofs must be sorted"
    assert SPAN_MAX == max(SPAN_ROOF_GRID)

    # --- spatial laplacian ---
    assert LAPLACIAN_K >= 1
    assert LAPLACIAN_K in LAPLACIAN_K_SANITY

    # --- Distance-to-References feature width (486 at m=9) ---
    assert N_REFERENCES == 3
    assert len(REFERENCE_NAMES) == N_REFERENCES
    assert feature_dim(9) == N_REFERENCES * 9 * N_CHANNELS == 486
    assert feature_dim(1) == N_REFERENCES * N_CHANNELS == 54
    assert feature_dim(5) == 270

    # --- labeling ---
    assert SPH_SECONDS == 300
    assert tuple(sorted(SOP_GRID_MINUTES)) == SOP_GRID_MINUTES, "SOP grid must be sorted"
    assert SOP_PRIMARY_MINUTES in SOP_GRID_MINUTES
    assert SOP_ASPIRATIONAL_MINUTES in SOP_GRID_MINUTES
    assert LABEL_INTERICTAL != LABEL_PREICTAL

    # --- symmetric exclusion buffer (v2) ---
    assert SEIZURE_EXCLUSION_SECONDS == 4 * 3600
    assert POSTICTAL_SECONDS == SEIZURE_EXCLUSION_SECONDS
    assert INTERICTAL_GUARD_SECONDS == SEIZURE_EXCLUSION_SECONDS
    assert INTERICTAL_POOL_MULTIPLIER > 0

    # --- continuity ---
    assert INTER_FILE_GAP_TOLERANCE_SECONDS >= 0
    assert "chb24" in PATIENTS_WITHOUT_CLOCKS

    # --- classifier (v2 Elastic-Net) ---
    assert LR_PARAMS["penalty"] == "elasticnet"
    assert LR_PARAMS["solver"] == "saga", "elasticnet requires the saga solver"
    assert 0.0 <= LR_PARAMS["l1_ratio"] <= 1.0
    assert all(0.0 <= r <= 1.0 for r in L1_RATIO_GRID)
    assert PRIMARY_CLASSIFIER in BENCHMARK_CLASSIFIERS

    # --- balancing (v2) ---
    assert USE_RESAMPLING is True
    assert UNDERSAMPLE_INTERICTAL_TO_PREICTAL_RATIO >= OVERSAMPLE_TARGET_RATIO
    assert OVERSAMPLE_TARGET_RATIO == 1.0

    # --- evaluation ---
    assert PRIMARY_TARGET_FPR_PER_HOUR in TARGET_FPR_PER_HOUR

    # --- paths are Path objects and dirs can be created ---
    assert isinstance(PROJECT_ROOT, Path)
    ensure_dirs()
    for d in ALL_DIRS:
        assert d.exists(), f"directory not created: {d}"

    print(summary())
    print("\nAll config.py self-tests passed.")
