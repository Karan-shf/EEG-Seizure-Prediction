# 01 — `config.py`

**Source:** `src/config.py`  ·  **Rebuild file 1 of ~19**  ·  Reqs: A–I (all)

---

## Purpose

`config.py` is the **single source of truth** for the whole project. Every
constant that defines the experiment — file paths, sampling rate, the channel
montage, the filter bank, window geometry, the SPH/SOP labeling scheme, feature
dimensions, classifier hyper-parameters and evaluation settings — lives here and
**nowhere else**.

**Design rule:** no other module may hard-code a pipeline number. If a value is
part of the experimental design (A–I), it is defined in `config.py` so that:

1. a single edit propagates everywhere (change the window once, everything
   downstream follows), and
2. an entire experiment is *fully described* by this one file — critical for
   reproducibility and for writing the paper's Methods section.

This directly addresses a failure mode of the old pipeline, where magic numbers
were scattered across files and quietly disagreed with one another.

---

## Why it's necessary

- **Reproducibility.** One `SEED`, one place. Re-running reproduces splits,
  SMOTE, and classifier initialization exactly.
- **Consistency.** Derived values (`WINDOW_SAMPLES`, `STRIDE_SAMPLES`,
  `FEATURE_DIM`) are *computed* from their definitions, so they can never drift
  out of sync with the settings they depend on.
- **Auditability.** The reviewer/professor can read one file and know the exact
  experimental configuration.

---

## Math (derived constants)

**Window length in samples** (Req B), for window duration $T_\text{win}$ and
sampling rate $f_s$:

$$\text{WINDOW\_SAMPLES} = \operatorname{round}(T_\text{win}\, f_s)
= \operatorname{round}(4.0 \times 256) = 1024.$$

**Stride in samples**, for overlap fraction $o$:

$$\text{STRIDE\_SAMPLES} = \operatorname{round}\big(\text{WINDOW\_SAMPLES}\,(1-o)\big)
= \operatorname{round}(1024 \times 0.60) = 614,$$

so the stride is $614/256 \approx 2.398\text{ s}$, i.e. 40% overlap.

**Tangent-space dimensionality** (Req C/D). A symmetric $n \times n$ matrix has

$$\frac{n(n+1)}{2}$$

unique entries (the upper triangle including the diagonal). This is exactly the
length of the vector obtained by projecting an SPD covariance matrix to the
tangent space. For $n = 18$ channels:

$$\text{TANGENT\_DIM\_PER\_BAND} = \frac{18 \times 19}{2} = 171.$$

**Total feature dimension** (Req D), concatenating all bands:

$$\text{FEATURE\_DIM} = \text{TANGENT\_DIM\_PER\_BAND} \times N_\text{bands}
= 171 \times 5 = 855.$$

**Nyquist check** (Req D). Every band's high edge must satisfy
$f_\text{high} < f_s/2 = 128\text{ Hz}$. Our top band (γ, 30–45 Hz) is well below
Nyquist.

---

## Section-by-section reference

| Section | Contents | Req |
|---------|----------|-----|
| **0. Paths** | `PROJECT_ROOT` derived from this file's location (`parents[1]`), then `data/`, `data/raw/chb-mit`, `data/processed`, `experiments/{logs,checkpoints,results}`. `ensure_dirs()` creates any missing folders (idempotent). | — |
| **1. Reproducibility** | `SEED = 42`, reused by classifiers and any RNG. | — |
| **2. Signal** | `FS = 256`; `CHANNELS` = the 18 canonical bipolar pairs in **fixed order** (the order indexes covariance rows/cols); `N_CHANNELS` derived. | A |
| **3. Filter bank** | `NOTCH_FREQ = 60`, `NOTCH_QUALITY = 30`, `FILTER_ORDER = 4`; `BANDS` = δ(0.5–4), θ(4–8), α(8–13), β(13–30), γ(30–45); `N_BANDS` derived. | D |
| **4. Windowing** | `WINDOW_SECONDS = 4.0`, `WINDOW_OVERLAP = 0.40`; derived `WINDOW_SAMPLES = 1024`, `STRIDE_SAMPLES = 614`, `STRIDE_SECONDS ≈ 2.398`. | B |
| **5. Features** | `COV_ESTIMATOR = 'oas'` (+ `'sch'` fallback); derived `TANGENT_DIM_PER_BAND = 171`, `FEATURE_DIM = 855`; `ALIGNMENT_METHOD = 'rct'`. | C, D, E |
| **6. Labeling** | `SPH_MINUTES = 5`; `SOP_GRID_MINUTES = (15,30,45,60)` with primary 30 / aspirational 60; `POSTICTAL_SECONDS = 3 h`; `LEAD_SEIZURE_MIN_PRECEDING_SECONDS = 4 h`; `INTERICTAL_GUARD_SECONDS = 4 h`; labels `0 = interictal`, `1 = preictal`. | G |
| **7. Classifier** | `PRIMARY_CLASSIFIER = 'lr'`; `BENCHMARK_CLASSIFIERS`; `LR_PARAMS` (`class_weight='balanced'`, `solver='liblinear'`); `SVM_PARAMS`; `USE_SMOTE = False`; `STANDARDIZE_FEATURES = True`. | F |
| **8. Evaluation** | `CV_SCHEME = 'lopo'`; `TARGET_FPR_PER_HOUR = (0.1,0.15,0.2,0.5)` (primary 0.15); `FIRING_POWER_THRESHOLD = 0.5`; `REFRACTORY_SECONDS`; `THRESHOLD_SELECTION = 'train_val_fixed_fpr'`. | I |
| **9. Helpers** | `summary()` returns a readable dump of the key derived settings. | — |

### Key functions

- **`ensure_dirs() -> None`** — creates every project directory if missing
  (`mkdir(parents=True, exist_ok=True)`); safe to call repeatedly.
- **`summary() -> str`** — returns a formatted multi-line string of the most
  important settings/derived values for quick console inspection.

---

## What the self-test proves

Run with `python -m src.config`. The `__main__` block asserts:

- **Derived geometry** is exactly right: `WINDOW_SAMPLES == 1024`,
  `STRIDE_SAMPLES == 614`, `TANGENT_DIM_PER_BAND == 171`, `FEATURE_DIM == 855`,
  `N_CHANNELS == 18`, `N_BANDS == 5`.
- **Montage integrity:** no duplicate channels.
- **Filter bank validity:** every band is ascending ($f_\text{low} < f_\text{high}$),
  non-overlapping (each band starts at or above the previous band's top), and
  below Nyquist (128 Hz).
- **Labeling validity:** `SPH_SECONDS == 300`, the SOP grid is sorted and
  contains both the primary and aspirational operating points, and the two class
  labels differ.
- **Evaluation validity:** the primary FPR target is in the target grid and the
  primary classifier is in the benchmark set.
- **Paths work:** `PROJECT_ROOT` is a `Path`, and every project directory can be
  created.

On success it prints the `summary()` dump and
`"All config.py self-tests passed."`

---

## Downstream consumers

Every other module imports from here, e.g. `from src import config as cfg` and
then `cfg.FS`, `cfg.CHANNELS`, `cfg.BANDS`, `cfg.WINDOW_SAMPLES`, etc. When we
revisit any locked decision, we change it **once here** and the whole pipeline
follows.
