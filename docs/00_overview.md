# Project Overview

## 1. Goal

Predict epileptic seizures **ahead of time** from scalp EEG on the **CHB-MIT**
dataset, in a **cross-patient (patient-independent)** setting.

- **Prediction, not detection.** We classify the *pre-seizure* (preictal) brain
  state versus the far-from-seizure (interictal) state — we never look at the
  seizure itself (ictal data is excluded).
- **Ultimate target:** raise an alarm up to **1 hour** before onset
  (SOP = 60 min, aspirational). Primary reported operating point is **30 min**.
- **Cross-patient:** the model is trained on some patients and tested on
  *unseen* patients via **Leave-One-Patient-Out (LOPO)** cross-validation. This
  is much harder than the patient-specific setups common in the literature, and
  it is the setting our professor asked us to target.

---

## 2. Why a clean rebuild

The previous CNN + LSTM pipeline performed **at chance** in the cross-patient
setting (LOPO mean-patient AUC ≈ 0.45, sensitivity ≈ 10%). Root causes were
structural: data leakage in fixed splits, tiling/augmentation artifacts, and an
architecture that never had a chance to generalize across patients. Rather than
patch it, we restart on a **Riemannian geometry** foundation that is the
competition-standard for small-sample, cross-subject EEG.

The new spatial branch is: **band-pass → covariance (OAS) → Riemannian
re-centering (RCT) → tangent-space projection → Logistic Regression**.

---

## 3. Project structure

```text
project/
├── data/
│   ├── raw/chb-mit/            # your local EDFs + *-summary.txt (untouched)
│   └── processed/              # cached features (X, y, patient ids) per SOP
├── src/
│   ├── config.py               # single source of truth (constants, paths, seeds)
│   ├── utils/
│   │   └── logger.py           # dual console+file logging
│   ├── io/
│   │   ├── summary_parser.py    # parse *-summary.txt -> seizure times + clocks
│   │   └── edf_loader.py        # load EDF, apply montage, validate channels   [A]
│   ├── preprocessing/
│   │   ├── montage.py           # canonical 18 bipolar channel definition       [A]
│   │   ├── filters.py           # 60 Hz notch + 5-band zero-phase Butterworth   [D]
│   │   └── windowing.py         # 4 s / 40% overlap segmentation                [B]
│   ├── labeling/
│   │   └── labeler.py           # SPH/SOP timeline, postictal+cluster exclusion,
│   │                            #   lead-seizure logic                          [G]
│   ├── features/
│   │   ├── covariance.py        # OAS covariance per band per window            [C]
│   │   ├── alignment.py         # calibration-anchored RCT recentering          [E]
│   │   └── tangent.py           # tangent projection + per-band concat -> 855-D [D/E]
│   ├── data/
│   │   └── dataset_builder.py    # orchestrate EDF->filter->window->label->features
│   ├── models/
│   │   └── classifier.py        # LR (+SVM/LDA) pipeline, scaler, class weights [F]
│   ├── evaluation/
│   │   ├── alarms.py            # firing-power moving avg + refractory -> alarms [I]
│   │   └── metrics.py           # ROC/PR-AUC, sens/spec/prec/F1/acc/bal, event  [I]
│   ├── experiment/
│   │   ├── lopo.py             # leave-one-patient-out loop, no-leakage wiring   [I]
│   │   ├── sop_grid.py        # SOP characteristic curve (Option 2)             [G/I]
│   │   └── inventory.py       # per-patient usable-seizure counts -> CSV        [G]
│   └── run.py                  # thin CLI entry point
├── docs/                        # this documentation tree (one .md per file)
├── notebooks/
├── experiments/{logs,checkpoints,results}
├── requirements.txt
└── .gitignore
```

---

## 4. The A to I locked decisions (implementation map)

| Req | Decision | Where it lives |
|-----|----------|----------------|
| **A** Channels | 18 canonical bipolar channels, fixed order, per-EDF validation | `preprocessing/montage.py`, `io/edf_loader.py`, `config.CHANNELS` |
| **B** Windows | 4 s windows (1024 samples @256 Hz), 40% overlap (614-sample stride) | `preprocessing/windowing.py`, `config.WINDOW_*` |
| **C** Covariance | OAS shrinkage per band per window; Schäfer-Strimmer fallback | `features/covariance.py`, `config.COV_*` |
| **D** Filter bank | δ/θ/α/β/γ (0.5–45 Hz), 60 Hz notch, 4th-order Butterworth (zero-phase); per-band tangent concat → 855-D | `preprocessing/filters.py`, `features/tangent.py`, `config.BANDS` |
| **E** Alignment | Calibration-anchored Riemannian re-centering (RCT), per band per patient, fit on TRAIN only | `features/alignment.py`, `config.ALIGNMENT_METHOD` |
| **F** Imbalance | Full clean interictal + `class_weight='balanced'`; SMOTE optional; OSPDIM skipped | `models/classifier.py`, `config.LR_PARAMS`, `config.USE_SMOTE` |
| **G** Labels | SPH = 5 min; SOP ∈ {15,30,45,60}; postictal + cluster exclusion (3 h); lead-seizure rule; 2-class | `labeling/labeler.py`, `config` §6 |
| **H** Tooling | pyRiemann (OAS covariance, TangentSpace, transfer.TLCenter) + scikit-learn | project-wide |
| **I** Evaluation | LOPO; ROC-AUC + PR-AUC (per-patient + aggregated); event-based sensitivity + FPR/h; Sens@fixed-FPR; balanced accuracy; firing-power alarms; thresholds frozen on train/val | `evaluation/*`, `experiment/*`, `config` §8 |

**Classifier (locked):** Logistic Regression primary on tangent features
(`class_weight='balanced'`); LDA / RBF-SVM / MDM / FgMDM benchmarked on TRAIN
folds only. LR is chosen for calibrated probabilities (needed for firing-power
alarm logic and threshold selection), few parameters, and being the canonical
pyRiemann tangent-space choice.

**RSMMTN (future):** a second, temporal branch runs in parallel to the spatial
Riemannian branch and is fused at the **feature level** (per-branch z-score →
concatenate → single LR). Not implemented yet; the spatial branch is built to
make this drop-in later.

---

## 5. End-to-end pipeline

```text
            ┌──────────────────────── per patient, per EDF ────────────────────────┐
 raw EDF ─► [A] montage/validate ─► [D] notch + band-pass (5 bands)
            │                                        │
            │                              [B] 4 s / 40% windows
            │                                        │
 [G] label each window (SPH/SOP) + EXCLUDE ictal, postictal (3 h), clusters
            │                                        │
            ▼                                        ▼
   ┌─ SPATIAL branch (now) ─────────────┐   ┌─ TEMPORAL branch: RSMMTN (later) ─┐
   │ [C] OAS covariance per band/window │   │  spatial/temporal/fusion sub-nets  │
   │ [E] RCT recenter (train-fit anchor)│   │  -> Euclidean/tangent feature vec  │
   │ [D] tangent map -> concat 855-D    │   └───────────────────────────────────┘
   └───────────────┬────────────────────┘                 │
                   └──────── feature-level fusion (z-score ⊕ concat) ───────┐
                                                                            ▼
                                        [F] class_weight balanced ─► LR classifier
                                                                            │
                                        [I] firing power + refractory ─► alarms
                                                                            │
                                        [I] LOPO metrics + SOP curve, no leakage
```

---

## 6. Build order (bottom-up)

Each file is written and self-tested before the next, so every layer stands on
verified foundations:

1. `config.py`  ← **done**
2. `utils/logger.py`
3. `io/summary_parser.py`
4. `preprocessing/montage.py` → `io/edf_loader.py`
5. `preprocessing/filters.py` → `preprocessing/windowing.py`
6. `labeling/labeler.py`  (+ `experiment/inventory.py` for the usable-count CSV)
7. `features/covariance.py` → `features/alignment.py` → `features/tangent.py`
8. `data/dataset_builder.py`  (first full raw→features run)
9. `models/classifier.py` → `evaluation/alarms.py` → `evaluation/metrics.py`
10. `experiment/lopo.py` → `experiment/sop_grid.py` → `run.py`

---

## 7. Conventions

**Documentation.** One Markdown file per source file (`docs/NN_<name>.md`) using a
fixed template: *Purpose → Why it's necessary → Math → Function-by-function
reference → What the self-test proves*. All formulas collected in
`docs/math_appendix.md`.

**Self-tests.** Every `.py` ends with an `if __name__ == "__main__":` block that
validates the file in isolation, preferring **synthetic data** so it runs even
without the full dataset. A green run = the file does its job. Run any file with
`python -m src.<module>` from the project root.

**No leakage (non-negotiable, Req I).** Anything learned from data — RCT
alignment anchors, feature scalers, decision thresholds, SOP choice — is fit on
**train/validation patients only** and merely *applied* to held-out test
patients. Test patients never influence any decision.

---

## 8. Glossary

- **Ictal** — during a seizure (excluded from modeling).
- **Preictal** — the window *before* a seizure we try to detect (positive class).
- **Interictal** — far from any seizure (negative class).
- **Postictal** — after a seizure; noisy/abnormal, excluded (3 h here).
- **SPH** (Seizure Prediction Horizon) — a mandatory gap right before onset that
  the alarm must precede; gives the patient usable warning time (5 min).
- **SOP** (Seizure Occurrence Period) — the window within which the seizure is
  expected to occur after the SPH; the preictal label spans this period.
- **Lead seizure** — a seizure preceded by enough seizure-free EEG to be treated
  as an independent event (clusters are merged/excluded).
- **SPD** — Symmetric Positive Definite (covariance matrices live here).
- **Tangent space** — a flat (Euclidean) linearization of the curved SPD
  manifold at a reference point, where ordinary classifiers like LR work well.
- **RCT** — Riemannian re-centering / transfer learning that moves each
  patient's covariance cloud to a common reference (the identity), reducing
  between-patient shift.
- **LOPO** — Leave-One-Patient-Out cross-validation.
- **FPR/h** — false alarms per hour (the key cost metric for prediction).
