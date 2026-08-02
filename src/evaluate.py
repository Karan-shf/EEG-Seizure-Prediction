"""
evaluate.py
-----------
Evaluation script for the trained SeizurePredictor model.

Runs the best saved checkpoint against the held-out test patients
and computes the full set of clinical and ML metrics.

Metrics reported
----------------
Standard ML metrics:
  - ROC-AUC
  - Sensitivity (recall for preictal class)
  - Specificity (recall for interictal class)
  - F1 score
  - Accuracy

Clinical metrics:
  - False Positive Rate per hour (FPR/h) — the alarm fatigue metric
  - Confusion matrix

Explainability outputs:
  - Per-patient attention weight heatmaps over time
  - Average attention weight curve across all correct preictal predictions
  - Per-patient prediction table

All outputs saved to experiments/results/evaluation/

Run
---
    python src/evaluate.py

Requirements
------------
A trained checkpoint must exist at experiments/checkpoints/best_model.pt
Run train.py first.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    f1_score, accuracy_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import SeizureDataset, get_splits, make_dataloaders
from model import SeizurePredictor, ModelConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Defaults used only when evaluate.py is run standalone.
# The grid runner passes these explicitly per configuration.
DEFAULT_METADATA_PATH   = 'data/processed/offset0_dur30/metadata.csv'
DEFAULT_SEQUENCES_DIR   = 'data/processed/offset0_dur30/sequences'
DEFAULT_CHECKPOINT_PATH = 'experiments/checkpoints/offset0_dur30/best_model.pt'
DEFAULT_OUTPUT_DIR      = 'experiments/results/offset0_dur30/evaluation'

# Classification threshold — sequences with P(preictal) >= this are
# predicted as preictal. 0.5 is standard; tuning this trades sensitivity
# for specificity. We report metrics at 0.5 but also plot the full ROC curve
# so readers can see the tradeoff at any threshold.
# THRESHOLD = 0.5
THRESHOLD = 0.35

# Frames advance by STRIDE (not WINDOW), so each sequence spans
# n_frames × STRIDE_SECONDS of EEG. Needed for a correct FPR/h.
WINDOW_SECONDS = 5
STRIDE_SECONDS = 3

DEVICE = (
    'mps'  if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available()          else
    'cpu'
)


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, logger) -> SeizurePredictor:
    """Load the best saved model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'Checkpoint not found at {checkpoint_path}. '
            f'Run train.py first.'
        )

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    logger.info(f'Loaded checkpoint from epoch {ckpt["epoch"]} '
          f'(val AUC = {ckpt["val_auc"]:.4f})')

    config = ModelConfig(**ckpt['model_config'])
    model  = SeizurePredictor(config)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Run inference on test set
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device) -> dict:
    """
    Run the model on a DataLoader and collect all outputs.

    Returns
    -------
    dict with keys:
        probas       : np.ndarray (N,)   — P(preictal) for each sequence
        labels       : np.ndarray (N,)   — true labels (1=preictal, 0=interictal)
        attn_weights : np.ndarray (N, 360) — attention weights per sequence
        patient_ids  : list of str       — patient ID per sequence
        seq_types    : list of str       — 'preictal' or 'interictal' per sequence
        filenames    : list of str       — .npy filename per sequence
    """
    model = model.to(device)
    model.eval()

    all_probas   = []
    all_labels   = []
    all_attn     = []
    all_patients = []
    all_types    = []
    all_files    = []

    meta_rows = loader.dataset.meta.reset_index(drop=True)

    for batch_idx, (seqs, labels) in enumerate(loader):
        seqs = seqs.to(device)

        probas, attn_weights = model.predict_proba(seqs)

        all_probas.append(probas.squeeze(1).cpu().numpy())
        all_labels.append(labels.numpy())
        all_attn.append(attn_weights.cpu().numpy())

        # Track metadata for each sample in this batch
        batch_size   = seqs.shape[0]
        start_idx    = batch_idx * loader.batch_size
        end_idx      = start_idx + batch_size
        batch_meta   = meta_rows.iloc[start_idx:end_idx]

        all_patients.extend(batch_meta['patient_id'].tolist())
        all_types.extend(batch_meta['type'].tolist())
        all_files.extend(batch_meta['filename'].tolist())

    return {
        'probas':       np.concatenate(all_probas),
        'labels':       np.concatenate(all_labels),
        'attn_weights': np.concatenate(all_attn),
        'patient_ids':  all_patients,
        'seq_types':    all_types,
        'filenames':    all_files,
    }


def pick_threshold_youden(probas: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising Youden's J (sensitivity + specificity - 1),
    chosen on the validation set. Falls back to 0.5 if degenerate."""
    fpr, tpr, thr = roc_curve(labels, probas)
    best = int(np.argmax(tpr - fpr))
    t = float(thr[best])
    return t if np.isfinite(t) else 0.5   # roc_curve prepends an inf threshold

# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------

def compute_metrics(probas: np.ndarray, labels: np.ndarray,
                    n_frames: int,
                    threshold: float = THRESHOLD) -> dict:
    """
    Compute all classification and clinical metrics.

    Parameters
    ----------
    probas    : P(preictal) for each sequence
    labels    : true binary labels
    threshold : classification cutoff

    Returns
    -------
    dict of metric name → value
    """
    preds = (probas >= threshold).astype(int)

    auc         = roc_auc_score(labels, probas)
    fpr_curve, tpr_curve, thresholds = roc_curve(labels, probas)
    acc         = accuracy_score(labels, preds)
    f1          = f1_score(labels, preds, zero_division=0)

    cm = confusion_matrix(labels, preds)
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall for preictal
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # recall for interictal
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # precision
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # negative predictive value

    # --- FPR/h (alarm fatigue metric) ---
    # Each sequence represents WINDOW_SECONDS × N_FRAMES seconds of EEG
    # A false positive is one false alarm per sequence
    # Convert to alarms per hour
    # 360 was only right for the old 30-min, non-overlapping design. For dur15
    # (299 frames × 3s ≈ 15 min) the old formula understated FPR/h ~2x
    # (reported 2.0 should have been ~4.0).
    total_interictal_sequences = int((labels == 0).sum())
    seconds_per_sequence       = n_frames * STRIDE_SECONDS
    total_interictal_hours     = (total_interictal_sequences * seconds_per_sequence) / 3600
    fpr_per_hour = fp / total_interictal_hours if total_interictal_hours > 0 else 0.0

    return {
        'auc':           auc,
        'accuracy':      acc,
        'sensitivity':   sensitivity,   # = recall for preictal
        'specificity':   specificity,
        'f1':            f1,
        'ppv':           ppv,           # precision
        'npv':           npv,
        'fpr_per_hour':  fpr_per_hour,
        'tp': int(tp), 'fp': int(fp),
        'tn': int(tn), 'fn': int(fn),
        'threshold':     threshold,
        'roc_fpr':       fpr_curve.tolist(),
        'roc_tpr':       tpr_curve.tolist(),
        'roc_thresholds': thresholds.tolist(),
    }


def compute_per_patient_metrics(results: dict, n_frames: int,
                                threshold: float = THRESHOLD) -> pd.DataFrame:
    """
    Compute metrics broken down by patient.
    Returns a DataFrame with one row per patient.
    """
    rows = []
    patient_ids = sorted(set(results['patient_ids']))

    for pid in patient_ids:
        mask   = np.array([p == pid for p in results['patient_ids']])
        probas = results['probas'][mask]
        labels = results['labels'][mask]
        preds  = (probas >= threshold).astype(int)

        n_preictal   = int((labels == 1).sum())
        n_interictal = int((labels == 0).sum())

        if len(np.unique(labels)) < 2:
            # Only one class present for this patient — can't compute AUC
            auc = float('nan')
        else:
            auc = roc_auc_score(labels, probas)

        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

        inter_hours = (n_interictal * n_frames * STRIDE_SECONDS) / 3600
        fpr_h = fp / inter_hours if inter_hours > 0 else float('nan')

        rows.append({
            'patient_id':   pid,
            'n_preictal':   n_preictal,
            'n_interictal': n_interictal,
            'auc':          round(auc, 4) if not np.isnan(auc) else 'N/A',
            'sensitivity':  round(sensitivity, 4),
            'specificity':  round(specificity, 4),
            'fpr_per_hour': round(fpr_h, 4),
            'tp': int(tp), 'fp': int(fp),
            'tn': int(tn), 'fn': int(fn),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc_curve(metrics: dict, save_path: str, logger):
    """Plot and save the ROC curve."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(metrics['roc_fpr'], metrics['roc_tpr'],
            color='steelblue', linewidth=2,
            label=f'ROC curve (AUC = {metrics["auc"]:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random baseline')

    # Mark the operating point at the chosen threshold
    fpr_at_threshold = 1 - metrics['specificity']
    tpr_at_threshold = metrics['sensitivity']
    ax.scatter([fpr_at_threshold], [tpr_at_threshold],
               color='red', s=100, zorder=5,
               label=f'Threshold={metrics["threshold"]} '
                     f'(Sens={tpr_at_threshold:.2f}, '
                     f'Spec={metrics["specificity"]:.2f})')

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)',      fontsize=12)
    ax.set_title('ROC Curve — Seizure Prediction\n(Test patients)',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'ROC curve saved to {save_path}')


def plot_confusion_matrix(metrics: dict, save_path: str, logger):
    """Plot and save the confusion matrix."""
    cm = np.array([
        [metrics['tn'], metrics['fp']],
        [metrics['fn'], metrics['tp']],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    classes = ['Interictal', 'Pre-ictal']
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    ax.set_ylabel('True label',      fontsize=12)
    ax.set_xlabel('Predicted label', fontsize=12)
    ax.set_title('Confusion Matrix — Test Set', fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Confusion matrix saved to {save_path}')


def plot_attention_heatmap(results: dict, save_path: str, logger,
                           offset_minutes: int = 0, duration_minutes: int = 30,
                           n_samples: int = 6):
    """
    Plot attention weight heatmaps for a sample of correctly predicted
    preictal sequences.

    Each row is one sequence. The x-axis is time (0 = 30 min before seizure,
    359 = seizure onset). Bright = high attention weight.
    """
    probas  = results['probas']
    labels  = results['labels']
    attns   = results['attn_weights']
    pids    = results['patient_ids']
    types   = results['seq_types']

    # Select correctly predicted preictal sequences
    preds            = (probas >= THRESHOLD).astype(int)
    correct_preictal = (
        (labels == 1) & (preds == 1)
    )
    indices = np.where(correct_preictal)[0]

    if len(indices) == 0:
        logger.info('[Attention] No correctly predicted preictal sequences found.')
        return

    # Take up to n_samples
    indices = indices[:n_samples]

    fig, axes = plt.subplots(len(indices), 1,
                             figsize=(16, 2.5 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    fig.suptitle(
        'Attention Weights — Correctly Predicted Pre-ictal Sequences\n'
        'Bright = high model attention  |  x-axis: time before seizure',
        fontsize=12, y=1.01
    )

    # This window spans from -(offset+duration) to -offset min before onset.
    # (Hardcoded -30..0 mislabeled every config where offset!=0 or duration!=30.)
    n_frames   = attns.shape[1]
    t_start    = -(offset_minutes + duration_minutes)
    t_end      = -offset_minutes
    time_mins  = np.linspace(t_start, t_end, n_frames)

    for ax, idx in zip(axes, indices):
        attn = attns[idx]                  # (360,)
        pid  = pids[idx]
        prob = probas[idx]

        # Plot as a heatmap (1 row, 360 columns)
        ax.imshow(
            attn[np.newaxis, :],
            aspect='auto',
            cmap='YlOrRd',
            extent=[t_start, t_end, 0, 1],
            vmin=0,
        )
        ax.set_yticks([])
        ax.set_xlabel('Minutes before seizure onset', fontsize=9)
        ax.set_title(
            f'{pid}  |  P(preictal) = {prob:.3f}',
            fontsize=9, pad=3
        )

        # Mark the peak attention frame
        peak_frame   = np.argmax(attn)
        peak_time    = time_mins[peak_frame]
        ax.axvline(x=peak_time, color='blue', linewidth=1.5,
                   alpha=0.7, linestyle='--')
        ax.text(peak_time + 0.3, 0.6,
                f'peak\n{peak_time:.1f} min',
                color='blue', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Attention heatmap saved to {save_path}')


def plot_mean_attention(results: dict, save_path: str, logger,
                        offset_minutes: int = 0, duration_minutes: int = 30):
    """
    Plot the mean attention weight curve across all correctly predicted
    preictal sequences, with standard deviation band.

    This is your key explainability figure: it shows WHEN in the
    30-minute window the model consistently focuses its attention.
    """
    probas  = results['probas']
    labels  = results['labels']
    attns   = results['attn_weights']

    preds            = (probas >= THRESHOLD).astype(int)
    correct_preictal = (labels == 1) & (preds == 1)
    indices          = np.where(correct_preictal)[0]

    if len(indices) == 0:
        logger.info('[Mean Attention] No correctly predicted preictal sequences.')
        return

    attn_subset = attns[indices]              # (n_correct, 360)
    mean_attn   = attn_subset.mean(axis=0)    # (360,)
    std_attn    = attn_subset.std(axis=0)     # (360,)

    n_frames  = attns.shape[1]
    t_start   = -(offset_minutes + duration_minutes)
    t_end     = -offset_minutes
    time_mins = np.linspace(t_start, t_end, n_frames)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(time_mins, mean_attn, color='steelblue', linewidth=2,
            label=f'Mean attention (n={len(indices)} sequences)')
    ax.fill_between(time_mins,
                    mean_attn - std_attn,
                    mean_attn + std_attn,
                    alpha=0.25, color='steelblue', label='±1 std')

    # Mark peak
    peak_idx  = np.argmax(mean_attn)
    peak_time = time_mins[peak_idx]
    ax.axvline(x=peak_time, color='red', linestyle='--', linewidth=1.5,
               label=f'Peak attention: {peak_time:.1f} min before seizure')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.text(0.2, mean_attn.max() * 0.95, 'Seizure\nonset', fontsize=9)

    ax.set_xlabel('Minutes before seizure onset', fontsize=12)
    ax.set_ylabel('Attention weight', fontsize=12)
    ax.set_title(
        'Mean Attention Weight Across Pre-ictal Sequences\n'
        'Shows WHEN the model detects pre-ictal activity',
        fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(t_start, t_end + 2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Mean attention curve saved to {save_path}')
    logger.info(f'  → Model peaks at {peak_time:.1f} minutes before seizure onset')
    logger.info(f'     (clinical implication: pre-ictal changes detectable '
          f'~{abs(peak_time):.0f} min before seizure)')


# ---------------------------------------------------------------------------
# Print final report
# ---------------------------------------------------------------------------

def print_report(metrics: dict, per_patient: pd.DataFrame,
                 checkpoint_path: str, logger):
    """Print a clean evaluation report to terminal."""
    logger.info('\n' + '=' * 55)
    logger.info('SeizureHorizon — Evaluation Report')
    logger.info('=' * 55)
    logger.info(f'Checkpoint : {checkpoint_path}')
    logger.info(f'Threshold  : {metrics["threshold"]}')
    logger.info("")
    logger.info('── Overall metrics (test patients) ─────────────────')
    logger.info(f'  ROC-AUC         : {metrics["auc"]:.4f}')
    logger.info(f'  Sensitivity      : {metrics["sensitivity"]:.4f}  '
          f'({metrics["tp"]} / {metrics["tp"] + metrics["fn"]} preictal detected)')
    logger.info(f'  Specificity      : {metrics["specificity"]:.4f}  '
          f'({metrics["tn"]} / {metrics["tn"] + metrics["fp"]} interictal correct)')
    logger.info(f'  F1 Score         : {metrics["f1"]:.4f}')
    logger.info(f'  Accuracy         : {metrics["accuracy"]:.4f}')
    logger.info(f'  PPV (Precision)  : {metrics["ppv"]:.4f}')
    logger.info(f'  NPV              : {metrics["npv"]:.4f}')
    logger.info("")
    logger.info('── Clinical metric ──────────────────────────────────')
    logger.info(f'  FPR/h            : {metrics["fpr_per_hour"]:.4f} false alarms per hour')
    clinical = ('✓ Clinically acceptable (< 0.1/h)'
                if metrics['fpr_per_hour'] < 0.1
                else '✗ Above clinical threshold (> 0.1/h)')
    logger.info(f'  Assessment       : {clinical}')
    logger.info("")
    logger.info('── Confusion matrix ─────────────────────────────────')
    logger.info(f'  TP={metrics["tp"]}  FP={metrics["fp"]}')
    logger.info(f'  FN={metrics["fn"]}  TN={metrics["tn"]}')
    logger.info("")
    logger.info('── Per-patient breakdown ────────────────────────────')
    logger.info(per_patient.to_string(index=False))
    logger.info('=' * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(
    metadata_path: str = DEFAULT_METADATA_PATH,
    sequences_dir: str = DEFAULT_SEQUENCES_DIR,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    threshold: float = THRESHOLD,
    config_name: str | None = None,
    offset_minutes: int = 0,
    duration_minutes: int = 30,
):
    """
    Parameters
    ----------
    metadata_path   : str — path to this config's metadata.csv
    sequences_dir   : str — path to this config's sequences/ folder
    checkpoint_path : str — path to this config's trained model
    output_dir      : str — where to save evaluation outputs
    threshold       : float — classification threshold
    config_name     : str — label for this run, used in printed report
    """

    from logger import get_logger

    os.makedirs(output_dir, exist_ok=True)
    
    logger = get_logger(
        name=f'{config_name}_eval' if config_name else 'evaluate',
        log_dir=os.path.join('experiments', 'logs', config_name or 'default'),
    )

    device = torch.device(DEVICE)
    logger.info(f'[Evaluate] Device: {device}')
    if config_name:
        print(f'[Evaluate] Config: {config_name}')

    # --- Load model ---
    model = load_model(checkpoint_path, logger)
    model = model.to(device)

    # --- Build test DataLoader ---
    meta  = pd.read_csv(metadata_path)
    folds = get_splits(meta, strategy='fixed')
    train_p, val_p, test_p = folds[0]

    _, val_loader, test_loader = make_dataloaders(
        meta, sequences_dir,
        train_p, val_p, test_p,
        batch_size=16,
        num_workers=0,
    )

    logger.info(f'\n[Evaluate] Running inference on {len(test_loader.dataset)} '
          f'test sequences from patients: {test_p}')

    # --- Inference ---
    results  = run_inference(model, test_loader, device)
    n_frames = results['attn_weights'].shape[1]

    # --- Calibrate threshold on VALIDATION (never on test) ---
    # 0.35 was hardcoded and uncalibrated -> all-positive operating point.
    val_results = run_inference(model, val_loader, device)
    if len(np.unique(val_results['labels'])) == 2:
        threshold = pick_threshold_youden(val_results['probas'], val_results['labels'])
        logger.info(f'[Evaluate] Calibrated threshold on val set: {threshold:.3f}')
    else:
        logger.info(f'[Evaluate] Val set single-class — keeping threshold {threshold:.3f}')

    # --- Metrics (at the calibrated threshold, with real frame count) ---
    metrics     = compute_metrics(results['probas'], results['labels'],
                                  n_frames=n_frames, threshold=threshold)
    per_patient = compute_per_patient_metrics(results, n_frames=n_frames,
                                              threshold=threshold)

    # --- Print report ---
    print_report(metrics, per_patient, checkpoint_path, logger)

    # --- Save metrics as JSON ---
    metrics_save = {k: v for k, v in metrics.items()
                    if k not in ('roc_fpr', 'roc_tpr', 'roc_thresholds')}
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_save, f, indent=2)
    logger.info(f'\nMetrics saved to {metrics_path}')

    # --- Save per-patient table ---
    table_path = os.path.join(output_dir, 'per_patient_metrics.csv')
    per_patient.to_csv(table_path, index=False)
    logger.info(f'Per-patient table saved to {table_path}')

    # --- Plots ---
    plot_roc_curve(
        metrics,
        os.path.join(output_dir, 'roc_curve.png'),
        logger
    )
    plot_confusion_matrix(
        metrics,
        os.path.join(output_dir, 'confusion_matrix.png'),
        logger
    )
    plot_attention_heatmap(
        results,
        os.path.join(output_dir, 'attention_heatmap.png'),
        logger,
        offset_minutes=offset_minutes,
        duration_minutes=duration_minutes,
    )
    plot_mean_attention(
        results,
        os.path.join(output_dir, 'mean_attention.png'),
        logger,
        offset_minutes=offset_minutes,
        duration_minutes=duration_minutes,
    )

    logger.info(f'\n[Evaluate] All outputs saved to {output_dir}/')
    return {
        'config_name': config_name,
        'metrics':     metrics,
        'per_patient': per_patient,
        'results':     results,
        'output_dir':  output_dir,
    }


if __name__ == '__main__':
    evaluate(config_name='offset0_dur30')