"""
train.py
--------
Training loop for the SeizurePredictor CNN-LSTM-Attention model.

What it does
------------
1. Loads metadata and builds train/val DataLoaders for the fixed patient split
2. Initialises the model, optimizer, and weighted loss function
3. Trains for up to max_epochs with early stopping on validation AUC
4. Saves the best checkpoint whenever validation AUC improves
5. Logs per-epoch metrics to experiments/logs/ for TensorBoard
6. logger.infos a clean training summary at the end

Run
---
    python src/train.py

Output
------
    experiments/checkpoints/best_model.pt   — best weights by val AUC
    experiments/logs/                        — TensorBoard event files
    experiments/results/training_curves.png — loss + AUC curves
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

# Make src/ importable when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import SeizureDataset, get_splits, make_dataloaders
from model import SeizurePredictor, ModelConfig, model_summary
from logger import get_logger

logger = get_logger(name='train')

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

class TrainConfig:
    # Paths
    metadata_path  = 'data/processed/metadata.csv'
    sequences_dir  = 'data/processed/sequences'
    checkpoint_dir = 'experiments/checkpoints'
    log_dir        = 'experiments/logs'
    results_dir    = 'experiments/results'

    # Split strategy: 'fixed' for development, 'lopo' for final publication
    split_strategy = 'fixed'

    # Training
    # max_epochs     = 100
    max_epochs     = 100
    batch_size     = 16
    learning_rate  = 1e-3
    weight_decay   = 1e-4    # L2 regularisation — important for small datasets

    # Early stopping: stop if val AUC doesn't improve for this many epochs
    patience       = 15

    # Learning rate scheduler: reduce LR when val AUC plateaus
    lr_patience    = 7
    lr_factor      = 0.5
    min_lr         = 1e-5

    # Device
    device = (
        'mps'  if torch.backends.mps.is_available() else   # Apple Silicon GPU
        'cuda' if torch.cuda.is_available()          else   # NVIDIA GPU
        'cpu'
    )

    # Reproducibility
    seed = 42


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, val_auc, config, path):
    """Save model weights and training state."""
    torch.save({
        'epoch':          epoch,
        'model_state':    model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'val_auc':        val_auc,
        'model_config':   config.__dict__,
    }, path)


def load_checkpoint(path, model, optimizer=None):
    """Load model weights from checkpoint."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt['epoch'], ckpt['val_auc']


# ---------------------------------------------------------------------------
# One epoch of training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one full pass over the training DataLoader.

    Returns
    -------
    avg_loss : float — mean loss over all batches
    auc      : float — ROC-AUC over all training samples
    """
    model.train()

    total_loss  = 0.0
    all_logits  = []
    all_labels  = []

    for seqs, labels in loader:
        seqs   = seqs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, _ = model(seqs)            # (batch, 1) — attention weights unused here
        logits    = logits.squeeze(1)      # (batch,)

        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTM
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    avg_loss   = total_loss / len(loader)
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probas     = 1 / (1 + np.exp(-all_logits))   # sigmoid

    # AUC requires both classes to be present
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, probas)
    else:
        auc = float('nan')

    return avg_loss, auc


# ---------------------------------------------------------------------------
# One epoch of validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Evaluate model on validation set without updating weights.

    Returns
    -------
    avg_loss : float
    auc      : float
    """
    model.eval()

    total_loss = 0.0
    all_logits = []
    all_labels = []

    for seqs, labels in loader:
        seqs   = seqs.to(device)
        labels = labels.to(device)

        logits, _ = model(seqs)
        logits    = logits.squeeze(1)

        loss = criterion(logits, labels)
        total_loss += loss.item()

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    avg_loss   = total_loss / len(loader)
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probas     = 1 / (1 + np.exp(-all_logits))

    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, probas)
    else:
        auc = float('nan')

    return avg_loss, auc


# ---------------------------------------------------------------------------
# Plot training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, save_path: str):
    """
    Plot loss and AUC curves for train and validation.
    Saves to experiments/results/training_curves.png.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Curves — SeizurePredictor', fontsize=13)

    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train loss',
                 color='steelblue', linewidth=1.5)
    axes[0].plot(epochs, history['val_loss'],   label='Val loss',
                 color='darkorange', linewidth=1.5)
    axes[0].axvline(x=history['best_epoch'], color='green',
                    linestyle='--', alpha=0.7, label=f'Best epoch ({history["best_epoch"]})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('BCE Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # AUC
    axes[1].plot(epochs, history['train_auc'], label='Train AUC',
                 color='steelblue', linewidth=1.5)
    axes[1].plot(epochs, history['val_auc'],   label='Val AUC',
                 color='darkorange', linewidth=1.5)
    axes[1].axvline(x=history['best_epoch'], color='green',
                    linestyle='--', alpha=0.7, label=f'Best epoch ({history["best_epoch"]})')
    axes[1].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random baseline')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].set_title('AUC')
    axes[1].set_ylim(0.4, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Training curves saved to {save_path}')


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(train_cfg: TrainConfig, model_cfg: ModelConfig):
    """
    Full training run for one fold.

    Parameters
    ----------
    train_cfg : TrainConfig — paths, epochs, lr, etc.
    model_cfg : ModelConfig — model hyperparameters
    """
    set_seed(train_cfg.seed)
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(train_cfg.log_dir,        exist_ok=True)
    os.makedirs(train_cfg.results_dir,    exist_ok=True)

    device = torch.device(train_cfg.device)
    logger.info(f'\n[Train] Device: {device}')

    # --- Data ---
    meta   = pd.read_csv(train_cfg.metadata_path)
    folds  = get_splits(meta, strategy=train_cfg.split_strategy)
    train_p, val_p, test_p = folds[0]   # fold 0 for fixed; loop over folds for LOPO

    train_loader, val_loader, _ = make_dataloaders(
        meta,
        train_cfg.sequences_dir,
        train_p, val_p, test_p,
        batch_size=train_cfg.batch_size,
        num_workers=0,
    )

    # --- Model ---
    model = SeizurePredictor(model_cfg).to(device)
    model_summary(model, model_cfg)

    # --- Loss: weighted BCE ---
    # pos_weight penalises false negatives (missed seizures) more than false positives
    train_ds  = train_loader.dataset
    weights   = train_ds.get_class_weights()
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1].to(device))

    # --- Optimizer ---
    optimizer = Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    # --- LR scheduler ---
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',              # maximise val AUC
        factor=train_cfg.lr_factor,
        patience=train_cfg.lr_patience,
        min_lr=train_cfg.min_lr,
        verbose=True,
    )

    # --- Training state ---
    best_val_auc   = 0.0
    best_epoch     = 0
    epochs_no_improve = 0
    checkpoint_path = os.path.join(train_cfg.checkpoint_dir, 'best_model.pt')

    history = {
        'train_loss': [], 'val_loss': [],
        'train_auc':  [], 'val_auc':  [],
        'lr':         [],
        'best_epoch': 1,
    }

    logger.info(f'\n[Train] Starting training — max {train_cfg.max_epochs} epochs '
          f'| patience {train_cfg.patience}\n')
    logger.info(f'{"Epoch":>6} {"Train Loss":>11} {"Val Loss":>10} '
          f'{"Train AUC":>10} {"Val AUC":>9} {"LR":>10} {"":>8}')
    logger.info('-' * 72)

    for epoch in range(1, train_cfg.max_epochs + 1):
        t0 = time.time()

        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)

        # Check for improvement
        improved = ''
        if val_auc > best_val_auc:
            best_val_auc   = val_auc
            best_epoch     = epoch
            epochs_no_improve = 0
            history['best_epoch'] = epoch
            save_checkpoint(model, optimizer, epoch, val_auc,
                            model_cfg, checkpoint_path)
            improved = '✓ best'
        else:
            epochs_no_improve += 1

        elapsed = time.time() - t0
        logger.info(f'{epoch:>6} {train_loss:>11.4f} {val_loss:>10.4f} '
              f'{train_auc:>10.4f} {val_auc:>9.4f} '
              f'{current_lr:>10.2e}  {improved}')

        # Early stopping
        if epochs_no_improve >= train_cfg.patience:
            logger.info(f'\n[Train] Early stopping at epoch {epoch} '
                  f'(no improvement for {train_cfg.patience} epochs)')
            break

    logger.info(f'\n[Train] Training complete.')
    logger.info(f'  Best val AUC : {best_val_auc:.4f} at epoch {best_epoch}')
    logger.info(f'  Checkpoint   : {checkpoint_path}')

    # Save training history as JSON
    history_path = os.path.join(train_cfg.results_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Plot curves
    curves_path = os.path.join(train_cfg.results_dir, 'training_curves.png')
    plot_training_curves(history, curves_path)

    return model, best_val_auc, history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    train_cfg = TrainConfig()
    model_cfg = ModelConfig()

    logger.info('=' * 55)
    logger.info('SeizureHorizon — Training')
    logger.info('=' * 55)
    logger.info(f'Split strategy : {train_cfg.split_strategy}')
    logger.info(f'Max epochs     : {train_cfg.max_epochs}')
    logger.info(f'Batch size     : {train_cfg.batch_size}')
    logger.info(f'Learning rate  : {train_cfg.learning_rate}')
    logger.info(f'Early stopping : patience={train_cfg.patience}')
    logger.info(f'Device         : {train_cfg.device}')

    model, best_auc, history = train(train_cfg, model_cfg)

    logger.info(f'\nFinal best validation AUC: {best_auc:.4f}')
    logger.info('Run evaluate.py next to test on held-out patients.')
