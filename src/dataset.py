"""
dataset.py
----------
PyTorch Dataset and split utilities for the SeizureHorizon project.

Responsibilities
----------------
1. Load metadata.csv and filter by patient split
2. Load .npy sequence files on demand (lazy loading — not all at once)
3. Normalise each sequence to zero mean and unit variance per channel per band
4. Restrict channels to the 17 globally valid ones
5. Return (sequence, label) pairs ready for the DataLoader
6. Provide get_splits() for both fixed and LOPO cross-validation strategies

Usage
-----
from src.dataset import SeizureDataset, get_splits

folds = get_splits(meta, strategy='fixed')
train_patients, val_patients, test_patients = folds[0]

train_ds = SeizureDataset(meta, sequences_dir, train_patients)
val_ds   = SeizureDataset(meta, sequences_dir, val_patients)
test_ds  = SeizureDataset(meta, sequences_dir, test_patients)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# N_FRAMES is no longer a fixed global — it varies per experiment
# configuration (offset/duration combination). It is now inferred
# directly from the .npy files themselves at dataset load time,
# so no manual bookkeeping is needed when running the grid search.

# Number of frequency bands
N_BANDS    = 5

# Number of channels is now defined by the canonical montage in preprocessing
# (Fix F1). build_dataset projects every window onto these channels in a fixed
# order, so channel i is always the same electrode across patients/files.
from preprocessing import CANONICAL_CHANNELS
N_CHANNELS = len(CANONICAL_CHANNELS)

# Fixed predetermined patient split
# 17 train / 4 val / 3 test  (patient-level — no patient appears in two splits)
FIXED_TEST_PATIENTS  = ['chb01', 'chb02', 'chb03']
FIXED_VAL_PATIENTS   = ['chb04', 'chb05', 'chb06', 'chb07']


# ---------------------------------------------------------------------------
# Split utility
# ---------------------------------------------------------------------------

def get_splits(meta: pd.DataFrame, strategy: str = 'fixed') -> list:
    """
    Return a list of (train_patients, val_patients, test_patients) tuples.

    Parameters
    ----------
    meta     : pd.DataFrame — loaded metadata.csv
    strategy : str
        'fixed' — one fold, predetermined split, fast (for development)
        'lopo'  — leave-one-patient-out, 24 folds (for final publication)

    Returns
    -------
    list of tuples: [(train_list, val_list, test_list), ...]
    Fixed strategy returns a list with exactly one tuple.
    LOPO returns a list with 24 tuples (one per patient as test).

    Switching from fixed to LOPO is a one-argument change at call site.
    """
    all_patients = sorted(meta['patient_id'].unique())

    if strategy == 'fixed':
        test_patients  = FIXED_TEST_PATIENTS
        val_patients   = FIXED_VAL_PATIENTS
        train_patients = [
            p for p in all_patients
            if p not in test_patients and p not in val_patients
        ]
        print(f'[Split] Fixed strategy')
        print(f'  Train : {len(train_patients)} patients — {train_patients}')
        print(f'  Val   : {len(val_patients)} patients — {val_patients}')
        print(f'  Test  : {len(test_patients)} patients — {test_patients}')
        return [(train_patients, val_patients, test_patients)]

    elif strategy == 'lopo':
        # Leave-one-patient-out: rotate through all patients as test
        # The patient immediately after test (cyclically) is validation
        folds = []
        n = len(all_patients)
        for i, test_p in enumerate(all_patients):
            val_p   = [all_patients[(i + 1) % n]]
            train_p = [
                p for p in all_patients
                if p != test_p and p not in val_p
            ]
            folds.append((train_p, val_p, [test_p]))
        print(f'[Split] LOPO strategy — {len(folds)} folds')
        return folds

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Use 'fixed' or 'lopo'.")


# ---------------------------------------------------------------------------
# Channel restriction utility
# ---------------------------------------------------------------------------

def get_valid_channel_indices(ch_names: list, montage_pos: dict) -> list:
    """
    Return indices of channels whose first electrode maps to a known
    10-20 scalp position.

    Parameters
    ----------
    ch_names    : list of str — channel names from the EDF file
    montage_pos : dict — channel name → (x, y, z) from MNE montage

    Returns
    -------
    list of int — indices into ch_names that are valid
    """
    valid = []
    for i, ch in enumerate(ch_names):
        first_el = ch.split('-')[0].strip().upper()
        if first_el in montage_pos:
            valid.append(i)
    return valid


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SeizureDataset(Dataset):
    """
    PyTorch Dataset for pre-ictal / interictal EEG sequences.

    Each item is a tuple:
        sequence : torch.Tensor, shape (N_FRAMES, N_BANDS, N_CHANNELS) float32
        label    : torch.Tensor, scalar, 1 for preictal / 0 for interictal

    Normalisation
    -------------
    Each sequence is normalised independently (per channel per band) to
    zero mean and unit variance. This is instance normalisation — it removes
    absolute power differences between patients and recording sessions,
    forcing the model to learn from relative spatial patterns and temporal
    changes rather than absolute amplitude levels.

    Channel restriction
    -------------------
    Only the first N_CHANNELS channels of each sequence are kept.
    build_dataset.py saves sequences with channels ordered consistently,
    so slicing [:N_CHANNELS] on axis 2 gives the 17 globally valid channels.

    Parameters
    ----------
    meta           : pd.DataFrame — full metadata.csv loaded with pd.read_csv
    sequences_dir  : str — path to data/processed/sequences/
    patient_list   : list of str — which patients to include in this split
    augment        : bool — apply time-shift augmentation (training only)
    """

    def __init__(
        self,
        meta: pd.DataFrame,
        sequences_dir: str,
        patient_list: list,
        augment: bool = False,
        norm_stats: dict | None = None,
    ):
        self.sequences_dir = sequences_dir
        self.augment       = augment
        self.norm_stats    = norm_stats   # Fix F3: train-only (mean, std) or None

        # Filter metadata to only the requested patients
        self.meta = meta[meta['patient_id'].isin(patient_list)].reset_index(drop=True)

        if len(self.meta) == 0:
            raise ValueError(
                f'No sequences found for patients: {patient_list}. '
                f'Check that metadata.csv contains these patient IDs.'
            )

        # Class counts — used externally to compute loss weights
        self.n_preictal   = int((self.meta['label'] == 1).sum())
        self.n_interictal = int((self.meta['label'] == 0).sum())

        print(f'[Dataset] {len(patient_list)} patients | '
              f'{len(self.meta)} sequences | '
              f'preictal={self.n_preictal} | '
              f'interictal={self.n_interictal} | '
              f'augment={augment}')

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> tuple:
        row = self.meta.iloc[idx]

        # --- Load sequence ---
        seq_path = os.path.join(self.sequences_dir, row['filename'])
        seq = np.load(seq_path)   # shape: (360, 5, n_channels)

        # --- Channels are already the canonical montage (Fix F1) ---
        # build_dataset now saves exactly N_CHANNELS canonical channels in a
        # fixed order. Slice/pad defensively in case an older cache is present.
        if seq.shape[2] >= N_CHANNELS:
            seq = seq[:, :, :N_CHANNELS]
        else:
            pad = np.zeros((seq.shape[0], seq.shape[1], N_CHANNELS - seq.shape[2]),
                           dtype=seq.dtype)
            seq = np.concatenate([seq, pad], axis=2)

        # --- Augmentation (training only) ---
        # Time-shift: randomly roll the sequence along the time axis
        # This simulates the model seeing different starting points
        # and prevents it from over-relying on absolute frame position.
        if self.augment:
            # --- Time-shift augmentation ---
            shift = np.random.randint(-18, 18)   # ± 3 minutes (18 × 5s windows)
            seq   = np.roll(seq, shift, axis=0)

            # Zero out frames that rolled in from the other end
            if shift > 0:
                seq[:shift] = 0.0
            elif shift < 0:
                seq[shift:] = 0.0

            # --- Gaussian noise injection ---
            # Adds small random noise scaled to each sequence's own std,
            # simulating natural measurement noise in EEG recordings.
            # Only applied to non-padded (real) frames.
            non_zero_mask = seq.sum(axis=(1, 2)) != 0
            if non_zero_mask.sum() > 0:
                noise_std = seq[non_zero_mask].std() * 0.02   # 2% (5% washed out the weak distal signal)
                noise = np.random.normal(0, noise_std, size=seq.shape).astype(np.float32)
                seq[non_zero_mask] += noise[non_zero_mask]

            # --- Channel dropout ---
            # Randomly zero out 1-2 channels across the entire sequence.
            # Forces the model to not over-rely on any single electrode,
            # improving robustness to electrode artifacts or disconnections.
            n_channels = seq.shape[2]
            n_drop = np.random.choice([0, 1], p=[0.8, 0.2])  # at most 1 dropped; usually none
            if n_drop > 0:
                drop_channels = np.random.choice(n_channels, size=n_drop, replace=False)
                seq[:, :, drop_channels] = 0.0

        # --- Normalisation (Fix F3: train-only global stats when available) ---
        # If per-(band, channel) statistics fitted on the TRAINING split were
        # supplied, standardise with those (no val/test leakage, shared scale
        # across patients). Otherwise fall back to per-sequence instance
        # normalisation (also leakage-free).
        if self.norm_stats is not None:
            seq = _global_normalise(seq, self.norm_stats['mean'],
                                    self.norm_stats['std'])
        else:
            seq = _instance_normalise(seq)

        # --- Convert to tensor ---
        sequence_tensor = torch.tensor(seq, dtype=torch.float32)
        label_tensor    = torch.tensor(float(row['label']), dtype=torch.float32)

        return sequence_tensor, label_tensor

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for weighted binary cross-entropy loss.

        Returns a tensor [weight_interictal, weight_preictal] where
        the preictal class is weighted 1.5× higher to penalise
        missed seizure predictions more than false alarms.

        Usage in training:
            weights = train_ds.get_class_weights()
            # Pass pos_weight to BCEWithLogitsLoss:
            criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1])
        """
        # pos_weight for BCEWithLogitsLoss should be n_negative / n_positive
        # = n_interictal / n_preictal. That alone corrects the imbalance.
        # The old extra 1.5x "preictal bias" double-counted on top of the
        # already-minority preictal class and forced every prediction positive
        # (test specificity collapsed to 0.0).
        w_interictal  = 1.0
        # pos_weight = n_neg/n_pos corrects the *usual* imbalance (interictal
        # majority). After F4 tiling the balance can invert (pre-ictal majority),
        # making this < 1, which punishes false alarms harder than missed
        # seizures and drives the model to never fire. Floor at 1.0 so positives
        # are never penalised below neutral (bump to ~1.25 if still trigger-shy).
        w_preictal    = max(1.0, self.n_interictal / self.n_preictal)
        weights = torch.tensor([w_interictal, w_preictal], dtype=torch.float32)
        print(f'[Dataset] Class weights — interictal: {w_interictal:.3f} | '
              f'preictal: {w_preictal:.3f}')
        return weights


# ---------------------------------------------------------------------------
# Fix F3 - train-only global normalisation
# ---------------------------------------------------------------------------

def fit_channel_stats(meta: pd.DataFrame, sequences_dir: str,
                      train_patients: list) -> dict:
    """
    Compute per-(band, channel) mean and std over the TRAINING split only,
    using non-zero (non-padded) frames. The same stats are applied to every
    split so no validation/test information leaks into normalisation.
    """
    sub = meta[meta['patient_id'].isin(train_patients)]
    sums = None
    sqs  = None
    count = 0
    for fname in sub['filename']:
        p = os.path.join(sequences_dir, fname)
        if not os.path.exists(p):
            continue
        seq = np.load(p)
        if seq.shape[2] >= N_CHANNELS:
            seq = seq[:, :, :N_CHANNELS]
        nz = seq.sum(axis=(1, 2)) != 0
        if nz.sum() == 0:
            continue
        real = seq[nz]                      # (n_real, N_BANDS, N_CHANNELS)
        if sums is None:
            sums = real.sum(axis=0)
            sqs  = (real ** 2).sum(axis=0)
        else:
            sums += real.sum(axis=0)
            sqs  += (real ** 2).sum(axis=0)
        count += real.shape[0]
    if not count:
        return {'mean': np.zeros((N_BANDS, N_CHANNELS), dtype=np.float32),
                'std':  np.ones((N_BANDS, N_CHANNELS), dtype=np.float32)}
    mean = (sums / count).astype(np.float32)
    var  = (sqs / count) - mean ** 2
    std  = np.sqrt(np.clip(var, 1e-12, None)).astype(np.float32)
    std  = np.where(std < 1e-8, 1e-8, std).astype(np.float32)
    return {'mean': mean, 'std': std}


def _global_normalise(seq: np.ndarray, mean: np.ndarray,
                      std: np.ndarray) -> np.ndarray:
    """Standardise non-padded frames with pre-fitted (train-only) per-(band,
    channel) statistics; padded (all-zero) frames stay zero."""
    seq = seq.copy()
    nz = seq.sum(axis=(1, 2)) != 0
    if nz.sum() == 0:
        return seq
    seq[nz] = (seq[nz] - mean) / std
    return seq


# ---------------------------------------------------------------------------
# Instance normalisation helper
# ---------------------------------------------------------------------------

def _instance_normalise(seq: np.ndarray) -> np.ndarray:
    """
    Normalise a single sequence to zero mean and unit variance,
    computed per band per channel using only non-zero (non-padded) frames.

    Parameters
    ----------
    seq : np.ndarray, shape (N_FRAMES, N_BANDS, N_CHANNELS)

    Returns
    -------
    np.ndarray, same shape, normalised
    """
    seq = seq.copy()

    # Identify non-padded frames (frames where any value is non-zero)
    non_zero_mask = seq.sum(axis=(1, 2)) != 0   # (N_FRAMES,) bool

    if non_zero_mask.sum() == 0:
        # Fully padded sequence — return as-is (all zeros)
        return seq

    real_frames = seq[non_zero_mask]   # (n_real, N_BANDS, N_CHANNELS)

    # Compute stats over real frames for each band and channel
    # mean/std shape: (N_BANDS, N_CHANNELS)
    mean = real_frames.mean(axis=0)
    std  = real_frames.std(axis=0)
    std  = np.where(std < 1e-8, 1e-8, std)   # avoid division by zero

    # Apply normalisation to real frames only; padded frames stay zero
    seq[non_zero_mask] = (real_frames - mean) / std

    return seq


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    meta: pd.DataFrame,
    sequences_dir: str,
    train_patients: list,
    val_patients: list,
    test_patients: list,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple:
    """
    Convenience function: create train, val, and test DataLoaders
    for one fold.

    Parameters
    ----------
    meta            : pd.DataFrame
    sequences_dir   : str
    train_patients  : list of str
    val_patients    : list of str
    test_patients   : list of str
    batch_size      : int (default 16 — safe for MacBook memory)
    num_workers     : int (default 0 — set to 4 on server with multiple cores)

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    # Fix F3: fit normalisation statistics on the TRAINING patients only,
    # then share them with val/test so no evaluation data leaks in.
    # norm_stats = fit_channel_stats(meta, sequences_dir, train_patients)
    norm_stats = None

    train_ds = SeizureDataset(meta, sequences_dir, train_patients, augment=True,
                              norm_stats=norm_stats)
    val_ds   = SeizureDataset(meta, sequences_dir, val_patients,   augment=False,
                              norm_stats=norm_stats)
    test_ds  = SeizureDataset(meta, sequences_dir, test_patients,  augment=False,
                              norm_stats=norm_stats)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,             # shuffle training order every epoch
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,           # drop incomplete last batch for stable BN
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader

def infer_n_frames(sequences_dir: str) -> int:
    """
    Inspect one .npy file in sequences_dir to determine the frame count
    for this configuration. Used to set ModelConfig.n_frames correctly
    before building the model, since frame count varies by offset/duration.

    Parameters
    ----------
    sequences_dir : str — path to a config's sequences/ folder

    Returns
    -------
    int — number of frames (first dimension) in the sequences
    """
    files = [f for f in os.listdir(sequences_dir) if f.endswith('.npy')]
    if not files:
        raise ValueError(f'No .npy files found in {sequences_dir}')

    sample = np.load(os.path.join(sequences_dir, files[0]))
    return sample.shape[0]


# ---------------------------------------------------------------------------
# Self-test — run directly to verify dataset loads correctly
#   python src/dataset.py
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    sequences_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/processed/offset0_dur30/sequences'
    metadata_path = sys.argv[2] if len(sys.argv) > 2 else 'data/processed/offset0_dur30/metadata.csv'

    print('=== SeizureDataset self-test ===\n')

    meta   = pd.read_csv(metadata_path)
    folds  = get_splits(meta, strategy='fixed')
    train_p, val_p, test_p = folds[0]

    train_loader, val_loader, test_loader = make_dataloaders(
        meta, sequences_dir, train_p, val_p, test_p,
        batch_size=4,
    )

    print('\n--- Inspecting one training batch ---')
    seqs, labels = next(iter(train_loader))
    print(f'Sequence batch shape : {seqs.shape}')   # (4, n_frames, 5, 17) — n_frames varies by config
    print(f'Label batch shape    : {labels.shape}') # (4,)
    print(f'Label values         : {labels}')
    print(f'Sequence dtype       : {seqs.dtype}')
    print(f'Sequence min/max     : {seqs.min():.4f} / {seqs.max():.4f}')
    print(f'Any NaN in sequence  : {torch.isnan(seqs).any()}')

    print('\n--- Class weights ---')
    train_ds = train_loader.dataset
    weights  = train_ds.get_class_weights()
    print(f'Weights tensor: {weights}')

    print('\nSelf-test complete. Dataset is ready for model training.')
