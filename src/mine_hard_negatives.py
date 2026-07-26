"""
mine_hard_negatives.py
----------------------
Hard negative mining for the SeizureHorizon pipeline.

What it does
------------
1. Loads the trained model from experiments/checkpoints/best_model.pt
2. Runs inference on ALL existing interictal sequences
3. Identifies sequences where P(preictal) is high despite true label=0
   — these are the "hard negatives" that confuse the model
4. For each hard negative sequence, finds the source EDF file and
   extracts additional interictal windows from neighbouring time regions
5. Saves the new sequences as .npy files and appends rows to metadata.csv

The result is a richer interictal set that specifically targets the model's
failure modes, forcing the next training run to learn finer distinctions
between pre-ictal and confusing interictal patterns.

When to run
-----------
After the FIRST complete training run (train.py).
Then rebuild the dataset partially (only hard negatives are added —
existing sequences are preserved) and retrain.

Workflow
--------
    python src/train.py                   # first training run
    python src/mine_hard_negatives.py     # find hard negatives
    python src/train.py                   # retrain on augmented dataset

Run
---
    cd seizure_horizon
    python src/mine_hard_negatives.py
"""

import os
import sys
import csv
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import SeizureDataset, get_splits, N_CHANNELS
from model import SeizurePredictor, ModelConfig
from preprocessing import build_sequence_from_edf, PRE_ICTAL_SECS, STRIDE_SECONDS, WINDOW_SECONDS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

METADATA_PATH    = 'data/processed/metadata.csv'
SEQUENCES_DIR    = 'data/processed/sequences'
CHECKPOINT_PATH  = 'experiments/checkpoints/best_model.pt'
CHBMIT_ROOT      = 'data/raw/chb-mit'

# Sequences with P(preictal) above this threshold are considered hard negatives
# 0.4 catches sequences the model is meaningfully confused about
# (not just borderline 0.5 — we want clearly wrong predictions too)
HARD_NEGATIVE_THRESHOLD = 0.4

# For each hard negative, how many neighbouring windows to extract
# A neighbour is a window offset by ±1 anchor step (30 minutes) from the
# original hard negative anchor point
NEIGHBOURS_PER_HARD_NEGATIVE = 2

# Minimum gap from any seizure for new interictal windows (1 hour)
MIN_GAP_FROM_SEIZURE = 3600

# Device
DEVICE = (
    'mps'  if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available()          else
    'cpu'
)


# ---------------------------------------------------------------------------
# Step 1 — Load model and run inference on interictal sequences
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str) -> SeizurePredictor:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'No checkpoint at {checkpoint_path}. Run train.py first.'
        )
    ckpt   = torch.load(checkpoint_path, map_location='cpu')
    config = ModelConfig(**ckpt['model_config'])
    model  = SeizurePredictor(config)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'Loaded checkpoint from epoch {ckpt["epoch"]} '
          f'(val AUC = {ckpt["val_auc"]:.4f})')
    return model


@torch.no_grad()
def score_interictal_sequences(
    model: SeizurePredictor,
    meta: pd.DataFrame,
    sequences_dir: str,
    device: torch.device,
) -> pd.DataFrame:
    """
    Run inference on all interictal sequences and return a DataFrame
    with each sequence's predicted P(preictal).

    Only training split interictal sequences are scored — we never
    mine from val or test patients to avoid data leakage.

    Returns
    -------
    pd.DataFrame with columns:
        filename, patient_id, source_edf, anchor_sec, p_preictal
    sorted by p_preictal descending (hardest negatives first)
    """
    # Only mine from training patients
    folds = get_splits(meta, strategy='fixed')
    train_patients = folds[0][0]

    interictal_meta = meta[
        (meta['type'] == 'interictal') &
        (meta['patient_id'].isin(train_patients))
    ].reset_index(drop=True)

    print(f'\nScoring {len(interictal_meta)} interictal sequences '
          f'from {len(train_patients)} training patients...')

    model = model.to(device)
    scores = []

    for _, row in interictal_meta.iterrows():
        seq_path = os.path.join(sequences_dir, row['filename'])
        if not os.path.exists(seq_path):
            continue

        seq = np.load(seq_path)                      # (n_frames, 5, n_ch)
        seq = seq[:, :, :N_CHANNELS]                 # restrict channels

        # Instance normalise (same as dataset.py)
        seq = _instance_normalise(seq)

        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        proba, _ = model.predict_proba(tensor)
        p_preictal = proba.squeeze().item()

        scores.append({
            'filename':   row['filename'],
            'patient_id': row['patient_id'],
            'source_edf': row['source_edf'],
            'anchor_sec': row['anchor_sec'],
            'p_preictal': p_preictal,
        })

    scored_df = pd.DataFrame(scores)
    scored_df = scored_df.sort_values('p_preictal', ascending=False)

    print(f'\nTop 10 hardest interictal sequences:')
    print(scored_df.head(10)[['filename', 'patient_id', 'p_preictal']].to_string(index=False))

    return scored_df


def _instance_normalise(seq: np.ndarray) -> np.ndarray:
    """Mirror of dataset.py's _instance_normalise."""
    seq = seq.copy()
    non_zero_mask = seq.sum(axis=(1, 2)) != 0
    if non_zero_mask.sum() == 0:
        return seq
    real_frames = seq[non_zero_mask]
    mean = real_frames.mean(axis=0)
    std  = real_frames.std(axis=0)
    std  = np.where(std < 1e-8, 1e-8, std)
    seq[non_zero_mask] = (real_frames - mean) / std
    return seq


# ---------------------------------------------------------------------------
# Step 2 — Extract neighbouring windows around hard negatives
# ---------------------------------------------------------------------------

def get_seizure_onsets_for_file(
    meta: pd.DataFrame,
    patient_id: str,
    source_edf: str,
) -> list:
    """
    Get all seizure onset times (in seconds) for a given patient+file
    from the existing metadata, to ensure new windows stay far from seizures.
    """
    seizure_rows = meta[
        (meta['patient_id'] == patient_id) &
        (meta['source_edf'] == source_edf) &
        (meta['type'] == 'preictal')
    ]
    return seizure_rows['anchor_sec'].tolist()


def is_safe_anchor(
    anchor: float,
    seizure_onsets: list,
    existing_anchors: list,
    min_gap: float = MIN_GAP_FROM_SEIZURE,
) -> bool:
    """
    Check whether a candidate anchor point is:
    1. At least min_gap seconds from every seizure onset
    2. Not already used as an anchor in this file
    3. At least 30 minutes from the start of the recording
    """
    if anchor < PRE_ICTAL_SECS:
        return False

    for onset in seizure_onsets:
        if abs(anchor - onset) < min_gap:
            return False

    for existing in existing_anchors:
        if abs(anchor - existing) < PRE_ICTAL_SECS:
            return False

    return True


def mine_neighbours_for_sequence(
    patient_id: str,
    source_edf: str,
    anchor_sec: float,
    meta: pd.DataFrame,
    hard_neg_counter: list,
    sequences_dir: str,
) -> list:
    """
    Extract up to NEIGHBOURS_PER_HARD_NEGATIVE additional interictal windows
    from time regions neighbouring the given anchor point.

    Neighbours are placed at anchor ± 30 minutes, ± 60 minutes, etc.
    until we find enough safe anchors or run out of recording.

    Returns list of metadata row dicts for successfully saved sequences.
    """
    edf_path = os.path.join(CHBMIT_ROOT, patient_id, source_edf)
    if not os.path.exists(edf_path):
        print(f'  [MISSING] {edf_path}')
        return []

    # Get recording duration
    try:
        import mne
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        recording_duration = raw.times[-1]
    except Exception as e:
        print(f'  [ERROR reading {source_edf}] {e}')
        return []

    # All seizure onsets in this file
    seizure_onsets = get_seizure_onsets_for_file(meta, patient_id, source_edf)

    # All existing anchor points in this file (to avoid duplication)
    existing_file_anchors = meta[
        (meta['patient_id'] == patient_id) &
        (meta['source_edf'] == source_edf)
    ]['anchor_sec'].tolist()

    # Generate candidate neighbour anchors at ±30, ±60, ±90 min offsets
    step          = PRE_ICTAL_SECS   # 30 minutes
    candidates    = []
    for multiplier in [1, -1, 2, -2, 3, -3]:
        candidate = anchor_sec + multiplier * step
        if 0 < candidate <= recording_duration:
            candidates.append(candidate)

    saved_rows = []
    found      = 0

    for candidate in candidates:
        if found >= NEIGHBOURS_PER_HARD_NEGATIVE:
            break

        if not is_safe_anchor(candidate, seizure_onsets, existing_file_anchors):
            continue

        # Extract sequence
        hard_neg_counter[0] += 1
        idx       = hard_neg_counter[0]
        save_name = f'{patient_id}_hardneg{idx:03d}_interictal.npy'
        save_path = os.path.join(sequences_dir, save_name)

        if os.path.exists(save_path):
            print(f'  [SKIP — exists] {save_name}')
            existing_file_anchors.append(candidate)
            seq = np.load(save_path)
            saved_rows.append(_make_row(
                save_name, patient_id, source_edf, candidate, seq.shape, 'cached'
            ))
            found += 1
            continue

        try:
            sequence, sfreq, n_ch = build_sequence_from_edf(
                edf_path,
                end_time=float(candidate),
                duration=float(PRE_ICTAL_SECS),
            )
            np.save(save_path, sequence)
            existing_file_anchors.append(candidate)

            print(f'  [SAVED] {save_name}  '
                  f'anchor={candidate:.0f}s  shape={sequence.shape}')

            saved_rows.append(_make_row(
                save_name, patient_id, source_edf, candidate, sequence.shape, 'hard_negative'
            ))
            found += 1

        except Exception as e:
            print(f'  [ERROR] {patient_id} {source_edf} anchor={candidate:.0f}s — {e}')

    return saved_rows


def _make_row(filename, patient_id, source_edf, anchor_sec, shape, note) -> dict:
    return {
        'filename':   filename,
        'patient_id': patient_id,
        'source_edf': source_edf,
        'type':       'interictal',
        'label':      0,
        'anchor_sec': anchor_sec,
        'n_frames':   shape[0],
        'n_bands':    shape[1],
        'n_channels': shape[2],
        'note':       note,
    }


# ---------------------------------------------------------------------------
# Step 3 — Append new rows to metadata.csv
# ---------------------------------------------------------------------------

def append_to_metadata(new_rows: list, metadata_path: str):
    """
    Append new sequence rows to the existing metadata.csv.
    Does not duplicate rows that already exist (checks by filename).
    """
    existing = pd.read_csv(metadata_path)
    existing_files = set(existing['filename'].tolist())

    truly_new = [r for r in new_rows if r['filename'] not in existing_files]

    if not truly_new:
        print('\nNo new rows to append — all hard negatives already in metadata.')
        return

    new_df   = pd.DataFrame(truly_new)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(metadata_path, index=False)

    print(f'\nAppended {len(truly_new)} new rows to {metadata_path}')
    print(f'Total sequences now: {len(combined)}')
    print(f'  preictal   : {(combined["label"] == 1).sum()}')
    print(f'  interictal : {(combined["label"] == 0).sum()}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def mine_hard_negatives():
    print('=' * 55)
    print('SeizureHorizon — Hard Negative Mining')
    print('=' * 55)
    print(f'Threshold : P(preictal) >= {HARD_NEGATIVE_THRESHOLD} '
          f'→ flagged as hard negative')
    print(f'Neighbours: {NEIGHBOURS_PER_HARD_NEGATIVE} per hard negative')
    print(f'Min gap   : {MIN_GAP_FROM_SEIZURE}s from any seizure')
    print()

    device = torch.device(DEVICE)
    print(f'Device: {device}')

    # Load model and metadata
    model = load_model(CHECKPOINT_PATH)
    meta  = pd.read_csv(METADATA_PATH)

    # Step 1 — Score all training interictal sequences
    scored = score_interictal_sequences(model, meta, SEQUENCES_DIR, device)

    # Step 2 — Filter to hard negatives
    hard_negatives = scored[scored['p_preictal'] >= HARD_NEGATIVE_THRESHOLD]
    print(f'\nHard negatives found: {len(hard_negatives)} / {len(scored)} '
          f'interictal sequences')
    print(f'(P(preictal) >= {HARD_NEGATIVE_THRESHOLD})')

    if len(hard_negatives) == 0:
        print('\nNo hard negatives found. Model classifies all interictal '
              'sequences confidently. No mining needed.')
        return

    # Step 3 — Mine neighbours for each hard negative
    print(f'\nMining {NEIGHBOURS_PER_HARD_NEGATIVE} neighbours '
          f'per hard negative...\n')

    all_new_rows    = []
    hard_neg_counter = [0]  # mutable counter

    for _, row in hard_negatives.iterrows():
        print(f'Hard negative: {row["filename"]}  '
              f'P(preictal)={row["p_preictal"]:.3f}')

        new_rows = mine_neighbours_for_sequence(
            patient_id=row['patient_id'],
            source_edf=row['source_edf'],
            anchor_sec=row['anchor_sec'],
            meta=meta,
            hard_neg_counter=hard_neg_counter,
            sequences_dir=SEQUENCES_DIR,
        )
        all_new_rows.extend(new_rows)

    # Step 4 — Append to metadata.csv
    print(f'\n{"=" * 55}')
    print(f'Mining complete. New sequences extracted: {len(all_new_rows)}')
    append_to_metadata(all_new_rows, METADATA_PATH)

    print('\nNext steps:')
    print('  1. Run python src/train.py  to retrain on the augmented dataset')
    print('  2. Run python src/evaluate.py  to compare results')


if __name__ == '__main__':
    mine_hard_negatives()
