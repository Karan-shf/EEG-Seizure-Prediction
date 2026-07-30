"""
build_dataset.py
----------------
Orchestrates the full dataset construction pipeline for SeizureHorizon.

What it does
------------
1. Reads all CHB-MIT summary files to locate seizure events
2. For each seizure event:
   - Extracts the 30-minute pre-ictal window ending at seizure onset
   - Computes the (360, 5, n_channels) band power sequence
   - Saves it as a .npy file in data/processed/sequences/
3. For each patient:
   - Finds a quiet interictal period (>= 4 hours from any seizure)
   - Extracts a matching 30-minute window
   - Computes and saves its sequence
4. Writes data/processed/metadata.csv indexing all saved sequences

Run once to build the dataset:
    python src/build_dataset.py

After this script completes, data/processed/sequences/ contains all
training samples and metadata.csv is the index your Dataset class will read.
"""

import os
import csv
import json
import numpy as np

from parse_summaries import parse_all_summaries, get_all_seizure_events
from preprocessing import build_sequence_from_edf, PRE_ICTAL_SECS, WINDOW_SECONDS, STRIDE_SECONDS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# AFTER
CHBMIT_ROOT = 'data/raw/chb-mit'

INTERICTAL_MIN_GAP = 3600
MAX_INTERICTAL_PER_PATIENT = 10

# TARGET_FRAMES is now computed per-config inside build_dataset(),
# since it depends on that config's duration_minutes — no longer a
# fixed global constant.


# ---------------------------------------------------------------------------
# Helper: collect all seizure times for a patient (across all files)
# in absolute seconds from the start of the *first* recording file.
# CHB-MIT files are independent recordings — we treat each file's time
# axis as starting from 0, so we track them per-file, not absolutely.
# ---------------------------------------------------------------------------

def get_patient_seizure_times(patient_data: dict) -> dict:
    """
    Return a dict mapping each .edf filename to a list of onset times (seconds).

    Parameters
    ----------
    patient_data : dict — the per-file seizure dict for one patient

    Returns
    -------
    dict { filename: [onset_sec, ...] }
    """
    return {
        fname: [s['onset'] for s in seizures]
        for fname, seizures in patient_data.items()
    }


def find_interictal_anchors(
    raw_duration: float,
    seizure_onsets: list,
    min_gap: int = INTERICTAL_MIN_GAP,
    n_anchors: int = MAX_INTERICTAL_PER_PATIENT,
    window_needed: float = PRE_ICTAL_SECS,
) -> list:
    """
    Find up to n_anchors time points within a recording that are at least
    min_gap seconds away from every seizure onset, and have at least
    window_needed seconds of EEG before them.

    Parameters
    ----------
    raw_duration   : float — total duration of the recording in seconds
    seizure_onsets : list  — list of seizure onset times in this recording
    min_gap        : int   — minimum distance from any seizure (seconds)
    n_anchors      : int   — maximum number of anchors to return
    window_needed  : float — how many seconds before the anchor we need

    Returns
    -------
    list of float anchor times (end of interictal window = anchor)
    """
    anchors = []

    # Candidate anchor points: every 30-minute mark through the recording
    candidate_step = window_needed  # check every 30 minutes
    t = window_needed  # first candidate: 30 minutes into the recording

    while t <= raw_duration and len(anchors) < n_anchors:
        # Check distance from every seizure in this file
        too_close = any(
            abs(t - onset) < min_gap
            for onset in seizure_onsets
        )
        if not too_close:
            anchors.append(t)

        t += candidate_step

    return anchors


# ---------------------------------------------------------------------------
# Core: process one seizure event
# ---------------------------------------------------------------------------

def process_preictal_event(
    patient_id: str,
    filename: str,
    onset: int,
    event_idx: int,
    edf_path: str,
    output_dir: str,
    offset_sec: float,
    duration_sec: float,
    target_frames: int,
    logger
) -> dict | None:
    """
    Extract a pre-ictal sequence for one seizure event and save it.

    Returns a metadata dict row on success, or None if the event is skipped.
    """
    save_name = f'{patient_id}_sz{event_idx:02d}_preictal.npy'
    save_path = os.path.join(output_dir, save_name)

    if os.path.exists(save_path):
        logger.info(f'  [SKIP — already exists] {save_name}')
        # Still return metadata so it appears in the CSV
        seq = np.load(save_path)
        return _make_metadata_row(
            save_name, patient_id, filename, 'preictal', 1,
            onset, seq.shape, note='cached'
        )

    try:
        sequence, sfreq, n_ch = build_sequence_from_edf(
            edf_path,
            anchor_time=float(onset),
            duration=float(duration_sec),
            offset=float(offset_sec),
            target_n_windows=target_frames,
        )
        np.save(save_path, sequence)
        logger.info(f'  [SAVED] {save_name}  shape={sequence.shape}  sfreq={sfreq}')

        # Count how many leading frames are zero-padded
        non_zero_frames = int(np.any(sequence != 0, axis=(1, 2)).sum())
        padded_frames   = target_frames - non_zero_frames
        note = f'padded_frames={padded_frames}' if padded_frames > 0 else 'full'

        return _make_metadata_row(
            save_name, patient_id, filename, 'preictal', 1,
            onset, sequence.shape, note=note
        )

    except ValueError as e:
        logger.info(f'  [SKIP] {patient_id} {filename} onset={onset}s — {e}')
        return None

    except Exception as e:
        logger.info(f'  [ERROR] {patient_id} {filename} onset={onset}s — {e}')
        return None


# ---------------------------------------------------------------------------
# Core: process interictal windows for one patient file
# ---------------------------------------------------------------------------

def process_interictal_events(
    patient_id: str,
    filename: str,
    edf_path: str,
    seizure_onsets: list,
    inter_event_counter: list,   # mutable counter shared across calls
    output_dir: str,
    offset_sec: float,
    duration_sec: float,
    target_frames: int,
    logger
) -> list:
    """
    Extract interictal sequences from a recording file and save them.

    Returns a list of metadata row dicts (one per saved sequence).
    """
    rows = []

    try:
        import mne
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        raw_duration = raw.times[-1]
    except Exception as e:
        logger.info(f'  [ERROR reading {filename} for interictal] {e}')
        return rows

    anchors = find_interictal_anchors(raw_duration, seizure_onsets, n_anchors=1)

    if not anchors:
        logger.info(f'  [NO INTERICTAL] {patient_id} {filename} — no safe window found')
        return rows

    for anchor in anchors:
        inter_event_counter[0] += 1
        idx       = inter_event_counter[0]
        save_name = f'{patient_id}_inter{idx:02d}_interictal.npy'
        save_path = os.path.join(output_dir, save_name)

        if os.path.exists(save_path):
            logger.info(f'  [SKIP — already exists] {save_name}')
            seq = np.load(save_path)
            rows.append(_make_metadata_row(
                save_name, patient_id, filename, 'interictal', 0,
                anchor, seq.shape, note='cached'
            ))
            continue

        try:
            sequence, sfreq, n_ch = build_sequence_from_edf(
                edf_path,
                anchor_time=float(anchor),
                duration=float(duration_sec),
                offset=float(offset_sec),
                target_n_windows=target_frames,
            )
            np.save(save_path, sequence)
            logger.info(f'  [SAVED] {save_name}  shape={sequence.shape}')

            rows.append(_make_metadata_row(
                save_name, patient_id, filename, 'interictal', 0,
                anchor, sequence.shape, note='full'
            ))

        except Exception as e:
            logger.info(f'  [ERROR] interictal {patient_id} {filename} anchor={anchor}s — {e}')

    return rows


# ---------------------------------------------------------------------------
# Helper: build a metadata CSV row dict
# ---------------------------------------------------------------------------

def _make_metadata_row(
    filename, patient_id, source_edf, split_type, label,
    anchor_sec, shape, note=''
) -> dict:
    return {
        'filename':   filename,
        'patient_id': patient_id,
        'source_edf': source_edf,
        'type':       split_type,       # 'preictal' or 'interictal'
        'label':      label,            # 1 or 0
        'anchor_sec': anchor_sec,
        'n_frames':   shape[0],
        'n_bands':    shape[1],
        'n_channels': shape[2],
        'note':       note,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def build_dataset(
    offset_minutes: float = 0,
    duration_minutes: float = 30,
    output_root: str = 'data/processed',
    config_name: str | None = None,
):
    """
    Build a full dataset for one (offset, duration) window configuration.

    Parameters
    ----------
    offset_minutes   : float — gap in minutes between window end and seizure onset
                        0 = window ends exactly at onset (original behaviour)
    duration_minutes : float — length of the window in minutes
    output_root       : str  — base directory; a config-specific subfolder
                                is created beneath it
    config_name        : str  — folder/file naming; auto-generated if None
                                e.g. 'offset15_dur30'

    Output structure
    -----------------
    {output_root}/{config_name}/sequences/*.npy
    {output_root}/{config_name}/metadata.csv
    """

    from logger import get_logger

    if config_name is None:
        config_name = f'offset{int(offset_minutes)}_dur{int(duration_minutes)}'

    logger = get_logger(
        name=f'{config_name}_dataset',
        log_dir=os.path.join('experiments', 'logs', config_name or 'default'),
    )

    output_dir    = os.path.join(output_root, config_name, 'sequences')
    metadata_path = os.path.join(output_root, config_name, 'metadata.csv')

    offset_sec   = offset_minutes * 60
    duration_sec = duration_minutes * 60

    # Compute target frame count for this specific duration
    # (must match the logic in preprocessing.build_sequence_from_edf)
    samples_per_win    = int(WINDOW_SECONDS * 256)     # 256 Hz standard CHB-MIT rate
    samples_per_stride = int(STRIDE_SECONDS * 256)
    target_frames = (
        (int(duration_sec * 256) - samples_per_win) // samples_per_stride
    ) + 1

    os.makedirs(output_dir, exist_ok=True)

    logger.info('=' * 60)
    logger.info(f'SeizureHorizon — Dataset Builder — [{config_name}]')
    logger.info('=' * 60)
    logger.info(f'CHB-MIT root   : {CHBMIT_ROOT}')
    logger.info(f'Output dir     : {output_dir}')
    logger.info(f'Metadata file  : {metadata_path}')
    logger.info(f'Offset         : {offset_minutes} min ({offset_sec:.0f}s)')
    logger.info(f'Duration       : {duration_minutes} min ({duration_sec:.0f}s)')
    logger.info(f'Target frames  : {target_frames}')
    logger.info('')

    logger.info('Step 1: Parsing summary files...')
    seizure_index = parse_all_summaries(CHBMIT_ROOT, logger)
    all_events    = get_all_seizure_events(seizure_index)
    logger.info(f'\nTotal seizure events found: {len(all_events)}\n')

    metadata_rows = []

    for patient_id, patient_data in sorted(seizure_index.items()):
        logger.info(f'\n{"─" * 50}')
        logger.info(f'Patient: {patient_id}')
        logger.info(f'{"─" * 50}')

        patient_dir    = os.path.join(CHBMIT_ROOT, patient_id)
        seizure_times  = get_patient_seizure_times(patient_data)
        preictal_count = 0
        inter_counter  = [0]

        # --- Pre-ictal sequences ---
        for filename, seizures in sorted(patient_data.items()):
            if not seizures:
                continue

            edf_path = os.path.join(patient_dir, filename)
            if not os.path.exists(edf_path):
                logger.info(f'  [MISSING EDF] {edf_path}')
                continue

            logger.info(f'\n  File: {filename}')
            prev_offset = None

            for sz in sorted(seizures, key=lambda x: x['onset']):
                onset  = sz['onset']
                offset = sz['offset']

                # Skip if too close to previous seizure given this window's reach
                min_gap_needed = offset_sec + duration_sec
                if prev_offset is not None and onset - prev_offset < min_gap_needed:
                    logger.info(f'  [SKIP — too close to previous seizure] '
                          f'onset={onset}s, prev_offset={prev_offset}s')
                    prev_offset = offset
                    continue

                preictal_count += 1
                row = process_preictal_event(
                    patient_id, filename, onset, preictal_count, edf_path, output_dir,
                    offset_sec=offset_sec,
                    duration_sec=duration_sec,
                    target_frames=target_frames,
                    logger=logger
                )
                if row:
                    metadata_rows.append(row)

                prev_offset = offset

        # --- Interictal sequences ---
        for filename, seizures in sorted(patient_data.items()):
            if inter_counter[0] >= MAX_INTERICTAL_PER_PATIENT:
                break

            edf_path = os.path.join(patient_dir, filename)
            if not os.path.exists(edf_path):
                continue

            all_onsets_in_file = seizure_times.get(filename, [])
            inter_rows = process_interictal_events(
                patient_id, filename, edf_path,
                all_onsets_in_file, inter_counter, output_dir,
                offset_sec=offset_sec,
                duration_sec=duration_sec,
                target_frames=target_frames,
                logger=logger
            )
            metadata_rows.extend(inter_rows)

    # --- Write metadata CSV ---
    logger.info(f'\n{"=" * 60}')
    logger.info(f'Writing metadata to {metadata_path} ...')

    fieldnames = [
        'filename', 'patient_id', 'source_edf', 'type', 'label',
        'anchor_sec', 'n_frames', 'n_bands', 'n_channels', 'note'
    ]

    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    preictal_rows   = [r for r in metadata_rows if r['label'] == 1]
    interictal_rows = [r for r in metadata_rows if r['label'] == 0]
    patients_done   = len(set(r['patient_id'] for r in metadata_rows))

    logger.info(f'\nDataset summary [{config_name}]')
    logger.info(f'  Patients processed  : {patients_done}')
    logger.info(f'  Pre-ictal sequences : {len(preictal_rows)}')
    logger.info(f'  Interictal sequences: {len(interictal_rows)}')
    logger.info(f'  Total sequences     : {len(metadata_rows)}')
    logger.info(f'  Metadata saved to   : {metadata_path}')
    logger.info(f'\nDataset build complete for [{config_name}].')

    return {
        'config_name':   config_name,
        'output_dir':    output_dir,
        'metadata_path': metadata_path,
        'target_frames': target_frames,
        'n_preictal':    len(preictal_rows),
        'n_interictal':  len(interictal_rows),
    }


if __name__ == '__main__':
    # Default: original 30-minute window ending at seizure onset
    build_dataset(offset_minutes=0, duration_minutes=30)
