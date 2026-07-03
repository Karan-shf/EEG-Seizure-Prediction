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
from preprocessing import build_sequence_from_edf, PRE_ICTAL_SECS, WINDOW_SECONDS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHBMIT_ROOT   = 'data/raw/chb-mit'
OUTPUT_DIR    = 'data/processed/sequences'
METADATA_PATH = 'data/processed/metadata.csv'

# Minimum gap in seconds between any seizure and an interictal anchor point
INTERICTAL_MIN_GAP = 3600

# How many interictal sequences to extract per patient (at most)
MAX_INTERICTAL_PER_PATIENT = 6

# Target sequence length in frames
TARGET_FRAMES = PRE_ICTAL_SECS // WINDOW_SECONDS  # 360


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
) -> dict | None:
    """
    Extract a pre-ictal sequence for one seizure event and save it.

    Returns a metadata dict row on success, or None if the event is skipped.
    """
    save_name = f'{patient_id}_sz{event_idx:02d}_preictal.npy'
    save_path = os.path.join(output_dir, save_name)

    if os.path.exists(save_path):
        print(f'  [SKIP — already exists] {save_name}')
        # Still return metadata so it appears in the CSV
        seq = np.load(save_path)
        return _make_metadata_row(
            save_name, patient_id, filename, 'preictal', 1,
            onset, seq.shape, note='cached'
        )

    try:
        sequence, sfreq, n_ch = build_sequence_from_edf(
            edf_path,
            end_time=float(onset),
            duration=float(PRE_ICTAL_SECS),
        )
        np.save(save_path, sequence)
        print(f'  [SAVED] {save_name}  shape={sequence.shape}  sfreq={sfreq}')

        # Count how many leading frames are zero-padded
        non_zero_frames = int(np.any(sequence != 0, axis=(1, 2)).sum())
        padded_frames   = TARGET_FRAMES - non_zero_frames
        note = f'padded_frames={padded_frames}' if padded_frames > 0 else 'full'

        return _make_metadata_row(
            save_name, patient_id, filename, 'preictal', 1,
            onset, sequence.shape, note=note
        )

    except ValueError as e:
        print(f'  [SKIP] {patient_id} {filename} onset={onset}s — {e}')
        return None

    except Exception as e:
        print(f'  [ERROR] {patient_id} {filename} onset={onset}s — {e}')
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
        print(f'  [ERROR reading {filename} for interictal] {e}')
        return rows

    anchors = find_interictal_anchors(raw_duration, seizure_onsets, n_anchors=1)

    if not anchors:
        print(f'  [NO INTERICTAL] {patient_id} {filename} — no safe window found')
        return rows

    for anchor in anchors:
        inter_event_counter[0] += 1
        idx       = inter_event_counter[0]
        save_name = f'{patient_id}_inter{idx:02d}_interictal.npy'
        save_path = os.path.join(output_dir, save_name)

        if os.path.exists(save_path):
            print(f'  [SKIP — already exists] {save_name}')
            seq = np.load(save_path)
            rows.append(_make_metadata_row(
                save_name, patient_id, filename, 'interictal', 0,
                anchor, seq.shape, note='cached'
            ))
            continue

        try:
            sequence, sfreq, n_ch = build_sequence_from_edf(
                edf_path,
                end_time=float(anchor),
                duration=float(PRE_ICTAL_SECS),
            )
            np.save(save_path, sequence)
            print(f'  [SAVED] {save_name}  shape={sequence.shape}')

            rows.append(_make_metadata_row(
                save_name, patient_id, filename, 'interictal', 0,
                anchor, sequence.shape, note='full'
            ))

        except Exception as e:
            print(f'  [ERROR] interictal {patient_id} {filename} anchor={anchor}s — {e}')

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

def build_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print('=' * 60)
    print('SeizureHorizon — Dataset Builder')
    print('=' * 60)
    print(f'CHB-MIT root  : {CHBMIT_ROOT}')
    print(f'Output dir    : {OUTPUT_DIR}')
    print(f'Metadata file : {METADATA_PATH}')
    print(f'Window size   : {WINDOW_SECONDS}s')
    print(f'Pre-ictal len : {PRE_ICTAL_SECS}s ({TARGET_FRAMES} frames)')
    print()

    # Step 1 — Parse all summary files
    print('Step 1: Parsing summary files...')
    seizure_index = parse_all_summaries(CHBMIT_ROOT)
    all_events    = get_all_seizure_events(seizure_index)
    print(f'\nTotal seizure events found: {len(all_events)}\n')

    metadata_rows = []

    # Step 2 — Process each patient
    for patient_id, patient_data in sorted(seizure_index.items()):
        print(f'\n{"─" * 50}')
        print(f'Patient: {patient_id}')
        print(f'{"─" * 50}')

        patient_dir    = os.path.join(CHBMIT_ROOT, patient_id)
        seizure_times  = get_patient_seizure_times(patient_data)
        preictal_count = 0
        inter_counter  = [0]   # mutable list so subfunction can increment

        # Group seizure events by file and sort by onset to detect close pairs
        for filename, seizures in sorted(patient_data.items()):
            if not seizures:
                continue

            edf_path = os.path.join(patient_dir, filename)
            if not os.path.exists(edf_path):
                print(f'  [MISSING EDF] {edf_path}')
                continue

            print(f'\n  File: {filename}')

            # --- Pre-ictal sequences ---
            prev_offset = None   # track previous seizure's end for close-pair detection

            for sz in sorted(seizures, key=lambda x: x['onset']):
                onset  = sz['onset']
                offset = sz['offset']

                # Skip if this seizure starts too soon after the previous one ended
                if prev_offset is not None and onset - prev_offset < PRE_ICTAL_SECS:
                    print(f'  [SKIP — too close to previous seizure] '
                          f'onset={onset}s, prev_offset={prev_offset}s')
                    prev_offset = offset
                    continue

                preictal_count += 1
                row = process_preictal_event(
                    patient_id, filename, onset, preictal_count, edf_path, OUTPUT_DIR
                )
                if row:
                    metadata_rows.append(row)

                prev_offset = offset

            # # --- Interictal sequences ---
            # all_onsets_in_file = seizure_times.get(filename, [])
            # inter_rows = process_interictal_events(
            #     patient_id, filename, edf_path,
            #     all_onsets_in_file, inter_counter, OUTPUT_DIR
            # )
            # metadata_rows.extend(inter_rows)

        # interictal sequences (ALL files, including seizure-free ones)
        patient_inter_count = 0
        for filename, seizures in sorted(patient_data.items()):
            if patient_inter_count >= MAX_INTERICTAL_PER_PATIENT:   
                break
            edf_path = os.path.join(patient_dir, filename)
            if not os.path.exists(edf_path):
                continue

            all_onsets_in_file = seizure_times.get(filename, [])
            inter_rows = process_interictal_events(
                patient_id, filename, edf_path,
                all_onsets_in_file, inter_counter, OUTPUT_DIR
            )
            metadata_rows.extend(inter_rows)
            patient_inter_count += len(inter_rows)

    # Step 3 — Write metadata CSV
    print(f'\n{"=" * 60}')
    print(f'Writing metadata to {METADATA_PATH} ...')

    fieldnames = [
        'filename', 'patient_id', 'source_edf', 'type', 'label',
        'anchor_sec', 'n_frames', 'n_bands', 'n_channels', 'note'
    ]

    with open(METADATA_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Summary statistics
    preictal_rows   = [r for r in metadata_rows if r['label'] == 1]
    interictal_rows = [r for r in metadata_rows if r['label'] == 0]
    patients_done   = len(set(r['patient_id'] for r in metadata_rows))

    print(f'\nDataset summary')
    print(f'  Patients processed  : {patients_done}')
    print(f'  Pre-ictal sequences : {len(preictal_rows)}')
    print(f'  Interictal sequences: {len(interictal_rows)}')
    print(f'  Total sequences     : {len(metadata_rows)}')
    print(f'  Metadata saved to   : {METADATA_PATH}')
    print('\nDataset build complete.')


if __name__ == '__main__':
    build_dataset()
