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

from parse_summaries import (
    parse_all_summaries, get_all_seizure_events, parse_all_file_times,
)
from preprocessing import (
    build_sequence_from_edf, build_sequence_multifile, build_patient_timeline,
    PRE_ICTAL_SECS, WINDOW_SECONDS, STRIDE_SECONDS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# AFTER
CHBMIT_ROOT = 'data/raw/chb-mit'

INTERICTAL_MIN_GAP = 3600
MAX_INTERICTAL_PER_PATIENT = 10

# ---------------------------------------------------------------------------
# Fix F2 - padding guard
# ---------------------------------------------------------------------------
# Windows whose leading frames are mostly zero-padding carry little real EEG
# and dominated some configs (e.g. 19 windows with ~1/3 padding). Any window
# more than this fraction padded is dropped; every saved window records its
# measured pad_fraction in metadata.
PAD_FRACTION_MAX = 0.30

# ---------------------------------------------------------------------------
# Fix F4 - preictal multi-window augmentation
# ---------------------------------------------------------------------------
# Instead of one window per seizure, tile up to PREICTAL_AUG_MAX windows across
# each seizure's lead-in, each shifted earlier by (1 - overlap) x duration.
# This multiplies the scarce positive class ~5x, reducing per-patient noise.
PREICTAL_AUG_MAX     = 5
PREICTAL_AUG_OVERLAP = 0.5    # 50% overlap -> stride = 0.5 x duration

# ---------------------------------------------------------------------------
# Fix F5 - proportional, diversified interictal sampling
# ---------------------------------------------------------------------------
# Negatives per patient scale with that patient's positive count (after F4)
# instead of a flat 10, and anchors are spread across the whole timeline.
INTERICTAL_NEG_POS_RATIO   = 2.0
INTERICTAL_MIN_PER_PATIENT = 4
INTERICTAL_MAX_PER_PATIENT = 40

# Interictal windows are scanned across the patient timeline (when clock
# stamps exist) or within each file (fallback), spaced by this stride and
# clamped to at least the window duration so negatives never overlap.
# Interictal windows use offset=0: the pre-ictal offset is meaningless when
# there is no seizure to lead up to.
INTERICTAL_STRIDE_FLOOR = 300  # seconds

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
    scan_lo: float,
    scan_hi: float,
    seizure_times: list,
    window_needed: float,
    min_gap: int = INTERICTAL_MIN_GAP,
    n_anchors: int = MAX_INTERICTAL_PER_PATIENT,
    stride: float | None = None,
) -> list:
    """
    Find up to n_anchors anchor points (each the END of an interictal window)
    inside [scan_lo, scan_hi] such that the whole window
    [anchor - window_needed, anchor] stays at least ``min_gap`` seconds clear of
    every seizure. Axis-agnostic: pass local file seconds with local onsets, or
    absolute patient-timeline seconds with absolute onsets.

    Parameters
    ----------
    scan_lo, scan_hi : float - inclusive range of the axis to scan (seconds)
    seizure_times    : list  - seizure onset times on the SAME axis
    window_needed    : float - seconds of data required before each anchor
                               (== duration_sec; interictal has no offset)
    min_gap          : int   - forbidden radius around each seizure (seconds)
    n_anchors        : int   - maximum number of anchors to return
    stride           : float - spacing between candidate anchors; defaults to
                               window_needed so windows never overlap

    Returns
    -------
    list of float anchor times (end of interictal window = anchor)
    """
    if stride is None:
        stride = window_needed
    stride = max(float(stride), float(INTERICTAL_STRIDE_FLOOR))

    anchors = []
    t = scan_lo + window_needed          # need `window_needed` of lead-in
    while t <= scan_hi and len(anchors) < n_anchors:
        win_lo = t - window_needed
        # Reject if window [win_lo, t] overlaps the forbidden zone
        # [onset - min_gap, onset + min_gap] of ANY seizure.
        clash = any(
            (win_lo < onset + min_gap) and (t > onset - min_gap)
            for onset in seizure_times
        )
        if not clash:
            anchors.append(t)
        t += stride

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
    logger,
    timeline: dict | None = None,
    patient_dir: str | None = None,
    abs_onset: float | None = None,
    earliest_allowed_abs: float | None = None,
) -> list:
    """
    Extract pre-ictal sequences for one seizure event and save them.

    Fix F4 tiles up to PREICTAL_AUG_MAX overlapping windows across the lead-in;
    Fix F2 drops any window that is more than PAD_FRACTION_MAX zero-padding.
    Returns a list of metadata rows (possibly empty).
    """
    rows = []
    aug_stride_sec = max(1.0, duration_sec * (1.0 - PREICTAL_AUG_OVERLAP))

    for k in range(PREICTAL_AUG_MAX):
        # Fix F4: k=0 is the original window ending (onset - offset); larger k
        # tiles further back into the lead-in, implemented as a larger offset.
        this_offset = float(offset_sec) + k * aug_stride_sec
        suffix    = '' if k == 0 else f'_aug{k}'
        save_name = f'{patient_id}_sz{event_idx:02d}{suffix}_preictal.npy'
        save_path = os.path.join(output_dir, save_name)

        if os.path.exists(save_path):
            logger.info(f'  [SKIP — already exists] {save_name}')
            seq = np.load(save_path)
            nz  = int(np.any(seq != 0, axis=(1, 2)).sum())
            pf  = 1.0 - (nz / seq.shape[0]) if seq.shape[0] else 1.0
            if pf > PAD_FRACTION_MAX:
                continue
            rows.append(_make_metadata_row(
                save_name, patient_id, filename, 'preictal', 1,
                onset, seq.shape, note='cached', pad_fraction=pf))
            continue

        try:
            if timeline and abs_onset is not None and patient_dir is not None and filename in timeline:
                # Stitch across contiguous files so the window can reach back
                # before this file's own t=0 (unlocks large offsets).
                sequence, sfreq, n_ch = build_sequence_multifile(
                    timeline, patient_dir,
                    abs_anchor=float(abs_onset),
                    duration=float(duration_sec),
                    offset=this_offset,
                    target_n_windows=target_frames,
                    earliest_allowed_abs=earliest_allowed_abs,
                    logger=logger,
                )
            else:
                # Fallback: single-file extraction (no clock stamps, e.g. chb24).
                sequence, sfreq, n_ch = build_sequence_from_edf(
                    edf_path,
                    anchor_time=float(onset),
                    duration=float(duration_sec),
                    offset=this_offset,
                    target_n_windows=target_frames,
                )
        except ValueError as e:
            # Ran out of usable EEG this far back — stop tiling earlier.
            logger.info(f'  [SKIP] {patient_id} {filename} onset={onset}s '
                        f'offset={this_offset:.0f}s — {e}')
            break
        except Exception as e:
            logger.info(f'  [ERROR] {patient_id} {filename} onset={onset}s — {e}')
            break

        # Fix F2: drop windows that are mostly zero-padding. Deeper windows only
        # pad more, so once one is too padded we stop tiling earlier.
        non_zero_frames = int(np.any(sequence != 0, axis=(1, 2)).sum())
        pad_fraction    = 1.0 - (non_zero_frames / target_frames) if target_frames else 1.0
        if pad_fraction > PAD_FRACTION_MAX:
            logger.info(f'  [DROP — pad {pad_fraction:.0%}] {save_name} '
                        f'({non_zero_frames}/{target_frames} real frames)')
            break

        np.save(save_path, sequence)
        note = 'full' if pad_fraction == 0 else f'padded_frames={target_frames - non_zero_frames}'
        logger.info(f'  [SAVED] {save_name}  shape={sequence.shape}  '
                    f'sfreq={sfreq}  pad={pad_fraction:.0%}')
        rows.append(_make_metadata_row(
            save_name, patient_id, filename, 'preictal', 1,
            onset, sequence.shape, note=note, pad_fraction=pad_fraction))

    return rows


# ---------------------------------------------------------------------------
# Core: process interictal windows for one patient file
# ---------------------------------------------------------------------------

def process_interictal_events(
    patient_id: str,
    patient_data: dict,
    seizure_times: dict,
    patient_dir: str,
    inter_event_counter: list,   # mutable [count] shared across the patient
    output_dir: str,
    offset_sec: float,           # accepted for symmetry; NOT used (see below)
    duration_sec: float,
    target_frames: int,
    logger,
    timeline: dict | None = None,
    target_interictal: int = MAX_INTERICTAL_PER_PATIENT,
) -> list:
    """
    Collect interictal sequences for ONE patient.

    Interictal windows are ``duration_sec`` seconds of quiet EEG ending at an
    anchor, extracted with offset=0 (the pre-ictal offset is meaningless
    without a seizure to lead up to). Anchors are chosen so the whole window
    stays >= INTERICTAL_MIN_GAP from every seizure.

    When a patient timeline is available the scan runs on the absolute axis and
    windows stitch across contiguous files (this is what lets long durations
    such as 45/60 min find windows at all). Otherwise it falls back to scanning
    each file independently on its local axis.

    Returns a list of metadata row dicts (one per saved sequence).
    """
    rows = []
    cap    = int(target_interictal)      # Fix F5: per-patient negative target
    budget = cap - inter_event_counter[0]
    if budget <= 0:
        return rows

    def _emit(source_edf, anchor, builder):
        """Increment the counter, honour the on-disk cache, build+save."""
        inter_event_counter[0] += 1
        idx       = inter_event_counter[0]
        save_name = f'{patient_id}_inter{idx:02d}_interictal.npy'
        save_path = os.path.join(output_dir, save_name)

        if os.path.exists(save_path):
            logger.info(f'  [SKIP - already exists] {save_name}')
            seq = np.load(save_path)
            nz  = int(np.any(seq != 0, axis=(1, 2)).sum())
            pf  = 1.0 - (nz / seq.shape[0]) if seq.shape[0] else 1.0
            if pf > PAD_FRACTION_MAX:            # Fix F2
                inter_event_counter[0] -= 1
                return None
            return _make_metadata_row(
                save_name, patient_id, source_edf, 'interictal', 0,
                anchor, seq.shape, note='cached', pad_fraction=pf
            )
        try:
            sequence = builder()
            non_zero = int(np.any(sequence != 0, axis=(1, 2)).sum())
            pad_fraction = 1.0 - (non_zero / target_frames) if target_frames else 1.0
            if pad_fraction > PAD_FRACTION_MAX:            # Fix F2
                inter_event_counter[0] -= 1   # this slot produced no window
                logger.info(f'  [DROP — pad {pad_fraction:.0%}] {save_name}')
                return None
            np.save(save_path, sequence)
            padded   = target_frames - non_zero
            note     = f'padded_frames={padded}' if padded > 0 else 'full'
            logger.info(f'  [SAVED] {save_name}  shape={sequence.shape}  ({note})')
            return _make_metadata_row(
                save_name, patient_id, source_edf, 'interictal', 0,
                anchor, sequence.shape, note=note, pad_fraction=pad_fraction
            )
        except Exception as e:
            logger.info(f'  [ERROR] interictal {patient_id} {source_edf} '
                        f'anchor={anchor:.0f}s - {e}')
            return None

    # -- Timeline (stitched, absolute-axis) path ---------------------------
    if timeline:
        abs_seizures = []
        for fname, onsets in seizure_times.items():
            if fname in timeline:
                base = timeline[fname]['abs_start']
                abs_seizures.extend(base + o for o in onsets)

        scan_lo = min(t['abs_start'] for t in timeline.values())
        scan_hi = max(t['abs_end']   for t in timeline.values())

        # Fix F5: spread anchors across the whole timeline instead of packing
        # them at the start — stride grows with the available span and budget.
        span = max(0.0, scan_hi - scan_lo)
        diversify_stride = max(duration_sec, span / (budget + 1)) if budget > 0 else duration_sec
        anchors = find_interictal_anchors(
            scan_lo, scan_hi, abs_seizures,
            window_needed=duration_sec,
            n_anchors=budget,
            stride=diversify_stride,
        )
        if not anchors:
            logger.info(f'  [NO INTERICTAL] {patient_id} - no safe window on '
                        f'timeline (dur={duration_sec:.0f}s, '
                        f'{len(abs_seizures)} seizures)')
            return rows

        for anchor in anchors:
            def _build_stitched(a=anchor):
                seq, _sfreq, _nch = build_sequence_multifile(
                    timeline, patient_dir,
                    abs_anchor=float(a),
                    duration=float(duration_sec),
                    offset=0.0,
                    target_n_windows=target_frames,
                    logger=logger,
                )
                return seq
            row = _emit('(stitched)', anchor, _build_stitched)
            if row:
                rows.append(row)
        return rows

    # -- Single-file fallback (no clock stamps, e.g. chb24) ----------------
    for filename, _seizures in sorted(patient_data.items()):
        if inter_event_counter[0] >= cap:
            break
        edf_path = os.path.join(patient_dir, filename)
        if not os.path.exists(edf_path):
            continue
        try:
            import mne
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            raw_duration = float(raw.times[-1])
        except Exception as e:
            logger.info(f'  [ERROR reading {filename} for interictal] {e}')
            continue

        remaining = cap - inter_event_counter[0]
        anchors = find_interictal_anchors(
            0.0, raw_duration, seizure_times.get(filename, []),
            window_needed=duration_sec,
            n_anchors=remaining,
            stride=duration_sec,
        )
        if not anchors:
            logger.info(f'  [NO INTERICTAL] {patient_id} {filename} - '
                        f'no safe window found')
            continue

        for anchor in anchors:
            def _build_single(a=anchor, p=edf_path):
                seq, _sfreq, _nch = build_sequence_from_edf(
                    p,
                    anchor_time=float(a),
                    duration=float(duration_sec),
                    offset=0.0,
                    target_n_windows=target_frames,
                )
                return seq
            row = _emit(filename, anchor, _build_single)
            if row:
                rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Helper: build a metadata CSV row dict
# ---------------------------------------------------------------------------

def _make_metadata_row(
    filename, patient_id, source_edf, split_type, label,
    anchor_sec, shape, note='', pad_fraction=0.0
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
        'pad_fraction': round(float(pad_fraction), 4),   # Fix F2
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

    logger.info('Step 1b: Parsing per-file clock times for cross-file stitching...')
    file_times_all = parse_all_file_times(CHBMIT_ROOT, logger)
    logger.info(f'Patients with usable file-time stamps: {len(file_times_all)}\n')

    metadata_rows = []

    for patient_id, patient_data in sorted(seizure_index.items()):
        logger.info(f'\n{"─" * 50}')
        logger.info(f'Patient: {patient_id}')
        logger.info(f'{"─" * 50}')

        patient_dir    = os.path.join(CHBMIT_ROOT, patient_id)
        seizure_times  = get_patient_seizure_times(patient_data)
        preictal_count = 0
        patient_preictal_built = 0     # Fix F5: negatives scale with this
        inter_counter  = [0]

        # Absolute-seconds timeline for this patient (empty if no clock stamps,
        # in which case preictal extraction falls back to single-file mode).
        ordered_files = sorted(patient_data.keys())
        patient_ft    = file_times_all.get(patient_id, {})
        timeline      = build_patient_timeline(patient_ft, ordered_files) if patient_ft else {}
        if timeline:
            logger.info(f'  Timeline built for {len(timeline)} files')

        # --- Pre-ictal sequences (absolute-timeline aware) ---
        # Flatten every seizure for this patient; attach absolute onset/offset
        # when the file is on the timeline so windows can stitch across files
        # and be ordered globally.
        flat = []
        for filename, seizures in patient_data.items():
            for sz in seizures:
                if filename in timeline:
                    base = timeline[filename]['abs_start']
                    abs_onset = base + sz['onset']
                    abs_off   = base + (sz['offset'] if sz['offset'] else sz['onset'])
                else:
                    abs_onset = abs_off = None
                flat.append({'filename': filename, 'onset': sz['onset'],
                            'offset': sz['offset'],
                            'abs_onset': abs_onset, 'abs_off': abs_off})

        flat.sort(key=lambda e: (
            e['abs_onset'] if e['abs_onset'] is not None else float('inf'),
            e['filename'], e['onset'],
        ))

        min_gap_needed   = offset_sec + duration_sec
        prev_abs_off     = None      # absolute end of previous seizure
        prev_local_off   = None      # local end (single-file fallback)
        prev_file        = None

        for ev in flat:
            filename = ev['filename']
            onset    = ev['onset']
            edf_path = os.path.join(patient_dir, filename)
            if not os.path.exists(edf_path):
                logger.info(f'  [MISSING EDF] {edf_path}')
                continue

            earliest_allowed_abs = None
            if ev['abs_onset'] is not None:
                if prev_abs_off is not None and (ev['abs_onset'] - prev_abs_off) < min_gap_needed:
                    logger.info(f'  [SKIP — too close to previous seizure] '
                                f'{filename} abs_onset={ev["abs_onset"]:.0f}s '
                                f'prev_off={prev_abs_off:.0f}s')
                    prev_abs_off = ev['abs_off']; prev_file = filename
                    continue
                earliest_allowed_abs = prev_abs_off   # never reach into prior seizure
            else:
                # single-file fallback: only compare within the same file
                if prev_file == filename and prev_local_off is not None \
                        and (onset - prev_local_off) < min_gap_needed:
                    logger.info(f'  [SKIP — too close to previous seizure] '
                                f'{filename} onset={onset}s')
                    prev_local_off = ev['offset']; prev_file = filename
                    continue

            preictal_count += 1
            new_rows = process_preictal_event(
                patient_id, filename, onset, preictal_count, edf_path, output_dir,
                offset_sec=offset_sec,
                duration_sec=duration_sec,
                target_frames=target_frames,
                logger=logger,
                timeline=timeline,
                patient_dir=patient_dir,
                abs_onset=ev['abs_onset'],
                earliest_allowed_abs=earliest_allowed_abs,
            )
            metadata_rows.extend(new_rows)
            patient_preictal_built += len(new_rows)

            prev_abs_off   = ev['abs_off']
            prev_local_off = ev['offset']
            prev_file      = filename

        # --- Interictal sequences (patient-level; stitched when possible) ---
        # Fix F5: target negatives proportional to this patient's positives.
        target_interictal = int(min(
            INTERICTAL_MAX_PER_PATIENT,
            max(INTERICTAL_MIN_PER_PATIENT,
                round(patient_preictal_built * INTERICTAL_NEG_POS_RATIO))))
        inter_rows = process_interictal_events(
            patient_id, patient_data, seizure_times, patient_dir,
            inter_counter, output_dir,
            offset_sec=offset_sec,
            duration_sec=duration_sec,
            target_frames=target_frames,
            logger=logger,
            timeline=timeline,
            target_interictal=target_interictal,
        )
        metadata_rows.extend(inter_rows)

    # --- Write metadata CSV ---
    logger.info(f'\n{"=" * 60}')
    logger.info(f'Writing metadata to {metadata_path} ...')

    fieldnames = [
        'filename', 'patient_id', 'source_edf', 'type', 'label',
        'anchor_sec', 'n_frames', 'n_bands', 'n_channels', 'pad_fraction', 'note'
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
