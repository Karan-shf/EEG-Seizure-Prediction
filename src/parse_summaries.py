"""
parse_summaries.py
------------------
Reads CHB-MIT chbXX-summary.txt files and extracts seizure onset/offset
times for every recording file of every patient.

Output
------
Returns a dict structured as:
{
    'chb01': {
        'chb01_03.edf': [
            {'onset': 2996, 'offset': 3036},
            ...
        ],
        'chb01_04.edf': [...],
        ...
    },
    'chb02': { ... },
    ...
}

Usage
-----
from src.parse_summaries import parse_all_summaries
seizure_index = parse_all_summaries('data/raw/chb-mit')
"""

import os
import re


def parse_patient_summary(summary_path: str) -> dict:
    """
    Parse a single chbXX-summary.txt file.

    Parameters
    ----------
    summary_path : str
        Full path to the summary .txt file.

    Returns
    -------
    dict
        Keys are .edf filenames (e.g. 'chb01_03.edf').
        Values are lists of dicts with 'onset' and 'offset' in seconds.
        Files with zero seizures map to an empty list.
    """
    result = {}
    current_file = None
    num_seizures = 0
    seizures_found = 0

    with open(summary_path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()

            # Detect which .edf file we are currently reading about
            file_match = re.match(r'File Name:\s+(\S+\.edf)', line, re.IGNORECASE)
            if file_match:
                current_file = file_match.group(1).strip()
                # Normalise to lowercase for consistent dict keys
                current_file = current_file.lower()
                result[current_file] = []
                num_seizures = 0
                seizures_found = 0
                continue

            # Number of seizures in this file
            seizure_count_match = re.match(r'Number of Seizures in File:\s+(\d+)', line, re.IGNORECASE)
            if seizure_count_match and current_file:
                num_seizures = int(seizure_count_match.group(1))
                continue

            # Seizure onset — handles both "Seizure Start Time" and
            # "Seizure N Start Time" formats found across patients
            onset_match = re.match(r'Seizure(?:\s+\d+)?\s+Start Time:\s+(\d+)\s+seconds?', line, re.IGNORECASE)
            if onset_match and current_file and num_seizures > 0:
                onset = int(onset_match.group(1))
                result[current_file].append({'onset': onset, 'offset': None})
                seizures_found += 1
                continue

            # Seizure offset — matches the most recently opened seizure entry
            offset_match = re.match(r'Seizure(?:\s+\d+)?\s+End Time:\s+(\d+)\s+seconds?', line, re.IGNORECASE)
            if offset_match and current_file and result.get(current_file):
                offset = int(offset_match.group(1))
                # Fill in the last seizure entry that is still missing an offset
                for entry in reversed(result[current_file]):
                    if entry['offset'] is None:
                        entry['offset'] = offset
                        break
                continue

    # Sanity check: drop any seizure entries still missing an offset
    for fname in result:
        result[fname] = [s for s in result[fname] if s['offset'] is not None]

    return result


def parse_all_summaries(chbmit_root: str) -> dict:
    """
    Walk the CHB-MIT root directory and parse every patient's summary file.

    Parameters
    ----------
    chbmit_root : str
        Path to the folder containing chb01/, chb02/, ... subfolders.
        e.g. 'data/raw/chb-mit'

    Returns
    -------
    dict
        Top-level keys are patient IDs (e.g. 'chb01').
        Values are the per-file seizure dicts returned by parse_patient_summary.
    """
    all_patients = {}

    # Sort so patients are processed in consistent order
    entries = sorted(os.listdir(chbmit_root))

    for entry in entries:
        patient_dir = os.path.join(chbmit_root, entry)

        # Only process directories that look like patient folders (chbXX)
        if not os.path.isdir(patient_dir):
            continue
        if not re.match(r'chb\d+', entry, re.IGNORECASE):
            continue

        patient_id = entry.lower()  # normalise to 'chb01', 'chb02', ...

        # Find the summary file — it may be named chbXX-summary.txt
        # or occasionally with slight variations
        summary_file = None
        for fname in os.listdir(patient_dir):
            if 'summary' in fname.lower() and fname.endswith('.txt'):
                summary_file = os.path.join(patient_dir, fname)
                break

        if summary_file is None:
            print(f'[WARNING] No summary file found for {patient_id} — skipping.')
            continue

        patient_data = parse_patient_summary(summary_file)
        all_patients[patient_id] = patient_data

        # Report what was found
        total_seizures = sum(len(v) for v in patient_data.values())
        files_with_seizures = sum(1 for v in patient_data.values() if len(v) > 0)
        print(f'[{patient_id}] {len(patient_data)} recording files | '
              f'{files_with_seizures} with seizures | '
              f'{total_seizures} seizure events total')

    print(f'\nDone. Parsed {len(all_patients)} patients.')
    return all_patients


def get_all_seizure_events(seizure_index: dict) -> list:
    """
    Flatten the nested seizure index into a simple list of seizure events.
    Useful for iterating over all events regardless of patient or file.

    Parameters
    ----------
    seizure_index : dict
        Output of parse_all_summaries.

    Returns
    -------
    list of dicts, each with keys:
        patient_id  : str   e.g. 'chb01'
        filename    : str   e.g. 'chb01_03.edf'
        onset       : int   seconds from start of that .edf file
        offset      : int   seconds from start of that .edf file
    """
    events = []
    for patient_id, files in seizure_index.items():
        for filename, seizures in files.items():
            for sz in seizures:
                events.append({
                    'patient_id': patient_id,
                    'filename':   filename,
                    'onset':      sz['onset'],
                    'offset':     sz['offset'],
                })
    return events


# ---------------------------------------------------------------------------
# Quick self-test — run this file directly to verify parsing on your data:
#   python src/parse_summaries.py
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/chb-mit'
    print(f'Parsing summaries in: {root}\n')

    index = parse_all_summaries(root)
    events = get_all_seizure_events(index)

    print(f'\nTotal seizure events across all patients: {len(events)}')
    print('\nFirst 5 events:')
    for e in events[:5]:
        print(f"  {e['patient_id']} | {e['filename']} | "
              f"onset={e['onset']}s | offset={e['offset']}s")
