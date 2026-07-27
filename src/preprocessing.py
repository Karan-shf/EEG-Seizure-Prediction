"""
preprocessing.py
----------------
Reusable preprocessing functions for the SeizureHorizon pipeline.

Responsibilities
----------------
1. Load and filter a raw .edf file
2. Extract a time segment (pre-ictal or interictal) from a loaded recording
3. Slide a 5-second window across a segment and compute per-window band power
4. Return a (n_windows, 5, n_channels) numpy array ready for build_dataset.py

Nothing in this file saves anything to disk — that is build_dataset.py's job.

Usage
-----
from src.preprocessing import load_and_filter, extract_segment, compute_sequence
"""

import numpy as np
import mne
from scipy.signal import welch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Frequency bands: (name, low_hz, high_hz)
FREQUENCY_BANDS = [
    ('delta', 0.5, 4.0),
    ('theta', 4.0, 8.0),
    ('alpha', 8.0, 13.0),
    ('beta',  13.0, 30.0),
    ('gamma', 30.0, 40.0),
]

WINDOW_SECONDS   = 5       # length of each analysis window in seconds
STRIDE_SECONDS  = 3        # 40% overlap: 5s window, 2s overlap, 3s step
PRE_ICTAL_SECS   = 1800    # 30 minutes in seconds (default duration)
DEFAULT_OFFSET_SECONDS = 0  # default: window ends exactly at seizure onset
NOTCH_FREQ       = 60.0    # US powerline frequency (change to 50.0 for Europe)
BANDPASS_LOW     = 0.5
BANDPASS_HIGH    = 40.0

# Minimum number of seconds of EEG required before seizure onset
# to include an event (we accept anything > 0 due to zero-padding strategy)
MIN_AVAILABLE_SECS = 10


# ---------------------------------------------------------------------------
# Step 1 — Load and filter
# ---------------------------------------------------------------------------

def load_and_filter(edf_path: str) -> mne.io.BaseRaw:
    """
    Load a CHB-MIT .edf file, apply bandpass and notch filters,
    and return the filtered Raw object.

    Parameters
    ----------
    edf_path : str
        Full path to the .edf file.

    Returns
    -------
    mne.io.Raw
        Filtered, preloaded Raw object. All 23 channels are retained.
        Channel types are set to 'eeg'.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Set all channels to EEG type so MNE handles them correctly
    raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})

    # Bandpass: remove DC drift and high-frequency noise
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, fir_design='firwin', verbose=False)

    # Notch: remove powerline interference
    raw.notch_filter(NOTCH_FREQ, verbose=False)

    return raw


# ---------------------------------------------------------------------------
# Step 2 — Extract a time segment
# ---------------------------------------------------------------------------

def extract_segment(
    raw: mne.io.BaseRaw,
    anchor_time: float,
    duration: float,
    offset: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Extract a segment of EEG ending `offset` seconds before `anchor_time`,
    spanning `duration` seconds. If the recording starts after the computed
    segment_start, the segment starts from t=0 instead (shorter than
    requested — caller handles zero-padding).

    The window is defined as:
        segment_end   = anchor_time - offset
        segment_start = segment_end - duration

    Example: anchor_time=seizure onset, offset=900 (15 min), duration=1800 (30 min)
    → window spans from 45 min before onset to 15 min before onset.

    Parameters
    ----------
    raw         : mne.io.Raw — filtered recording
    anchor_time : float      — reference point in seconds (e.g. seizure onset)
    duration    : float      — length of the window in seconds
    offset      : float      — gap in seconds between the window's end and
                                anchor_time. offset=0 means the window ends
                                exactly at anchor_time (original behaviour).

    Returns
    -------
    np.ndarray, shape (n_channels, n_samples)
        Raw voltage values for the extracted segment.
    float
        Actual duration in seconds of the returned segment.
        May be less than `duration` if the recording started late.
    """
    recording_start = 0.0
    segment_end      = anchor_time - offset
    segment_start    = max(recording_start, segment_end - duration)

    if segment_end <= recording_start:
        raise ValueError(
            f'Segment end ({segment_end:.1f}s) is at or before recording '
            f'start. anchor_time={anchor_time}s, offset={offset}s. '
            f'This configuration has no valid EEG to extract.'
        )

    actual_duration = segment_end - segment_start

    if actual_duration < MIN_AVAILABLE_SECS:
        raise ValueError(
            f'Segment too short: only {actual_duration:.1f}s available '
            f'(minimum required: {MIN_AVAILABLE_SECS}s). '
            f'Recording ends at {raw.times[-1]:.1f}s, '
            f'anchor_time={anchor_time}s, offset={offset}s.'
        )

    raw_segment = raw.copy().crop(tmin=segment_start, tmax=segment_end)
    data = np.asarray(raw_segment.get_data())  # shape: (n_channels, n_samples)

    return data, actual_duration

# ---------------------------------------------------------------------------
# Step 3 — Band power for one window
# ---------------------------------------------------------------------------

def _band_power_single_window(window_data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Compute average power in each frequency band for one EEG window.

    Parameters
    ----------
    window_data : np.ndarray, shape (n_channels, n_samples)
    sfreq       : float — sampling frequency in Hz

    Returns
    -------
    np.ndarray, shape (5, n_channels)
        Row 0 = delta, 1 = theta, 2 = alpha, 3 = beta, 4 = gamma.
    """
    n_channels = window_data.shape[0]
    n_bands    = len(FREQUENCY_BANDS)
    result     = np.zeros((n_bands, n_channels), dtype=np.float32)

    # Welch's method: nperseg = 2 seconds of data for frequency resolution
    nperseg = int(sfreq * 2)

    freqs, psd = welch(window_data, fs=sfreq, nperseg=nperseg)
    # psd shape: (n_channels, n_freqs)

    for band_idx, (_, low, high) in enumerate(FREQUENCY_BANDS):
        freq_mask = (freqs >= low) & (freqs <= high)
        result[band_idx, :] = np.mean(psd[:, freq_mask], axis=1)

    return result


# ---------------------------------------------------------------------------
# Step 4 — Slide windows and build sequence
# ---------------------------------------------------------------------------

def compute_sequence(
    segment_data: np.ndarray,
    actual_duration: float,
    sfreq: float,
    target_n_windows: int | None = None,
) -> np.ndarray:
    """
    Slide a window across segment_data with STRIDE_SECONDS step and compute
    band power for each window. Zero-pad at the beginning if the segment is
    shorter than the target number of windows.

    Parameters
    ----------
    segment_data     : np.ndarray, shape (n_channels, n_samples)
    actual_duration  : float  — real duration of segment_data in seconds
    sfreq            : float  — sampling frequency in Hz
    target_n_windows : int    — desired sequence length
                                default: computed from PRE_ICTAL_SECS and STRIDE_SECONDS

    Returns
    -------
    np.ndarray, shape (target_n_windows, 5, n_channels)
        Time-ordered sequence of band power maps.
        Early frames are zeros if segment was shorter than target.
    """
    samples_per_win    = int(WINDOW_SECONDS * sfreq)
    samples_per_stride = int(STRIDE_SECONDS * sfreq)

    if target_n_windows is None:
        # How many overlapping windows fit in a full PRE_ICTAL_SECS segment
        target_n_windows = (
            (int(PRE_ICTAL_SECS * sfreq) - samples_per_win) // samples_per_stride
        ) + 1

    n_channels = segment_data.shape[0]
    n_bands    = len(FREQUENCY_BANDS)

    # How many windows fit in the actual available segment
    n_samples_available = int(actual_duration * sfreq)
    if n_samples_available < samples_per_win:
        return np.zeros((target_n_windows, n_bands, n_channels), dtype=np.float32)

    n_available_windows = (
        (n_samples_available - samples_per_win) // samples_per_stride
    ) + 1

    # Cap at target in case segment is longer than PRE_ICTAL_SECS
    n_windows_to_compute = min(n_available_windows, target_n_windows)

    # Initialise output with zeros (zero-padding handled here)
    sequence = np.zeros((target_n_windows, n_bands, n_channels), dtype=np.float32)

    # Computed windows sit at the END of the sequence
    # (most recent frames = just before seizure onset)
    start_frame = target_n_windows - n_windows_to_compute

    for i in range(n_windows_to_compute):
        sample_start = i * samples_per_stride
        sample_end   = sample_start + samples_per_win

        if sample_end > segment_data.shape[1]:
            break

        window_data = segment_data[:, sample_start:sample_end]
        band_map    = _band_power_single_window(window_data, sfreq)

        sequence[start_frame + i] = band_map

    return sequence


# ---------------------------------------------------------------------------
# Convenience wrapper: full pipeline for one EEG file + one time point
# ---------------------------------------------------------------------------

def build_sequence_from_edf(
    edf_path: str,
    anchor_time: float,
    duration: float = PRE_ICTAL_SECS,
    offset: float = 0.0,
    target_n_windows: int | None = None,
) -> tuple:
    """
    Full pipeline: load EDF → filter → extract segment → compute sequence.

    Parameters
    ----------
    edf_path         : str   — path to .edf file
    anchor_time       : float — reference point in seconds (seizure onset
                                 or interictal anchor point)
    duration         : float — length of the window in seconds
                                (default 1800 = 30 min)
    offset           : float — gap in seconds between window end and
                                anchor_time (default 0 = window ends
                                exactly at anchor_time, original behaviour)
    target_n_windows : int   — desired sequence length in frames.
                                If None, computed automatically from
                                duration, WINDOW_SECONDS, and STRIDE_SECONDS.

    Returns
    -------
    sequence  : np.ndarray, shape (target_n_windows, 5, n_channels)
    sfreq     : float — sampling frequency (needed by caller for reference)
    n_channels: int   — number of EEG channels in this recording
    """
    raw = load_and_filter(edf_path)
    sfreq = raw.info['sfreq']

    segment_data, actual_duration = extract_segment(
        raw, anchor_time, duration, offset=offset
    )

    # Compute target_n_windows from THIS call's duration if not provided,
    # rather than relying on the global PRE_ICTAL_SECS default
    if target_n_windows is None:
        samples_per_win    = int(WINDOW_SECONDS * sfreq)
        samples_per_stride = int(STRIDE_SECONDS * sfreq)
        target_n_windows = (
            (int(duration * sfreq) - samples_per_win) // samples_per_stride
        ) + 1

    sequence = compute_sequence(
        segment_data, actual_duration, sfreq,
        target_n_windows=target_n_windows
    )

    return sequence, sfreq, segment_data.shape[0]


if __name__ == "__main__":

    for duration_min in [15, 30, 45, 60]:
        duration_sec = duration_min * 60
        samples_per_win    = int(WINDOW_SECONDS * 256)   # sfreq=256 typical CHB-MIT
        samples_per_stride = int(STRIDE_SECONDS * 256)
        n_frames = ((int(duration_sec * 256) - samples_per_win) // samples_per_stride) + 1
        print(f'{duration_min:>3} min duration → {n_frames} frames')
