"""
edf_loader.py
=============
Load a CHB-MIT EDF recording and reduce it to the canonical 18-channel montage
(Req A), in fixed order, at the expected sampling rate.

MNE is imported lazily inside `load_edf`, so this module (and its self-test) can
be imported without MNE installed. The channel selection/reordering logic is
factored into `select_and_order`, a pure NumPy function tested without any EDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from src import config as cfg
from src.utils.logger import get_logger
from src.preprocessing import montage

log = get_logger(__name__)


@dataclass
class EdfRecording:
    """A loaded EDF reduced to the canonical montage."""
    name: str
    data: np.ndarray            # shape (18, n_samples), canonical channel order
    sfreq: float
    ch_names: tuple[str, ...]   # canonical names, aligned to `data` rows

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration_sec(self) -> float:
        return self.n_samples / self.sfreq


def select_and_order(data: np.ndarray, ch_names: Sequence[str]) -> np.ndarray:
    """Select the canonical 18 channels from `data` and return them in order.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels_in, n_samples)
    ch_names : names for each row of `data`.

    Returns
    -------
    np.ndarray, shape (18, n_samples), rows in canonical order.

    Raises
    ------
    ValueError if the shapes disagree or any canonical channel is missing.
    """
    if data.shape[0] != len(ch_names):
        raise ValueError(
            f"data has {data.shape[0]} rows but {len(ch_names)} channel names"
        )
    picks = montage.validate_channels(ch_names)  # raises on any missing channel
    return data[picks, :]


def load_edf(
    path: str | Path,
    *,
    to_microvolts: bool = True,
    resample: bool = True,
) -> EdfRecording:
    """Load an EDF, reduce to the canonical montage, and check the sampling rate.

    Parameters
    ----------
    path : path to the .edf file.
    to_microvolts : convert MNE's SI volts to microvolts (CHB-MIT native unit).
    resample : if the file's sampling rate differs from config.FS, resample to it
        (raises instead when False).
    """
    import mne  # lazy import so the project doesn't require MNE at import time

    path = Path(path)
    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")

    data: np.ndarray = raw.get_data()   # (n_channels, n_samples), SI volts. # type: ignore
    ch_names = list(raw.ch_names)
    sfreq = float(raw.info["sfreq"])

    data = select_and_order(data, ch_names)

    if to_microvolts:
        data = data * 1e6                 # volts -> microvolts

    if sfreq != cfg.FS:
        if resample:
            log.warning("%s: sfreq %.1f != %d Hz, resampling",
                        path.name, sfreq, cfg.FS)
            data = mne.filter.resample(data, up=float(cfg.FS), down=sfreq, axis=1)
            sfreq = float(cfg.FS)
        else:
            raise ValueError(f"{path.name}: sfreq {sfreq} != expected {cfg.FS}")

    rec = EdfRecording(
        name=path.name,
        data=np.ascontiguousarray(data, dtype=np.float64),
        sfreq=sfreq,
        ch_names=montage.CANONICAL_CHANNELS,
    )
    log.info("Loaded %s: %d ch x %d samples (%.1fs @ %dHz)",
             rec.name, rec.data.shape[0], rec.n_samples,
             rec.duration_sec, int(rec.sfreq))
    return rec


# ---------------------------------------------------------------------------
# Self-test (pure NumPy -> no MNE / no EDF file needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    from src.preprocessing.montage import normalize_name, CANONICAL_CHANNELS

    print("Running edf_loader.py self-test ...\n")

    n_samples = 1000
    # Tag each row with the canonical index of its channel (junk rows -> -1),
    # so we can verify select_and_order both SELECTS and REORDERS correctly.
    canon_index = {normalize_name(c): k for k, c in enumerate(CANONICAL_CHANNELS)}

    names = list(CANONICAL_CHANNELS)
    random.Random(0).shuffle(names)
    names = names + ["ECG", "VNS"]            # extra channels to be dropped
    data = np.full((len(names), n_samples), -1.0)
    for i, nm in enumerate(names):
        data[i, :] = canon_index.get(normalize_name(nm), -1)

    out = select_and_order(data, names)
    assert out.shape == (18, n_samples), out.shape
    for k in range(18):
        assert np.all(out[k] == k), f"row {k} is misordered"

    # shape/name mismatch raises
    try:
        select_and_order(data, names[:-1])
        raise AssertionError("expected ValueError on length mismatch")
    except ValueError:
        pass

    # missing canonical channel raises
    missing_names = [c for c in CANONICAL_CHANNELS if c != "CZ-PZ"]
    d2 = np.zeros((len(missing_names), 10))
    try:
        select_and_order(d2, missing_names)
        raise AssertionError("expected ValueError on missing channel")
    except ValueError:
        pass

    # EdfRecording derived properties
    rec = EdfRecording(name="x", data=np.zeros((18, 512)),
                       sfreq=256.0, ch_names=CANONICAL_CHANNELS)
    assert rec.n_samples == 512
    assert abs(rec.duration_sec - 2.0) < 1e-9

    print("All edf_loader.py self-tests passed.")
