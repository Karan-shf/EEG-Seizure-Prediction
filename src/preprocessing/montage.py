"""
montage.py
==========
The canonical 18-channel bipolar montage (Req A) and channel-resolution logic.

Different CHB-MIT EDF files list their channels in different orders, sometimes
include extra channels (ECG, VNS, dummy '-' channels), use duplicate names (MNE
renames a second 'T8-P8' to 'T8-P8-1'), and vary in capitalization. This module
maps whatever an EDF provides onto our fixed 18-channel montage, in a fixed
order, so every covariance matrix is indexed identically across patients.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from src import config as cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

# The canonical montage is defined once, in config, and re-exported here.
CANONICAL_CHANNELS: tuple[str, ...] = tuple(cfg.CHANNELS)
N_CHANNELS: int = len(CANONICAL_CHANNELS)

# Matches an MNE-style duplicate suffix, e.g. 'T8-P8-1' -> base 'T8-P8'.
_DUP_SUFFIX = re.compile(r"^(.*?)-(\d+)$")


def normalize_name(name: str) -> str:
    """Uppercase, trim, and remove internal spaces from a channel name."""
    return name.strip().upper().replace(" ", "")


def _build_lookup(available: Sequence[str]) -> dict[str, int]:
    """Map normalized available names -> index (first occurrence wins).

    Also registers a duplicate-suffix-stripped alias (e.g. 'T8-P8-1' -> 'T8-P8')
    so canonical names still match MNE-renamed duplicates. Note that names like
    'P7-O1' do NOT match the suffix pattern (the digits are not preceded by '-').
    """
    lookup: dict[str, int] = {}
    for i, nm in enumerate(available):
        key = normalize_name(nm)
        lookup.setdefault(key, i)
        m = _DUP_SUFFIX.match(key)
        if m:
            lookup.setdefault(m.group(1), i)
    return lookup


def find_channel_indices(available: Sequence[str]) -> tuple[list[int], list[str]]:
    """Resolve the canonical montage against an EDF's channel names.

    Returns
    -------
    picks : list[int]
        Indices into `available`, in canonical order, for the channels found.
    missing : list[str]
        Canonical channel names that could not be found.
    """
    lookup = _build_lookup(available)
    picks: list[int] = []
    missing: list[str] = []
    for ch in CANONICAL_CHANNELS:
        idx = lookup.get(normalize_name(ch))
        if idx is None:
            missing.append(ch)
        else:
            picks.append(idx)
    return picks, missing


def validate_channels(available: Sequence[str]) -> list[int]:
    """Return canonical-order picks, raising if any canonical channel is missing."""
    picks, missing = find_channel_indices(available)
    if missing:
        raise ValueError(
            f"EDF is missing {len(missing)} canonical channel(s): {missing}"
        )
    return picks


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    print("Running montage.py self-test ...\n")
    assert N_CHANNELS == 18, N_CHANNELS

    # 1. Canonical channels, one lowercased, plus junk + duplicate, shuffled.
    avail = list(CANONICAL_CHANNELS)
    avail[0] = avail[0].lower()                   # case-insensitive match
    avail += ["ECG", "VNS", "T8-P8-1", "-"]        # junk + duplicate suffix
    random.Random(cfg.SEED).shuffle(avail)

    picks, missing = find_channel_indices(avail)
    assert missing == [], f"unexpected missing: {missing}"
    assert len(picks) == 18
    # picks must reproduce the canonical order exactly
    resolved = [normalize_name(avail[i]) for i in picks]
    assert resolved == [normalize_name(c) for c in CANONICAL_CHANNELS]

    # 2. Missing a channel -> reported, and validate_channels raises.
    broken = [c for c in CANONICAL_CHANNELS if c != "FZ-CZ"]
    _, missing2 = find_channel_indices(broken)
    assert missing2 == ["FZ-CZ"], missing2
    try:
        validate_channels(broken)
        raise AssertionError("expected ValueError for missing channel")
    except ValueError:
        pass

    # 3. Normalization + duplicate-suffix alias helpers.
    assert normalize_name(" fp1-f7 ") == "FP1-F7"
    assert _build_lookup(["T8-P8-1"]).get("T8-P8") == 0

    print("\nAll montage.py self-tests passed.")
