"""
summary_parser.py
=================
Parse CHB-MIT ``chbXX-summary.txt`` annotation files.

Each patient folder ships one summary text file that lists, for every EDF
recording: its file name, wall-clock start/end time, and any seizures (with
onset/offset in seconds *relative to the start of that file*). This module turns
that semi-structured text into clean Python objects that the labeler and the
data-inventory steps consume.

This file does NOT read EDF signal data -- it only parses the text annotations.
EDF loading lives in ``io/edf_loader.py``.

Handled format quirks
---------------------
* Single-seizure files use ``Seizure Start Time:`` while multi-seizure files use
  ``Seizure 1 Start Time:``, ``Seizure 2 Start Time:``, ... -- both are parsed.
* The trailing ``seconds`` unit is optional.
* Some patients (e.g. chb24) omit ``File Start Time`` / ``File End Time`` -- those
  become ``None`` and clock-based duration is unavailable.
* Wall clocks may wrap past midnight (end < start): duration adds 24 h.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.logger import get_logger

log = get_logger(__name__)

_SECONDS_PER_DAY = 24 * 3600

# --- regular expressions -----------------------------------------------------
_RE_SAMPLING = re.compile(r"Data Sampling Rate:\s*([\d.]+)\s*Hz", re.IGNORECASE)
_RE_CHANNEL = re.compile(r"Channel\s+\d+:\s*(.+?)\s*$", re.IGNORECASE)
_RE_FILE = re.compile(r"File Name:\s*(\S+)", re.IGNORECASE)
_RE_START_CLK = re.compile(r"File Start Time:\s*([\d]{1,2}:[\d]{2}:[\d]{2})", re.IGNORECASE)
_RE_END_CLK = re.compile(r"File End Time:\s*([\d]{1,2}:[\d]{2}:[\d]{2})", re.IGNORECASE)
_RE_NSEIZ = re.compile(r"Number of Seizures in File:\s*(\d+)", re.IGNORECASE)
_RE_SEIZ_START = re.compile(r"Seizure\s*(?:\d+)?\s*Start Time:\s*([\d.]+)", re.IGNORECASE)
_RE_SEIZ_END = re.compile(r"Seizure\s*(?:\d+)?\s*End Time:\s*([\d.]+)", re.IGNORECASE)


# --- data structures ---------------------------------------------------------
@dataclass
class Seizure:
    """One seizure, in seconds relative to the start of its EDF file."""
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class EdfFile:
    """One EDF recording and its annotations."""
    name: str
    start_clock: str | None = None
    end_clock: str | None = None
    n_seizures: int = 0
    seizures: list[Seizure] = field(default_factory=list)

    @property
    def clock_duration_sec(self) -> float | None:
        """Recording length from the wall clocks, or None if unavailable.

        Adds 24 h when the end clock is earlier than the start (midnight wrap).
        """
        if self.start_clock is None or self.end_clock is None:
            return None
        dur = clock_to_seconds(self.end_clock) - clock_to_seconds(self.start_clock)
        if dur < 0:
            dur += _SECONDS_PER_DAY
        return float(dur)


@dataclass
class PatientSummary:
    """Everything parsed from one chbXX-summary.txt."""
    patient_id: str
    sampling_rate: int | None
    channels: list[str]
    files: list[EdfFile]

    @property
    def total_seizures(self) -> int:
        return sum(f.n_seizures for f in self.files)

    @property
    def files_with_seizures(self) -> list[EdfFile]:
        return [f for f in self.files if f.n_seizures > 0]


# --- helpers -----------------------------------------------------------------
def clock_to_seconds(clock: str) -> int:
    """Convert an ``HH:MM:SS`` wall clock to seconds since midnight.

    Hours may exceed 23 (some CHB-MIT summaries keep counting past 24 h); the
    value is used only for differences, so large hours are fine.
    """
    h, m, s = (int(x) for x in clock.split(":"))
    return h * 3600 + m * 60 + s


def _derive_patient_id(path: Path) -> str:
    """chb01-summary.txt -> 'chb01'."""
    stem = path.name
    return stem.split("-")[0] if "-" in stem else path.stem


# --- core parsing ------------------------------------------------------------
def parse_summary_text(text: str, patient_id: str = "unknown") -> PatientSummary:
    """Parse the full text of a chbXX-summary.txt into a PatientSummary."""
    lines = text.splitlines()

    sampling_rate: int | None = None
    channels: list[str] = []
    files: list[EdfFile] = []

    current: EdfFile | None = None
    seen_first_file = False
    seiz_starts: list[float] = []
    seiz_ends: list[float] = []

    def _flush(cur: EdfFile | None) -> None:
        """Pair up the collected seizure starts/ends and attach to a file."""
        if cur is None:
            return
        cur.seizures = [Seizure(s, e) for s, e in zip(seiz_starts, seiz_ends)]
        if cur.n_seizures != len(cur.seizures):
            log.warning(
                "%s: declared %d seizures but parsed %d for %s",
                patient_id, cur.n_seizures, len(cur.seizures), cur.name,
            )
            # Trust the explicit parsed intervals if the header count is wrong.
            if cur.n_seizures == 0:
                cur.n_seizures = len(cur.seizures)
        files.append(cur)

    for line in lines:
        # Sampling rate (appears once, before the file blocks).
        m = _RE_SAMPLING.search(line)
        if m and sampling_rate is None:
            sampling_rate = int(float(m.group(1)))
            continue

        # Header channel list: only lines before the first "File Name:".
        if not seen_first_file:
            m = _RE_CHANNEL.search(line)
            if m:
                name = m.group(1).strip()
                # Skip placeholder / empty channel entries like "-" or ".".
                if name and name not in {"-", ".", "--"} and name not in channels:
                    channels.append(name)
                continue

        # New file block: flush the previous file first.
        m = _RE_FILE.search(line)
        if m:
            _flush(current)
            seiz_starts, seiz_ends = [], []
            current = EdfFile(name=m.group(1).strip())
            seen_first_file = True
            continue

        if current is None:
            continue  # ignore anything before the first file block

        m = _RE_START_CLK.search(line)
        if m:
            current.start_clock = m.group(1)
            continue
        m = _RE_END_CLK.search(line)
        if m:
            current.end_clock = m.group(1)
            continue
        m = _RE_NSEIZ.search(line)
        if m:
            current.n_seizures = int(m.group(1))
            continue
        m = _RE_SEIZ_START.search(line)
        if m:
            seiz_starts.append(float(m.group(1)))
            continue
        m = _RE_SEIZ_END.search(line)
        if m:
            seiz_ends.append(float(m.group(1)))
            continue

    _flush(current)  # don't forget the last file

    log.info(
        "Parsed %s: %d files, %d seizures, sampling_rate=%s",
        patient_id, len(files), sum(f.n_seizures for f in files), sampling_rate,
    )
    return PatientSummary(
        patient_id=patient_id,
        sampling_rate=sampling_rate,
        channels=channels,
        files=files,
    )


def parse_summary_file(path: str | Path) -> PatientSummary:
    """Read and parse a chbXX-summary.txt file from disk."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_summary_text(text, patient_id=_derive_patient_id(path))


# ---------------------------------------------------------------------------
# Self-test (synthetic summary -> no real dataset needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running summary_parser.py self-test ...\n")

    SYNTHETIC = """\
Data Sampling Rate: 256 Hz
*************************

Channels in EDF Files:
**********************
Channel 1: FP1-F7
Channel 2: F7-T7
Channel 3: T7-P7
Channel 4: -

File Name: chbTT_01.edf
File Start Time: 11:42:54
File End Time: 12:42:54
Number of Seizures in File: 0

File Name: chbTT_02.edf
File Start Time: 12:42:57
File End Time: 13:42:57
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

File Name: chbTT_03.edf
File Start Time: 23:30:00
File End Time: 00:30:00
Number of Seizures in File: 2
Seizure 1 Start Time: 100 seconds
Seizure 1 End Time: 130 seconds
Seizure 2 Start Time: 500 seconds
Seizure 2 End Time: 560 seconds

File Name: chbTT_04.edf
Number of Seizures in File: 0
"""

    s = parse_summary_text(SYNTHETIC, patient_id="chbTT")

    # --- header ---
    assert s.sampling_rate == 256, s.sampling_rate
    assert s.channels == ["FP1-F7", "F7-T7", "T7-P7"], s.channels  # "-" skipped
    assert len(s.files) == 4, len(s.files)

    # --- file 1: no seizures ---
    f1 = s.files[0]
    assert f1.name == "chbTT_01.edf"
    assert f1.n_seizures == 0 and f1.seizures == []
    assert f1.clock_duration_sec == 3600.0  # 11:42:54 -> 12:42:54

    # --- file 2: one seizure, "Seizure Start Time" form ---
    f2 = s.files[1]
    assert f2.n_seizures == 1 and len(f2.seizures) == 1
    assert f2.seizures[0].start_sec == 2996.0
    assert f2.seizures[0].end_sec == 3036.0
    assert f2.seizures[0].duration_sec == 40.0

    # --- file 3: two seizures, numbered form + midnight wrap ---
    f3 = s.files[2]
    assert f3.n_seizures == 2 and len(f3.seizures) == 2
    assert f3.seizures[1].start_sec == 500.0 and f3.seizures[1].end_sec == 560.0
    assert f3.clock_duration_sec == 3600.0  # 23:30 -> 00:30 wraps midnight

    # --- file 4: missing clocks ---
    f4 = s.files[3]
    assert f4.start_clock is None and f4.end_clock is None
    assert f4.clock_duration_sec is None

    # --- aggregates & helpers ---
    assert s.total_seizures == 3, s.total_seizures
    assert len(s.files_with_seizures) == 2
    assert clock_to_seconds("11:42:54") == 42174
    assert _derive_patient_id(Path("chb01-summary.txt")) == "chb01"

    print("Parsed summary:")
    print(f"  patient          : {s.patient_id}")
    print(f"  sampling rate    : {s.sampling_rate} Hz")
    print(f"  channels         : {s.channels}")
    print(f"  files            : {len(s.files)}")
    print(f"  total seizures   : {s.total_seizures}")

    print("\nAll summary_parser.py self-tests passed.")
