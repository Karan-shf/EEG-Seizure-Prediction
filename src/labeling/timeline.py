"""
timeline.py
===========
Stage 2 (Req G): build a per-patient GLOBAL absolute timeline from CHB-MIT
summary clocks, then partition it into contiguous SEGMENTS for windowing.

Policy (locked with the user)
-----------------------------
* STITCH: consecutive EDF files whose inter-file gap <= tolerance are merged
  into ONE contiguous segment on a single monotonic absolute timeline, so that
  labels (preictal / interictal) flow correctly across the small seam.
* NEVER CROSS A SEAM: even inside a stitched segment, the exact file->file
  seam (the sample index where one EDF ends and the next begins) is recorded,
  and NO single analysis window may span it. `iter_windowable_subruns()` yields
  the seam-free [start, end) sample ranges that windowing.py slides within, and
  `crosses_seam()` is the guard windowing.py checks.

A gap > tolerance is a HARD BREAK: a new segment starts and nothing (window or
transition) ever spans it. Patients in config.PATIENTS_WITHOUT_CLOCKS (e.g.
chb24) have no wall clocks -> one segment per file, every boundary a hard break,
absolute times undefined; pass real per-file sample counts to place them.

Depends only on summary_parser objects + config. No EDF or SciPy needed.
Per-file sample counts default to clock-duration estimates (round(dur * fs));
pass `file_samples` (from edf_loader) to make seam offsets sample-exact.

Midnight wrap: summary clocks are within-day (00:00:00..23:59:59). An inter-file
clock difference below -MAX_OVERLAP_SEC is read as a midnight rollover (+24 h);
a small negative (<=MAX_OVERLAP_SEC, i.e. clock-rounding overlap) is clamped to
zero. This assumes each consecutive inter-file gap is under 24 h.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from src import config as cfg
from src.io.summary_parser import PatientSummary, clock_to_seconds
from src.utils.logger import get_logger

log = get_logger(__name__)

_SECONDS_PER_DAY = 24 * 3600
_MAX_OVERLAP_SEC = 5.0  # inter-file overlaps beyond this are read as midnight wraps


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MemberFile:
    """One EDF file placed inside a segment."""
    name: str
    n_samples: int | None          # None only for no-clock patients w/o counts
    seg_offset_samples: int        # start sample of this file within its segment
    abs_start_sec: float | None    # absolute time on the patient timeline
    abs_end_sec: float | None
    has_clock: bool


@dataclass(frozen=True)
class Segment:
    """A maximal run of files joined by <= tolerance gaps (one timeline)."""
    index: int
    member_files: tuple[MemberFile, ...]
    n_samples: int | None
    seam_offsets_samples: tuple[int, ...]   # internal file seams; NO window may cross
    abs_start_sec: float | None
    abs_end_sec: float | None

    @property
    def subruns(self) -> tuple[tuple[int, int], ...]:
        """Seam-free [start, end) sample ranges (the windowable regions).

        Windows are generated strictly inside one sub-run, so they can start or
        end exactly on a seam but never straddle one.
        """
        if self.n_samples is None:
            return ()
        bounds = (0, *self.seam_offsets_samples, self.n_samples)
        return tuple((bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1))


@dataclass(frozen=True)
class Gap:
    """Gap between two consecutive files (NaN gap_sec = unknown, no clocks)."""
    after_file: str
    before_file: str
    gap_sec: float
    is_hard_break: bool


@dataclass
class PatientTimeline:
    patient_id: str
    fs: int
    has_clocks: bool
    gap_tolerance_sec: float
    segments: tuple[Segment, ...]
    gaps: tuple[Gap, ...]
    files_by_name: Mapping[str, MemberFile]
    seg_of_file: Mapping[str, int]

    # -- counts --
    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_hard_breaks(self) -> int:
        return sum(1 for g in self.gaps if g.is_hard_break)

    @property
    def total_recorded_sec(self) -> float:
        return sum(
            mf.abs_end_sec - mf.abs_start_sec
            for seg in self.segments
            for mf in seg.member_files
            if mf.abs_start_sec is not None and mf.abs_end_sec is not None
        )

    # -- absolute-time mapping --
    def file_rel_to_abs(self, name: str, rel_sec: float) -> float | None:
        """Absolute time of a file-relative offset (seconds), or None w/o clocks."""
        mf = self.files_by_name[name]
        if mf.abs_start_sec is None:
            return None
        return mf.abs_start_sec + float(rel_sec)

    def seizure_abs_intervals(
        self, summary: PatientSummary
    ) -> list[tuple[float, float]]:
        """Absolute (start, end) seconds for every seizure, sorted."""
        out: list[tuple[float, float]] = []
        for f in summary.files:
            for sz in f.seizures:
                a = self.file_rel_to_abs(f.name, sz.start_sec)
                b = self.file_rel_to_abs(f.name, sz.end_sec)
                if a is not None and b is not None:
                    out.append((a, b))
        out.sort()
        return out

    # -- windowing support --
    def iter_windowable_subruns(self) -> Iterator[tuple[int, int, int]]:
        """Yield (segment_index, start_sample, end_sample) for every seam-free run."""
        for seg in self.segments:
            for start, end in seg.subruns:
                yield seg.index, start, end

    def crosses_seam(
        self, seg_index: int, offset_samples: int, n_win_samples: int
    ) -> bool:
        """True if window [offset, offset+n) would straddle an internal seam."""
        end = offset_samples + n_win_samples
        return any(
            offset_samples < seam < end
            for seam in self.segments[seg_index].seam_offsets_samples
        )

    def window_abs_bounds(
        self, seg_index: int, offset_samples: int, n_win_samples: int
    ) -> tuple[float | None, float | None]:
        """Absolute (start, end) seconds of a window given by (segment, offset).

        Because a window never crosses a seam it lies entirely in one member
        file, so the mapping is exact even across a stitched sub-tolerance gap.
        """
        seg = self.segments[seg_index]
        for mf in seg.member_files:
            if mf.n_samples is None:
                continue
            lo = mf.seg_offset_samples
            hi = lo + mf.n_samples
            if lo <= offset_samples < hi:
                if mf.abs_start_sec is None:
                    return (None, None)
                start = mf.abs_start_sec + (offset_samples - lo) / self.fs
                return (start, start + n_win_samples / self.fs)
        raise ValueError(
            f"offset {offset_samples} out of range for segment {seg_index}"
        )

    def describe(self) -> str:
        lines = [
            f"Patient {self.patient_id}: {len(self.files_by_name)} file(s), "
            f"{self.n_segments} segment(s), {self.n_hard_breaks} hard break(s), "
            f"clocks={self.has_clocks}",
        ]
        for seg in self.segments:
            span = (
                f"abs {seg.abs_start_sec:.0f}-{seg.abs_end_sec:.0f}s"
                if seg.abs_start_sec is not None
                else "abs=unknown"
            )
            lines.append(
                f"  seg{seg.index}: {len(seg.member_files)} file(s), "
                f"{seg.n_samples} samples, "
                f"{len(seg.seam_offsets_samples)} seam(s), {span}"
            )
        if self.total_recorded_sec:
            lines.append(f"  total recorded: {self.total_recorded_sec / 3600:.2f} h")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _inter_file_gap_sec(prev_end_clock: str, cur_start_clock: str) -> float:
    gap = clock_to_seconds(cur_start_clock) - clock_to_seconds(prev_end_clock)
    if gap < -_MAX_OVERLAP_SEC:      # crossed midnight
        gap += _SECONDS_PER_DAY
    elif gap < 0:                    # tiny clock-rounding overlap
        gap = 0
    return float(gap)


def _file_n_samples(f, fs: int, file_samples: Mapping[str, int] | None) -> int | None:
    if file_samples is not None and f.name in file_samples:
        return int(file_samples[f.name])
    dur = f.clock_duration_sec
    return None if dur is None else int(round(dur * fs))


def build_timeline(
    summary: PatientSummary,
    *,
    fs: int = cfg.FS,
    gap_tolerance_sec: float = cfg.INTER_FILE_GAP_TOLERANCE_SECONDS,
    file_samples: Mapping[str, int] | None = None,
    patients_without_clocks: tuple[str, ...] = cfg.PATIENTS_WITHOUT_CLOCKS,
) -> PatientTimeline:
    """Build a PatientTimeline (stitched segments + seams) from a PatientSummary."""
    files = list(summary.files)
    pid = summary.patient_id
    has_clocks = (
        pid not in set(patients_without_clocks)
        and len(files) > 0
        and all(f.start_clock is not None and f.end_clock is not None for f in files)
    )
    if not has_clocks:
        return _build_fallback(summary, fs, gap_tolerance_sec, file_samples)

    gaps: list[Gap] = []
    segments: list[Segment] = []
    files_by_name: dict[str, MemberFile] = {}
    seg_of_file: dict[str, int] = {}

    cur_members: list[MemberFile] = []
    cur_seams: list[int] = []
    seg_cum = 0
    seg_index = 0
    abs_cursor = 0.0  # absolute end of the previous file

    def _close() -> None:
        nonlocal cur_members, cur_seams, seg_cum, seg_index
        if not cur_members:
            return
        seg = Segment(
            index=seg_index,
            member_files=tuple(cur_members),
            n_samples=seg_cum,
            seam_offsets_samples=tuple(cur_seams),
            abs_start_sec=cur_members[0].abs_start_sec,
            abs_end_sec=cur_members[-1].abs_end_sec,
        )
        segments.append(seg)
        for mf in cur_members:
            files_by_name[mf.name] = mf
            seg_of_file[mf.name] = seg_index
        seg_index += 1
        cur_members = []
        cur_seams = []
        seg_cum = 0

    for i, f in enumerate(files):
        n = _file_n_samples(f, fs, file_samples)
        dur = f.clock_duration_sec  # not None (clocks verified above)
        assert dur is not None
        assert n is not None

        if i == 0:
            abs_start = 0.0
            start_new = True
        else:
            prev_end = files[i - 1].end_clock
            cur_start = f.start_clock
            assert prev_end is not None and cur_start is not None
            gap = _inter_file_gap_sec(prev_end, cur_start)
            is_hard = gap > gap_tolerance_sec
            gaps.append(
                Gap(after_file=files[i - 1].name, before_file=f.name,
                    gap_sec=gap, is_hard_break=is_hard)
            )
            abs_start = abs_cursor + gap
            start_new = is_hard

        if start_new:
            _close()

        abs_end = abs_start + dur
        if cur_members:  # subsequent file in this segment -> record its seam
            cur_seams.append(seg_cum)
        cur_members.append(
            MemberFile(
                name=f.name, n_samples=n, seg_offset_samples=seg_cum,
                abs_start_sec=abs_start, abs_end_sec=abs_end, has_clock=True,
            )
        )
        seg_cum += int(n)
        abs_cursor = abs_end

    _close()

    tl = PatientTimeline(
        patient_id=pid, fs=fs, has_clocks=True,
        gap_tolerance_sec=float(gap_tolerance_sec),
        segments=tuple(segments), gaps=tuple(gaps),
        files_by_name=files_by_name, seg_of_file=seg_of_file,
    )
    log.info("timeline %s: %d seg", pid, tl.n_segments)
    return tl


def _build_fallback(
    summary: PatientSummary,
    fs: int,
    gap_tolerance_sec: float,
    file_samples: Mapping[str, int] | None,
) -> PatientTimeline:
    """No-clock patients: one segment per file, all boundaries hard breaks."""
    segments: list[Segment] = []
    gaps: list[Gap] = []
    files_by_name: dict[str, MemberFile] = {}
    seg_of_file: dict[str, int] = {}

    for i, f in enumerate(summary.files):
        n = _file_n_samples(f, fs, file_samples)
        mf = MemberFile(
            name=f.name, n_samples=n, seg_offset_samples=0,
            abs_start_sec=None, abs_end_sec=None, has_clock=False,
        )
        segments.append(
            Segment(index=i, member_files=(mf,), n_samples=n,
                    seam_offsets_samples=(), abs_start_sec=None, abs_end_sec=None)
        )
        files_by_name[f.name] = mf
        seg_of_file[f.name] = i
        if i > 0:
            gaps.append(
                Gap(after_file=summary.files[i - 1].name, before_file=f.name,
                    gap_sec=float("nan"), is_hard_break=True)
            )

    return PatientTimeline(
        patient_id=summary.patient_id, fs=fs, has_clocks=False,
        gap_tolerance_sec=float(gap_tolerance_sec),
        segments=tuple(segments), gaps=tuple(gaps),
        files_by_name=files_by_name, seg_of_file=seg_of_file,
    )


def build_timeline_from_file(path, **kwargs) -> PatientTimeline:
    """Convenience: parse a chbXX-summary.txt then build its timeline."""
    from src.io.summary_parser import parse_summary_file
    return build_timeline(parse_summary_file(path), **kwargs)


# ---------------------------------------------------------------------------
# Self-test (pure Python; no EDF / SciPy needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.io.summary_parser import EdfFile, Seizure

    print("Running timeline.py self-test ...\n")
    FS = 256
    TOL = 10.0
    assert cfg.INTER_FILE_GAP_TOLERANCE_SECONDS == 10.0, "config tolerance drifted"
    HOUR = 3600 * FS  # samples in a 1 h file

    def mk(name, start, end, seizures=()):
        return EdfFile(
            name=name, start_clock=start, end_clock=end,
            n_seizures=len(seizures),
            seizures=[Seizure(s, e) for s, e in seizures],
        )

    files = [
        mk("f1", "12:00:00", "13:00:00"),
        mk("f2", "13:00:05", "14:00:05", seizures=[(100.0, 140.0)]),  # +5 s stitch
        mk("f3", "14:00:12", "15:00:12"),                             # +7 s stitch
        mk("f4", "16:30:00", "17:30:00"),                             # +1h29m48s BREAK
        mk("f5", "23:30:00", "00:30:00"),                             # +6h BREAK, wraps
        mk("f6", "00:30:05", "01:30:05"),                             # +5 s stitch (post-wrap)
    ]
    summary = PatientSummary("chb99", FS, ["FP1-F7"], files)
    tl = build_timeline(summary, fs=FS, gap_tolerance_sec=TOL)
    print(tl.describe(), "\n")

    # segmentation
    assert tl.n_segments == 3, tl.n_segments
    assert tl.n_hard_breaks == 2, tl.n_hard_breaks
    assert tl.n_segments == tl.n_hard_breaks + 1

    seg0, seg1, seg2 = tl.segments
    assert tuple(m.name for m in seg0.member_files) == ("f1", "f2", "f3")
    assert tuple(m.name for m in seg1.member_files) == ("f4",)
    assert tuple(m.name for m in seg2.member_files) == ("f5", "f6")

    # seams + windowable sub-runs
    assert seg0.seam_offsets_samples == (HOUR, 2 * HOUR), seg0.seam_offsets_samples
    assert seg0.n_samples == 3 * HOUR
    assert seg0.subruns == ((0, HOUR), (HOUR, 2 * HOUR), (2 * HOUR, 3 * HOUR))
    assert seg1.seam_offsets_samples == ()
    assert seg2.seam_offsets_samples == (HOUR,)

    # sub-runs partition the segment exactly and never contain a seam interior
    for seg in tl.segments:
        covered = sum(e - s for s, e in seg.subruns)
        assert covered == seg.n_samples
        for (s, e) in seg.subruns:
            assert not any(s < seam < e for seam in seg.seam_offsets_samples)

    # THE headline guarantee: no window may cross a seam
    win = cfg.WINDOW_SAMPLES
    assert tl.crosses_seam(0, HOUR - 100, win) is True     # straddles the seam
    assert tl.crosses_seam(0, HOUR, win) is False          # starts exactly on seam
    assert tl.crosses_seam(0, HOUR - win, win) is False    # ends exactly on seam

    # absolute timeline + seizure placement (f2 starts at 3600 + 5 s)
    assert tl.file_rel_to_abs("f2", 0.0) == 3605.0
    assert tl.file_rel_to_abs("f3", 0.0) == 3605.0 + 3600.0 + 7.0  # 7212
    assert tl.seizure_abs_intervals(summary) == [(3705.0, 3745.0)]

    # monotonic across the midnight wrap (f5 dur must be 3600 s, not negative)
    f5 = tl.files_by_name["f5"]
    assert f5.abs_end_sec is not None
    assert f5.abs_start_sec is not None
    assert abs((f5.abs_end_sec - f5.abs_start_sec) - 3600.0) < 1e-6
    assert tl.files_by_name["f6"].abs_start_sec == f5.abs_end_sec + 5.0

    # window_abs_bounds lands in the right member file across a stitched gap
    start, end = tl.window_abs_bounds(0, HOUR, win)  # first window of f2
    assert end is not None
    assert start == 3605.0 and abs(end - (3605.0 + win / FS)) < 1e-9

    # actual sample-count override makes seams exact
    tl_ov = build_timeline(summary, fs=FS, gap_tolerance_sec=TOL,
                           file_samples={"f1": 900000})
    assert tl_ov.segments[0].seam_offsets_samples == (900000, 900000 + HOUR)

    # no-clock fallback (chb24): one segment per file, abs undefined, all breaks
    nc = PatientSummary("chb24", FS, [], [mk("c1", "12:00:00", "13:00:00"),
                                          EdfFile("c2")])
    tl_nc = build_timeline(nc, fs=FS, gap_tolerance_sec=TOL,
                           file_samples={"c2": 2000})
    assert tl_nc.has_clocks is False
    assert tl_nc.n_segments == 2
    assert tl_nc.segments[0].n_samples == HOUR         # from clocks
    assert tl_nc.segments[1].n_samples == 2000         # from override
    assert tl_nc.file_rel_to_abs("c1", 5.0) is None
    assert all(g.is_hard_break for g in tl_nc.gaps)

    print("All timeline.py self-tests passed.")
