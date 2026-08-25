"""
labeler.py
==========
Stage 2 / Req G (continued): assign a class label to every candidate analysis
window on a patient's GLOBAL absolute timeline.

    LABEL_PREICTAL   (1) : window lies fully inside a lead seizure's
                           [onset - SPH - SOP, onset - SPH] prediction window.
    LABEL_INTERICTAL (0) : window sits at least SEIZURE_EXCLUSION_SECONDS
                           (+/- 4 h, Truong 2018) away from EVERY seizure.
    None (dropped)       : everything else -- ictal, the SPH intervention gap,
                           the pre-guard / postictal buffers, preictal of a
                           NON-lead seizure, and any window that would straddle
                           an inter-file seam.

Separation of concerns (locked with the user, 25 Aug 2026)
----------------------------------------------------------
* timeline.py and labeler.py stay SEPARATE modules. timeline.py owns recording
  continuity (stitching sub-tolerance gaps into contiguous segments, recording
  seams, absolute-time mapping). labeler.py is a PURE function of that timeline
  plus the SPH/SOP + exclusion policy in config -- it never re-derives geometry.
* SEAM POLICY: only the single window that physically straddles an inter-file
  seam is dropped. Windows that sit perfectly connected on either side of the
  seam (within the <= tolerance stitch) are KEPT and labeled normally. This is
  exactly what timeline's seam-free sub-runs already guarantee: window offsets
  are generated strictly inside one sub-run, so a window may start or end ON a
  seam but never span it. `crosses_seam()` is re-checked here as a belt-and-
  suspenders guard.

Coordinate system
-----------------
All comparisons happen in a single monotonic "reference-seconds" axis per
window:
* clock patients   -> ABSOLUTE seconds on the patient timeline. Exclusion is
                      enforced GLOBALLY, so an interictal window is rejected
                      even if the nearest seizure lives in a different segment
                      across a multi-hour hard break.
* no-clock patients -> SEGMENT-LOCAL seconds (sample offset / fs). Each file is
                      its own segment (timeline fallback), so exclusion / lead
                      logic can only be enforced WITHIN a file; cross-file
                      relationships are unknown and treated conservatively.

Depends only on config + summary_parser + timeline objects. No EDF / SciPy.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from src import config as cfg
from src.io.summary_parser import PatientSummary
from src.labeling.timeline import PatientTimeline, build_timeline
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SeizureEvent:
    """One seizure placed on the reference axis (abs, or seg-local w/o clocks)."""
    index: int                 # global chronological index
    file_name: str
    seg_index: int
    onset_sec: float
    offset_sec: float
    is_lead: bool              # >= LEAD_SEIZURE_MIN_PRECEDING_SECONDS seizure-free before


@dataclass(frozen=True)
class PreictalInterval:
    """[start, end) prediction window for one LEAD seizure (reference seconds)."""
    seizure_index: int
    seg_index: int
    start_sec: float
    end_sec: float


@dataclass
class LabelPlan:
    """Immutable-ish labeling policy resolved for one patient + one SOP."""
    patient_id: str
    fs: int
    has_clocks: bool
    sph_seconds: int
    sop_seconds: int
    exclusion_seconds: int
    lead_min_preceding_seconds: int
    require_preictal_same_segment: bool
    seizures: tuple[SeizureEvent, ...]
    preictal_intervals: tuple[PreictalInterval, ...]
    exclusion_intervals: tuple[tuple[float, float, int], ...]  # (start, end, seg_index)
    timeline: PatientTimeline

    # -- counts --
    @property
    def n_lead_seizures(self) -> int:
        return sum(1 for s in self.seizures if s.is_lead)

    # -- core: label a single window ------------------------------------------
    def label_window(
        self, seg_index: int, offset_samples: int, n_win_samples: int
    ) -> int | None:
        """Return LABEL_PREICTAL / LABEL_INTERICTAL, or None to DROP the window.

        `seg_index, offset_samples` identify the window exactly as produced by
        timeline.iter_windowable_subruns(); `n_win_samples` is its length.
        """
        # (0) Seam guard: never label a window that straddles an inter-file seam.
        if self.timeline.crosses_seam(seg_index, offset_samples, n_win_samples):
            return None

        w_start, w_end = self._window_ref_bounds(seg_index, offset_samples, n_win_samples)
        if w_start is None or w_end is None:
            return None

        # (1) Preictal wins: fully inside a LEAD seizure's prediction window,
        #     and (by default) in the SAME contiguous segment as that seizure
        #     so a preictal span never reaches across a hard break.
        for pz in self.preictal_intervals:
            if pz.seg_index != seg_index and (
                self.require_preictal_same_segment or not self.has_clocks
            ):
                continue
            if w_start >= pz.start_sec and w_end <= pz.end_sec:
                return cfg.LABEL_PREICTAL

        # (2) Interictal: window must NOT overlap any seizure's +/- exclusion
        #     buffer (this buffer also swallows the ictal span itself and the
        #     preictal window of non-lead seizures -> those become drops).
        for ex_start, ex_end, ex_seg in self.exclusion_intervals:
            if not self.has_clocks and ex_seg != seg_index:
                continue  # no-clock: cross-file distance is unknown, ignore
            if w_start < ex_end and w_end > ex_start:
                return None
        return cfg.LABEL_INTERICTAL

    def _window_ref_bounds(
        self, seg_index: int, offset_samples: int, n_win_samples: int
    ) -> tuple[float | None, float | None]:
        if self.has_clocks:
            return self.timeline.window_abs_bounds(
                seg_index, offset_samples, n_win_samples
            )
        return (offset_samples / self.fs, (offset_samples + n_win_samples) / self.fs)

    # -- iteration over all candidate windows ---------------------------------
    def iter_labeled_windows(
        self,
        *,
        window_samples: int = cfg.WINDOW_SAMPLES,
        stride_samples: int = cfg.STRIDE_SAMPLES,
        include_dropped: bool = False,
    ) -> Iterator[tuple[int, int, int | None]]:
        """Yield (seg_index, offset_samples, label) for every candidate window.

        Windows are generated strictly inside timeline sub-runs, so the window
        that would straddle a seam is simply never produced (its connected
        neighbours on both sides ARE produced). Dropped (None) windows are
        skipped unless include_dropped=True.
        """
        for seg_index, start, end in self.timeline.iter_windowable_subruns():
            offset = start
            while offset + window_samples <= end:
                label = self.label_window(seg_index, offset, window_samples)
                if label is not None or include_dropped:
                    yield seg_index, offset, label
                offset += stride_samples

    def counts(
        self,
        *,
        window_samples: int = cfg.WINDOW_SAMPLES,
        stride_samples: int = cfg.STRIDE_SAMPLES,
    ) -> dict[str, int]:
        """Tally preictal / interictal / dropped candidate windows."""
        out = {"preictal": 0, "interictal": 0, "dropped": 0}
        for _, _, label in self.iter_labeled_windows(
            window_samples=window_samples,
            stride_samples=stride_samples,
            include_dropped=True,
        ):
            if label == cfg.LABEL_PREICTAL:
                out["preictal"] += 1
            elif label == cfg.LABEL_INTERICTAL:
                out["interictal"] += 1
            else:
                out["dropped"] += 1
        return out

    def describe(self) -> str:
        n_lead = self.n_lead_seizures
        lines = [
            f"LabelPlan {self.patient_id}: {len(self.seizures)} seizure(s), "
            f"{n_lead} lead, SPH={self.sph_seconds}s, SOP={self.sop_seconds}s, "
            f"exclusion=+/-{self.exclusion_seconds}s, clocks={self.has_clocks}",
        ]
        for s in self.seizures:
            lines.append(
                f"  sz{s.index} [{s.file_name} seg{s.seg_index}] "
                f"{s.onset_sec:.0f}-{s.offset_sec:.0f}s "
                f"{'LEAD' if s.is_lead else 'non-lead'}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_label_plan(
    summary: PatientSummary,
    timeline: PatientTimeline,
    *,
    sph_seconds: int = cfg.SPH_SECONDS,
    sop_minutes: int = cfg.SOP_PRIMARY_MINUTES,
    exclusion_seconds: int = cfg.SEIZURE_EXCLUSION_SECONDS,
    lead_min_preceding_seconds: int = cfg.LEAD_SEIZURE_MIN_PRECEDING_SECONDS,
    require_preictal_same_segment: bool = True,
    assume_lead_across_unknown_gap: bool = False,
) -> LabelPlan:
    """Resolve the SPH/SOP + exclusion labeling policy for one patient + SOP.

    Seizures are placed on the reference axis (absolute seconds when the
    timeline has clocks, else segment-local seconds). A seizure is a LEAD
    seizure iff it is preceded by >= lead_min_preceding_seconds of seizure-free
    EEG (the first seizure counts as lead). For no-clock patients, a seizure
    that opens a new file has an unknown preceding gap: it is treated as lead
    only if assume_lead_across_unknown_gap is True (default: conservative False).
    """
    fs = timeline.fs
    has_clocks = timeline.has_clocks
    sop_seconds = int(sop_minutes) * 60

    # --- place every seizure on the reference axis ---
    raw: list[tuple[int, float, float, str]] = []  # (seg_index, onset, offset, file)
    for f in summary.files:
        if f.name not in timeline.files_by_name:
            continue
        mf = timeline.files_by_name[f.name]
        seg_index = timeline.seg_of_file[f.name]
        for sz in f.seizures:
            if has_clocks:
                onset = timeline.file_rel_to_abs(f.name, sz.start_sec)
                offset = timeline.file_rel_to_abs(f.name, sz.end_sec)
            else:
                base = mf.seg_offset_samples / fs
                onset = base + sz.start_sec
                offset = base + sz.end_sec
            if onset is None or offset is None:
                continue
            raw.append((seg_index, float(onset), float(offset), f.name))

    raw.sort(key=(lambda r: r[1]) if has_clocks else (lambda r: (r[0], r[1])))

    # --- lead-seizure detection ---
    seizures: list[SeizureEvent] = []
    prev_offset: float | None = None
    prev_seg: int | None = None
    for idx, (seg_index, onset, offset, fname) in enumerate(raw):
        if prev_offset is None:
            is_lead = True
        elif has_clocks:
            is_lead = (onset - prev_offset) >= lead_min_preceding_seconds
        elif seg_index != prev_seg:
            is_lead = assume_lead_across_unknown_gap
        else:
            is_lead = (onset - prev_offset) >= lead_min_preceding_seconds
        seizures.append(
            SeizureEvent(idx, fname, seg_index, onset, offset, is_lead)
        )
        prev_offset = offset
        prev_seg = seg_index

    # --- derive preictal + exclusion intervals ---
    preictal = tuple(
        PreictalInterval(
            seizure_index=s.index,
            seg_index=s.seg_index,
            start_sec=s.onset_sec - sph_seconds - sop_seconds,
            end_sec=s.onset_sec - sph_seconds,
        )
        for s in seizures
        if s.is_lead
    )
    exclusion = tuple(
        (s.onset_sec - exclusion_seconds, s.offset_sec + exclusion_seconds, s.seg_index)
        for s in seizures
    )

    plan = LabelPlan(
        patient_id=summary.patient_id,
        fs=fs,
        has_clocks=has_clocks,
        sph_seconds=int(sph_seconds),
        sop_seconds=int(sop_seconds),
        exclusion_seconds=int(exclusion_seconds),
        lead_min_preceding_seconds=int(lead_min_preceding_seconds),
        require_preictal_same_segment=require_preictal_same_segment,
        seizures=tuple(seizures),
        preictal_intervals=preictal,
        exclusion_intervals=exclusion,
        timeline=timeline,
    )
    log.info(
        "labelplan %s: %d seizure(s), %d lead, SOP=%dmin",
        summary.patient_id, len(seizures), plan.n_lead_seizures, sop_minutes,
    )
    return plan


def build_label_plan_from_summary(
    summary: PatientSummary, **kwargs
) -> LabelPlan:
    """Convenience: build the timeline (via timeline.py) then the label plan.

    Keeps the two modules separate -- this only calls timeline's PUBLIC builder.
    Pass-through kwargs are split between build_timeline and build_label_plan.
    """
    tl_keys = {"fs", "gap_tolerance_sec", "file_samples", "patients_without_clocks"}
    tl_kwargs = {k: v for k, v in kwargs.items() if k in tl_keys}
    lp_kwargs = {k: v for k, v in kwargs.items() if k not in tl_keys}
    timeline = build_timeline(summary, **tl_kwargs)
    return build_label_plan(summary, timeline, **lp_kwargs)


# ---------------------------------------------------------------------------
# Self-test (pure Python; no EDF / SciPy needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.io.summary_parser import EdfFile, Seizure

    print("Running labeler.py self-test ...\n")
    FS = 256
    WIN = cfg.WINDOW_SAMPLES        # 1536 (6 s)
    # Small policy so a few-minute synthetic recording exercises every branch.
    SPH, SOP_MIN, EXCL, LEAD = 10, 1, 120, 120  # SOP_MIN in minutes -> 60 s

    def mk(name, start, end, seizures=()):
        return EdfFile(
            name=name, start_clock=start, end_clock=end,
            n_seizures=len(seizures),
            seizures=[Seizure(s, e) for s, e in seizures],
        )

    # --- clock patient: one stitched segment of three 600 s files (+3 s gaps) ---
    # abs: f1 0-600, f2 603-1203 (sz1 @ rel 500-520 -> abs 1103-1123),
    #      f3 1206-1806 (sz2 @ rel 10-20 -> abs 1216-1226, only 93 s after sz1)
    files = [
        mk("f1", "12:00:00", "12:10:00"),
        mk("f2", "12:10:03", "12:20:03", seizures=[(500.0, 520.0)]),
        mk("f3", "12:20:06", "12:30:06", seizures=[(10.0, 20.0)]),
    ]
    summary = PatientSummary("chb99", FS, ["FP1-F7"], files)
    tl = build_timeline(summary, fs=FS, gap_tolerance_sec=10.0)
    assert tl.n_segments == 1 and tl.n_hard_breaks == 0, tl.describe()
    seg0 = tl.segments[0]
    seam1 = seg0.seam_offsets_samples[0]      # 600 s * 256 = 153600
    assert seam1 == 600 * FS, seg0.seam_offsets_samples

    plan = build_label_plan(
        summary, tl,
        sph_seconds=SPH, sop_minutes=SOP_MIN,
        exclusion_seconds=EXCL, lead_min_preceding_seconds=LEAD,
    )
    print(plan.describe(), "\n")

    # lead detection: sz0 is first (lead); sz1 is only 93 s after sz0 (< 120 s) -> non-lead
    assert plan.seizures[0].is_lead is True
    assert plan.seizures[1].is_lead is False
    assert len(plan.preictal_intervals) == 1               # only the lead seizure
    pz = plan.preictal_intervals[0]
    assert (pz.start_sec, pz.end_sec) == (1103 - SPH - 60, 1103 - SPH), (pz.start_sec, pz.end_sec)

    # helper: sample offset for an absolute time inside member file `mf`
    def off_for_abs(mf, abs_sec):
        return mf.seg_offset_samples + int(round((abs_sec - mf.abs_start_sec) * FS))

    f2 = tl.files_by_name["f2"]
    f1 = tl.files_by_name["f1"]

    # (a) window fully inside preictal [1033,1093] -> PREICTAL
    o_pre = off_for_abs(f2, 1050.0)
    assert plan.label_window(0, o_pre, WIN) == cfg.LABEL_PREICTAL

    # (b) window far from every seizure -> INTERICTAL
    o_int = off_for_abs(f1, 200.0)
    assert plan.label_window(0, o_int, WIN) == cfg.LABEL_INTERICTAL

    # (c) window in the postictal buffer of sz0 (abs ~1150, within +/-120 s) -> drop
    o_post = off_for_abs(f2, 1150.0)
    assert plan.label_window(0, o_post, WIN) is None

    # (d) seam policy: straddling window dropped; connected neighbours kept
    assert tl.crosses_seam(0, seam1 - 100, WIN) is True
    assert plan.label_window(0, seam1 - 100, WIN) is None          # straddles seam -> drop
    assert tl.crosses_seam(0, seam1 - WIN, WIN) is False           # ends exactly on seam
    assert tl.crosses_seam(0, seam1, WIN) is False                 # starts exactly on seam
    # both connected neighbours are real, labelable candidates (interictal here)
    assert plan.label_window(0, seam1 - WIN, WIN) == cfg.LABEL_INTERICTAL
    assert plan.label_window(0, seam1, WIN) == cfg.LABEL_INTERICTAL

    # (e) no yielded window ever crosses a seam; tallies are positive
    for seg_i, off, label in plan.iter_labeled_windows():
        assert not tl.crosses_seam(seg_i, off, WIN)
        assert label in (cfg.LABEL_PREICTAL, cfg.LABEL_INTERICTAL)
    counts = plan.counts()
    print("clock-patient window counts:", counts, "\n")
    assert counts["preictal"] > 0 and counts["interictal"] > 0 and counts["dropped"] > 0

    # --- no-clock patient (chb24-style): per-file segments, seg-local labeling ---
    nc_files = [mk("n1", None, None, seizures=[(300.0, 310.0)])]
    nc_summary = PatientSummary("chb24", FS, ["FP1-F7"], nc_files)
    nc_tl = build_timeline(
        nc_summary, fs=FS, file_samples={"n1": 600 * FS},
        patients_without_clocks=("chb24",),
    )
    assert nc_tl.has_clocks is False and nc_tl.n_segments == 1
    nc_plan = build_label_plan(
        nc_summary, nc_tl,
        sph_seconds=SPH, sop_minutes=SOP_MIN,
        exclusion_seconds=EXCL, lead_min_preceding_seconds=LEAD,
    )
    # seg-local: onset=300 -> preictal [230,290]; window @ 240 s -> PREICTAL
    assert nc_plan.label_window(0, int(round(240 * FS)), WIN) == cfg.LABEL_PREICTAL
    # window @ 50 s (>120 s from the [180,430] buffer) -> INTERICTAL
    assert nc_plan.label_window(0, int(round(50 * FS)), WIN) == cfg.LABEL_INTERICTAL
    # window @ 305 s (ictal) -> drop
    assert nc_plan.label_window(0, int(round(305 * FS)), WIN) is None
    print("no-clock window counts:", nc_plan.counts(), "\n")

    print("OK - labeler.py self-test passed.")
