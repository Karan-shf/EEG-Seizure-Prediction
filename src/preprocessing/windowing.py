"""
windowing.py
============
Stage 2 / Req B + F (v2): turn a labeled patient timeline into the concrete list
of analysis windows the feature pipeline will consume.

Responsibilities
----------------
1. GENERATE candidate windows by sliding a fixed-length window (WINDOW_SECONDS,
   WINDOW_OVERLAP) strictly INSIDE each seam-free sub-run produced by
   timeline.py. Because generation is sub-run-bounded, a window may start or end
   exactly ON an inter-file seam but never straddles one -- the single window
   that would cross a seam is simply never produced, while its perfectly
   connected neighbours on both sides are kept (locked with the user 25 Aug 2026).
2. LABEL each window via labeler.LabelPlan (preictal / interictal / drop).
3. SUBSAMPLE the interictal candidate pool [Req F, v2]: CHB-MIT has ~40 h/patient
   so the full interictal set is intractable and wildly imbalanced. Keep a
   GENEROUS pool ~= INTERICTAL_POOL_MULTIPLIER x the preictal count, stratified
   across recording files (spread over time-of-day) and seeded for
   reproducibility. This is NOT class balancing -- it only bounds compute while
   preserving diversity; true balancing (cluster-centroid -> Borderline-SMOTE)
   happens later, in-fold, on the 486-D distance features.

The window offsets produced here are exactly the (seg_index, offset_samples)
pairs that edf_loader / the feature stages slice out of the signal, and that
timeline.window_abs_bounds() maps back to absolute time.

Depends only on config + timeline + labeler objects. No EDF / SciPy / NumPy.
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict
from dataclasses import dataclass

from src import config as cfg
from src.labeling.labeler import LabelPlan
from src.labeling.timeline import PatientTimeline
from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Window:
    """One analysis window, ready to be sliced from the signal."""
    seg_index: int
    offset_samples: int          # start sample within its segment
    n_samples: int
    label: int                   # cfg.LABEL_PREICTAL / cfg.LABEL_INTERICTAL
    file_name: str               # the member EDF this window lives in
    abs_start_sec: float | None  # absolute start (None for no-clock patients)

    @property
    def end_samples(self) -> int:
        return self.offset_samples + self.n_samples


@dataclass
class WindowSet:
    """All windows for one patient at one (window length, SOP) configuration."""
    patient_id: str
    window_samples: int
    stride_samples: int
    windows: tuple[Window, ...]        # final, chronologically sorted
    n_preictal: int
    n_interictal_kept: int             # after pool subsampling
    n_interictal_total: int            # before pool subsampling
    n_dropped: int
    subsampled: bool

    def by_label(self, label: int) -> list[Window]:
        return [w for w in self.windows if w.label == label]

    def counts(self) -> dict[str, int]:
        return {
            "preictal": self.n_preictal,
            "interictal_kept": self.n_interictal_kept,
            "interictal_total": self.n_interictal_total,
            "dropped": self.n_dropped,
        }

    def describe(self) -> str:
        ratio = (
            self.n_interictal_kept / self.n_preictal
            if self.n_preictal else float("nan")
        )
        return (
            f"WindowSet {self.patient_id}: win={self.window_samples} "
            f"stride={self.stride_samples} | preictal={self.n_preictal} "
            f"interictal={self.n_interictal_kept}/{self.n_interictal_total} "
            f"(kept:pre = {ratio:.1f}x) dropped={self.n_dropped} "
            f"subsampled={self.subsampled}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _file_of(tl: PatientTimeline, seg_index: int, offset_samples: int) -> str:
    """Name of the member EDF that contains `offset_samples` in `seg_index`.

    A window never crosses a seam, so it lies entirely within one member file.
    """
    seg = tl.segments[seg_index]
    for mf in seg.member_files:
        if mf.n_samples is None:
            continue
        lo = mf.seg_offset_samples
        if lo <= offset_samples < lo + mf.n_samples:
            return mf.name
    return seg.member_files[-1].name  # fallback (should not happen)


def _stratified_subsample(
    windows: list[Window], n_target: int, *, stratify: str, seed: int
) -> list[Window]:
    """Deterministically keep `n_target` interictal windows.

    stratify == "recording_file": Hamilton (largest-remainder) allocation of the
    quota across files proportional to each file's interictal count, capped at
    availability, then sample without replacement inside each file. Any other
    value falls back to a single global random sample. Seeded for reproducibility.
    """
    if n_target >= len(windows):
        return list(windows)
    rng = random.Random(seed)

    if stratify != "recording_file":
        return rng.sample(windows, n_target)

    groups: "OrderedDict[str, list[Window]]" = OrderedDict()
    for w in sorted(windows, key=lambda w: (w.file_name, w.seg_index, w.offset_samples)):
        groups.setdefault(w.file_name, []).append(w)

    total = len(windows)
    quota = {f: n_target * len(ws) / total for f, ws in groups.items()}
    alloc = {f: min(len(groups[f]), int(math.floor(q))) for f, q in quota.items()}

    # Distribute the remainder by largest fractional part, respecting capacity.
    order = sorted(groups, key=lambda f: quota[f] - math.floor(quota[f]), reverse=True)
    remaining = n_target - sum(alloc.values())
    while remaining > 0:
        progressed = False
        for f in order:
            if remaining <= 0:
                break
            if alloc[f] < len(groups[f]):
                alloc[f] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break  # every file capped (cannot happen while n_target < total)

    out: list[Window] = []
    for f, ws in groups.items():
        k = alloc[f]
        out.extend(ws if k >= len(ws) else rng.sample(ws, k))
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_windows(
    plan: LabelPlan,
    *,
    window_seconds: float = cfg.WINDOW_SECONDS,
    overlap: float = cfg.WINDOW_OVERLAP,
    subsample_pool: bool = True,
    pool_multiplier: float = cfg.INTERICTAL_POOL_MULTIPLIER,
    stratify: str = cfg.INTERICTAL_POOL_STRATIFY,
    seed: int = cfg.INTERICTAL_POOL_SEED,
) -> WindowSet:
    """Generate, label, and pool-subsample all windows for one patient.

    All preictal windows are always kept. The interictal pool is trimmed to
    ~= pool_multiplier x (preictal count) when subsample_pool is True and there
    is at least one preictal window; otherwise every interictal window is kept.
    """
    tl = plan.timeline
    win = cfg.window_samples(window_seconds)
    stride = cfg.stride_samples(window_seconds, overlap)

    preictal: list[Window] = []
    interictal: list[Window] = []
    n_dropped = 0

    for seg_index, start, end in tl.iter_windowable_subruns():
        offset = start
        while offset + win <= end:
            label = plan.label_window(seg_index, offset, win)
            if label is None:
                n_dropped += 1
            else:
                abs_start = None
                if tl.has_clocks:
                    abs_start = tl.window_abs_bounds(seg_index, offset, win)[0]
                w = Window(
                    seg_index=seg_index,
                    offset_samples=offset,
                    n_samples=win,
                    label=label,
                    file_name=_file_of(tl, seg_index, offset),
                    abs_start_sec=abs_start,
                )
                (preictal if label == cfg.LABEL_PREICTAL else interictal).append(w)
            offset += stride

    n_pre = len(preictal)
    n_inter_total = len(interictal)

    do_sub = subsample_pool and n_pre > 0
    if do_sub:
        n_target = int(round(pool_multiplier * n_pre))
        interictal_kept = _stratified_subsample(
            interictal, n_target, stratify=stratify, seed=seed
        )
    else:
        interictal_kept = list(interictal)

    windows = tuple(
        sorted(preictal + interictal_kept, key=lambda w: (w.seg_index, w.offset_samples))
    )

    ws = WindowSet(
        patient_id=plan.patient_id,
        window_samples=win,
        stride_samples=stride,
        windows=windows,
        n_preictal=n_pre,
        n_interictal_kept=len(interictal_kept),
        n_interictal_total=n_inter_total,
        n_dropped=n_dropped,
        subsampled=do_sub and len(interictal_kept) < n_inter_total,
    )
    log.info(
        "windows %s: %d preictal, %d/%d interictal, %d dropped",
        plan.patient_id, n_pre, ws.n_interictal_kept, n_inter_total, n_dropped,
    )
    return ws


def build_windows_from_summary(summary, **kwargs) -> WindowSet:
    """Convenience: summary -> timeline -> label plan -> windows.

    Keeps modules decoupled -- delegates to labeler.build_label_plan_from_summary
    (which itself calls timeline's public builder). Pass-through kwargs are split
    between the labeling layer and this windowing layer.
    """
    from src.labeling.labeler import build_label_plan_from_summary

    win_keys = {
        "window_seconds", "overlap", "subsample_pool",
        "pool_multiplier", "stratify", "seed",
    }
    win_kwargs = {k: v for k, v in kwargs.items() if k in win_keys}
    plan_kwargs = {k: v for k, v in kwargs.items() if k not in win_keys}
    plan = build_label_plan_from_summary(summary, **plan_kwargs)
    return build_windows(plan, **win_kwargs)


# ---------------------------------------------------------------------------
# Self-test (pure Python; no EDF / SciPy / NumPy needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.io.summary_parser import EdfFile, PatientSummary, Seizure
    from src.labeling.labeler import build_label_plan
    from src.labeling.timeline import build_timeline

    print("Running windowing.py self-test ...\n")
    FS = 256
    SPH, SOP_MIN, EXCL, LEAD = 10, 1, 120, 120  # small policy for a short synthetic run

    def mk(name, start, end, seizures=()):
        return EdfFile(
            name=name, start_clock=start, end_clock=end,
            n_seizures=len(seizures),
            seizures=[Seizure(s, e) for s, e in seizures],
        )

    # One stitched segment of three 600 s files (+3 s sub-tolerance gaps).
    files = [
        mk("f1", "12:00:00", "12:10:00"),
        mk("f2", "12:10:03", "12:20:03", seizures=[(500.0, 520.0)]),  # lead sz -> abs 1103
        mk("f3", "12:20:06", "12:30:06", seizures=[(10.0, 20.0)]),    # non-lead (93 s later)
    ]
    summary = PatientSummary("chb99", FS, ["FP1-F7"], files)
    tl = build_timeline(summary, fs=FS, gap_tolerance_sec=10.0)
    plan = build_label_plan(
        summary, tl, sph_seconds=SPH, sop_minutes=SOP_MIN,
        exclusion_seconds=EXCL, lead_min_preceding_seconds=LEAD,
    )

    ws = build_windows(plan)  # default 6 s / 40% overlap, 5x pool
    print(ws.describe(), "\n")

    win = cfg.WINDOW_SAMPLES
    seams = tl.segments[0].seam_offsets_samples

    # every window: valid label, correct length, never crosses a seam
    for w in ws.windows:
        assert w.label in (cfg.LABEL_PREICTAL, cfg.LABEL_INTERICTAL)
        assert w.n_samples == win
        assert not tl.crosses_seam(w.seg_index, w.offset_samples, win)
        assert w.file_name in tl.files_by_name

    # chronological + de-duplicated offsets
    keys = [(w.seg_index, w.offset_samples) for w in ws.windows]
    assert keys == sorted(keys) and len(keys) == len(set(keys))

    # pool subsampling: interictal trimmed to 5x preictal, preictal untouched
    assert ws.n_preictal > 0
    assert ws.n_interictal_total > 5 * ws.n_preictal            # subsampling is exercised
    assert ws.n_interictal_kept == round(5.0 * ws.n_preictal)   # == pool multiplier
    assert ws.subsampled is True
    assert len(ws.by_label(cfg.LABEL_PREICTAL)) == ws.n_preictal

    # stratification spread the pool across multiple recording files
    inter_files = {w.file_name for w in ws.by_label(cfg.LABEL_INTERICTAL)}
    assert len(inter_files) > 1, inter_files

    # determinism: same seed -> identical selection
    ws2 = build_windows(plan)
    assert [(w.seg_index, w.offset_samples) for w in ws2.windows] == keys

    # no subsampling path keeps the full pool
    ws_full = build_windows(plan, subsample_pool=False)
    assert ws_full.n_interictal_kept == ws_full.n_interictal_total
    assert ws_full.subsampled is False

    # no-clock patient still produces windows (segment-local labeling)
    nc = PatientSummary("chb24", FS, ["FP1-F7"], [mk("n1", None, None, seizures=[(300.0, 310.0)])])
    nc_tl = build_timeline(nc, fs=FS, file_samples={"n1": 600 * FS},
                           patients_without_clocks=("chb24",))
    nc_plan = build_label_plan(nc, nc_tl, sph_seconds=SPH, sop_minutes=SOP_MIN,
                               exclusion_seconds=EXCL, lead_min_preceding_seconds=LEAD)
    nc_ws = build_windows(nc_plan)
    print(nc_ws.describe(), "\n")
    assert nc_ws.n_preictal > 0
    assert all(w.abs_start_sec is None for w in nc_ws.windows)  # no clocks -> no abs time

    print("OK - windowing.py self-test passed.")
