"""
inventory.py
============
Stage (experiment): dataset / patient / seizure INVENTORY + LOPO eligibility.

Before running the (expensive) LOPO sweep it is worth a fast, EDF-free census of
what each patient actually contributes:

  * how many files / stitched segments / hard breaks,
  * how many seizures, and how many of those are usable LEAD seizures,
  * how many preictal / interictal candidate windows the labeling policy yields,
  * and therefore whether the patient can serve as a LOPO fold at all
    (needs >= 1 lead seizure AND both preictal and interictal windows).

This reuses exactly the same timeline -> labeler -> windowing stack the feature
pipeline uses, so the window counts here are the same ones the provider will
stream -- but computed WITHOUT reading a single EDF sample. For clock patients
the segment lengths come from the summary clocks (round(duration * fs)); this is
an estimate, accurate to ~1 window. No-clock patients (chb24) have no sample
counts from the summary, so their window counts are only available if you pass
`file_samples` or `probe=True` (which reads EDF headers via MNE, header-only).

Dependency-light: pure Python for clock patients (no NumPy / SciPy / EDF). MNE is
imported lazily and only when `probe=True`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from src import config as cfg
from src.utils.logger import get_logger
from src.io.summary_parser import PatientSummary, parse_summary_file
from src.labeling.timeline import build_timeline
from src.labeling.labeler import build_label_plan
from src.preprocessing.windowing import build_windows

log = get_logger(__name__)

# CHB-MIT is chb01..chb24 (chb24 has no wall clocks).
DEFAULT_PATIENTS: tuple[str, ...] = tuple(f"chb{i:02d}" for i in range(1, 25))
# Minimum LEAD seizures for a patient to be a usable LOPO fold. A single lead
# seizure already yields a preictal block; raise this (e.g. to 2-3) to drop the
# fragile few-seizure patients (chb02, chb07, chb11, chb17, chb19, chb22).
MIN_LEAD_SEIZURES: int = 1


@dataclass
class PatientInventory:
    patient_id: str
    has_clocks: bool
    sized: bool                 # were window counts computable (clocks or samples)?
    n_files: int
    n_segments: int
    n_hard_breaks: int
    recorded_hours: float
    total_seizures: int
    n_lead_seizures: int
    n_preictal_windows: int
    n_interictal_kept: int
    n_interictal_total: int
    n_dropped: int
    eligible: bool
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Discovery / path resolution
# ---------------------------------------------------------------------------
def _summary_path(patient_id: str, raw_dir: Path) -> Path:
    """Resolve chbXX-summary.txt under raw_dir/chbXX/ or flat under raw_dir/."""
    for cand in (raw_dir / patient_id / f"{patient_id}-summary.txt",
                 raw_dir / f"{patient_id}-summary.txt"):
        if cand.exists():
            return cand
    return raw_dir / patient_id / f"{patient_id}-summary.txt"


def _edf_path(patient_id: str, name: str, raw_dir: Path) -> Path:
    for cand in (raw_dir / patient_id / name, raw_dir / name):
        if cand.exists():
            return cand
    return raw_dir / patient_id / name


def discover_patients(raw_dir: Optional[Path] = None) -> tuple[str, ...]:
    """Patient ids that have a *-summary.txt under raw_dir (recursively one level)."""
    raw_dir = Path(raw_dir) if raw_dir is not None else cfg.RAW_DIR
    found: set[str] = set()
    if raw_dir.exists():
        for p in list(raw_dir.glob("*-summary.txt")) + list(raw_dir.glob("*/*-summary.txt")):
            found.add(p.name.split("-")[0])
    return tuple(sorted(found))


def _probe_file_samples(summary: PatientSummary, raw_dir: Path) -> Dict[str, int]:
    """Read EDF headers (no signal) to get exact per-file sample counts."""
    import mne  # lazy; header-only read

    counts: Dict[str, int] = {}
    for f in summary.files:
        path = _edf_path(summary.patient_id, f.name, raw_dir)
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        n = int(raw.n_times)
        sf = float(raw.info["sfreq"])
        if sf != cfg.FS:
            n = int(round(n * cfg.FS / sf))
        counts[f.name] = n
    return counts


# ---------------------------------------------------------------------------
# Core (pure function of a parsed summary)
# ---------------------------------------------------------------------------
def inventory_from_summary(
    summary: PatientSummary,
    *,
    file_samples: Optional[Mapping[str, int]] = None,
    sop_minutes: Optional[int] = None,
    window_seconds: Optional[float] = None,
    overlap: Optional[float] = None,
    min_lead_seizures: int = MIN_LEAD_SEIZURES,
    subsample_pool: bool = True,
) -> PatientInventory:
    """Build a PatientInventory from an already-parsed summary (no EDF read)."""
    sop = int(cfg.SOP_PRIMARY_MINUTES if sop_minutes is None else sop_minutes)
    win_s = float(cfg.WINDOW_SECONDS if window_seconds is None else window_seconds)
    ovlp = float(cfg.WINDOW_OVERLAP if overlap is None else overlap)

    tl = build_timeline(summary, fs=cfg.FS, file_samples=file_samples)
    plan = build_label_plan(summary, tl, sop_minutes=sop)

    sized = any(seg.n_samples for seg in tl.segments)
    n_pre = n_int_kept = n_int_total = n_dropped = 0
    if sized:
        ws = build_windows(plan, window_seconds=win_s, overlap=ovlp,
                           subsample_pool=subsample_pool)
        n_pre = ws.n_preictal
        n_int_kept = ws.n_interictal_kept
        n_int_total = ws.n_interictal_total
        n_dropped = ws.n_dropped

    n_lead = plan.n_lead_seizures
    if not sized:
        eligible = False
        reason = "no sample counts (no clocks; pass file_samples or probe=True)"
    elif n_lead < min_lead_seizures:
        eligible = False
        reason = f"{n_lead} lead seizure(s) < required {min_lead_seizures}"
    elif n_pre == 0:
        eligible = False
        reason = "no preictal windows"
    elif n_int_kept == 0:
        eligible = False
        reason = "no interictal windows"
    else:
        eligible = True
        reason = "ok"

    return PatientInventory(
        patient_id=summary.patient_id,
        has_clocks=tl.has_clocks,
        sized=sized,
        n_files=len(summary.files),
        n_segments=tl.n_segments,
        n_hard_breaks=tl.n_hard_breaks,
        recorded_hours=round(tl.total_recorded_sec / 3600.0, 3),
        total_seizures=summary.total_seizures,
        n_lead_seizures=n_lead,
        n_preictal_windows=n_pre,
        n_interictal_kept=n_int_kept,
        n_interictal_total=n_int_total,
        n_dropped=n_dropped,
        eligible=eligible,
        reason=reason,
    )


def build_patient_inventory(
    patient_id: str,
    *,
    raw_dir: Optional[Path] = None,
    probe: bool = False,
    **kwargs,
) -> PatientInventory:
    """Parse chbXX-summary.txt from disk, then inventory it.

    probe=True reads EDF headers (via MNE) for exact sample counts -- required to
    size no-clock patients (chb24); clock patients do not need it.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else cfg.RAW_DIR
    summary = parse_summary_file(_summary_path(patient_id, raw_dir))
    file_samples = kwargs.pop("file_samples", None)
    if probe and file_samples is None:
        file_samples = _probe_file_samples(summary, raw_dir)
    return inventory_from_summary(summary, file_samples=file_samples, **kwargs)


def build_inventory(
    patients: Optional[Sequence[str]] = None,
    *,
    raw_dir: Optional[Path] = None,
    probe: bool = False,
    **kwargs,
) -> List[PatientInventory]:
    """Inventory every requested patient (defaults to whatever is on disk)."""
    raw_dir = Path(raw_dir) if raw_dir is not None else cfg.RAW_DIR
    if patients is None:
        patients = discover_patients(raw_dir) or DEFAULT_PATIENTS
    out: List[PatientInventory] = []
    for pid in patients:
        try:
            out.append(build_patient_inventory(pid, raw_dir=raw_dir, probe=probe, **kwargs))
        except FileNotFoundError:
            log.warning("inventory: no summary for %s (skipped)", pid)
        except Exception as exc:  # keep the census going on a bad file
            log.warning("inventory: %s failed: %s", pid, exc)
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize_inventory(items: Sequence[PatientInventory]) -> Dict[str, object]:
    elig = [it for it in items if it.eligible]
    return {
        "n_patients": len(items),
        "n_eligible": len(elig),
        "eligible_patients": [it.patient_id for it in elig],
        "total_lead_seizures": sum(it.n_lead_seizures for it in items),
        "total_preictal_windows": sum(it.n_preictal_windows for it in items),
        "total_interictal_kept": sum(it.n_interictal_kept for it in items),
        "total_recorded_hours": round(sum(it.recorded_hours for it in items), 2),
    }


def format_inventory(items: Sequence[PatientInventory]) -> str:
    hdr = (f"{'patient':<8} {'clk':>3} {'files':>5} {'seg':>3} {'brk':>3} "
           f"{'hours':>6} {'sz':>3} {'lead':>4} {'pre':>6} {'int':>7} "
           f"{'elig':>4}  reason")
    lines = [hdr, "-" * len(hdr)]
    for it in items:
        lines.append(
            f"{it.patient_id:<8} {('Y' if it.has_clocks else 'n'):>3} "
            f"{it.n_files:>5} {it.n_segments:>3} {it.n_hard_breaks:>3} "
            f"{it.recorded_hours:>6.1f} {it.total_seizures:>3} "
            f"{it.n_lead_seizures:>4} {it.n_preictal_windows:>6} "
            f"{it.n_interictal_kept:>7} {('Y' if it.eligible else 'n'):>4}  {it.reason}"
        )
    s = summarize_inventory(items)
    lines.append("-" * len(hdr))
    lines.append(f"eligible {s['n_eligible']}/{s['n_patients']} | "
                 f"lead seizures {s['total_lead_seizures']} | "
                 f"preictal windows {s['total_preictal_windows']} | "
                 f"interictal kept {s['total_interictal_kept']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test (pure Python; no EDF / SciPy / NumPy / sklearn needed)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.io.summary_parser import EdfFile, Seizure

    print("Running inventory.py self-test ...\n")

    def mk(name, start, end, seizures=()):
        return EdfFile(name=name, start_clock=start, end_clock=end,
                       n_seizures=len(seizures),
                       seizures=[Seizure(s, e) for s, e in seizures])

    # --- clock patient: 8 stitched 1 h files, one lead seizure near the END so
    #     the early hours survive the +/-4 h seizure-exclusion as interictal ---
    #     (a short recording gets fully swallowed by the exclusion -> 0 interictal)
    files = [
        mk("c1", "12:00:00", "13:00:00"),
        mk("c2", "13:00:04", "14:00:04"),
        mk("c3", "14:00:08", "15:00:08"),
        mk("c4", "15:00:12", "16:00:12"),
        mk("c5", "16:00:16", "17:00:16"),
        mk("c6", "17:00:20", "18:00:20"),
        mk("c7", "18:00:24", "19:00:24"),
        mk("c8", "19:00:28", "20:00:28", seizures=[(1800.0, 1860.0)]),  # lead, late
    ]
    summary = PatientSummary("chb99", cfg.FS, ["FP1-F7"], files)
    inv = inventory_from_summary(summary, sop_minutes=30)
    print(inv.to_dict(), "\n")
    assert inv.has_clocks and inv.sized
    assert inv.n_files == 8 and inv.n_segments == 1 and inv.n_hard_breaks == 0
    assert inv.total_seizures == 1 and inv.n_lead_seizures == 1
    assert inv.n_preictal_windows > 0 and inv.n_interictal_kept > 0
    assert inv.eligible and inv.reason == "ok"
    # pool subsampling caps interictal at ~5x preictal
    assert inv.n_interictal_kept <= round(cfg.INTERICTAL_POOL_MULTIPLIER * inv.n_preictal_windows) + 1

    # --- patient with NO seizures -> not eligible ---
    dull = PatientSummary("chb98", cfg.FS, ["FP1-F7"],
                          [mk("d1", "09:00:00", "10:00:00")])
    inv0 = inventory_from_summary(dull)
    assert not inv0.eligible and inv0.n_lead_seizures == 0
    assert inv0.n_preictal_windows == 0

    # --- no-clock patient without sample counts -> unsized, not eligible ---
    ncs = PatientSummary("chb24", cfg.FS, ["FP1-F7"],
                         [mk("n1", None, None, seizures=[(1800.0, 1860.0)])])
    inv_nc = inventory_from_summary(ncs)
    assert not inv_nc.has_clocks and not inv_nc.sized and not inv_nc.eligible
    # ...but pass explicit sample counts (8 h so interictal survives the +/-4 h
    # exclusion around the early seizure) and it sizes + becomes eligible
    inv_nc2 = inventory_from_summary(ncs, file_samples={"n1": 8 * 3600 * cfg.FS})
    assert inv_nc2.sized and inv_nc2.n_preictal_windows > 0
    assert inv_nc2.n_interictal_kept > 0 and inv_nc2.eligible

    # --- min_lead_seizures gate ---
    strict = inventory_from_summary(summary, min_lead_seizures=2)
    assert not strict.eligible and "lead seizure" in strict.reason

    # --- aggregate report ---
    items = [inv, inv0, inv_nc2]
    rep = summarize_inventory(items)
    assert rep["n_patients"] == 3 and rep["n_eligible"] == 2
    eligible = rep["eligible_patients"]
    assert isinstance(eligible, (list, tuple, set))
    assert "chb99" in eligible
    print(format_inventory(items))

    print("\nOK - inventory.py self-test passed.")
