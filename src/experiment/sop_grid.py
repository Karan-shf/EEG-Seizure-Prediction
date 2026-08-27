"""
sop_grid.py
===========
Stage (experiment): the HEADLINE sweep -- alpha x span-roof x LOPO.

The design's headline experiment varies exactly three axes (config section 9):
alpha, span roof, and the leave-one-patient-out fold. This module drives the
first two by calling `lopo.run_lopo` for every (alpha, span_roof) cell and
collecting each cell's pooled LOPO summary into one comparable table. Window
length and SOP stay frozen at their config defaults (sweep those later by
pointing SWEEP_* at the matching *_GRID -- no code change needed here).

Name note: "sop_grid" is the experiment-grid driver (kept for continuity with
the build order). It does NOT sweep SOP by default -- SOP is frozen at
cfg.SOP_PRIMARY_MINUTES; pass a different `sop_minutes` to move the whole grid
to another operating period.

Efficiency: for a fixed alpha, all span roofs reuse ONE provider, so the
expensive fold-invariant artifacts (g_patient, anchors, d_baseline) are built
and cached once per alpha and merely re-sliced per span roof.

Results are written to cfg.RESULTS_DIR as JSON (full per-patient detail + a flat
comparison table). Dependency-light import; running the grid needs the same deps
as lopo.run_lopo. The self-test is dual-mode (structural always; full tiny grid
only when scikit-learn + imbalanced-learn are present).
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from src import config as cfg
from src.utils.logger import get_logger
from src.experiment.lopo import LopoResult, run_lopo

log = get_logger(__name__)

HAVE_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAVE_IMBLEARN = importlib.util.find_spec("imblearn") is not None

_TABLE_METRICS = (
    "pooled_auc", "mean_patient_auc", "mean_event_sensitivity",
    "pooled_event_sensitivity", "pooled_fpr_per_hour", "mean_warning_seconds",
    "n_events", "n_predicted", "n_patients",
)


@dataclass
class GridResult:
    cells: List[LopoResult]
    sop_minutes: int
    classifier_name: str
    target_fpr_per_hour: float

    def table(self) -> List[Dict[str, object]]:
        """One flat row per (alpha, span_roof) cell for easy comparison."""
        rows: List[Dict[str, object]] = []
        for c in self.cells:
            row: Dict[str, object] = {"alpha": c.alpha, "span_roof": c.span_roof}
            for k in _TABLE_METRICS:
                row[k] = c.summary.get(k)
            rows.append(row)
        return rows

    def best(self, metric: str = "pooled_auc", *, maximize: bool = True) -> Optional[LopoResult]:
        """Best cell by a summary metric (ignoring NaN). maximize=False for FPR."""
        scored = [(c.summary.get(metric), c) for c in self.cells]
        scored = [(v, c) for v, c in scored if isinstance(v, (int, float)) and v == v]
        if not scored:
            return None
        return (max if maximize else min)(scored, key=lambda t: t[0])[1]

    def to_dict(self) -> Dict[str, object]:
        return {
            "sop_minutes": self.sop_minutes,
            "classifier_name": self.classifier_name,
            "target_fpr_per_hour": self.target_fpr_per_hour,
            "n_cells": len(self.cells),
            "table": self.table(),
            "cells": [c.to_dict() for c in self.cells],
        }


def _jsonable(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serializable: {type(o)!r}")


def _save_grid(result: GridResult, results_dir: Optional[Path], tag: Optional[str]) -> Path:
    results_dir = Path(results_dir) if results_dir is not None else cfg.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"sop{result.sop_minutes}_{result.classifier_name}"
    path = results_dir / f"lopo_grid_{tag}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, default=_jsonable, indent=2)
    log.info("sop_grid: wrote %d cells -> %s", len(result.cells), path)
    return path


def run_grid(*, alphas: Optional[Sequence[float]] = None,
             span_roofs: Optional[Sequence[int]] = None,
             patients: Optional[Sequence[str]] = None,
             raw_dir: Optional[Path] = None,
             sop_minutes: Optional[int] = None,
             classifier_name: Optional[str] = None,
             target_fpr_per_hour: Optional[float] = None,
             provider_factory: Optional[Callable[[float], object]] = None,
             seed: Optional[int] = None,
             mode: str = "exact",
             save: bool = True,
             results_dir: Optional[Path] = None,
             tag: Optional[str] = None) -> GridResult:
    """Sweep every (alpha, span_roof) cell via lopo.run_lopo and collect results.

    provider_factory(alpha) -> SPD provider lets callers inject data or reuse a
    warmed provider; by default each alpha builds its own ChbSpdProvider inside
    run_lopo. One provider is reused across all span roofs of a given alpha.
    """
    alphas = tuple(cfg.SWEEP_ALPHA if alphas is None else alphas)
    span_roofs = tuple(cfg.SWEEP_SPAN_ROOF if span_roofs is None else span_roofs)
    sop = int(cfg.SOP_PRIMARY_MINUTES if sop_minutes is None else sop_minutes)
    cname = classifier_name or cfg.PRIMARY_CLASSIFIER
    target = float(cfg.PRIMARY_TARGET_FPR_PER_HOUR
                   if target_fpr_per_hour is None else target_fpr_per_hour)

    cells: List[LopoResult] = []
    for a in alphas:
        provider = provider_factory(a) if provider_factory is not None else None
        for m in span_roofs:
            res = run_lopo(alpha=a, span_roof=m, provider=provider, patients=patients,
                           raw_dir=raw_dir, sop_minutes=sop, classifier_name=cname,
                           target_fpr_per_hour=target, seed=seed, mode=mode)
            cells.append(res)
            log.info("grid cell alpha=%.2f m=%d: pooled_auc=%.3f event_sens=%.3f fpr/h=%.3f",
                     a, m, res.summary["pooled_auc"],
                     res.summary["mean_event_sensitivity"],
                     res.summary["pooled_fpr_per_hour"])

    result = GridResult(cells, sop, cname, target)
    if save:
        _save_grid(result, results_dir, tag)
    return result


def format_grid(result: GridResult) -> str:
    hdr = (f"{'alpha':>5} {'m':>2} {'pool_auc':>8} {'mean_auc':>8} "
           f"{'ev_sens':>7} {'fpr/h':>6} {'warn_s':>7}")
    lines = [hdr, "-" * len(hdr)]
    for row in result.table():
        def _f(x, nd=3):
            return f"{x:.{nd}f}" if isinstance(x, (int, float)) and x == x else "  nan"
        lines.append(
            f"{row['alpha']:>5.2f} {row['span_roof']:>2} "
            f"{_f(row['pooled_auc']):>8} {_f(row['mean_patient_auc']):>8} "
            f"{_f(row['mean_event_sensitivity']):>7} "
            f"{_f(row['pooled_fpr_per_hour'], 2):>6} "
            f"{_f(row['mean_warning_seconds'], 0):>7}"
        )
    best = result.best("pooled_auc")
    if best is not None:
        lines.append("-" * len(hdr))
        lines.append(f"best pooled_auc: alpha={best.alpha:.2f} m={best.span_roof} "
                     f"({best.summary['pooled_auc']:.3f})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test (dual-mode)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    import shutil

    print("Running sop_grid.py self-test ...\n")

    # --- structural: GridResult table / best / to_dict / save over synthetic cells ---
    def _mk(alpha, m, auc, sens):
        summary = {
            "n_patients": 2, "pooled_auc": auc, "mean_patient_auc": auc,
            "mean_event_sensitivity": sens, "pooled_event_sensitivity": sens,
            "n_events": 2, "n_predicted": int(round(2 * sens)),
            "pooled_fpr_per_hour": 0.12, "total_false_alarms": 1,
            "total_interictal_hours": 8.0, "mean_warning_seconds": 300.0,
        }
        return LopoResult(alpha, m, cfg.SOP_PRIMARY_MINUTES, "lr", 0.15, [], summary, {})

    gr = GridResult([_mk(0.0, 5, 0.60, 0.5), _mk(1.0, 5, 0.72, 0.8)],
                    cfg.SOP_PRIMARY_MINUTES, "lr", 0.15)
    tbl = gr.table()
    assert len(tbl) == 2 and {"alpha", "span_roof", "pooled_auc"} <= set(tbl[0])
    best_pooled_auc = gr.best("pooled_auc")
    assert best_pooled_auc is not None
    assert best_pooled_auc.alpha == 1.0
    best_pooled_fpr_per_hour = gr.best("pooled_fpr_per_hour", maximize=False)
    assert best_pooled_fpr_per_hour is not None
    assert best_pooled_fpr_per_hour.alpha in (0.0, 1.0)
    payload = json.dumps(gr.to_dict(), default=_jsonable)
    assert json.loads(payload)["n_cells"] == 2
    print(format_grid(gr), "\n")

    tmp = Path(tempfile.mkdtemp(prefix="sh_grid_test_"))
    saved = _save_grid(gr, tmp, tag="selftest")
    assert saved.exists()
    reloaded = json.loads(saved.read_text())
    assert reloaded["n_cells"] == 2 and len(reloaded["table"]) == 2

    if HAVE_SKLEARN and HAVE_IMBLEARN:
        from src.data import cache

        rng = np.random.default_rng(cfg.SEED)
        nc, sr, dim = 3, 3, 4
        eye = np.eye(dim)

        def _spd(shift=0.0):
            A = rng.standard_normal((nc, sr, dim, dim))
            return A @ np.swapaxes(A, -1, -2) + (dim + shift) * eye

        class _Meta:
            __slots__ = ("seg_index", "label")
            def __init__(self, seg_index, label):
                self.seg_index = seg_index
                self.label = label

        class _MemProvider:
            def __init__(self, data, channels):
                self._data = data
                self._channels = tuple(channels)
            def patient_ids(self):
                return list(self._data)
            def channels(self):
                return self._channels
            def iter_windows(self, patient_id):
                for C, lab, _seg in self._data[patient_id]:
                    yield np.array(C, dtype=float), int(lab)
            def window_meta(self, patient_id):
                return [_Meta(seg, lab) for _C, lab, seg in self._data[patient_id]]

        data = {}
        for p in ("A", "B", "C", "D"):
            windows = []
            windows += [(_spd(), 0, 0) for _ in range(6)]
            windows += [(_spd(1.0), 1, 0) for _ in range(3)]
            windows += [(_spd(), 0, 1) for _ in range(6)]
            windows += [(_spd(1.0), 1, 1) for _ in range(3)]
            data[p] = windows
        provider = _MemProvider(data, tuple(f"c{i}" for i in range(nc)))

        cfg.CACHE_DIR = Path(tempfile.mkdtemp(prefix="sh_grid_ds_"))
        cfg.CACHE_ENABLED = True
        cache._MEM.clear()

        res = run_grid(alphas=(0.5,), span_roofs=(sr,),
                       provider_factory=lambda _a: provider,
                       target_fpr_per_hour=1e9, save=False)
        assert len(res.cells) == 1
        assert res.cells[0].summary["n_patients"] == 4
        assert len(res.table()) == 1
        shutil.rmtree(cfg.CACHE_DIR, ignore_errors=True)
        print("  full integration OK (sklearn + imbalanced-learn present)")
    else:
        print("  sklearn/imbalanced-learn absent -> structural checks only")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nOK - sop_grid.py self-test passed.")
