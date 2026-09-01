"""
run.py
======
Top-level command-line ENTRY POINT for the SeizureHorizon rebuild.

Everything below `run.py` is a library; this file is the one place a human (or a
shell script / SLURM job) invokes. It stitches the experiment layer together:

    inventory  ->  (eligibility gate)  ->  lopo / grid  ->  results JSON

Subcommands
-----------
  config      Print the resolved configuration (cfg.summary()).
  inventory   Fast, EDF-free cohort census + LOPO eligibility table.
  lopo        One leave-one-patient-out run at a single (alpha, span_roof).
  grid        The headline alpha x span-roof x LOPO sweep -> results JSON.
  all         inventory (eligibility gate) -> grid over the ELIGIBLE patients.
  selftest    Dependency-light plumbing self-test (no EDF / sklearn / imblearn).

Examples
--------
  cd /data && python3 -m src.run config
  cd /data && python3 -m src.run inventory --probe
  cd /data && python3 -m src.run lopo  --alpha 0.5 --span-roof 5
  cd /data && python3 -m src.run grid  --alphas 0 0.5 1 --span-roofs 3 5 7
  cd /data && python3 -m src.run all   --probe --target-fpr 0.15
  cd /data && python3 -m src.run selftest

Patient selectors accept explicit ids and inclusive ranges, e.g.
  --patients chb01 chb03 chb05
  --patients chb01..chb12 chb20
Omitting --patients lets the drivers default to every patient found on disk
(inventory) or chb01..chb24 (lopo/grid).

Actually running lopo/grid needs the full stack (mne + scipy for the provider,
scikit-learn for the classifier, imbalanced-learn for balancing) and the raw
EDFs under cfg.RAW_DIR. `config`, `inventory` (without --probe) and `selftest`
are dependency-light.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from src import config as cfg
from src.utils.logger import get_logger
from src.experiment import inventory as inventory_mod
from src.experiment import lopo as lopo_mod
from src.experiment import sop_grid as grid_mod
from src.experiment import parallel_build

log = get_logger("src.run")


# ===========================================================================
# Patient selectors
# ===========================================================================
def _expand_patients(tokens: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Expand a --patients selector into an explicit ordered id list.

    None -> None (let the driver choose its default). Each token is either a
    plain id ("chb07") or an inclusive numeric range ("chb01..chb12"). Order is
    preserved and duplicates removed.
    """
    if tokens is None:
        return None
    out: List[str] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if ".." in tok:
            lo, hi = tok.split("..", 1)
            try:
                a, b = int(lo[3:]), int(hi[3:])
            except ValueError:
                raise SystemExit(f"bad patient range {tok!r} (expected chbNN..chbMM)")
            prefix = lo[:3] or "chb"
            step = 1 if b >= a else -1
            for i in range(a, b + step, step):
                out.append(f"{prefix}{i:02d}")
        else:
            out.append(tok)
    # de-dupe, preserve order
    seen: set = set()
    uniq = [p for p in out if not (p in seen or seen.add(p))]
    return uniq


def _filter_eligible(items) -> List[str]:
    """Eligible patient ids (in inventory order) from a list of PatientInventory."""
    return [it.patient_id for it in items if it.eligible]


# ===========================================================================
# Subcommand handlers
# ===========================================================================
def _run_config(args: argparse.Namespace) -> int:
    print(cfg.summary())
    return 0


def _run_inventory(args: argparse.Namespace) -> int:
    patients = _expand_patients(args.patients)
    items = inventory_mod.build_inventory(
        patients, raw_dir=args.raw_dir, probe=args.probe,
        min_lead_seizures=args.min_lead)
    if not items:
        log.warning("no patients inventoried (is --raw-dir correct? %s)",
                    args.raw_dir or cfg.RAW_DIR)
        return 1
    print(inventory_mod.format_inventory(items))
    return 0


def _resolve_eligible(args: argparse.Namespace) -> Optional[List[str]]:
    """Run the inventory, print it, and return the eligible ids (or None if the
    census turned up nothing so the driver falls back to its own default)."""
    patients = _expand_patients(args.patients)
    items = inventory_mod.build_inventory(
        patients, raw_dir=args.raw_dir, probe=args.probe,
        min_lead_seizures=args.min_lead)
    if not items:
        log.warning("eligibility census empty; using driver default patient set")
        return None
    print(inventory_mod.format_inventory(items))
    eligible = _filter_eligible(items)
    if not eligible:
        raise SystemExit("no eligible patients -- nothing to run")
    log.info("eligibility gate: %d/%d patients eligible", len(eligible), len(items))
    return eligible


def _run_precompute(args: argparse.Namespace) -> int:
    patients = _resolve_eligible(args) if args.eligible_only else _expand_patients(args.patients)
    pats = patients if patients else list(lopo_mod.DEFAULT_PATIENTS)
    alphas = tuple(cfg.SWEEP_ALPHA if args.alphas is None else args.alphas)
    print(f"precompute: {len(pats)} patient(s) x {len(alphas)} alpha(s) "
          f"{list(alphas)}, workers={args.n_workers or '(auto)'}")
    for a in alphas:
        print(f"\n=== precompute alpha={a:.2f} ===")
        out = parallel_build.run_precompute_parallel(
            pats, alpha=a, raw_dir=args.raw_dir, sop_minutes=args.sop,
            n_workers=args.n_workers)
        n1_ok = sum(r.ok for r in out["level1"])
        n2_ok = sum(r.ok for r in out["level2"])
        print(f"  level-1: {n1_ok}/{len(out['level1'])} OK   "
              f"level-2: {n2_ok}/{len(out['level2'])} OK")
        if n1_ok < len(out["level1"]) or n2_ok < len(out["level2"]):
            log.error("precompute alpha=%.2f: incomplete -- see logged errors above", a)
            return 1
    print("\nprecompute complete. Run e.g.:")
    print("  python3 -m src.run grid --mode precomputed --eligible-only --probe")
    return 0


def _print_lopo(res) -> None:
    s = res.summary

    def g(key, nd=3):
        v = s.get(key)
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) and v == v else "nan"

    print(f"\nLOPO  alpha={res.alpha:.2f}  span_roof={res.span_roof}  "
          f"SOP={res.sop_minutes}min  clf={res.classifier_name}  "
          f"target_fpr/h={res.target_fpr_per_hour}")
    hdr = f"{'patient':<8} {'auc':>6} {'ev_sens':>7} {'fpr/h':>6} {'warn_s':>7} {'thr':>5}"
    print(hdr)
    print("-" * len(hdr))
    for r in res.per_patient:
        ev = r.get("event", {})
        def f(x, nd=3):
            return f"{x:.{nd}f}" if isinstance(x, (int, float)) and x == x else "nan"
        print(f"{str(r.get('patient_id','?')):<8} {f(r.get('auc')):>6} "
              f"{f(ev.get('sensitivity')):>7} {f(ev.get('fpr_per_hour'),2):>6} "
              f"{f(ev.get('mean_warning_seconds'),0):>7} {f(r.get('threshold'),2):>5}")
    print("-" * len(hdr))
    print(f"patients={s.get('n_patients','?')}  pooled_auc={g('pooled_auc')}  "
          f"mean_patient_auc={g('mean_patient_auc')}  "
          f"mean_event_sens={g('mean_event_sensitivity')}  "
          f"pooled_event_sens={g('pooled_event_sensitivity')}  "
          f"pooled_fpr/h={g('pooled_fpr_per_hour',2)}")


def _run_lopo(args: argparse.Namespace) -> int:
    patients = _resolve_eligible(args) if args.eligible_only else _expand_patients(args.patients)
    if getattr(args, "validate_anchors", False):
        return _validate_anchors(args, patients)
    res = lopo_mod.run_lopo(
        alpha=args.alpha, span_roof=args.span_roof, patients=patients,
        raw_dir=args.raw_dir, sop_minutes=args.sop,
        classifier_name=args.classifier, target_fpr_per_hour=args.target_fpr,
        seed=args.seed, mode=args.mode)
    _print_lopo(res)
    return 0


def _validate_anchors(args: argparse.Namespace, patients) -> int:
    """Run the SAME LOPO both ways (exact vs fast/purist) and print a
    side-by-side metric comparison, so the anchor approximation can be trusted
    before committing to the full grid."""
    common = dict(
        alpha=args.alpha, span_roof=args.span_roof, patients=patients,
        raw_dir=args.raw_dir, sop_minutes=args.sop,
        classifier_name=args.classifier, target_fpr_per_hour=args.target_fpr,
        seed=args.seed)
    log.info("validate-anchors: running EXACT mode ...")
    exact = lopo_mod.run_lopo(mode="exact", **common)
    log.info("validate-anchors: running FAST (purist) mode ...")
    fast = lopo_mod.run_lopo(mode="fast", **common)

    metrics = ("pooled_auc", "mean_patient_auc", "mean_event_sensitivity",
               "pooled_event_sensitivity", "pooled_fpr_per_hour", "mean_warning_seconds")

    def _fmt(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) and x == x else "nan"

    hdr = f"{'metric':<26} {'exact':>10} {'fast':>10} {'delta':>10}"
    print(f"\nAnchor-mode validation  alpha={args.alpha}  span_roof={args.span_roof}")
    print(hdr)
    print("-" * len(hdr))
    for k in metrics:
        ve, vf = exact.summary.get(k), fast.summary.get(k)
        both_num = all(isinstance(v, (int, float)) and v == v for v in (ve, vf))
        dv = (vf - ve) if both_num else float("nan") # type: ignore
        print(f"{k:<26} {_fmt(ve):>10} {_fmt(vf):>10} {_fmt(dv):>10}")
    print("-" * len(hdr))
    print("Small deltas (esp. pooled_auc / event sensitivity) => the purist "
          "speedup is safe to adopt for the grid.")
    return 0


def _print_grid(grid) -> None:
    print(grid_mod.format_grid(grid))
    for metric, maximize in (("pooled_auc", True), ("mean_event_sensitivity", True),
                             ("pooled_fpr_per_hour", False)):
        best = grid.best(metric, maximize=maximize)
        if best is not None:
            val = best.summary.get(metric)
            val = f"{val:.3f}" if isinstance(val, (int, float)) and val == val else "nan"
            print(f"best {metric:<24}: alpha={best.alpha:.2f} m={best.span_roof} ({val})")


def _run_grid(args: argparse.Namespace) -> int:
    patients = _resolve_eligible(args) if args.eligible_only else _expand_patients(args.patients)
    grid = grid_mod.run_grid(
        alphas=args.alphas, span_roofs=args.span_roofs, patients=patients,
        raw_dir=args.raw_dir, sop_minutes=args.sop,
        classifier_name=args.classifier, target_fpr_per_hour=args.target_fpr,
        seed=args.seed, mode=args.mode, save=not args.no_save,
        results_dir=args.results_dir, tag=args.tag)
    _print_grid(grid)
    if not args.no_save:
        print(f"\nresults JSON written under {args.results_dir or cfg.RESULTS_DIR}")
    return 0


def _run_all(args: argparse.Namespace) -> int:
    # Force the eligibility gate on, then run the full sweep over survivors.
    args.eligible_only = True
    return _run_grid(args)


# ===========================================================================
# Argument parser
# ===========================================================================
def _add_common(sp: argparse.ArgumentParser, *, with_eligible: bool) -> None:
    sp.add_argument("--patients", nargs="+", metavar="ID",
                    help="patient ids and/or ranges (chb01..chb12); default: all")
    sp.add_argument("--raw-dir", type=Path, default=None,
                    help=f"raw CHB-MIT dir (default: {cfg.RAW_DIR})")
    sp.add_argument("--sop", type=int, default=None,
                    help=f"SOP minutes (default: {cfg.SOP_PRIMARY_MINUTES})")
    sp.add_argument("--classifier", default=None,
                    help=f"classifier name (default: {cfg.PRIMARY_CLASSIFIER})")
    sp.add_argument("--target-fpr", type=float, default=None, dest="target_fpr",
                    help=f"target FPR/h (default: {cfg.PRIMARY_TARGET_FPR_PER_HOUR})")
    sp.add_argument("--seed", type=int, default=None,
                    help=f"random seed (default: {cfg.SEED})")
    sp.add_argument("--mode", choices=("exact", "fast", "precomputed"), default=cfg.FEATURE_ANCHOR_MODE,
                    help="feature-anchor mode: 'fast' = Tier-1 purist cached global "
                         "anchor (~10x less streaming, small approximation); "
                         "'precomputed' = exact per-fold leave-one-out anchors for "
                         "EVERY fold, each patient streamed exactly once for the "
                         "whole sweep -- same speed as 'fast' with zero approximation; "
                         f"default: {cfg.FEATURE_ANCHOR_MODE}")
    if with_eligible:
        sp.add_argument("--eligible-only", action="store_true",
                        help="run inventory first and keep only eligible patients")
        sp.add_argument("--probe", action="store_true",
                        help="probe EDF headers to size no-clock patients (chb24)")
        sp.add_argument("--min-lead", type=int,
                        default=inventory_mod.MIN_LEAD_SEIZURES,
                        help="min lead seizures for eligibility")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m src.run",
        description="SeizureHorizon top-level runner (inventory -> lopo/grid -> results).")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("config", help="print resolved configuration")

    inv = sub.add_parser("inventory", help="cohort census + LOPO eligibility")
    inv.add_argument("--patients", nargs="+", metavar="ID",
                     help="patient ids and/or ranges; default: all on disk")
    inv.add_argument("--raw-dir", type=Path, default=None)
    inv.add_argument("--sop", type=int, default=None)
    inv.add_argument("--probe", action="store_true",
                     help="probe EDF headers to size no-clock patients (chb24)")
    inv.add_argument("--min-lead", type=int, default=inventory_mod.MIN_LEAD_SEIZURES)

    pre = sub.add_parser("precompute",
                         help="Tier-2: parallel level-1 + level-2 cache build "
                              "(run BEFORE lopo/grid --mode precomputed)")
    pre.add_argument("--patients", nargs="+", metavar="ID",
                     help="patient ids and/or ranges; default: all")
    pre.add_argument("--raw-dir", type=Path, default=None)
    pre.add_argument("--sop", type=int, default=None)
    pre.add_argument("--alphas", nargs="+", type=float, default=None,
                     help=f"default: {list(cfg.SWEEP_ALPHA)}")
    pre.add_argument("--n-workers", type=int, default=None, dest="n_workers",
                     help="override src.utils.parallel.resolve_n_workers()'s "
                          "auto-detected worker count")
    pre.add_argument("--eligible-only", action="store_true",
                     help="run inventory first and keep only eligible patients")
    pre.add_argument("--probe", action="store_true",
                     help="probe EDF headers to size no-clock patients (chb24)")
    pre.add_argument("--min-lead", type=int, default=inventory_mod.MIN_LEAD_SEIZURES,
                     help="min lead seizures for eligibility")

    lo = sub.add_parser("lopo", help="single (alpha, span_roof) LOPO run")
    lo.add_argument("--alpha", type=float, required=True)
    lo.add_argument("--span-roof", type=int, default=None, dest="span_roof")
    lo.add_argument("--validate-anchors", action="store_true", dest="validate_anchors",
                    help="run BOTH exact and fast modes and print a side-by-side "
                         "comparison (trust the purist speedup before the full grid)")
    _add_common(lo, with_eligible=True)

    gr = sub.add_parser("grid", help="alpha x span-roof x LOPO sweep")
    gr.add_argument("--alphas", nargs="+", type=float, default=None,
                    help=f"default: {list(cfg.SWEEP_ALPHA)}")
    gr.add_argument("--span-roofs", nargs="+", type=int, default=None,
                    dest="span_roofs", help=f"default: {list(cfg.SWEEP_SPAN_ROOF)}")
    gr.add_argument("--tag", default=None, help="results filename tag")
    gr.add_argument("--results-dir", type=Path, default=None, dest="results_dir")
    gr.add_argument("--no-save", action="store_true", help="do not write results JSON")
    _add_common(gr, with_eligible=True)

    al = sub.add_parser("all", help="eligibility gate -> full grid over survivors")
    al.add_argument("--alphas", nargs="+", type=float, default=None)
    al.add_argument("--span-roofs", nargs="+", type=int, default=None, dest="span_roofs")
    al.add_argument("--tag", default=None)
    al.add_argument("--results-dir", type=Path, default=None, dest="results_dir")
    al.add_argument("--no-save", action="store_true")
    _add_common(al, with_eligible=True)

    sub.add_parser("selftest", help="dependency-light plumbing self-test")
    return p


_DISPATCH = {
    "config": _run_config,
    "inventory": _run_inventory,
    "precompute": _run_precompute,
    "lopo": _run_lopo,
    "grid": _run_grid,
    "all": _run_all,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "selftest":
        return _selftest()
    cfg.ensure_dirs()
    return _DISPATCH[args.command](args)


# ===========================================================================
# Self-test (dependency-light: patches the drivers, so no EDF / sklearn needed)
# ===========================================================================
def _selftest() -> int:
    print("Running run.py self-test ...\n")

    # --- patient selector parsing ---
    assert _expand_patients(None) is None
    assert _expand_patients(["chb01", "chb03"]) == ["chb01", "chb03"]
    assert _expand_patients(["chb01..chb03"]) == ["chb01", "chb02", "chb03"]
    assert _expand_patients(["chb03..chb01"]) == ["chb03", "chb02", "chb01"]
    assert _expand_patients(["chb01..chb02", "chb02", "chb05"]) == ["chb01", "chb02", "chb05"]

    # --- eligibility filter over real PatientInventory objects ---
    def mk_inv(pid, eligible):
        return inventory_mod.PatientInventory(
            patient_id=pid, has_clocks=True, sized=True, n_files=1, n_segments=1,
            n_hard_breaks=0, recorded_hours=8.0, total_seizures=1, n_lead_seizures=1,
            n_preictal_windows=10, n_interictal_kept=(50 if eligible else 0),
            n_interictal_total=50, n_dropped=0, eligible=eligible,
            reason=("ok" if eligible else "no interictal windows"))
    items = [mk_inv("chb01", True), mk_inv("chb02", False), mk_inv("chb03", True)]
    assert _filter_eligible(items) == ["chb01", "chb03"]

    # --- parser wiring ---
    p = _build_parser()
    ns = p.parse_args(["grid", "--alphas", "0", "0.5", "--span-roofs", "3", "5",
                       "--no-save", "--patients", "chb01..chb03"])
    assert ns.command == "grid" and ns.alphas == [0.0, 0.5] and ns.span_roofs == [3, 5]
    assert ns.no_save is True
    ns2 = p.parse_args(["lopo", "--alpha", "0.75", "--span-roof", "7"])
    assert ns2.alpha == 0.75 and ns2.span_roof == 7 and ns2.eligible_only is False

    # --- dispatch wiring: patch the drivers so nothing heavy runs ---
    calls = {}
    orig = (grid_mod.run_grid, lopo_mod.run_lopo, inventory_mod.build_inventory,
            parallel_build.run_precompute_parallel)
    try:
        def fake_grid(**kw):
            calls["grid"] = kw
            return grid_mod.GridResult([], kw.get("sop_minutes") or cfg.SOP_PRIMARY_MINUTES,
                                       kw.get("classifier_name") or cfg.PRIMARY_CLASSIFIER,
                                       kw.get("target_fpr_per_hour") or cfg.PRIMARY_TARGET_FPR_PER_HOUR)

        def fake_lopo(**kw):
            calls["lopo"] = kw
            summ = {"n_patients": 0, "pooled_auc": float("nan"),
                    "mean_patient_auc": float("nan"), "mean_event_sensitivity": float("nan"),
                    "pooled_event_sensitivity": float("nan"),
                    "pooled_fpr_per_hour": float("nan"), "mean_warning_seconds": float("nan")}
            return lopo_mod.LopoResult(
                kw["alpha"], kw.get("span_roof") or cfg.SPAN_MAX,
                kw.get("sop_minutes") or cfg.SOP_PRIMARY_MINUTES,
                kw.get("classifier_name") or cfg.PRIMARY_CLASSIFIER,
                kw.get("target_fpr_per_hour") or cfg.PRIMARY_TARGET_FPR_PER_HOUR,
                [], summ, {})

        def fake_inv(patients=None, **kw):
            calls["inv"] = {"patients": patients, **kw}
            return items  # chb01 + chb03 eligible

        def fake_precompute(patients, *, alpha, **kw):
            calls.setdefault("precompute_alphas", []).append(alpha)
            calls["precompute_patients"] = patients
            return {"level1": [], "level2": []}

        grid_mod.run_grid = fake_grid
        lopo_mod.run_lopo = fake_lopo
        inventory_mod.build_inventory = fake_inv
        parallel_build.run_precompute_parallel = fake_precompute

        # grid, explicit patients, no eligibility gate
        assert main(["grid", "--alphas", "0.5", "--span-roofs", "3",
                     "--patients", "chb01..chb03", "--no-save"]) == 0
        assert calls["grid"]["alphas"] == [0.5] and calls["grid"]["span_roofs"] == [3]
        assert calls["grid"]["save"] is False
        assert calls["grid"]["patients"] == ["chb01", "chb02", "chb03"]

        # lopo with eligibility gate -> only eligible patients passed through
        assert main(["lopo", "--alpha", "0.5", "--eligible-only", "--probe"]) == 0
        assert calls["lopo"]["alpha"] == 0.5
        assert calls["lopo"]["patients"] == ["chb01", "chb03"]

        # all -> forces eligibility gate, drives the grid over survivors
        calls.pop("grid", None)
        assert main(["all", "--alphas", "1.0", "--span-roofs", "9", "--no-save"]) == 0
        assert calls["grid"]["patients"] == ["chb01", "chb03"]

        # precompute: loops run_precompute_parallel once per alpha
        assert main(["precompute", "--alphas", "0.5", "1.0",
                     "--patients", "chb01..chb02"]) == 0
        assert calls["precompute_alphas"] == [0.5, 1.0]
        assert calls["precompute_patients"] == ["chb01", "chb02"]

        # config prints without touching the drivers
        assert main(["config"]) == 0
    finally:
        (grid_mod.run_grid, lopo_mod.run_lopo, inventory_mod.build_inventory,
         parallel_build.run_precompute_parallel) = orig

    print("\nOK - run.py self-test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
