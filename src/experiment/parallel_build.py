"""
parallel_build.py
==================
Tier-2 CPU multiprocessing orchestration: the patient-parallel level-1
(per-patient anchors) and level-2 (all-fold precomputed features) sweeps.

Two SEPARATE process pools, run in strict sequence, with a hard barrier
between them:

    pool 1 (level-1, ALL patients in parallel)
        |
        |  <- barrier: every patient's g_patient / d_baseline /
        |     patient_anchor_means must be cached before level-2 can start,
        |     since level-2's fold anchors are built by averaging EVERY
        |     patient's level-1 result. Enforced in code, not just by
        |     convention: run_precompute_parallel() raises if any level-1
        |     patient failed, refusing to start level-2 on incomplete anchors.
        v
    build_all_fold_references()   <- sequential, MAIN process, cheap (only
                                      averages already-cached small matrices,
                                      never touches raw windows)
        |
        v
    pool 2 (level-2, ALL patients in parallel)

Design choices and why (full locks/race/starvation analysis was worked
through in conversation before this was written):
  * Each worker builds its OWN single-patient ChbSpdProvider locally -- no
    provider object crosses the process boundary. Only plain, cheaply-
    picklable arguments (patient id, alpha, raw_dir, sop_minutes, the
    fingerprint/tag strings) are passed to a worker.
  * Workers are dispatched with imap_unordered + chunksize=1: patients vary
    hugely in window count (chb12: 40 seizures; others: 3), so static equal
    chunking would starve idle workers behind one unlucky worker stuck with
    the heavy patients. chunksize=1 means a worker grabs the next available
    patient the INSTANT it finishes its current one.
  * COMPUTE_BACKEND is forced to "cpu" INSIDE each worker's own process, via
    _worker_init. This is NOT optional under Windows' spawn: a runtime
    mutation made in the PARENT process (e.g. `cfg.COMPUTE_BACKEND = "cpu"`
    typed in a script before creating the Pool) is INVISIBLE to spawned
    children -- spawn re-imports config.py fresh from disk in every child,
    getting the FILE's declared default, not the parent's in-memory state.
    It must be set from inside the child itself, which is exactly what
    _worker_init (passed as the Pool's `initializer`) does.
  * BOTH pools are kept CPU-only in this version. Level-1's spd_log is
    eigh-based, where GPU measurably LOST to CPU on this hardware (5.6s vs
    3.0s at the earlier per-window benchmark scale) -- clear-cut. Level-2's
    JBLD distances is the step GPU measurably WON at (0.30s vs 0.73s,
    full-pipeline float32) -- but that was measured at 2 references/window,
    not the 48 references/window level-2 actually batches, and mixing GPU
    into a multi-worker pool reintroduces the single-shared-device
    contention question that was deliberately sidestepped rather than
    solved. CPU-only-both-pools is the safe, already-reasoned-through
    default to ship first; level-2's backend choice is a DEFERRED decision,
    worth a dedicated benchmark once it is running for real, not something
    to assume settled by this comment.
  * BLAS threads are pinned to 1 inside each worker BEFORE numpy is first
    imported there. This module's OWN top-level imports (config, logger,
    parallel utils) are deliberately numpy-free so that _worker_init runs
    before any numpy import happens in a fresh worker process; the
    numpy-pulling imports (ChbSpdProvider, dataset_builder) are deferred to
    INSIDE the job functions for exactly this reason -- moving them to this
    module's top level would defeat the pinning.
  * Each job is wrapped in try/except and returns a structured JobResult
    rather than letting an exception propagate: with imap_unordered, an
    unhandled exception in one task kills the whole pool iteration, which
    would be a bad way to discover a problem 6 hours into an unattended run.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src import config as cfg
from src.utils.logger import get_logger
from src.utils.parallel import resolve_n_workers, pin_blas_threads

log = get_logger(__name__)


@dataclass
class JobResult:
    patient_id: str
    ok: bool
    seconds: float
    n_windows: int = 0
    error: Optional[str] = None


def _worker_init() -> None:
    """Runs ONCE per worker process, before any job is dispatched to it.
    MUST pin BLAS threads and force the CPU backend before this process's
    first numpy import (see module docstring)."""
    pin_blas_threads(1)
    cfg.COMPUTE_BACKEND = "cpu"


def _level1_job(args) -> JobResult:
    """One patient's level-1 anchor build (g_patient + the merged d_baseline
    / patient_anchor_means pass). Runs inside a worker process.

    Explicit PID-tagged START/END log lines exist because, with only END
    times, "started together and finished at naturally different times
    because patients are different sizes" and "ran strictly one after
    another" are IMPOSSIBLE to tell apart from the outside. Don't eyeball
    that distinction from completion timestamps alone -- check these START
    lines (or Task Manager's process list) if parallelism is ever in doubt.
    """
    patient_id, alpha, raw_dir, sop_minutes, fingerprint = args
    pid = os.getpid()
    t0 = time.perf_counter()
    log.info("level1 START patient=%s pid=%d", patient_id, pid)
    try:
        # Deferred imports: guarantees _worker_init's env-var pinning ran
        # before numpy is first imported in THIS process.
        from src.experiment.lopo import ChbSpdProvider
        from src.data import dataset_builder as db

        provider = ChbSpdProvider([patient_id], alpha=alpha, raw_dir=raw_dir,
                                  sop_minutes=sop_minutes)
        db.patient_g(provider, patient_id, fingerprint=fingerprint)
        baseline, _means = db._pass2_artifacts(provider, patient_id, fingerprint=fingerprint)
        elapsed = time.perf_counter() - t0
        log.info("level1 END   patient=%s pid=%d elapsed=%.1fs windows=%d",
                 patient_id, pid, elapsed, len(baseline.labels))
        return JobResult(patient_id, True, elapsed, n_windows=len(baseline.labels))
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the pool
        elapsed = time.perf_counter() - t0
        log.error("level1 FAILED patient=%s pid=%d elapsed=%.1fs error=%r",
                  patient_id, pid, elapsed, exc)
        return JobResult(patient_id, False, elapsed, error=repr(exc))


def _level2_job(args) -> JobResult:
    """One patient's level-2 all-fold feature stream. Runs inside a worker
    process. Assumes level-1 AND build_all_fold_references are already
    cached on disk (the barrier run_level2_parallel enforces before
    dispatching this) -- the build_all_fold_references call below is
    therefore expected to be a cache HIT (a disk read), never a recompute,
    regardless of which single patient this worker's own provider knows
    about (ChbSpdProvider resolves any patient id by path directly; it does
    not gate iter_windows on what was passed to its constructor).

    NOTE: no span_roof here on purpose. patient_all_fold_features always
    caches the distance tensor at the FULL SPAN_MAX -- span roof is a cheap
    slice applied later, per fold, in dataset_builder.build_fold_precomputed.

    Also no longer calls build_all_fold_references itself: that already ran
    (per-fold, one at a time) in the main process before this pool spawned
    (see run_level2_parallel), so every fold reference this worker needs is
    already an individual, cheap cache hit -- loaded lazily, group-at-a-time,
    inside patient_all_fold_features -> _patient_all_fold_distance_tensor."""
    patient_id, alpha, raw_dir, sop_minutes, fingerprint, tag, patient_ids = args
    pid = os.getpid()
    t0 = time.perf_counter()
    log.info("level2 START patient=%s pid=%d", patient_id, pid)
    try:
        from src.experiment.lopo import ChbSpdProvider
        from src.data import dataset_builder as db

        provider = ChbSpdProvider([patient_id], alpha=alpha, raw_dir=raw_dir,
                                  sop_minutes=sop_minutes)
        fold_order = tuple(patient_ids)
        feats = db.patient_all_fold_features(provider, patient_id, fold_order=fold_order,
                                             fingerprint=fingerprint, tag=tag)
        elapsed = time.perf_counter() - t0
        log.info("level2 END   patient=%s pid=%d elapsed=%.1fs windows=%d",
                 patient_id, pid, elapsed, len(feats["y"]))
        return JobResult(patient_id, True, elapsed, n_windows=len(feats["y"]))
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        log.error("level2 FAILED patient=%s pid=%d elapsed=%.1fs error=%r",
                  patient_id, pid, elapsed, exc)
        return JobResult(patient_id, False, elapsed, error=repr(exc))


def _run_pool(job_fn, job_args: List[tuple], *, n_workers: Optional[int] = None) -> List[JobResult]:
    n = n_workers if n_workers is not None else resolve_n_workers()
    log.info("parallel pool: %d job(s) across %d worker(s)", len(job_args), n)

    # Pin BLAS threads HERE, in the PARENT, before any child is spawned --
    # not just inside _worker_init. Spawned children inherit the parent's
    # OS-level environment at creation time, so this guarantees every
    # worker sees the pinned values from its very first bootstrap step,
    # before anything in that child (including run.py's own import chain
    # re-executing under spawn) can import numpy and lock in BLAS's thread
    # count first. _worker_init's own call is kept too, as a second layer
    # for any code path that creates a Pool without going through here.
    pin_blas_threads(1)

    ctx = get_context("spawn")   # explicit: the correct default on Windows,
                                  # and explicit beats relying on whatever a
                                  # different OS's platform default happens
                                  # to be
    results: List[JobResult] = []
    with ctx.Pool(processes=n, initializer=_worker_init) as pool:
        for res in pool.imap_unordered(job_fn, job_args, chunksize=1):
            status = "OK" if res.ok else f"FAILED: {res.error}"
            log.info("  %s: %s (%.1fs, %d windows)",
                     res.patient_id, status, res.seconds, res.n_windows)
            results.append(res)
    return results


def run_level1_parallel(patients: Sequence[str], *, alpha: float,
                        raw_dir: Optional[Path] = None,
                        sop_minutes: Optional[int] = None,
                        n_workers: Optional[int] = None) -> List[JobResult]:
    """Level-1: build every patient's g_patient + d_baseline + anchor means,
    in parallel. Must fully complete (every patient OK) before level-2 runs.
    """
    from src.experiment.lopo import _alpha_fingerprint
    fp = _alpha_fingerprint(alpha)
    job_args = [(p, alpha, raw_dir, sop_minutes, fp) for p in patients]

    t0 = time.perf_counter()
    results = _run_pool(_level1_job, job_args, n_workers=n_workers)
    elapsed = time.perf_counter() - t0

    failed = [r for r in results if not r.ok]
    log.info("level-1 parallel: %d/%d patients OK in %.1fs (%.1f min)",
             len(results) - len(failed), len(results), elapsed, elapsed / 60)
    if failed:
        log.error("level-1 parallel: %d patient(s) FAILED: %s",
                  len(failed), [r.patient_id for r in failed])
    return results


def run_level2_parallel(patients: Sequence[str], *, alpha: float,
                        raw_dir: Optional[Path] = None,
                        sop_minutes: Optional[int] = None,
                        n_workers: Optional[int] = None) -> List[JobResult]:
    """Level-2: build all N fold-reference sets (sequential, main process,
    cheap), THEN stream every patient's windows exactly once against all of
    them, in parallel. Requires level-1 already complete for every patient
    (call run_level1_parallel first, or use run_precompute_parallel).

    No span_roof parameter: patient_all_fold_features always caches the
    distance tensor at the full SPAN_MAX. Any span roof is a cheap slice
    applied later, per fold, in dataset_builder.build_fold_precomputed --
    that is exactly what makes the span-roof sweep free after this runs.
    """
    from src.experiment.lopo import ChbSpdProvider, _alpha_fingerprint
    from src.data import dataset_builder as db

    fp = _alpha_fingerprint(alpha)
    ids = list(patients)
    tag = db._fast_tag(ids, alpha)

    # --- sequential barrier: build + cache the shared fold-reference set
    # ONCE, in the main process, BEFORE spawning workers. Each worker then
    # gets a cache HIT (cheap disk read) instead of racing to build it. ---
    log.info("level-2: building all %d fold-reference sets (main process)...", len(ids))
    provider = ChbSpdProvider(ids, alpha=alpha, raw_dir=raw_dir, sop_minutes=sop_minutes)
    t0 = time.perf_counter()
    db.build_all_fold_references(provider, ids, fingerprint=fp, tag=tag)
    log.info("level-2: fold references ready in %.1fs", time.perf_counter() - t0)

    job_args = [(p, alpha, raw_dir, sop_minutes, fp, tag, ids) for p in ids]

    t1 = time.perf_counter()
    results = _run_pool(_level2_job, job_args, n_workers=n_workers)
    elapsed = time.perf_counter() - t1

    failed = [r for r in results if not r.ok]
    log.info("level-2 parallel: %d/%d patients OK in %.1fs (%.1f min)",
             len(results) - len(failed), len(results), elapsed, elapsed / 60)
    if failed:
        log.error("level-2 parallel: %d patient(s) FAILED: %s",
                  len(failed), [r.patient_id for r in failed])
    return results


def run_precompute_parallel(patients: Sequence[str], *, alpha: float,
                            raw_dir: Optional[Path] = None,
                            sop_minutes: Optional[int] = None,
                            n_workers: Optional[int] = None) -> Dict[str, List[JobResult]]:
    """Convenience: run level-1 to completion, THEN level-2. Raises if any
    level-1 patient failed -- level-2's fold references would silently be
    wrong/incomplete otherwise, which is worse than stopping loudly."""
    l1 = run_level1_parallel(patients, alpha=alpha, raw_dir=raw_dir,
                             sop_minutes=sop_minutes, n_workers=n_workers)
    if any(not r.ok for r in l1):
        raise RuntimeError(
            "level-1 had failures; refusing to start level-2 (its fold "
            "references need EVERY patient's level-1 result). See the "
            "logged errors above.")
    l2 = run_level2_parallel(patients, alpha=alpha,
                             raw_dir=raw_dir, sop_minutes=sop_minutes,
                             n_workers=n_workers)
    return {"level1": l1, "level2": l2}


# ---------------------------------------------------------------------------
# Self-test: proves the POOL PLUMBING (spawn, pickling function references,
# initializer invocation, imap_unordered dispatch across distinct worker
# PIDs, result aggregation) works correctly on THIS platform, using a
# trivial dummy job -- deliberately NOT the real EDF-dependent level-1/
# level-2 jobs, so this needs no dataset and runs in seconds. This is what
# catches a spawn/pickling problem cheaply, before trusting an unattended
# multi-hour real run to it. Test the REAL jobs against real data with e.g.:
#
#   from src.experiment.parallel_build import run_precompute_parallel
#   run_precompute_parallel(["chb01", "chb02"], alpha=0.5, span_roof=5)
# ---------------------------------------------------------------------------
def _dummy_job(x: int):
    import os
    return (x, os.getpid())


if __name__ == "__main__":
    print("Running parallel_build.py self-test (pool plumbing only, no EDF data)...\n")

    n = min(3, resolve_n_workers())
    ctx = get_context("spawn")
    with ctx.Pool(processes=n, initializer=_worker_init) as pool:
        results = list(pool.imap_unordered(_dummy_job, range(6), chunksize=1))

    xs = sorted(r[0] for r in results)
    pids = {r[1] for r in results}
    assert xs == list(range(6)), f"expected all 6 inputs processed exactly once, got {xs}"
    assert len(results) == 6

    print(f"OK - 6 jobs completed across {len(pids)} distinct worker PID(s): {pids}")
    print("\nTo run the REAL level-1/level-2 build against actual patient data, "
          "from a script or REPL:")
    print('  from src.experiment.parallel_build import run_precompute_parallel')
    print('  run_precompute_parallel(["chb01", "chb02"], alpha=0.5, span_roof=5)')