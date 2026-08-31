"""
parallel.py
===========
Tier-2 CPU multiprocessing sizing + hygiene helpers.

The number of worker PROCESSES used by the patient-parallel sweep (level-1
anchor building, level-2 all-fold feature streaming) is NEVER hard-coded at a
call site -- every driver reads it from `resolve_n_workers()`, which reads
`cfg.N_WORKER_PROCESSES` / `cfg.MULTIPROCESSING_RESERVED_CORES`, so moving to
different hardware is a one-line config edit, not a code change.

Physical vs logical cores matters here: hyperthread sibling cores share
execution units and buy little to nothing for this workload (dense
Cholesky/eigh-heavy floating point, not I/O-bound), so auto-detection prefers
the PHYSICAL core count. `psutil` gives that directly; it is an OPTIONAL,
lazily-imported dependency (not in requirements.txt) -- if it is absent, this
falls back to `os.cpu_count()` (LOGICAL cores, may overcount on hyperthreaded
CPUs) and says so via the logger, so the fallback is never silent.
"""
from __future__ import annotations

import os

from src import config as cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


def _physical_core_count() -> tuple[int, bool]:
    """Return (count, is_physical). Prefers psutil's physical count; falls
    back to os.cpu_count() (logical) if psutil isn't installed."""
    try:
        import psutil  # optional, lazy -- not in requirements.txt
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n), True
    except Exception:
        pass
    return (os.cpu_count() or 1), False


def resolve_n_workers() -> int:
    """Worker-process count for the Tier-2 CPU pool.

    cfg.N_WORKER_PROCESSES set (not None) -> that value, verbatim (the
    explicit per-machine override). Otherwise: physical core count minus
    cfg.MULTIPROCESSING_RESERVED_CORES, floored at 1.
    """
    if cfg.N_WORKER_PROCESSES is not None:
        n = max(1, int(cfg.N_WORKER_PROCESSES))
        log.info("resolve_n_workers: N_WORKER_PROCESSES=%d (explicit override)", n)
        return n

    count, is_physical = _physical_core_count()
    n = max(1, count - int(cfg.MULTIPROCESSING_RESERVED_CORES))
    kind = ("physical" if is_physical else
            "logical (psutil not installed -- may overcount on hyperthreaded CPUs)")
    log.info("resolve_n_workers: %d %s cores - %d reserved -> %d workers",
             count, kind, cfg.MULTIPROCESSING_RESERVED_CORES, n)
    return n


def pin_blas_threads(n_threads: int = 1) -> None:
    """Pin every BLAS backend's internal thread count to `n_threads`.

    MUST be called before numpy (and therefore its BLAS backend) is first
    imported in a process -- BLAS reads these environment variables once, at
    its own init time, not on every call. In the Tier-2 pool this means
    calling it at the very top of each worker's entry point, before any
    `from src...` import that pulls numpy in transitively.

    Without this, N worker PROCESSES each spawning their own M BLAS THREADS
    oversubscribes the machine (N x M contending for the same physical
    cores), which measurably HURTS throughput rather than helping -- this is
    not an optional tuning knob for the Tier-2 pool, it is required.
    """
    val = str(int(n_threads))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = val


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running parallel.py self-test ...\n")

    prev = cfg.N_WORKER_PROCESSES
    try:
        cfg.N_WORKER_PROCESSES = 4
        assert resolve_n_workers() == 4

        cfg.N_WORKER_PROCESSES = None
        n = resolve_n_workers()
        assert n >= 1
        print(f"auto-detected worker count on this machine: {n}")
    finally:
        cfg.N_WORKER_PROCESSES = prev

    import os as _os
    prev_env = _os.environ.get("OMP_NUM_THREADS")
    try:
        pin_blas_threads(1)
        assert _os.environ["OMP_NUM_THREADS"] == "1"
        assert _os.environ["OPENBLAS_NUM_THREADS"] == "1"
        assert _os.environ["MKL_NUM_THREADS"] == "1"
    finally:
        if prev_env is None:
            _os.environ.pop("OMP_NUM_THREADS", None)
        else:
            _os.environ["OMP_NUM_THREADS"] = prev_env

    print("\nOK - parallel.py self-test passed.")
