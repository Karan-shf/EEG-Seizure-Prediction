"""
cache.py
========
Stage 5/6: a small on-disk cache for the LOPO **fold-invariant** artifacts.

The expensive per-patient quantities do NOT depend on which patient is held out:

    clean_signal          filtered signal (produced upstream)
    window_plan           window index + labels + timeline (produced upstream)
    g_patient             per-patient baseline SPD (Frechet mean of interictal C)
    d_baseline            per-window baseline distances  delta_R(C', I)
    patient_anchor_means  per-patient level-1 interictal/preictal Frechet means

These are listed in `cfg.CACHE_FOLD_INVARIANT`. Computing them ONCE and reusing
them across all folds is what makes LOPO affordable; only the per-fold
population distances are recomputed each fold. The dense SPD tensor is never
cached (`cfg.CACHE_DENSE_SPD = False`) - it is streamed and discarded.

Keys are `(kind, key)` (e.g. `("g_patient", "chb07")`). A `fingerprint` (hash of
the relevant config) is stored alongside each entry so a stale cache from a
different configuration is transparently ignored.

Dependency-light: standard library + numpy only (no torch / scipy).
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

from src import config as cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

# In-run memoization (only consulted when caching is enabled). Bounded LRU:
# every entry also has (or will have) a durable copy on disk for
# fold-invariant kinds, so eviction is never a correctness risk -- it just
# means the next access re-reads from disk. See cfg.CACHE_MEM_MAX_ENTRIES.
_MEM: OrderedDict[tuple[str, str], Any] = OrderedDict()


def _mem_set(mk: tuple[str, str], value: Any) -> None:
    """Insert/update an entry and mark it most-recently-used, evicting the
    least-recently-used entry if this pushes _MEM over its configured cap."""
    _MEM[mk] = value
    _MEM.move_to_end(mk)
    max_entries = int(getattr(cfg, "CACHE_MEM_MAX_ENTRIES", 40))
    while len(_MEM) > max_entries:
        evicted_mk, _ = _MEM.popitem(last=False)
        log.debug("cache: _MEM evicted %s (LRU, cap=%d)", evicted_mk, max_entries)


def cache_dir() -> Path:
    """Root cache directory (read from config each call so tests can override)."""
    return Path(cfg.CACHE_DIR)


def is_enabled() -> bool:
    return bool(getattr(cfg, "CACHE_ENABLED", True))


def is_fold_invariant(kind: str) -> bool:
    """Whether `kind` is whitelisted for persistent caching in config."""
    return kind in tuple(getattr(cfg, "CACHE_FOLD_INVARIANT", ()))


def _safe(name: str) -> str:
    s = "".join(c if (c.isalnum() or c in "-._") else "_" for c in str(name))
    if len(s) > 100:
        s = s[:80] + "_" + hashlib.sha1(str(name).encode()).hexdigest()[:12]
    return s or "_"


def _paths(kind: str, key: str) -> tuple[Path, Path]:
    base = cache_dir() / _safe(kind)
    return base / f"{_safe(key)}.pkl", base / f"{_safe(key)}.meta.json"


def _mk(kind: str, key: str) -> tuple[str, str]:
    return (str(kind), str(key))


def has(kind: str, key: str, *, fingerprint: Optional[str] = None) -> bool:
    if not is_enabled():
        return False
    mk = _mk(kind, key)
    if mk in _MEM:
        _MEM.move_to_end(mk)  # touched -> most-recently-used
        return fingerprint is None or _MEM[mk][0] == fingerprint
    data_p, meta_p = _paths(kind, key)
    if not data_p.exists():
        return False
    if fingerprint is not None:
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            return False
        if meta.get("fingerprint") != fingerprint:
            return False
    return True


def load(kind: str, key: str, *, fingerprint: Optional[str] = None,
         default: Any = None) -> Any:
    if not is_enabled():
        return default
    mk = _mk(kind, key)
    if mk in _MEM:
        fp_stored, obj = _MEM[mk]
        _MEM.move_to_end(mk)  # touched -> most-recently-used
        return obj if (fingerprint is None or fp_stored == fingerprint) else default
    data_p, meta_p = _paths(kind, key)
    if not data_p.exists():
        return default
    if fingerprint is not None:
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            return default
        if meta.get("fingerprint") != fingerprint:
            return default
    with open(data_p, "rb") as f:
        obj = pickle.load(f)
    _mem_set(mk, (fingerprint, obj))
    return obj


def save(kind: str, key: str, obj: Any, *,
         fingerprint: Optional[str] = None) -> None:
    if not is_enabled():
        return
    _mem_set(_mk(kind, key), (fingerprint, obj))
    data_p, meta_p = _paths(kind, key)
    data_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = data_p.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, data_p)          # atomic publish
    meta_p.write_text(json.dumps({
        "kind": str(kind), "key": str(key),
        "fingerprint": fingerprint, "saved_at": time.time(),
    }))


def get_or_compute(kind: str, key: str, compute_fn: Callable[[], Any], *,
                   fingerprint: Optional[str] = None, force: bool = False) -> Any:
    """Return cached value or compute it.

    - Caching disabled -> always compute fresh (no memory, no disk).
    - Enabled + whitelisted kind (or force) -> persisted to disk + memory.
    - Enabled + non-whitelisted kind -> memoized in memory for this run only.
    """
    if not is_enabled():
        return compute_fn()
    if not force and has(kind, key, fingerprint=fingerprint):
        return load(kind, key, fingerprint=fingerprint)
    obj = compute_fn()
    if is_fold_invariant(kind) or force:
        save(kind, key, obj, fingerprint=fingerprint)
    else:
        _mem_set(_mk(kind, key), (fingerprint, obj))
    return obj


def clear(kind: Optional[str] = None) -> None:
    """Drop cached entries from memory and disk (all, or one kind)."""
    if kind is None:
        _MEM.clear()
        base = cache_dir()
    else:
        for k in [k for k in _MEM if k[0] == str(kind)]:
            _MEM.pop(k, None)
        base = cache_dir() / _safe(kind)
    if base.exists():
        shutil.rmtree(base)


# ---------------------------------------------------------------------------
# Self-test (standard library + numpy only)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running cache.py self-test ...\n")
    import tempfile
    import numpy as np

    # Redirect the cache into a throwaway directory.
    cfg.CACHE_DIR = Path(tempfile.mkdtemp(prefix="sh_cache_test_"))
    prev_enabled = getattr(cfg, "CACHE_ENABLED", True)
    cfg.CACHE_ENABLED = True
    _MEM.clear()

    fp = "fp-v1"
    # pick a whitelisted (fold-invariant) kind and a non-whitelisted one
    inv_kind = cfg.CACHE_FOLD_INVARIANT[0]
    assert is_fold_invariant(inv_kind) and not is_fold_invariant("scratch_kind")

    # --- roundtrip an ndarray + a nested object ---
    arr = np.arange(12, dtype=float).reshape(3, 4)
    obj = {"a": arr, "b": [1, 2, {"c": np.eye(2)}]}
    save(inv_kind, "chbXX", obj, fingerprint=fp)
    got = load(inv_kind, "chbXX", fingerprint=fp)
    assert np.allclose(got["a"], arr) and np.allclose(got["b"][2]["c"], np.eye(2))

    # --- fingerprint mismatch is a miss ---
    assert has(inv_kind, "chbXX", fingerprint=fp)
    assert not has(inv_kind, "chbXX", fingerprint="other-fp")
    assert load(inv_kind, "chbXX", fingerprint="other-fp", default="MISS") == "MISS"

    # --- get_or_compute computes once, then serves cache ---
    calls = {"n": 0}
    def compute():
        calls["n"] += 1
        return np.array([calls["n"]], dtype=float)
    v1 = get_or_compute(inv_kind, "once", compute, fingerprint=fp)
    v2 = get_or_compute(inv_kind, "once", compute, fingerprint=fp)
    assert calls["n"] == 1 and np.allclose(v1, v2)

    # --- non-whitelisted kind: memory only, nothing written to disk ---
    get_or_compute("scratch_kind", "k", lambda: 123, fingerprint=fp)
    data_p, _ = _paths("scratch_kind", "k")
    assert not data_p.exists(), "non-fold-invariant kind must not persist to disk"
    _MEM.clear()
    assert not has("scratch_kind", "k", fingerprint=fp)

    # --- disabled mode always recomputes ---
    cfg.CACHE_ENABLED = False
    calls["n"] = 0
    get_or_compute(inv_kind, "dis", compute, fingerprint=fp)
    get_or_compute(inv_kind, "dis", compute, fingerprint=fp)
    assert calls["n"] == 2, "disabled cache must recompute every call"
    cfg.CACHE_ENABLED = True

    # --- clear removes a kind ---
    clear(inv_kind)
    assert not has(inv_kind, "once", fingerprint=fp)

    # --- bounded _MEM: LRU eviction never breaks correctness (fold-invariant
    # kinds always have a disk copy to fall back to) ---
    prev_cap = getattr(cfg, "CACHE_MEM_MAX_ENTRIES", 40)
    cfg.CACHE_MEM_MAX_ENTRIES = 3
    _MEM.clear()
    try:
        for i in range(6):
            save(inv_kind, f"lru{i}", np.array([i]), fingerprint=fp)
        assert len(_MEM) <= 3, f"_MEM exceeded cap: {len(_MEM)} entries"
        assert (inv_kind, "lru0") not in _MEM, "oldest entry should have been evicted"
        # DATA is still correct even after eviction -- served from disk instead
        assert load(inv_kind, "lru0", fingerprint=fp)[0] == 0
        assert load(inv_kind, "lru5", fingerprint=fp)[0] == 5
        assert has(inv_kind, "lru0", fingerprint=fp)
    finally:
        cfg.CACHE_MEM_MAX_ENTRIES = prev_cap
        _MEM.clear()

    cfg.CACHE_ENABLED = prev_enabled
    shutil.rmtree(cfg.CACHE_DIR, ignore_errors=True)
    print("OK - cache.py self-test passed.")
