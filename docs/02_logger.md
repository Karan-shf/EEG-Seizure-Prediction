# 02 — `utils/logger.py`

**Source:** `src/utils/logger.py`  ·  **Rebuild file 2 of ~19**  ·  Req: cross-cutting utility

---

## Purpose

A single, project-wide logging helper. `get_logger(name)` returns a Python
`logging.Logger` that writes **both** to the console (so you see progress live)
**and** to a timestamped file under `experiments/logs/` (so every run leaves a
permanent, greppable record).

---

## Why it's necessary

- **Long runs need a paper trail.** Building features and running LOPO across 24
  patients takes hours. `print()` scattered across files disappears when the
  terminal closes; a log file persists and is timestamped.
- **Consistency.** Every module logs in the *same* format
  (`time | module | level | message`), so logs from different stages line up and
  are easy to filter (e.g. `grep ERROR`).
- **This bit us before.** A real bug in the old pipeline was a logger that was
  passed as `None` and crashed a run mid-LOPO. Centralizing logger creation here
  removes that whole class of error — modules just call `get_logger(__name__)`.

---

## Math

None — this is pure infrastructure.

---

## Function reference

### `get_logger(name, *, level=INFO, to_file=True, log_dir=None, filename=None) -> logging.Logger`

| Argument | Meaning |
|----------|---------|
| `name` | Logger name, usually `__name__`; printed in every line so you know which module spoke. |
| `level` | Minimum severity to emit (default `INFO`). |
| `to_file` | If `True`, also write to a log file. Set `False` for quick console-only use (e.g. self-tests). |
| `log_dir` | Where the log file goes. Defaults to `config.LOG_DIR` (`experiments/logs/`). |
| `filename` | Log file name. Defaults to `"<name>_<YYYYmmdd-HHMMSS>.log"`. |

**Key behaviours**

- **No duplicate handlers.** If a logger with that `name` already has handlers,
  it is returned unchanged. This makes it safe for every module to call
  `get_logger(__name__)` at import time without stacking up repeated console
  lines.
- **`propagate = False`.** Records do not also bubble to the root logger, which
  would otherwise double-print.
- **Two handlers when `to_file=True`:** one `StreamHandler` to `stdout`, one
  `FileHandler` (UTF-8) to the timestamped file.
- **Config-aware but standalone-safe.** It imports `config.LOG_DIR` when the
  package is importable, but falls back to `../experiments/logs` if run in
  isolation, so the file's own self-test always works.

**Typical use**

```python
from src.utils.logger import get_logger
log = get_logger(__name__)
log.info("Loaded %d EDF files", n)
```

---

## What the self-test proves

Run with `python -m src.utils.logger`. The `__main__` block asserts:

1. A log **file is actually created** in the target directory.
2. The file **contains the logged messages** *and* their level names
   (`INFO`/`WARNING`/`ERROR`), confirming the format works.
3. Calling `get_logger` again with the same name returns the **same instance**
   and does **not** add duplicate handlers.
4. With `to_file=False` there is **exactly one** (console) handler.
5. The default log directory resolves to `.../logs`.

On success it prints `"All logger.py self-tests passed."`
