"""
logger.py
=========
Project-wide logging utility for SeizureHorizon.

Provides a single helper, `get_logger`, that returns a configured
`logging.Logger` writing to BOTH the console and a timestamped file under
`experiments/logs/`. Using one logging setup everywhere means every stage
(parsing, preprocessing, feature extraction, LOPO) produces consistent,
timestamped, greppable logs -- essential for debugging the long, multi-hour
runs this project involves.
"""

from __future__ import annotations
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Resolve the default log directory from config when available, but still work
# if this file is run in isolation (e.g. a quick standalone test).
try:
    from src import config as cfg
    _DEFAULT_LOG_DIR = cfg.LOG_DIR
except Exception:  # pragma: no cover - fallback for standalone execution
    _DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "experiments" / "logs"

# One consistent line format everywhere:
#   2026-08-11 02:56:30 | dataset_builder   | INFO    | Built 1234 windows
_LOG_FORMAT = "%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    *,
    level: int = logging.INFO,
    to_file: bool = True,
    log_dir: Path | None = None,
    filename: str | None = None,
) -> logging.Logger:
    """Return a logger that writes to the console (and optionally a file).

    Parameters
    ----------
    name : str
        Logger name, usually the module name; shown in every line.
    level : int
        Minimum level to emit (default ``logging.INFO``).
    to_file : bool
        If True, also write to a file under ``log_dir``.
    log_dir : Path | None
        Directory for the log file. Defaults to ``config.LOG_DIR``.
    filename : str | None
        Log file name. Defaults to ``"<name>_<YYYYmmdd-HHMMSS>.log"``.

    Returns
    -------
    logging.Logger
        A ready-to-use logger. Calling again with the same ``name`` returns the
        same instance without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Do not also bubble records up to the root logger (avoids double printing).
    logger.propagate = False

    # If this logger was already configured, reuse it as-is. This is what makes
    # repeated get_logger("x") calls across modules safe.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler -> stdout
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler -> experiments/logs/<name>_<timestamp>.log
    if to_file:
        log_dir = Path(log_dir) if log_dir is not None else _DEFAULT_LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            safe_name = name.replace(".", "_")
            # PID included so concurrent processes (Tier-2 multiprocessing)
            # NEVER collide on the same log filename, even if two workers
            # spawn within the same second -- a real risk under
            # multiprocessing that a timestamp alone doesn't rule out.
            filename = f"{safe_name}_{stamp}_p{os.getpid()}.log"
        file_path = log_dir / filename
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.debug("Logging to %s", file_path)

    return logger


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    print("Running logger.py self-test ...\n")

    # 1. Create a logger writing to a temp dir with a fixed filename.
    tmp = Path(tempfile.mkdtemp())
    log = get_logger("selftest", log_dir=tmp, filename="selftest.log")
    log.info("info message")
    log.warning("warning message")
    log.error("error message")

    # 2. The file exists and contains our messages and their levels.
    log_file = tmp / "selftest.log"
    assert log_file.exists(), "log file was not created"
    text = log_file.read_text(encoding="utf-8")
    for token in ("info message", "warning message", "error message",
                  "INFO", "WARNING", "ERROR"):
        assert token in text, f"missing {token!r} in log file"

    # 3. Calling again with the same name must NOT duplicate handlers.
    n_handlers = len(log.handlers)
    log_again = get_logger("selftest", log_dir=tmp, filename="selftest.log")
    assert log_again is log, "should return the same logger instance"
    assert len(log_again.handlers) == n_handlers, "duplicate handlers added!"

    # 4. to_file=False -> console handler only.
    console_only = get_logger("console_only", to_file=False)
    assert len(console_only.handlers) == 1, "expected exactly one handler"

    # 5. Default log dir resolves under experiments/logs.
    assert _DEFAULT_LOG_DIR.name == "logs", "default log dir should be .../logs"

    print("\nAll logger.py self-tests passed.")
