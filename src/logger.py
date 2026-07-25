"""
logger.py
---------
Logging utility for SeizureHorizon.

Routes all output to both the terminal (console) and a .log file
simultaneously. Uses Python's built-in logging module.

Every call to logger.info() prints to terminal AND writes to the log file.
No changes needed to existing print() calls — just replace print() with
logger.info() in train.py and evaluate.py.

Log files are saved to experiments/logs/ with a timestamp in the filename
so each run produces its own log and nothing is overwritten.

Usage
-----
from src.logger import get_logger

logger = get_logger(name='train')    # creates experiments/logs/train_YYYYMMDD_HHMMSS.log
logger.info('Starting training...')  # prints to terminal + writes to log
logger.warning('Low val AUC')        # same, but prefixed with WARNING
logger.error('File not found')       # same, prefixed with ERROR
"""

import logging
import os
from datetime import datetime


def get_logger(
    name: str = 'seizure_horizon',
    log_dir: str = 'experiments/logs',
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and return a logger that writes to both terminal and a log file.

    Parameters
    ----------
    name    : str — used as the log filename prefix and logger identifier
                    use 'train' for training runs, 'evaluate' for evaluation
    log_dir : str — directory where log files are saved
    level   : int — minimum log level (default INFO captures everything
                    except DEBUG messages)

    Returns
    -------
    logging.Logger
        Configured logger instance. Call .info(), .warning(), .error() on it.

    Log file naming
    ---------------
    Each call creates a new uniquely named file:
        experiments/logs/train_20260101_143022.log
        experiments/logs/evaluate_20260101_151500.log
    This means every training run has its own log — nothing is overwritten.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Unique filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = os.path.join(log_dir, f'{name}_{timestamp}.log')

    # Create logger
    logger = logging.getLogger(f'{name}_{timestamp}')
    logger.setLevel(level)

    # Avoid adding duplicate handlers if get_logger is called twice
    if logger.handlers:
        return logger

    # Format: timestamp — level — message
    formatter = logging.Formatter(
        fmt='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler 1: write to log file
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Handler 2: print to terminal (same output as before)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # First log line records where the file is being saved
    logger.info(f'Log file: {os.path.abspath(log_path)}')

    return logger