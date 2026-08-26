"""
alarms.py
=========
Stage I (evaluation): Firing-Power alarm logic + Option-2 threshold calibration.

A trained classifier emits a per-window preictal probability. This module turns
that probability stream into discrete seizure ALARMS:

    1. Firing Power (FP): a causal moving average of the preictal probability
       over the last ~SOP worth of windows -- smooths single-window spikes.
    2. Alarm: raised when FP crosses an operating threshold, after which a
       refractory period (SPH + SOP) silences further alarms.
    3. Threshold calibration (Option 2): the operating threshold is chosen on
       the TRAIN pool to hit a target false-alarm rate (FPR/h), then applied
       unchanged to the held-out patient.

Everything is per contiguous SEGMENT: RSMMTN windows never cross an inter-file
gap, so Firing Power is never smoothed across a seam (it resets each segment).
Dependency-light: standard library + numpy only.
"""
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import numpy as np

from src import config as cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


def firing_power_window_count(sop_minutes: Optional[float] = None, *,
                              stride_seconds: Optional[float] = None) -> int:
    """Number of windows spanning one SOP -- the Firing-Power averaging length."""
    sop_minutes = cfg.SOP_PRIMARY_MINUTES if sop_minutes is None else sop_minutes
    stride = cfg.STRIDE_SECONDS if stride_seconds is None else stride_seconds
    return max(1, int(round(sop_minutes * 60.0 / stride)))


def refractory_window_count(refractory_seconds: Optional[float] = None, *,
                            stride_seconds: Optional[float] = None) -> int:
    """Number of windows silenced after an alarm (default SPH + SOP)."""
    r = cfg.REFRACTORY_SECONDS if refractory_seconds is None else refractory_seconds
    stride = cfg.STRIDE_SECONDS if stride_seconds is None else stride_seconds
    return max(0, int(round(r / stride)))


def firing_power(probs: Sequence[float], n_windows: int) -> np.ndarray:
    """Causal moving average of preictal probability over the last n_windows.

    Ramp-up (fewer than n_windows seen) uses a partial average so early windows
    still produce a valid FP value.
    """
    probs_array = np.asarray(probs, dtype=float).ravel()
    n = probs_array.shape[0]
    if n == 0:
        return probs_array.copy()
    n_windows = max(1, int(n_windows))
    csum = np.concatenate(([0.0], np.cumsum(probs_array)))
    idx = np.arange(1, n + 1)
    lo = np.maximum(0, idx - n_windows)
    counts = idx - lo
    return (csum[idx] - csum[lo]) / counts


def generate_alarms(fp: Sequence[float], *, threshold: Optional[float] = None,
                    refractory_windows: int = 0) -> List[int]:
    """Indices where FP crosses threshold, honoring a refractory period."""
    thr = cfg.FIRING_POWER_THRESHOLD if threshold is None else float(threshold)
    fp_array = np.asarray(fp, dtype=float).ravel()
    r = max(0, int(refractory_windows))
    alarms: List[int] = []
    next_ok = 0
    for i in range(fp_array.shape[0]):
        if i >= next_ok and fp[i] >= thr:
            alarms.append(i)
            next_ok = i + r + 1
    return alarms


def alarms_per_segment(prob_segments, *, threshold: Optional[float] = None,
                       fp_windows: int, refractory_windows: int) -> List[List[int]]:
    """Run FP + alarm generation independently on each contiguous segment."""
    out = []
    for probs in prob_segments:
        fp = firing_power(probs, fp_windows)
        out.append(generate_alarms(fp.tolist(), threshold=threshold,
                                   refractory_windows=refractory_windows))
    return out


def count_false_alarms(alarm_idxs, labels) -> int:
    """Alarms landing on an interictal window are false alarms."""
    labels = np.asarray(labels)
    return int(sum(1 for i in alarm_idxs if labels[i] == cfg.LABEL_INTERICTAL))


def calibrate_threshold(prob_segments, label_segments, *,
                        target_fpr_per_hour: Optional[float] = None,
                        fp_windows: int, refractory_windows: int,
                        stride_seconds: Optional[float] = None,
                        thresholds: Optional[Sequence[float]] = None
                        ) -> Tuple[float, float]:
    """Option 2: pick the operating threshold on the TRAIN pool to hit a target
    FPR/h. Returns (threshold, achieved_fpr_per_hour).

    Scans thresholds ascending and returns the SMALLEST one whose pooled
    interictal FPR/h is <= target (most sensitive within budget). If none meet
    the budget, returns the threshold with the lowest achievable FPR/h.
    Pass flat lists of per-segment arrays pooled across all TRAIN patients.
    """
    target = (cfg.PRIMARY_TARGET_FPR_PER_HOUR if target_fpr_per_hour is None
              else float(target_fpr_per_hour))
    stride = cfg.STRIDE_SECONDS if stride_seconds is None else stride_seconds
    if thresholds is None:
        threshold_values = np.linspace(0.0, 1.0, 101)
    else:
        threshold_values = thresholds

    fps = [firing_power(p, fp_windows) for p in prob_segments]
    interictal_windows = sum(int(np.sum(np.asarray(l) == cfg.LABEL_INTERICTAL))
                             for l in label_segments)
    interictal_hours = interictal_windows * stride / 3600.0

    results: List[Tuple[float, float]] = []
    for thr in sorted(float(t) for t in threshold_values):
        false_alarms = 0
        for fp, lab in zip(fps, label_segments):
            al = generate_alarms(fp.tolist(), threshold=thr,
                                 refractory_windows=refractory_windows)
            false_alarms += count_false_alarms(al, lab)
        fpr = (false_alarms / interictal_hours) if interictal_hours > 0 else 0.0
        results.append((thr, fpr))
        if fpr <= target:
            log.info("calibrate_threshold: thr=%.3f -> FPR/h=%.3f (<= target %.3f)",
                     thr, fpr, target)
            return thr, fpr

    best = min(results, key=lambda t: t[1])
    log.warning("calibrate_threshold: target %.3f unmet; best thr=%.3f FPR/h=%.3f",
                target, best[0], best[1])
    return best


# ---------------------------------------------------------------------------
# self-test (author runs it; not run automatically)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running alarms.py self-test ...")

    # firing power smoothing
    probs = np.array([0.1] * 10 + [0.9] * 10 + [0.1] * 10)
    fp = firing_power(probs.tolist(), 3)
    assert fp.shape == probs.shape
    assert abs(fp[0] - 0.1) < 1e-9            # partial average at ramp-up
    assert abs(fp[12] - 0.9) < 1e-9           # fully inside the high region

    # alarms + refractory
    al = generate_alarms(fp.tolist(), threshold=0.5, refractory_windows=4)
    assert al, "expected at least one alarm"
    assert 10 <= al[0] <= 13
    assert np.all(np.diff(al) >= 5)           # refractory_windows + 1

    # calibration: all-interictal with a high spike -> threshold must suppress it
    seg = np.concatenate([np.full(50, 0.2), np.full(10, 0.95)])
    lab = np.zeros(60, dtype=int)
    thr, fpr = calibrate_threshold([seg], [lab], target_fpr_per_hour=0.5,
                                   fp_windows=3, refractory_windows=4,
                                   stride_seconds=6.0)
    assert 0.0 <= thr <= 1.0
    assert fpr <= 0.5 + 1e-9

    # window-count helpers
    assert firing_power_window_count(30, stride_seconds=3.6) == int(round(1800 / 3.6))
    assert refractory_window_count(2100, stride_seconds=3.6) == int(round(2100 / 3.6))

    print("alarms.py self-test OK")
