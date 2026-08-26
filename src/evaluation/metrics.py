"""
metrics.py
==========
Stage I (evaluation): window-level and event-level seizure-PREDICTION metrics,
plus per-patient and pooled LOPO aggregation.

Window-level  : threshold-free discrimination (ROC AUC) + confusion metrics.
Event-level   : the headline seizure-prediction metrics -- event sensitivity
                (fraction of seizures with >=1 alarm inside their preictal
                block), false-alarm rate FPR/h, and mean warning time -- built
                on the Firing-Power alarms from alarms.py.

Definitions (self-contained; the 0/1 labels carry all needed structure):
  * Each contiguous run of preictal (label==1) windows within a segment is ONE
    seizure opportunity (\"event\").
  * An event is PREDICTED if >=1 alarm falls inside its preictal block.
  * A FALSE ALARM is any alarm on an interictal (label==0) window.
  * FPR/h = false_alarms / interictal_hours, interictal_hours from the
    interictal window count x stride.

Dependency-light: standard library + numpy only (AUC via the rank statistic, so
the evaluation layer never needs scikit-learn).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, cast, Any
import numpy as np

from src import config as cfg
from src.utils.logger import get_logger
from src.evaluation import alarms as alarms_mod

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# window-level discrimination
# ---------------------------------------------------------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (like scipy.stats.rankdata)."""
    a = np.asarray(a, dtype=float).ravel()
    n = a.shape[0]
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(y_true, scores) -> float:
    """ROC AUC via the Mann-Whitney U rank statistic. NaN if a class is absent."""
    y = np.asarray(y_true).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    pos = y == cfg.LABEL_PREICTAL
    neg = y == cfg.LABEL_INTERICTAL
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(s)
    sum_pos = float(np.sum(ranks[pos]))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def confusion_at_threshold(y_true, scores, threshold: float) -> Tuple[int, int, int, int]:
    y = np.asarray(y_true).ravel()
    pred = (np.asarray(scores, dtype=float).ravel() >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == cfg.LABEL_PREICTAL)))
    fp = int(np.sum((pred == 1) & (y == cfg.LABEL_INTERICTAL)))
    tn = int(np.sum((pred == 0) & (y == cfg.LABEL_INTERICTAL)))
    fn = int(np.sum((pred == 0) & (y == cfg.LABEL_PREICTAL)))
    return tp, fp, tn, fn


def window_metrics(y_true, scores, *, threshold: float = 0.5) -> Dict[str, float]:
    tp, fp, tn, fn = confusion_at_threshold(y_true, scores, threshold)
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else float("nan")
    f1 = (2 * prec * sens / (prec + sens)
          if prec == prec and sens == sens and (prec + sens) else float("nan"))
    return {
        "auc": roc_auc(y_true, scores),
        "sensitivity": sens, "specificity": spec, "precision": prec,
        "accuracy": acc, "f1": f1,
        "n_preictal": tp + fn, "n_interictal": tn + fp,
    }


# ---------------------------------------------------------------------------
# event-level (Firing-Power alarms)
# ---------------------------------------------------------------------------
@dataclass
class PatientPredictions:
    """Per-window preictal probabilities + labels for one patient, split into
    contiguous segments (no segment crosses an inter-file seam)."""
    patient_id: str
    prob_segments: List[np.ndarray]
    label_segments: List[np.ndarray]
    stride_seconds: Optional[float] = None

    def all_scores(self) -> np.ndarray:
        return (np.concatenate([np.asarray(p, dtype=float).ravel()
                                for p in self.prob_segments])
                if self.prob_segments else np.array([]))

    def all_labels(self) -> np.ndarray:
        return (np.concatenate([np.asarray(l).ravel() for l in self.label_segments])
                if self.label_segments else np.array([]))


def _contiguous_blocks(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Inclusive (start, end) index ranges where mask is True."""
    mask = np.asarray(mask).astype(bool)
    blocks: List[Tuple[int, int]] = []
    n = mask.shape[0]
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            blocks.append((i, j))
            i = j + 1
        else:
            i += 1
    return blocks


def event_metrics(preds: "PatientPredictions", *, threshold: Optional[float] = None,
                  fp_windows: Optional[int] = None,
                  refractory_windows: Optional[int] = None,
                  stride_seconds: Optional[float] = None) -> Dict[str, float]:
    thr = cfg.FIRING_POWER_THRESHOLD if threshold is None else float(threshold)
    if stride_seconds is not None:
        stride = stride_seconds
    elif preds.stride_seconds:
        stride = preds.stride_seconds
    else:
        stride = cfg.STRIDE_SECONDS
    if fp_windows is None:
        fp_windows = alarms_mod.firing_power_window_count(stride_seconds=stride)
    if refractory_windows is None:
        refractory_windows = alarms_mod.refractory_window_count(stride_seconds=stride)

    n_events = n_pred = n_false = interictal_windows = 0
    warn_times: List[float] = []
    for probs, labels in zip(preds.prob_segments, preds.label_segments):
        labels = np.asarray(labels)
        fp = alarms_mod.firing_power(cast(Sequence[float], probs), fp_windows)
        al = alarms_mod.generate_alarms(cast(Sequence[float], fp), threshold=thr,
                                        refractory_windows=refractory_windows)
        interictal_windows += int(np.sum(labels == cfg.LABEL_INTERICTAL))
        n_false += alarms_mod.count_false_alarms(al, labels)
        for (s, e) in _contiguous_blocks(labels == cfg.LABEL_PREICTAL):
            n_events += 1
            hits = [i for i in al if s <= i <= e]
            if hits:
                n_pred += 1
                onset_time = (e + 1) * stride + cfg.SPH_SECONDS
                warn_times.append(onset_time - min(hits) * stride)

    interictal_hours = interictal_windows * stride / 3600.0
    return {
        "sensitivity": (n_pred / n_events) if n_events else float("nan"),
        "n_events": n_events, "n_predicted": n_pred,
        "false_alarms": n_false,
        "interictal_hours": interictal_hours,
        "fpr_per_hour": (n_false / interictal_hours) if interictal_hours > 0 else float("nan"),
        "mean_warning_seconds": float(np.mean(warn_times)) if warn_times else float("nan"),
    }


def evaluate_patient(preds: "PatientPredictions", *, threshold: Optional[float] = None,
                     fp_windows: Optional[int] = None,
                     refractory_windows: Optional[int] = None,
                     window_threshold: float = 0.5) -> Dict[str, object]:
    wm = window_metrics(preds.all_labels(), preds.all_scores(),
                        threshold=window_threshold)
    em = event_metrics(preds, threshold=threshold, fp_windows=fp_windows,
                       refractory_windows=refractory_windows)
    return {"patient_id": preds.patient_id, "auc": wm["auc"],
            "window": wm, "event": em}


def evaluate_lopo(preds_list: Sequence["PatientPredictions"], *,
                  threshold: Optional[float] = None,
                  fp_windows: Optional[int] = None,
                  refractory_windows: Optional[int] = None) -> Dict[str, object]:
    """Aggregate per-patient reports into pooled + macro LOPO summaries."""
    per_patient: List[Dict[str, object]] = []
    all_y: List[np.ndarray] = []
    all_s: List[np.ndarray] = []
    tot_false = 0.0
    tot_hours = 0.0
    sens_list: List[float] = []
    auc_list: List[float] = []
    for preds in preds_list:
        rep = evaluate_patient(preds, threshold=threshold, fp_windows=fp_windows,
                               refractory_windows=refractory_windows)
        per_patient.append(rep)
        all_y.append(preds.all_labels())
        all_s.append(preds.all_scores())
        em = cast(Dict[str, Any], rep["event"])
        tot_false += em["false_alarms"]
        tot_hours += em["interictal_hours"]
        if em["sensitivity"] == em["sensitivity"]:      # not NaN
            sens_list.append(em["sensitivity"])
        if rep["auc"] == rep["auc"]:
            auc_list.append(cast(float, rep["auc"]))

    pooled_y = np.concatenate(all_y) if all_y else np.array([])
    pooled_s = np.concatenate(all_s) if all_s else np.array([])
    summary = {
        "n_patients": len(preds_list),
        "pooled_auc": roc_auc(pooled_y, pooled_s) if pooled_y.size else float("nan"),
        "mean_patient_auc": float(np.mean(auc_list)) if auc_list else float("nan"),
        "mean_event_sensitivity": float(np.mean(sens_list)) if sens_list else float("nan"),
        "pooled_fpr_per_hour": (tot_false / tot_hours) if tot_hours > 0 else float("nan"),
        "total_false_alarms": tot_false,
        "total_interictal_hours": tot_hours,
    }
    return {"per_patient": per_patient, "summary": summary}


# ---------------------------------------------------------------------------
# self-test (author runs it; not run automatically)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running metrics.py self-test ...")

    # ROC AUC
    assert abs(roc_auc([0, 0, 0, 1, 1, 1], [.1, .2, .3, .7, .8, .9]) - 1.0) < 1e-9
    assert abs(roc_auc([0, 1, 0, 1], [.5, .5, .5, .5]) - 0.5) < 1e-9   # ties -> 0.5

    # contiguous blocks helper
    assert _contiguous_blocks(np.array([0, 1, 1, 0, 1])) == [(1, 2), (4, 4)]

    # event: one preictal block at the end, high prob -> predicted, no false alarms
    seg = np.concatenate([np.full(20, 0.1), np.full(10, 0.9)])
    lab = np.array([cfg.LABEL_INTERICTAL] * 20 + [cfg.LABEL_PREICTAL] * 10)
    preds = PatientPredictions("t1", [seg], [lab], stride_seconds=6.0)
    em = event_metrics(preds, threshold=0.5, fp_windows=3, refractory_windows=2,
                       stride_seconds=6.0)
    assert em["n_events"] == 1 and em["n_predicted"] == 1
    assert em["sensitivity"] == 1.0 and em["false_alarms"] == 0
    assert em["mean_warning_seconds"] > 0

    # false alarm in interictal, event still predicted
    seg2 = np.concatenate([np.full(5, 0.9), np.full(15, 0.1), np.full(10, 0.9)])
    lab2 = np.array([cfg.LABEL_INTERICTAL] * 20 + [cfg.LABEL_PREICTAL] * 10)
    preds2 = PatientPredictions("t2", [seg2], [lab2], stride_seconds=6.0)
    em2 = event_metrics(preds2, threshold=0.5, fp_windows=3, refractory_windows=2,
                        stride_seconds=6.0)
    assert em2["false_alarms"] >= 1 and em2["n_predicted"] == 1
    assert em2["fpr_per_hour"] > 0

    # LOPO aggregation
    rep = cast(Dict[str, Any], evaluate_lopo([preds, preds2], threshold=0.5, fp_windows=3, refractory_windows=2))
    assert rep["summary"]["n_patients"] == 2
    assert 0.0 <= rep["summary"]["pooled_auc"] <= 1.0
    assert rep["summary"]["mean_event_sensitivity"] == 1.0

    print("metrics.py self-test OK")
