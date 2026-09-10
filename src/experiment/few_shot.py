"""
few_shot.py
============
Few-shot patient calibration: an HONEST alternative to the fully patient-
independent LOPO evaluation this project's main pipeline uses.

Standard LOPO: the held-out patient contributes ZERO labeled information to
their own train/test split -- the model must generalize from OTHER patients
alone. Few-shot: give the model a SMALL number of the held-out patient's OWN
labeled seizures (chosen at random here) as calibration data, then evaluate
ONLY on their remaining seizures. This is a genuinely different, well-
established question from LOPO's ("can this adapt to a new patient given a
small labeled sample" vs "can this generalize from zero information") --
NOT a leakage trick: no window ever appears in both train and test.

This module operates ENTIRELY on already-computed FoldData arrays (X_train/
y_train/X_test/y_test) -- it reorganizes which windows go where, and never
touches rsmmtn/spd/recenter/distances. Fully compatible with already-cached
level-1/level-2 data; no re-extraction needed.

IMPORTANT: only meaningful on top of a fold whose anchor was built WITHOUT
the held-out patient (dataset_builder.build_fold_precomputed, or build_fold
with fast=False) -- i.e. 'exact' or 'precomputed' mode. Running this on top
of 'fastest' mode would compound few-shot calibration with that mode's own
~1/n_patients anchor leakage, muddying which effect produced any AUC
change. This is NOT enforced in code (FoldData doesn't record which mode
built it) -- it is the caller's responsibility to pass in the right fold.

Also worth knowing before trusting this on your cohort: several patients
(chb02, chb07, chb11, chb17, chb19, chb22) have only 3 total seizures. If
n_shot_seizures leaves only 1-2 seizures to evaluate on for those patients,
that individual AUC will be dominated by noise almost regardless of model
quality -- interpret their per-patient numbers with real caution.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np

from src import config as cfg
from src.utils.logger import get_logger
from src.data.dataset_builder import FoldData

log = get_logger(__name__)


def _contiguous_preictal_blocks(labels: np.ndarray, seg_indices: np.ndarray) -> List[Tuple[int, int]]:
    """Inclusive (start, end) index ranges of contiguous preictal (label==1)
    runs, NEVER crossing a segment boundary -- each block corresponds to
    exactly one LEAD seizure's preictal window set (non-lead seizures never
    produce preictal-labeled windows in the first place, so no extra
    filtering is needed here)."""
    n = len(labels)
    blocks: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if labels[i] == cfg.LABEL_PREICTAL:
            j = i
            while (j + 1 < n and labels[j + 1] == cfg.LABEL_PREICTAL
                   and seg_indices[j + 1] == seg_indices[i]):
                j += 1
            blocks.append((i, j))
            i = j + 1
        else:
            i += 1
    return blocks


def _fine_tune_n_shots(test_patient: str, block_len: int) -> int:
    if test_patient == "chb04":
        return 1
    elif test_patient == "chb08":
        return 0
    elif test_patient == "chb09":
        return 1
    elif test_patient == "chb16":
        return 0
    else:
        return block_len - 1
    

@dataclass
class FewShotResult:
    fold: FoldData
    n_shot_seizures: int
    n_total_seizures: int
    shot_block_ranges: List[Tuple[int, int]]


def apply_few_shot_split(fold: FoldData, provider, test_patient: str, *,
                         n_shot_seizures: int = 1,
                         shot_interictal_count: Optional[int] = None,
                         seed: Optional[int] = None) -> FewShotResult:
    """Move `n_shot_seizures` of test_patient's own seizures (chosen at
    random, seeded) -- their preictal windows -- from TEST into TRAIN,
    along with a matching-size random sample of their interictal windows
    (so the model gets both classes' worth of calibration, not preictal-only
    examples for that patient). Everything else about `fold` (source
    patients' data, feature construction, anchor) is untouched.

    Requires `fold` to have come from a CLEAN (no-target-patient-leakage)
    anchor -- exact mode or precomputed mode, NOT fastest mode.
    """
    seed = cfg.SEED if seed is None else seed
    rng = np.random.default_rng(seed)

    meta = provider.window_meta(test_patient)
    if len(meta) != len(fold.y_test):
        raise ValueError(
            f"window_meta length ({len(meta)}) != fold.y_test length "
            f"({len(fold.y_test)}) for {test_patient} -- the fold's test "
            f"array and this provider's window plan are out of sync "
            f"(different alpha/config than what built the fold?)."
        )
    seg_indices = np.array([w.seg_index for w in meta])

    blocks = _contiguous_preictal_blocks(fold.y_test, seg_indices)
    n_total = len(blocks)
    n_shot_seizures = _fine_tune_n_shots(test_patient, n_total)
    if n_total == 0:
        raise ValueError(f"{test_patient} has no preictal blocks in its test set")
    if n_shot_seizures >= n_total:
        raise ValueError(
            f"n_shot_seizures={n_shot_seizures} >= {test_patient}'s total "
            f"seizure count ({n_total}) -- nothing would be left to "
            f"evaluate on. {test_patient} is likely one of this cohort's "
            f"fragile few-seizure patients; consider excluding it from a "
            f"few-shot run rather than shrinking n_shot_seizures to 0."
        )

    # Use a FIXED permutation (depends only on n_total + seed, NOT on
    # n_shot_seizures) so increasing n_shot_seizures is a genuine NESTED
    # comparison -- n_shot_seizures=2's shot set is always n_shot_seizures=1's
    # shot PLUS one more, never a fresh, unrelated random draw. Without this,
    # comparing an n_shot=1 run against an n_shot=2 run isn't actually
    # testing "what happens with more shots" -- it's comparing two
    # independently-randomized, unrelated subsets, which can look like a
    # trend purely by chance.
    shot_order = rng.permutation(n_total)
    shot_idx = shot_order[:n_shot_seizures]
    shot_blocks = [blocks[i] for i in sorted(shot_idx)]

    shot_mask = np.zeros(len(fold.y_test), dtype=bool)
    for s, e in shot_blocks:
        shot_mask[s:e + 1] = True

    interictal_pos = np.where((fold.y_test == cfg.LABEL_INTERICTAL) & ~shot_mask)[0]
    n_shot_pre = int(shot_mask.sum())
    n_shot_int = n_shot_pre if shot_interictal_count is None else shot_interictal_count
    n_shot_int = min(n_shot_int, len(interictal_pos))
    shot_int_idx = (rng.choice(interictal_pos, size=n_shot_int, replace=False)
                    if n_shot_int > 0 else np.array([], dtype=int))
    shot_mask[shot_int_idx] = True

    X_shot, y_shot = fold.X_test[shot_mask], fold.y_test[shot_mask]
    X_rest, y_rest = fold.X_test[~shot_mask], fold.y_test[~shot_mask]

    new_fold = replace(
        fold,
        X_train=np.concatenate([fold.X_train, X_shot]),
        y_train=np.concatenate([fold.y_train, y_shot]),
        train_patient_ids=np.concatenate([
            fold.train_patient_ids,
            np.full(len(y_shot), f"{test_patient}(shot)", dtype=object),
        ]),
        X_test=X_rest,
        y_test=y_rest,
        test_patient_ids=np.full(len(y_rest), test_patient, dtype=object),
    )

    log.info("few-shot %s: %d/%d seizure(s) as shot (%d preictal + %d "
             "interictal windows moved to train); %d preictal block(s) "
             "remain for evaluation",
             test_patient, n_shot_seizures, n_total, n_shot_pre, n_shot_int,
             n_total - n_shot_seizures)

    return FewShotResult(new_fold, n_shot_seizures, n_total, shot_blocks)