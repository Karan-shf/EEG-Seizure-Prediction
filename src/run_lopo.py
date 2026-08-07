"""
run_lopo.py
-----------
Pooled leave-one-patient-out (LOPO) evaluation for one (offset, duration)
config. Trains one model per held-out patient, pools all held-out test
predictions, calibrates a single decision threshold on the pooled validation
predictions, and reports one honest set of metrics.

Depends on the already-applied fixes:
  - evaluate.pick_threshold_youden  (Fix 6)
  - evaluate.compute_metrics(..., n_frames=...)  (Fix 5)

Run:
    python src/run_lopo.py --offset 15 --duration 60
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_dataset import build_dataset
from dataset import get_splits, make_dataloaders, infer_n_frames, FIXED_TEST_PATIENTS
from model import ModelConfig
from train import TrainConfig, train
from evaluate import (load_model, run_inference, compute_metrics,
                      pick_threshold_youden, pick_threshold_fpr,
                      TARGET_FPR_PER_HOUR,
                      compute_per_patient_metrics, compute_pooled_zscored_auc,
                      summarize_per_patient_auc)

DATA_ROOT       = 'data/processed'
CHECKPOINT_ROOT = 'experiments/checkpoints'
RESULTS_ROOT    = 'experiments/results'


def _infer(model, loader, device):
    """Run inference; return (probas, labels) or (empty, empty) if loader empty."""
    if len(loader.dataset) == 0:
        return np.array([]), np.array([])
    res = run_inference(model, loader, device)
    return np.asarray(res['probas']), np.asarray(res['labels'])


def run_one_config_lopo(offset_minutes: int, duration_minutes: int,
                        max_epochs: int | None = None) -> dict:
    config_name = f'offset{offset_minutes}_dur{duration_minutes}'
    print('#' * 65)
    print(f'#  LOPO CONFIG: {config_name}')
    print('#' * 65)

    # --- Step 1: build dataset (cross-file stitching happens here) ---
    build_info = build_dataset(
        offset_minutes=offset_minutes,
        duration_minutes=duration_minutes,
        output_root=DATA_ROOT,
        config_name=config_name,
    )
    seq_dir = build_info['output_dir']
    meta    = pd.read_csv(build_info['metadata_path'])
    n_frames = infer_n_frames(seq_dir)

    device = torch.device(
        'mps' if torch.backends.mps.is_available()
        else 'cuda' if torch.cuda.is_available() else 'cpu'
    )

    folds = get_splits(meta, strategy='lopo')

    pooled_test_p, pooled_test_y, pooled_test_pat = [], [], []
    pooled_val_p,  pooled_val_y                    = [], []
    fold_rows = []

    for train_p, val_p, test_p in folds:
        test_patient = test_p[0]
        # Skip folds whose held-out patient has no sequences at all
        n_test = int(meta['patient_id'].isin(test_p).sum())
        if n_test == 0:
            print(f'  [skip] {test_patient}: no sequences at this config')
            continue

        fold_cfg = TrainConfig(
            metadata_path=build_info['metadata_path'],
            sequences_dir=seq_dir,
            config_name=f'{config_name}_lopo_{test_patient}',
            checkpoint_dir=CHECKPOINT_ROOT,
            log_dir=os.path.join('experiments', 'logs'),
            results_dir=RESULTS_ROOT,
        )
        if max_epochs is not None:
            fold_cfg.max_epochs = max_epochs

        tr = train(fold_cfg, ModelConfig(n_frames=n_frames),
                   fold=(train_p, val_p, test_p))
        if tr.get('checkpoint_path') is None:
            print(f'  [skip] {test_patient}: training produced no checkpoint')
            continue

        model = load_model(tr['checkpoint_path'], logger=None).to(device)

        _, val_loader, test_loader = make_dataloaders(
            meta, seq_dir, train_p, val_p, test_p, batch_size=16, num_workers=0,
        )
        te_p, te_y = _infer(model, test_loader, device)
        va_p, va_y = _infer(model, val_loader, device)

        if te_p.size:
            pooled_test_p.append(te_p); pooled_test_y.append(te_y)
            pooled_test_pat.append(np.array([test_patient] * te_p.size))
        if va_p.size:
            pooled_val_p.append(va_p); pooled_val_y.append(va_y)

        fold_rows.append({
            'test_patient': test_patient,
            'n_test': int(te_p.size),
            'n_test_pos': int((te_y == 1).sum()) if te_y.size else 0,
            'best_val_auc': tr.get('best_val_auc'),
        })

    # --- Pool + metrics ---
    if not pooled_test_p:
        raise RuntimeError(f'{config_name}: no LOPO folds produced predictions.')

    test_probas = np.concatenate(pooled_test_p)
    test_labels = np.concatenate(pooled_test_y)
    test_pat    = np.concatenate(pooled_test_pat)

    # Calibrate ONE threshold on pooled validation predictions (Fix 6),
    # falling back to 0.5 if the pooled val set is single-class.
    if pooled_val_p:
        val_probas = np.concatenate(pooled_val_p)
        val_labels = np.concatenate(pooled_val_y)
        if len(np.unique(val_labels)) == 2:
            threshold = pick_threshold_fpr(val_probas, val_labels,
                                           n_frames=n_frames,
                                           target_fpr_per_hour=TARGET_FPR_PER_HOUR)
        else:
            threshold = 0.5
    else:
        threshold = 0.5

    metrics = compute_metrics(test_probas, test_labels,
                              threshold=threshold, n_frames=n_frames)

    # The pooled AUC above ranks raw probabilities from 24 SEPARATE fold-models
    # (each on its own score scale) in one list, which collapses the number even
    # when every patient is well-ranked internally. Report the honest cross-
    # subject metrics too, using the per-prediction patient ids already gathered:
    # macro-average per-patient AUC, and the F9 z-pooled AUC (per-patient
    # z-scoring removes the cross-patient/model offsets).
    results = {
        'probas':      np.asarray(test_probas, dtype=float),
        'labels':      np.asarray(test_labels),
        'patient_ids': np.asarray(test_pat),
    }
    per_patient = compute_per_patient_metrics(results, n_frames=n_frames,
                                              threshold=threshold)
    mean_pt_auc = summarize_per_patient_auc(per_patient)
    zpooled_auc = compute_pooled_zscored_auc(results)

    out_dir = os.path.join(RESULTS_ROOT, config_name, 'lopo')
    os.makedirs(out_dir, exist_ok=True)
    save = {k: v for k, v in metrics.items()
            if k not in ('roc_fpr', 'roc_tpr', 'roc_thresholds')}
    save.update({
        'config_name': config_name,
        'strategy': 'pooled_lopo',
        'auc_pooled':       float(metrics['auc']),
        'auc_mean_patient': float(mean_pt_auc),
        'auc_zpooled':      float(zpooled_auc),
        'threshold': float(threshold),
        'n_test_total': int(test_labels.size),
        'n_test_pos': int((test_labels == 1).sum()),
        'n_folds_used': len(fold_rows),
    })
    with open(os.path.join(out_dir, 'lopo_metrics.json'), 'w') as f:
        json.dump(save, f, indent=2)
    pd.DataFrame(fold_rows).to_csv(
        os.path.join(out_dir, 'lopo_per_fold.csv'), index=False)
    per_patient.to_csv(
        os.path.join(out_dir, 'lopo_per_patient.csv'), index=False)
    # Persist raw pooled predictions so any metric can be recomputed WITHOUT
    # retraining all 24 folds again.
    pd.DataFrame({
        'patient_id': np.asarray(test_pat),
        'proba':      np.asarray(test_probas, dtype=float),
        'label':      np.asarray(test_labels),
    }).to_csv(os.path.join(out_dir, 'lopo_pooled_predictions.csv'), index=False)

    print(f'\n[{config_name}] POOLED LOPO  '
          f"pooled_AUC={metrics['auc']:.3f}  mean_patient_AUC={mean_pt_auc:.3f}  "
          f"z_pooled_AUC={zpooled_auc:.3f}\n"
          f"                    sens={metrics['sensitivity']:.3f}  "
          f"spec={metrics['specificity']:.3f}  FPR/h={metrics['fpr_per_hour']:.2f}  "
          f'(thr={threshold:.3f}, pos={save["n_test_pos"]}/{save["n_test_total"]})')
    print(f'  saved -> {out_dir}/lopo_metrics.json')
    return save


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--offset', type=int, required=True)
    ap.add_argument('--duration', type=int, required=True)
    ap.add_argument('--max-epochs', type=int, default=None)
    args = ap.parse_args()
    run_one_config_lopo(args.offset, args.duration, max_epochs=args.max_epochs)