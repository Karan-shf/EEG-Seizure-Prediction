"""
run_experiment_grid.py
-----------------------
Orchestrates the full offset x duration grid search for SeizureHorizon.

Tests 16 window configurations:
    offsets   = [15, 30, 45, 60] minutes before seizure onset
    durations = [15, 30, 45, 60] minutes long

For each of the 16 (offset, duration) pairs:
    1. build_dataset(offset, duration)  → sequences + metadata for this config
    2. train(...)                        → trains a fresh model on this config
    3. evaluate(...)                     → tests on held-out patients

Results are collected into a summary table and a heatmap showing which
window configuration produces the best seizure prediction performance.

Run
---
    python src/run_experiment_grid.py

This will take a long time (16x full train+eval cycles). On Intel Mac CPU
expect several hours per config depending on sequence length — run this
on better hardware, or reduce GRID_OFFSETS / GRID_DURATIONS for a quick
pilot run first.

Output
------
    experiments/results/grid_search_summary.csv
    experiments/results/grid_search_heatmap.png
"""

import os
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_dataset import build_dataset
from dataset import infer_n_frames
from model import ModelConfig
from train import TrainConfig, train
from evaluate import evaluate


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

GRID_OFFSETS   = [15, 30, 45, 60]   # minutes before seizure onset (window end)
GRID_DURATIONS = [15, 30, 45, 60]   # minutes (window length)

DATA_ROOT        = 'data/processed'
CHECKPOINT_ROOT  = 'experiments/checkpoints'
RESULTS_ROOT     = 'experiments/results'
SUMMARY_CSV      = os.path.join(RESULTS_ROOT, 'grid_search_summary.csv')
HEATMAP_PATH     = os.path.join(RESULTS_ROOT, 'grid_search_heatmap.png')

# Set to a small number (e.g. 5) for a fast pilot run to sanity-check
# the whole pipeline before committing to the full grid.
MAX_EPOCHS_OVERRIDE = None   # None = use TrainConfig default (100)


# ---------------------------------------------------------------------------
# One full config run: build -> train -> evaluate
# ---------------------------------------------------------------------------

def run_one_config(offset_minutes: int, duration_minutes: int) -> dict:
    """
    Run the full pipeline for one (offset, duration) configuration.

    Returns
    -------
    dict with config name, timing, and all key metrics.
    On failure, returns a dict with 'status': 'failed' and the error message
    instead of raising — this lets the grid continue past a single bad config.
    """
    config_name = f'offset{offset_minutes}_dur{duration_minutes}'
    print('\n' + '#' * 65)
    print(f'#  CONFIG: {config_name}  '
          f'(offset={offset_minutes}min, duration={duration_minutes}min)')
    print('#' * 65)

    t_start = time.time()

    try:
        # --- Step 1: Build dataset for this config ---
        print(f'\n[{config_name}] Step 1/3 — Building dataset...')
        build_info = build_dataset(
            offset_minutes=offset_minutes,
            duration_minutes=duration_minutes,
            output_root=DATA_ROOT,
            config_name=config_name,
        )

        if build_info['n_preictal'] == 0 or build_info['n_interictal'] == 0:
            raise RuntimeError(
                f'Dataset build produced no usable sequences for '
                f'{config_name} (preictal={build_info["n_preictal"]}, '
                f'interictal={build_info["n_interictal"]}). Skipping.'
            )

        # --- Step 2: Train ---
        print(f'\n[{config_name}] Step 2/3 — Training...')
        n_frames = infer_n_frames(build_info['output_dir'])

        train_cfg = TrainConfig(
            metadata_path=build_info['metadata_path'],
            sequences_dir=build_info['output_dir'],
            config_name=config_name,
            checkpoint_dir=CHECKPOINT_ROOT,
            log_dir=os.path.join('experiments', 'logs'),
            results_dir=RESULTS_ROOT,
        )
        if MAX_EPOCHS_OVERRIDE is not None:
            train_cfg.max_epochs = MAX_EPOCHS_OVERRIDE

        model_cfg = ModelConfig(n_frames=n_frames)

        train_result = train(train_cfg, model_cfg)

        # --- Step 3: Evaluate ---
        print(f'\n[{config_name}] Step 3/3 — Evaluating...')
        eval_result = evaluate(
            metadata_path=build_info['metadata_path'],
            sequences_dir=build_info['output_dir'],
            checkpoint_path=train_result['checkpoint_path'],
            output_dir=os.path.join(RESULTS_ROOT, config_name, 'evaluation'),
            config_name=config_name,
        )

        elapsed = time.time() - t_start
        metrics = eval_result['metrics']

        print(f'\n[{config_name}] Complete in {elapsed/60:.1f} minutes. '
              f'Test AUC={metrics["auc"]:.4f}')

        return {
            'config_name':      config_name,
            'offset_minutes':   offset_minutes,
            'duration_minutes': duration_minutes,
            'status':           'success',
            'elapsed_minutes':  round(elapsed / 60, 1),
            'n_preictal':       build_info['n_preictal'],
            'n_interictal':     build_info['n_interictal'],
            'n_frames':         n_frames,
            'best_val_auc':     train_result['best_val_auc'],
            'best_epoch':       train_result['best_epoch'],
            'test_auc':         metrics['auc'],
            'test_sensitivity': metrics['sensitivity'],
            'test_specificity': metrics['specificity'],
            'test_f1':          metrics['f1'],
            'test_fpr_per_hour': metrics['fpr_per_hour'],
        }

    except Exception as e:
        elapsed = time.time() - t_start
        print(f'\n[{config_name}] FAILED after {elapsed/60:.1f} minutes: {e}')
        traceback.print_exc()

        return {
            'config_name':      config_name,
            'offset_minutes':   offset_minutes,
            'duration_minutes': duration_minutes,
            'status':           'failed',
            'elapsed_minutes':  round(elapsed / 60, 1),
            'error':            str(e),
        }


# ---------------------------------------------------------------------------
# Summary table and heatmap
# ---------------------------------------------------------------------------

def save_summary(all_results: list, csv_path: str):
    """Save the full grid search results as a CSV table."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(all_results)
    df = df.sort_values('test_auc', ascending=False, na_position='last')
    df.to_csv(csv_path, index=False)
    print(f'\nSummary table saved to {csv_path}')
    return df


def plot_heatmap(df: pd.DataFrame, save_path: str):
    """
    Plot a 4x4 heatmap of test AUC across offset (x) and duration (y).
    Failed configs are shown as gray/blank cells.
    """
    successful = df[df['status'] == 'success']
    if len(successful) == 0:
        print('No successful runs to plot.')
        return

    pivot = successful.pivot_table(
        index='duration_minutes',
        columns='offset_minutes',
        values='test_auc',
    )
    pivot = pivot.reindex(index=GRID_DURATIONS, columns=GRID_OFFSETS)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pivot.values, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')

    ax.set_xticks(range(len(GRID_OFFSETS)))
    ax.set_xticklabels([f'{o} min' for o in GRID_OFFSETS])
    ax.set_yticks(range(len(GRID_DURATIONS)))
    ax.set_yticklabels([f'{d} min' for d in GRID_DURATIONS])

    ax.set_xlabel('Offset before seizure onset', fontsize=12)
    ax.set_ylabel('Window duration', fontsize=12)
    ax.set_title('Test AUC — Window Configuration Grid Search', fontsize=13)

    # Annotate each cell with its AUC value
    for i in range(len(GRID_DURATIONS)):
        for j in range(len(GRID_OFFSETS)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='white' if val < 0.65 or val > 0.9 else 'black')
            else:
                ax.text(j, i, 'FAILED', ha='center', va='center',
                        fontsize=9, color='gray')

    plt.colorbar(im, ax=ax, label='Test AUC')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Heatmap saved to {save_path}')


def print_final_report(df: pd.DataFrame):
    """Print the ranked results and the best configuration."""
    print('\n' + '=' * 65)
    print('GRID SEARCH COMPLETE — FINAL RANKING')
    print('=' * 65)

    successful = df[df['status'] == 'success'].sort_values('test_auc', ascending=False)
    failed     = df[df['status'] == 'failed']

    if len(successful) > 0:
        print('\nTop 5 configurations by test AUC:')
        cols = ['config_name', 'offset_minutes', 'duration_minutes',
                'test_auc', 'test_sensitivity', 'test_specificity',
                'test_fpr_per_hour']
        print(successful[cols].head(5).to_string(index=False))

        best = successful.iloc[0]
        print(f'\nBest configuration: {best["config_name"]}')
        print(f'  Window: {best["duration_minutes"]} min long, '
              f'ending {best["offset_minutes"]} min before seizure onset')
        print(f'  Test AUC        : {best["test_auc"]:.4f}')
        print(f'  Sensitivity     : {best["test_sensitivity"]:.4f}')
        print(f'  Specificity     : {best["test_specificity"]:.4f}')
        print(f'  FPR/h           : {best["test_fpr_per_hour"]:.4f}')

    if len(failed) > 0:
        print(f'\n{len(failed)} configuration(s) failed:')
        print(failed[['config_name', 'error']].to_string(index=False))

    print('=' * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_grid_search():
    print('=' * 65)
    print('SeizureHorizon — Window Configuration Grid Search')
    print('=' * 65)
    print(f'Offsets   : {GRID_OFFSETS} minutes')
    print(f'Durations : {GRID_DURATIONS} minutes')
    print(f'Total configs: {len(GRID_OFFSETS) * len(GRID_DURATIONS)}')
    print()

    all_results = []
    grid_start  = time.time()

    total_configs = len(GRID_OFFSETS) * len(GRID_DURATIONS)
    config_num    = 0

    for duration in GRID_DURATIONS:
        for offset in GRID_OFFSETS:
            config_num += 1
            print(f'\n\n>>> Running config {config_num}/{total_configs} <<<')

            result = run_one_config(offset, duration)
            all_results.append(result)

            # Save incremental progress after every config, so a crash
            # partway through the grid doesn't lose completed results
            df = save_summary(all_results, SUMMARY_CSV)

    total_elapsed = time.time() - grid_start
    print(f'\n\nFull grid search completed in {total_elapsed/3600:.2f} hours.')

    # Final summary and visualization
    df = save_summary(all_results, SUMMARY_CSV)
    plot_heatmap(df, HEATMAP_PATH)
    print_final_report(df)

    return df


if __name__ == '__main__':
    run_grid_search()
