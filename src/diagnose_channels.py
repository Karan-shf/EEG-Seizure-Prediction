# diagnose_channels.py  — run on your machine, no rebuild needed
import os, numpy as np, pandas as pd

CFG = 'offset15_dur15'
BASE = os.path.join('data', 'processed', CFG)
SEQ_DIR = os.path.join(BASE, 'sequences')
META = os.path.join(BASE, 'metadata.csv')

CANON = ['FP1-F7','F7-T7','T7-P7','P7-O1','FP1-F3','F3-C3','C3-P3','P3-O1',
         'FP2-F4','F4-C4','C4-P4','P4-O2','FP2-F8','F8-T8','T8-P8','P8-O2','FZ-CZ','CZ-PZ']

TRAIN = [f'chb{n:02d}' for n in range(8, 25)]
VAL   = ['chb04','chb05','chb06','chb07']
TEST  = ['chb01','chb02','chb03']
def split_of(p): return 'train' if p in TRAIN else 'val' if p in VAL else 'test' if p in TEST else '?'

meta = pd.read_csv(META)
# dead[patient][channel] = count of sequences where that channel is entirely zero on REAL frames
from collections import defaultdict
dead  = defaultdict(lambda: np.zeros(18, dtype=int))
total = defaultdict(int)

for _, row in meta.iterrows():
    p = row['patient_id']
    seq = np.load(os.path.join(SEQ_DIR, row['filename']))      # (frames, 5, 18)
    real = seq[seq.sum(axis=(1, 2)) != 0]                      # drop padded frames
    if real.size == 0:
        continue
    ch_energy = np.abs(real).sum(axis=(0, 1))                  # (18,) energy per channel
    dead[p] += (ch_energy == 0).astype(int)
    total[p] += 1

print(f'{"patient":8} {"split":6} {"seqs":>5} {"avg_dead/18":>12}   always-dead canonical channels')
for p in sorted(total):
    always = [CANON[i] for i in range(18) if dead[p][i] == total[p]]
    avg = dead[p].sum() / max(total[p], 1)
    print(f'{p:8} {split_of(p):6} {total[p]:>5} {avg:>12.1f}   {always}')

print('\n=== per-split mean dead channels ===')
for s in ('train','val','test'):
    ps = [p for p in total if split_of(p)==s]
    if ps:
        m = np.mean([dead[p].sum()/total[p] for p in ps])
        print(f'{s:6}: {m:.2f} dead channels / 18 (patients: {ps})')