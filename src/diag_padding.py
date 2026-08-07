# diag_padding.py — is zero-padding a class/split confound?
import os, numpy as np, pandas as pd
BASE = 'data/processed/offset15_dur15'; SEQ = os.path.join(BASE, 'sequences')
m = pd.read_csv(os.path.join(BASE, 'metadata.csv'))
TRAIN = [f'chb{p:02d}' for p in range(8, 25)]
VAL   = ['chb04', 'chb05', 'chb06', 'chb07']
def sp(p): return 'train' if p in TRAIN else ('val' if p in VAL else 'test')

rows = []
for _, r in m.iterrows():
    f = os.path.join(SEQ, str(r['filename']))
    if not os.path.exists(f):
        continue
    seq = np.load(f)                                   # (frames, 5, 18)
    pad = float((seq.sum(axis=(1, 2)) == 0).mean())    # fraction of all-zero frames
    rows.append({'split': sp(r['patient_id']),
                 'cls': 'preictal' if int(r['label']) == 1 else 'interictal',
                 'pad': pad})

d = pd.DataFrame(rows)
print(d.groupby(['split', 'cls'])['pad'].agg(['mean', 'median', 'max', 'count']).round(3))
print('\n% of sequences with ANY zero-pad frame:')
print(d.assign(p=d['pad'] > 0).groupby(['split', 'cls'])['p'].mean().round(3))