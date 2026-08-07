# diag_val_auc.py — per-patient vs pooled val AUC on the saved checkpoint
import os, numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from model import SeizurePredictor, ModelConfig
from dataset import SeizureDataset, FIXED_VAL_PATIENTS

CFG  = 'offset15_dur15'
BASE = os.path.join('data', 'processed', CFG)
META = os.path.join(BASE, 'metadata.csv')
SEQ  = os.path.join(BASE, 'sequences')
CKPT = os.path.join('experiments', 'checkpoints', CFG, 'best_model.pt')
NORM_STATS = None          # match the run you're diagnosing (norm_stats=None)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt  = torch.load(CKPT, map_location='cpu')
model = SeizurePredictor(ModelConfig(**ckpt['model_config']))
model.load_state_dict(ckpt['model_state']); model.to(device).eval()
meta = pd.read_csv(META)

def infer(patients):
    ds = SeizureDataset(meta, SEQ, patients, augment=False, norm_stats=NORM_STATS)
    probs, labs = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            seq, y = ds[i]
            logit, _ = model(seq.unsqueeze(0).to(device))
            probs.append(torch.sigmoid(logit).item()); labs.append(int(y))
    return np.array(probs), np.array(labs)

print(f'{"patient":8} {"n":>4} {"pos":>4} {"AUC":>7}')
all_p, all_l, per_aucs, zparts = [], [], [], []
for p in FIXED_VAL_PATIENTS:
    pr, la = infer([p]); all_p.append(pr); all_l.append(la)
    if len(np.unique(la)) > 1:
        a = roc_auc_score(la, pr); per_aucs.append(a)
        z = (pr - pr.mean()) / (pr.std() + 1e-8); zparts.append((z, la))
        print(f'{p:8} {len(la):>4} {int(la.sum()):>4} {a:>7.3f}')
    else:
        print(f'{p:8} {len(la):>4} {int(la.sum()):>4}   single-class')

P = np.concatenate(all_p); L = np.concatenate(all_l)
print(f'\nPOOLED val AUC       : {roc_auc_score(L, P):.3f}')
print(f'MEAN per-patient AUC : {np.mean(per_aucs):.3f}')
if zparts:
    Z  = np.concatenate([z for z, _ in zparts])
    ZL = np.concatenate([l for _, l in zparts])
    print(f'Z-POOLED val AUC     : {roc_auc_score(ZL, Z):.3f}')