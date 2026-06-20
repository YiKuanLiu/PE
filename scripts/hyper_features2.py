"""Test an expanded set of hyperdense / high-HU features (per the user's list).
測試擴充的 hyperdense/high-HU 特徵:看哪些有用,值不值得加進去。

Per-lobe (T00, within lobe): max HU, p99.9, p99.5, n_above(>=50), largest-CC volume, mean high-HU.
Patient-level: left-right high-HU asymmetry; central-crop high-HU count (hilar/central-vessel proxy).
Threshold fixed at 50 HU for this feasibility test (tuning it in nested CV is a separate step).
"""
import json
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import yaml
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

cfg = yaml.safe_load(open("configs/swinunetr_ie.yaml")); raw = cfg["data"]["raw_dir"]
d0 = np.load("results/mil/instances_lobe21.npz", allow_pickle=True)
y = d0["y"].astype(int); files = [str(f) for f in d0["files"]]
bags21 = d0["bags"]; mask21 = d0["mask"]
folds = json.load(open("results/swinunetr_i/splits.json"))["folds"]
LO, HI = 50.0, 300.0                      # high-HU band (clot+soft tissue, exclude bone) / 高 HU 帶
PL = ["maxHU", "p99.9", "p99.5", "nAbove", "largestCC", "meanHigh"]
PT = ["LRasym", "centralHigh"]


def loc(fn):
    for s in ("Positive_Anon", "Negative_Anon"):
        p = os.path.join(raw, s, fn)
        if os.path.exists(p):
            return p


N = len(files)
PLF = np.full((N, 5, len(PL)), np.nan)
PTF = np.full((N, len(PT)), np.nan)
for i, fn in enumerate(files):
    p = loc(fn)
    if not p:
        continue
    m = sio.loadmat(p, variable_names=["T00", "T00_Lobe", "xymm", "zmm"])
    xymm = float(np.ravel(m["xymm"])[0]); zmm = float(np.ravel(m["zmm"])[0]); vox = xymm * xymm * zmm
    hu = np.asarray(m["T00"]).astype(np.float32) - 1024.0
    lob = np.asarray(m["T00_Lobe"]).astype(np.int16)
    rl = [0.0, 0.0]                        # right(L1-3) / left(L4-5) high-HU counts
    for k, l in enumerate(range(1, 6)):
        mk = lob == l
        if mk.sum() < 50:
            continue
        H = hu[mk]; high = H[(H >= LO) & (H < HI)]
        PLF[i, k, 0] = min(float(H.max()), 500.0)
        PLF[i, k, 1] = float(np.percentile(H, 99.9))
        PLF[i, k, 2] = float(np.percentile(H, 99.5))
        PLF[i, k, 3] = float(high.size)
        PLF[i, k, 5] = float(high.mean()) if high.size else 0.0
        ixk = np.where(mk)                 # crop to lobe bbox -> CC labeling ~10x faster / 裁切加速
        slk = tuple(slice(int(c.min()), int(c.max()) + 1) for c in ixk)
        hbc = (hu[slk] >= LO) & (hu[slk] < HI) & mk[slk]
        if hbc.sum() > 0:
            lab, _ = ndimage.label(hbc)
            PLF[i, k, 4] = float(np.bincount(lab.ravel())[1:].max()) * vox
        else:
            PLF[i, k, 4] = 0.0
        rl[0 if l <= 3 else 1] += high.size
    PTF[i, 0] = abs(rl[0] - rl[1]) / (rl[0] + rl[1] + 1.0)
    lung = lob > 0
    if lung.any():
        ix = np.where(lung)
        bb = [(c.min(), c.max()) for c in ix]
        ctr = []
        for lo_, hi_ in bb[:2]:                # central third in x,y
            d = hi_ - lo_; ctr.append((int(lo_ + d / 3), int(hi_ - d / 3)))
        cz = (bb[2][0], bb[2][1])
        sub = hu[ctr[0][0]:ctr[0][1], ctr[1][0]:ctr[1][1], cz[0]:cz[1]]
        subl = lung[ctr[0][0]:ctr[0][1], ctr[1][0]:ctr[1][1], cz[0]:cz[1]]
        PTF[i, 1] = float((((sub >= LO) & (sub < HI)) & subl).sum())
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{N}", flush=True)

np.savez("results/mil/hyper2.npz", PLF=PLF, PTF=PTF, PL=np.array(PL), PT=np.array(PT), y=y)


def uni(v):
    ok = np.isfinite(v)
    a = roc_auc_score(y[ok], v[ok]); return max(a, 1 - a)


print("\n=== univariate AUC (per-lobe feats: max & mean over lobes) ===")
for j, nm in enumerate(PL):
    mx = np.nanmax(np.where(np.isfinite(PLF[:, :, j]), PLF[:, :, j], -np.inf), axis=1)
    mn = np.nanmean(np.where(np.isfinite(PLF[:, :, j]), PLF[:, :, j], np.nan), axis=1)
    print(f"  {nm:<12} max-lobe {uni(mx):.3f}   mean-lobe {uni(mn):.3f}")
print("=== univariate AUC (patient-level) ===")
for j, nm in enumerate(PT):
    print(f"  {nm:<12} {uni(PTF[:, j]):.3f}")
print("  (ref: hyper_frac 0.703 ; full 21-feat model 0.710)")


# combined nested RF: 21 vs 21+new
def nested_rf(Xb, extra=None):
    py, pp = [], []
    for f in folds:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        def flat(ix):
            X = Xb[ix].reshape(len(ix), -1)
            if extra is not None:
                X = np.concatenate([X, extra[ix]], axis=1)
            return X
        Xt, Xe = flat(tr), flat(te)
        med = np.nanmedian(Xt, 0); med = np.where(np.isfinite(med), med, 0)
        Xt = np.where(np.isfinite(Xt), Xt, med); Xe = np.where(np.isfinite(Xe), Xe, med)
        c = RandomForestClassifier(400, max_depth=3, random_state=0, n_jobs=-1).fit(Xt, y[tr])
        py += y[te].tolist(); pp += c.predict_proba(Xe)[:, 1].tolist()
    return roc_auc_score(py, pp)


aug_pl = np.concatenate([bags21, PLF], axis=2)            # 21 + 6 per-lobe
print(f"\n=== combined nested RF ===")
print(f"  21 features (baseline)           {nested_rf(bags21):.3f}")
print(f"  21 + 6 new per-lobe              {nested_rf(aug_pl):.3f}")
print(f"  21 + 6 per-lobe + 2 patient      {nested_rf(aug_pl, extra=PTF):.3f}")
print(f"  new per-lobe ONLY (6)            {nested_rf(PLF):.3f}")
