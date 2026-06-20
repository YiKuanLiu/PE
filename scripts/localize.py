"""Mechanistic localization of hyper_frac: is the [50,150] HU signal vascular/central (lumen sign)
or subpleural/parenchymal (infarct/atelectasis)? Cheap version: shell-vs-core + component morphology.
機制定位:hyperdense 訊號偏中央血管(lumen sign)還是胸膜下實質(梗塞/塌陷)?周邊vs中央 + 形態學。
"""
import glob
import os

import numpy as np
import scipy.io as sio
import yaml
from scipy import ndimage
from sklearn.metrics import roc_auc_score

cfg = yaml.safe_load(open("configs/swinunetr_ie.yaml")); raw = cfg["data"].get("raw_dir", "")
d0 = np.load("results/mil/instances_lobe21.npz", allow_pickle=True)
files = [str(f) for f in d0["files"]]; y = d0["y"].astype(int); N = len(y)
sp = np.load("results/mil/spacing.npz", allow_pickle=True); XY = sp["xymm"].astype(float); ZZ = sp["zmm"].astype(float)
idx = {}
for r in [raw, "/home/yikuan/PE"]:
    if r and os.path.isdir(r):
        for p in glob.glob(os.path.join(r, "**", "*.mat"), recursive=True):
            idx.setdefault(os.path.basename(p), p)
print(f"indexed {len(idx)} mats", flush=True)

LO, HI = 50.0, 150.0; SHELL_MM = 5.0; SMALL_MM3 = 65.0
core_frac = np.full(N, np.nan); shell_frac = np.full(N, np.nan); focality = np.full(N, np.nan)
smallfrac = np.full(N, np.nan); ncomp = np.full(N, np.nan); tot_mm3 = np.full(N, np.nan); largest_mm3 = np.full(N, np.nan)

for i, fn in enumerate(files):
    p = idx.get(fn) or idx.get(os.path.basename(fn))
    if not p:
        continue
    m = sio.loadmat(p, variable_names=["T00", "T00_Lobe"])
    hu = np.asarray(m["T00"]).astype(np.float32) - 1024.0
    lung = np.asarray(m["T00_Lobe"]).astype(np.int16) > 0
    if lung.sum() < 1000:
        continue
    a0 = np.where(lung.any((1, 2)))[0]; a1 = np.where(lung.any((0, 2)))[0]; a2 = np.where(lung.any((0, 1)))[0]
    sl = (slice(a0[0], a0[-1] + 1), slice(a1[0], a1[-1] + 1), slice(a2[0], a2[-1] + 1))
    hu, lung = hu[sl], lung[sl]
    xy = XY[i] if np.isfinite(XY[i]) else 0.98; zz = ZZ[i] if np.isfinite(ZZ[i]) else 2.5
    vox = float(xy * xy * zz)
    dist = ndimage.distance_transform_edt(lung, sampling=(float(xy), float(xy), float(zz)))   # dist from pleura
    hyp = (hu >= LO) & (hu < HI) & lung
    shell = lung & (dist <= SHELL_MM); core = lung & (dist > SHELL_MM)
    shell_frac[i] = (hyp & shell).sum() / max(int(shell.sum()), 1)
    core_frac[i] = (hyp & core).sum() / max(int(core.sum()), 1)
    lab, nc = ndimage.label(hyp)
    if nc > 0:
        sizes = np.bincount(lab.ravel())[1:].astype(float) * vox    # mm^3 per component
        tot = sizes.sum(); tot_mm3[i] = tot; ncomp[i] = nc; largest_mm3[i] = sizes.max()
        if tot > 0:
            focality[i] = sizes.max() / tot; smallfrac[i] = sizes[sizes < SMALL_MM3].sum() / tot
    else:
        tot_mm3[i] = 0; ncomp[i] = 0; largest_mm3[i] = 0
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{N}", flush=True)

np.savez("results/mil/localize.npz", core_frac=core_frac, shell_frac=shell_frac, focality=focality,
         smallfrac=smallfrac, ncomp=ncomp, tot_mm3=tot_mm3, largest_mm3=largest_mm3, y=y)


def summ(name, s):
    m = np.isfinite(s)
    if m.sum() < 10:
        print(f"{name}: n<10 skip"); return
    a = roc_auc_score(y[m], s[m]); a = max(a, 1 - a)
    print(f"{name:<28} AUC {a:.3f} | PE+ {np.nanmean(s[y == 1]):.4g} vs PE- {np.nanmean(s[y == 0]):.4g}", flush=True)


print("\n=== A. subpleural SHELL (<=5mm from pleura) vs central CORE ===", flush=True)
summ("hyper_frac CORE (central)", core_frac)
summ("hyper_frac SHELL (subpleural)", shell_frac)
print("\n=== B. morphology of [50,150] connected components ===", flush=True)
summ("focality (largest/total)", focality)
summ("small-comp frac (<65mm3)", smallfrac)
summ("n components", ncomp)
summ("total hyper (mm3)", tot_mm3)
summ("largest comp (mm3)", largest_mm3)
print("\nPE+ vs PE- medians:", flush=True)
for nm, arr in [("ncomp", ncomp), ("largest_mm3", largest_mm3), ("focality", focality), ("tot_mm3", tot_mm3)]:
    print(f"  {nm}: PE+ {np.nanmedian(arr[y == 1]):.4g} | PE- {np.nanmedian(arr[y == 0]):.4g}", flush=True)
print("ALLDONE", flush=True)
