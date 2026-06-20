"""Hybrid (no-mask) CNN but with the hyper_frac FEATURE recomputed per fold using the
train-selected HU band (tuned_bands.npz), to see if it differs from the fixed-[50,150] 0.716.
Hybrid(no mask)CNN,但 hyper_frac 特徵改成每折用 train 選的 band 重算,對照固定 [50,150] 的 0.716。
"""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import scripts.mil_cnn_mask as M

M.CH = 1                                              # no-mask, CT-only image channel / 純 CT,不加 mask

dfe = np.load("results/mil/instances_lobe21.npz", allow_pickle=True)
FEAT = dfe["bags"].astype(np.float64); MASK = dfe["mask"].astype(np.float64); y = dfe["y"].astype(int)
names = [str(x) for x in dfe["names"]]; HF = names.index("hyper_frac")
IMG = np.load("results/mil/patches_t00_96.npz")["bags_img"]
folds = json.load(open("results/swinunetr_i/splits.json"))["folds"]
sel = [tuple(int(v) for v in b) for b in np.load("results/mil/tuned_bands.npz")["sel"]]

z = np.load("results/mil/lobe_hist.npz"); HST, TOT = z["HST"], z["TOT"]
EDGES = np.arange(-1000, 402, 2); CENT = (EDGES[:-1] + EDGES[1:]) / 2.0


def frac(lo, hi):                                    # hyper_frac per lobe for band [lo,hi) -> [N,5]
    s = (CENT >= lo) & (CENT < hi)
    f = HST[:, :, s].sum(2) / np.maximum(TOT, 1)
    f[TOT < 50] = np.nan
    return f


# --- sanity: histogram frac(50,150) must match the cached hyper_frac column (alignment) ---
fc = frac(50, 150); dd = np.abs(fc - FEAT[:, :, HF])
print(f"[sanity] |frac(50,150) - cached hyper_frac|  max={np.nanmax(dd):.5f}  mean={np.nanmean(dd):.5f}", flush=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
IMG_t = torch.from_numpy(IMG).to(dev)
SEEDS, EP, BS, POOL, MODE = 5, 60, 32, "max", "hybrid"
print(f"hybrid no-mask, FEATURE band per-fold train-selected | dev {dev} | bands {sel}", flush=True)

oof = np.full(len(y), np.nan); tr_aucs = []
for fi, f in enumerate(folds):
    tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
    lo, hi = sel[fi]
    FEATf = FEAT.copy(); FEATf[:, :, HF] = frac(lo, hi)      # swap in per-fold hyper_frac / 換成本折 band 的 hyper_frac
    pte, ptr = [], []
    for s in range(SEEDS):
        a, b = M.run_fold(IMG_t, FEATf, MASK, y, tr, te, MODE, POOL, s, dev, EP, BS, band=(lo, hi))
        pte.append(a); ptr.append(b)
    oof[te] = np.mean(pte, axis=0)
    if len(set(y[tr])) > 1:
        tr_aucs.append(roc_auc_score(y[tr], np.mean(ptr, axis=0)))
    print(f"  fold {fi} band {(lo,hi)} done", flush=True)

auc = roc_auc_score(y, oof)
print(f"\nRESULT hybrid no-mask, FEATURE band train-selected: AUC {auc:.3f} | "
      f"train {np.mean(tr_aucs):.3f} (gap {np.mean(tr_aucs)-auc:+.3f})", flush=True)
print("ref: hybrid no-mask FIXED [50,150] = 0.716", flush=True)
json.dump({"oof": oof.tolist(), "y": y.tolist(), "auc": auc, "sel": sel,
           "train_auc": float(np.mean(tr_aucs))}, open("results/mil/oof_hybrid_nomask_featband.json", "w"))
print("DONE", flush=True)
