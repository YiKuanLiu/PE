"""Nested-CV threshold selection for hyper_frac + sensitivity analysis (proposal 1).
在 nested CV 裡用 train 選 hyperdense 的 HU 上下限 + 敏感度分析。

One mat pass caches per-lobe HU histograms; hyper_frac(lo,hi) is then instant for any band, so we
can (a) inner-CV-select (lo,hi) per outer fold and apply to test, and (b) sweep bands to show the
result is robust to the band choice (and that an upper bound can exclude calcification).
一次 mat pass 快取逐肺葉 HU 直方圖;之後任何 [lo,hi] 的 hyper_frac 瞬間可得。
"""
import json
import os

import numpy as np
import scipy.io as sio
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

cfg = yaml.safe_load(open("configs/swinunetr_ie.yaml")); raw = cfg["data"]["raw_dir"]
d0 = np.load("results/mil/instances_lobe21.npz", allow_pickle=True)
y = d0["y"].astype(int); files = [str(f) for f in d0["files"]]
folds = json.load(open("results/swinunetr_i/splits.json"))["folds"]
HPATH = "results/mil/lobe_hist.npz"
EDGES = np.arange(-1000, 402, 2)                 # 2-HU bins / 2 HU 一格
CENT = (EDGES[:-1] + EDGES[1:]) / 2.0


def loc(fn):
    for s in ("Positive_Anon", "Negative_Anon"):
        p = os.path.join(raw, s, fn)
        if os.path.exists(p):
            return p


if os.path.exists(HPATH):
    z = np.load(HPATH); HST, TOT = z["HST"], z["TOT"]
else:
    N = len(files); HST = np.zeros((N, 5, len(EDGES) - 1), np.int32); TOT = np.zeros((N, 5))
    for i, fn in enumerate(files):
        p = loc(fn)
        if not p:
            continue
        m = sio.loadmat(p, variable_names=["T00", "T00_Lobe"])
        hu = np.asarray(m["T00"]).astype(np.float32) - 1024.0
        lob = np.asarray(m["T00_Lobe"]).astype(np.int16)
        for k, l in enumerate(range(1, 6)):
            mk = lob == l
            if mk.sum() < 50:
                continue
            HST[i, k] = np.histogram(hu[mk], bins=EDGES)[0]
            TOT[i, k] = mk.sum()
        if (i + 1) % 25 == 0:
            print(f"  hist {i+1}/{len(files)}", flush=True)
    np.savez(HPATH, HST=HST, TOT=TOT)
    print("cached", HPATH, flush=True)


def frac(lo, hi):                                # hyper_frac per lobe for band [lo,hi) -> [N,5]
    sel = (CENT >= lo) & (CENT < hi)
    f = HST[:, :, sel].sum(2) / np.maximum(TOT, 1)
    f[TOT < 50] = np.nan
    return f


def rf_oof(F):
    py, pp = [], []
    for fld in folds:
        tr, te = np.array(fld["train_idx"]), np.array(fld["test_idx"])
        Xt, Xe = F[tr].copy(), F[te].copy()
        med = np.nanmedian(Xt, 0); med = np.where(np.isfinite(med), med, 0)
        Xt = np.where(np.isfinite(Xt), Xt, med); Xe = np.where(np.isfinite(Xe), Xe, med)
        c = RandomForestClassifier(400, max_depth=3, random_state=0, n_jobs=-1).fit(Xt, y[tr])
        py += y[te].tolist(); pp += c.predict_proba(Xe)[:, 1].tolist()
    return roc_auc_score(py, pp)


LOS = [40, 50, 60, 70]; HIS = [100, 150, 200, 300]
bands = [(lo, hi) for lo in LOS for hi in HIS if hi > lo]

print("\n=== sensitivity: hyper_frac(5-lobe) RF AUC across bands (fixed) ===")
print("        " + "".join(f"hi={h:<6}" for h in HIS))
for lo in LOS:
    row = "".join(f"{rf_oof(frac(lo,h)):<8.3f}" if h > lo else f"{'-':<8}" for h in HIS)
    print(f"lo={lo:<4}{row}")
print(f"  fixed [50,150] = {rf_oof(frac(50,150)):.3f}")


# nested-CV-tuned threshold
def inner_pick(tr):
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    best, bb = -1, (50, 150)
    for lo, hi in bands:
        F = frac(lo, hi); a = []
        for it, iv in skf.split(F[tr], y[tr]):
            tri, vai = tr[it], tr[iv]
            Xt, Xe = F[tri].copy(), F[vai].copy()
            med = np.nanmedian(Xt, 0); med = np.where(np.isfinite(med), med, 0)
            Xt = np.where(np.isfinite(Xt), Xt, med); Xe = np.where(np.isfinite(Xe), Xe, med)
            c = RandomForestClassifier(300, max_depth=3, random_state=0, n_jobs=-1).fit(Xt, y[tri])
            if len(set(y[vai])) > 1:
                a.append(roc_auc_score(y[vai], c.predict_proba(Xe)[:, 1]))
        if a and np.mean(a) > best:
            best, bb = np.mean(a), (lo, hi)
    return bb


oof = np.full(len(y), np.nan); sel = []
for fld in folds:
    tr, te = np.array(fld["train_idx"]), np.array(fld["test_idx"])
    lo, hi = inner_pick(tr); sel.append((lo, hi))
    F = frac(lo, hi); Xt, Xe = F[tr].copy(), F[te].copy()
    med = np.nanmedian(Xt, 0); med = np.where(np.isfinite(med), med, 0)
    Xt = np.where(np.isfinite(Xt), Xt, med); Xe = np.where(np.isfinite(Xe), Xe, med)
    c = RandomForestClassifier(400, max_depth=3, random_state=0, n_jobs=-1).fit(Xt, y[tr])
    oof[te] = c.predict_proba(Xe)[:, 1]
print(f"\n=== nested-CV-tuned threshold ===")
print(f"  pooled AUC (threshold inner-selected per fold) = {roc_auc_score(y, oof):.3f}")
print(f"  selected (lo,hi) per fold: {sel}")
from collections import Counter
print(f"  most common band: {Counter(sel).most_common(3)}")
