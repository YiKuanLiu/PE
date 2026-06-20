"""Standalone hyperdense-sign results (nested OOF + bootstrap CI).
單獨 hyperdense 徵象的結果(nested OOF + bootstrap CI),並對比聚合方式造成的差異。
"""
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

d = np.load("results/mil/instances_lobe21.npz", allow_pickle=True)
bags = d["bags"].astype(float); y = d["y"].astype(int); mask = d["mask"].astype(bool)
names = [str(x) for x in d["names"]]
folds = json.load(open("results/swinunetr_i/splits.json"))["folds"]
N = len(y)
HMAX, HP99, HFRAC = names.index("hyper_max"), names.index("hyper_p99"), names.index("hyper_frac")


def bootci(yy, ss, B=5000, seed=0):
    rng = np.random.default_rng(seed); a = []
    for _ in range(B):
        i = rng.integers(0, N, N)
        if len(set(yy[i])) > 1:
            a.append(roc_auc_score(yy[i], ss[i]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def meanpool(col):                                  # mean of feature over valid lobes / 逐病人對有效肺葉取平均
    x = bags[:, :, col]
    return np.array([np.nanmean(x[i][mask[i]]) if mask[i].any() else np.nanmean(x[i]) for i in range(N)])


def maxpool(col):                                   # worst-lobe / 取最大葉
    x = bags[:, :, col]
    return np.array([np.nanmax(x[i][mask[i]]) if mask[i].any() else np.nanmax(x[i]) for i in range(N)])


def rf_oof(cols):                                   # nested RF on per-lobe features (flattened 5xlen) / 攤平丟 RF
    oof = np.full(N, np.nan)
    for f in folds:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        Xt = bags[tr][:, :, cols].reshape(len(tr), -1); Xe = bags[te][:, :, cols].reshape(len(te), -1)
        med = np.nanmedian(Xt, 0); med = np.where(np.isfinite(med), med, 0)
        Xt = np.where(np.isfinite(Xt), Xt, med); Xe = np.where(np.isfinite(Xe), Xe, med)
        oof[te] = RandomForestClassifier(400, max_depth=3, random_state=0, n_jobs=-1).fit(Xt, y[tr]).predict_proba(Xe)[:, 1]
    return oof


def show(tag, score):
    a = roc_auc_score(y, score)
    if a < 0.5:
        a = roc_auc_score(y, -score); score = -score
    lo, hi = bootci(y, score)
    print(f"{tag:<46} AUC {a:.3f} [{lo:.2f}, {hi:.2f}]")


print("=== single hyperdense feature, simple aggregation (no training) ===")
show("hyper_frac  (mean over lobes)", meanpool(HFRAC))
show("hyper_frac  (max  over lobes)", maxpool(HFRAC))
show("hyper_p99   (mean over lobes)", meanpool(HP99))
show("hyper_max   (mean over lobes)", meanpool(HMAX))
print("\n=== aggregation artifact: SAME single feature, but via RF on 5 lobes ===")
show("hyper_frac  (RF on 5-lobe vector)", rf_oof([HFRAC]))
print("\n=== hyperdense GROUP (3 feats) and contrasts, nested RF OOF ===")
show("hyperdense 3 (hyper_max,p99,frac), RF", rf_oof([HMAX, HP99, HFRAC]))
show("all 21 features, RF", rf_oof(list(range(21))))
show("18 features WITHOUT hyperdense, RF", rf_oof(list(range(18))))
