"""Comprehensive final stats: per-method AUC/CI/metrics + pairwise paired-bootstrap Δ & p-values.
完整統計:各方法 AUC/CI/全指標 + 兩兩配對 bootstrap 差值與 p value。所有方法以同一份 nested 10-fold OOF。
"""
import glob
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

d0 = np.load("results/mil/instances_lobe21.npz", allow_pickle=True)
bags = d0["bags"].astype(float); y = d0["y"].astype(int); files = [str(f) for f in d0["files"]]
pos = {f: i for i, f in enumerate(files)}; N = len(y)
folds = json.load(open("results/swinunetr_i/splits.json"))["folds"]


def rf_oof(X):
    oof = np.full(N, np.nan)
    for f in folds:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"]); Xt, Xe = X[tr], X[te]
        med = np.nanmedian(Xt, 0); med = np.where(np.isfinite(med), med, 0)
        Xt = np.where(np.isfinite(Xt), Xt, med); Xe = np.where(np.isfinite(Xe), Xe, med)
        oof[te] = RandomForestClassifier(400, max_depth=3, random_state=0, n_jobs=-1).fit(Xt, y[tr]).predict_proba(Xe)[:, 1]
    return oof


OOF, TRAIN = {}, {}
OOF["Features (21)"] = np.load("results/mil/oof_feat_mean.npz")["oof_feat"]
for key, fn in [("Image CNN (no mask)", "oof_image_nomask"), ("Hybrid (no mask)", "oof_hybrid_nomask"),
                ("Image CNN + mask", "oof_image_mask"), ("Hybrid + mask", "oof_hybrid_mask")]:
    d = json.load(open(f"results/mil/{fn}.json")); OOF[key] = np.array(d["oof"]); TRAIN[key] = d["train_auc"]
deep = np.full(N, np.nan)
for k in range(10):
    js = glob.glob(f"results/swinunetr_i/jobs/o{k}_refit_*.json")
    if js:
        dd = json.load(open(js[0]))
        for f_, s_ in zip(dd["test"]["filenames"], dd["test"]["y_score"]):
            if f_ in pos:
                deep[pos[f_]] = s_
OOF["SwinUNETR-I (DL)"] = deep
dr = np.load("results/radiomics/features.npz", allow_pickle=True); Xr = np.full((N, dr["X"].shape[1]), np.nan)
for i, f_ in enumerate([str(x) for x in dr["files"]]):
    if f_ in pos:
        Xr[pos[f_]] = dr["X"][i]
OOF["Radiomics"] = rf_oof(Xr)
OOF["V/Q features"] = np.load("results/mil/oof_feat_mean.npz")["oof_vq"]

rng = np.random.default_rng(0); B = 5000
idxs = [rng.integers(0, N, N) for _ in range(B)]


def auc_b(oof, idx):
    o = oof[idx]; m = np.isfinite(o); yy = y[idx][m]
    return roc_auc_score(yy, o[m]) if len(set(yy)) > 1 else np.nan


def metrics(oof):
    m = np.isfinite(oof); yy = y[m]; pr = (oof[m] > 0.5).astype(int)
    tp = ((pr == 1) & (yy == 1)).sum(); fn = ((pr == 0) & (yy == 1)).sum()
    tn = ((pr == 0) & (yy == 0)).sum(); fp = ((pr == 1) & (yy == 0)).sum()
    return dict(AUC=roc_auc_score(yy, oof[m]), Sens=tp/(tp+fn) if tp+fn else np.nan,
                Spec=tn/(tn+fp) if tn+fp else np.nan, PPV=tp/(tp+fp) if tp+fp else np.nan,
                NPV=tn/(tn+fn) if tn+fn else np.nan, F1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else np.nan)


BA = {k: np.array([auc_b(o, ix) for ix in idxs]) for k, o in OOF.items()}
order = ["Hybrid + mask", "Image CNN + mask", "Features (21)", "Hybrid (no mask)",
         "Radiomics", "V/Q features", "SwinUNETR-I (DL)", "Image CNN (no mask)"]
print("=== per-method (nested 10-fold OOF) ===")
print(f"{'method':<22}{'AUC[95% CI]':>20}{'Se':>6}{'Sp':>6}{'PPV':>6}{'NPV':>6}{'F1':>6}{'train/gap':>14}")
res = {}
for k in order:
    mt = metrics(OOF[k]); lo, hi = np.nanpercentile(BA[k], 2.5), np.nanpercentile(BA[k], 97.5)
    g = f"{TRAIN[k]:.2f}/{TRAIN[k]-mt['AUC']:+.2f}" if k in TRAIN else "—"
    aucstr = f"{mt['AUC']:.3f}[{lo:.2f},{hi:.2f}]"
    print(f"{k:<22}{aucstr:>20}{mt['Sens']:>6.2f}{mt['Spec']:>6.2f}{mt['PPV']:>6.2f}{mt['NPV']:>6.2f}{mt['F1']:>6.2f}{g:>14}")
    res[k] = dict(metrics=mt, ci=[float(lo), float(hi)], train_auc=TRAIN.get(k))

print("\n=== paired bootstrap Δ (A − B), 95% CI, two-sided p (same resamples) ===")
pairs = [("Hybrid + mask", "Features (21)"), ("Hybrid + mask", "Hybrid (no mask)"),
         ("Image CNN + mask", "Image CNN (no mask)"), ("Features (21)", "SwinUNETR-I (DL)"),
         ("Features (21)", "Radiomics"), ("Hybrid + mask", "SwinUNETR-I (DL)"),
         ("Image CNN + mask", "SwinUNETR-I (DL)")]
res["pairs"] = []
for a, b in pairs:
    d = BA[a] - BA[b]; d = d[np.isfinite(d)]
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    sig = "  SIG" if (lo > 0 or hi < 0) else ""
    print(f"  {a:<20} − {b:<20} Δ={d.mean():+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}{sig}")
    res["pairs"].append(dict(a=a, b=b, dAUC=float(d.mean()), ci=[float(lo), float(hi)], p=float(p)))
json.dump(res, open("results/mil/final_stats.json", "w"), indent=2)
print("\nsaved results/mil/final_stats.json")
