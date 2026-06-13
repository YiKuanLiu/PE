"""Optimized abstention (paper-style) on fused features, nested 10-fold.
最佳化棄答（論文風格）套在融合特徵上，nested 10 折。

Per outer fold: fit logistic on train, then choose (low, high) probability thresholds on the
TRAIN set that maximise decisive accuracy subject to train-abstention <= budget; apply to TEST.
Pool test decisive predictions -> decisive accuracy / sens / spec + abstention rate.
每折：訓練集擬合 logistic，於訓練集選 (low,high) 閾值最大化「決策後 accuracy」（棄答率<=上限），套到測試折。
直接對標論文 17% 棄答 -> 0.72 accuracy。
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

GRID = np.linspace(0.20, 0.80, 31)


def opt_lohi(p, y, budget):
    """Pick (low,high) maximising decisive accuracy with abstention<=budget on this set. / 選棄答閾值。"""
    best = (-1, 0.5, 0.5, 0.0)
    for lo in GRID:
        for hi in GRID:
            if hi <= lo:
                continue
            dec = (p < lo) | (p > hi)
            if dec.sum() == 0:
                continue
            ab = 1.0 - dec.mean()
            if ab > budget:
                continue
            pred = (p[dec] > hi).astype(int)
            acc = (pred == y[dec]).mean()
            # prefer higher accuracy, then lower abstention / 先比 accuracy 再比棄答率
            if acc > best[0] + 1e-9 or (abs(acc - best[0]) < 1e-9 and ab < best[3]):
                best = (acc, lo, hi, ab)
    return best[1], best[2]


def run(X, y, folds, budget, seed=42, tag=""):
    yt, pt, abst = [], [], 0
    for f in folds:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        Xtr, Xte = X[tr].copy(), X[te].copy()
        med = np.nanmedian(Xtr, axis=0); med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med); Xte = np.where(np.isfinite(Xte), Xte, med)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), y[tr])
        ptr = clf.predict_proba(sc.transform(Xtr))[:, 1]
        pte = clf.predict_proba(sc.transform(Xte))[:, 1]
        lo, hi = opt_lohi(ptr, y[tr], budget)
        dec = (pte < lo) | (pte > hi)
        abst += int((~dec).sum())
        for j, d in enumerate(dec):
            if d:
                yt.append(int(y[te[j]])); pt.append(int(pte[j] > hi))
    yt = np.array(yt); pt = np.array(pt)
    acc = (yt == pt).mean()
    tp = int(((pt == 1) & (yt == 1)).sum()); fn = int(((pt == 0) & (yt == 1)).sum())
    tn = int(((pt == 0) & (yt == 0)).sum()); fp = int(((pt == 1) & (yt == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ab_rate = abst / len(y)
    print(f"  {tag:<26} budget {int(budget*100):>2}%  ->  decisive acc {acc:.3f} "
          f"(sens {sens:.3f} spec {spec:.3f}) | abstained {ab_rate*100:.0f}% (n_decisive={len(yt)})")
    return acc, ab_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    splits = json.load(open(args.splits)); folds = splits["folds"]; files = list(splits["filenames"])
    pos = {f: i for i, f in enumerate(files)}; n = len(files)
    df = pd.read_csv(cfg["data"]["label_file"], header=None)
    lab = dict(zip(df.iloc[:, 0], df.iloc[:, 1])); y = np.array([int(lab[f]) for f in files])

    def align(p):
        d = np.load(p, allow_pickle=True); src = list(d["files"]); X = d["X"]
        out = np.full((n, X.shape[1]), np.nan)
        for i, f in enumerate(src):
            if f in pos:
                out[pos[f]] = X[i]
        return out

    Xperf = align("results/perfusion/perf_features.npz")
    Xrad = align("results/radiomics/features.npz")
    deep = np.full(n, np.nan)
    for k in range(len(folds)):
        js = glob.glob(f"results/swinunetr_i/jobs/o{k}_refit_*.json")
        if js:
            d = json.load(open(js[0]))
            for fn, scr in zip(d["test"]["filenames"], d["test"]["y_score"]):
                if fn in pos:
                    deep[pos[fn]] = scr

    feats = {"perf+radiomics": np.column_stack([Xperf, Xrad]),
             "deep+perf+radiomics": np.column_stack([deep, Xperf, Xrad])}
    print("=== optimized abstention (paper-style), nested 10-fold ===")
    print("paper reference: 17% abstain -> 0.72 acc (0.75 sens / 0.69 spec); forced 0.66 acc\n")
    for name, X in feats.items():
        print(f"AUC ref / feature set: {name}")
        for b in (0.0, 0.10, 0.17, 0.25):
            run(X, y, folds, b, tag=name)
        print()


if __name__ == "__main__":
    main()
