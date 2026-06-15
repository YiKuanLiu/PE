"""Per-fold breakdown + ablation + per-feature importance for the 21-feature lobe MIL.
21 特徵肺葉 MIL 的:逐折表現 + 消融 + 個別特徵重要度。

A) per-fold AUC (10 outer folds) for mean & rf pooling
B) feature-group ablation: each group ALONE + LEAVE-ONE-GROUP-OUT (marginal contribution)
C) individual feature ranking: RF importance (summed over lobes) + best-lobe univariate AUC

    python -m scripts.mil_report --feat results/mil/instances_lobe21.npz --splits results/swinunetr_i/splits.json
"""
import argparse
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from scripts.mil_train import load_bags, run_pooling

HP = dict(H=16, dropout=0.3, wd=1e-2, lr=1e-3, epochs=150, lam_ent=0.01)
GROUPS = {
    "VQ (M,V,R)":      [0, 1, 2],
    "HU mean/chg":     [3, 4, 5],
    "vol-ratio":       [8, 9],
    "vol-abs":         [6, 7],
    "mass/air-abs":    [10, 11, 12, 13],
    "first-order":     [14, 15, 16, 17],
    "HYPERDENSE":      [18, 19, 20],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    X, mask, y, files, names = load_bags(args.feat)          # X [125,5,21]
    folds = json.load(open(args.splits))["folds"]
    allidx = list(range(X.shape[2]))

    # ---- A) per-fold AUC ----
    for pool in ("mean", "rf"):
        auc, lo, hi, oof, _ = run_pooling(X, mask, y, folds, pool, args.seeds, HP, "cpu")
        print(f"\n=== per-fold AUC ({pool}); pooled {auc:.3f} [{lo:.3f}, {hi:.3f}] ===")
        print(f"{'fold':<6}{'n':>4}{'pos':>5}{'AUC':>8}")
        for i, f in enumerate(folds):
            te = np.array(f["test_idx"])
            a = roc_auc_score(y[te], oof[te]) if len(set(y[te])) > 1 else float("nan")
            print(f"{i:<6}{len(te):>4}{int(y[te].sum()):>5}{a:>8.3f}")

    # ---- B) feature-group ablation (rf) ----
    full = run_pooling(X, mask, y, folds, "rf", args.seeds, HP, "cpu")[0]
    print(f"\n=== feature-group ablation (rf); FULL(21) = {full:.3f} ===")
    print(f"{'group':<16}{'ALONE':>8}{'leave-out':>11}{'Δ vs full':>11}")
    for nm, g in GROUPS.items():
        alone = run_pooling(X[:, :, g], mask, y, folds, "rf", args.seeds, HP, "cpu")[0]
        rest = [j for j in allidx if j not in g]
        loo = run_pooling(X[:, :, rest], mask, y, folds, "rf", args.seeds, HP, "cpu")[0]
        print(f"{nm:<16}{alone:>8.3f}{loo:>11.3f}{loo-full:>+11.3f}")

    # ---- C) individual feature ranking ----
    Xf = X.reshape(X.shape[0], -1).copy()                    # [125, 5*21] lobe-major
    med = np.nanmedian(Xf, 0); med = np.where(np.isfinite(med), med, 0.0)
    Xf = np.where(np.isfinite(Xf), Xf, med)
    rf = RandomForestClassifier(600, max_depth=3, random_state=42, n_jobs=-1).fit(Xf, y)
    imp = rf.feature_importances_.reshape(5, X.shape[2]).sum(0)   # sum over lobes / 加總各肺葉
    uni = []
    for j in range(X.shape[2]):
        best = 0.5
        for l in range(5):
            c = X[:, l, j]; ok = np.isfinite(c)
            if ok.sum() > 50 and np.std(c[ok]) > 0:
                a = roc_auc_score(y[ok], c[ok]); best = max(best, a, 1 - a)
        uni.append(best)
    print("\n=== individual feature ranking (by RF importance summed over 5 lobes) ===")
    print(f"{'feature':<14}{'RF-imp':>9}{'best-lobe uniAUC':>18}")
    for j in np.argsort(-imp):
        print(f"{names[j]:<14}{imp[j]:>9.3f}{uni[j]:>18.3f}")


if __name__ == "__main__":
    main()
