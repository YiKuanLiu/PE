"""Supplementary Figure S2: learning curve of the per-lobe feature model (RF, 21 features, fixed [50,150]).
補充圖 S2:特徵模型 learning curve(訓練比例 50-100% → pooled OOF AUC,5 重抽 mean±std),與 consolidation.py 一致。
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

d0 = np.load("results/mil/instances_lobe21.npz", allow_pickle=True)
y = d0["y"].astype(int); N = len(y)
X = d0["bags"].astype(float).reshape(N, -1)                  # (125, 5*21)
folds = json.load(open("results/swinunetr_i/splits.json"))["folds"]


def imp(Xt, Xe):
    med = np.nanmedian(Xt, 0); med = np.where(np.isfinite(med), med, 0)
    return np.where(np.isfinite(Xt), Xt, med), np.where(np.isfinite(Xe), Xe, med)


def RF():
    return RandomForestClassifier(400, max_depth=3, random_state=0, n_jobs=-1)


FRACS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
means, stds = [], []
for frac in FRACS:
    a = []
    for ss in range(5):
        rr = np.random.default_rng(100 + ss); oof = np.full(N, np.nan)
        for f in folds:
            tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
            sub = rr.choice(tr, max(12, int(round(len(tr) * frac))), replace=False)
            if len(set(y[sub])) < 2:
                continue
            Xt, Xe = imp(X[sub], X[te]); oof[te] = RF().fit(Xt, y[sub]).predict_proba(Xe)[:, 1]
        m = np.isfinite(oof); a.append(roc_auc_score(y[m], oof[m]))
    means.append(float(np.mean(a))); stds.append(float(np.std(a)))
    print(f"  {int(frac*100)}% (~{int(round(112*frac))} train): {np.mean(a):.3f} ± {np.std(a):.3f}", flush=True)

means, stds = np.array(means), np.array(stds)
xp = [int(f * 100) for f in FRACS]
fig, ax = plt.subplots(figsize=(5.8, 4.6))
ax.plot(xp, means, "-o", color="#1f77b4", lw=2, ms=6)
ax.fill_between(xp, means - stds, means + stds, color="#1f77b4", alpha=0.18, label="± 1 SD (5 subsamples)")
for x, m in zip(xp, means):
    ax.text(x, m + 0.013, f"{m:.2f}", ha="center", fontsize=8.5)
ax.axhline(0.5, ls="--", c="0.6", lw=1)
ax.set_xlabel("Training data used (% of the full training set)")
ax.set_ylabel("Nested 10-fold pooled OOF AUC")
ax.set_ylim(0.5, 0.80); ax.set_xticks(xp)
ax.set_title("Learning curve of the per-lobe feature model", fontsize=11.5)
ax.legend(fontsize=8.5, loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig("results/figs/figS2_learning_curve.png", dpi=150, bbox_inches="tight")
print("saved results/figs/figS2_learning_curve.png")
