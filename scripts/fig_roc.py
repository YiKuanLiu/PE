"""Figure 4: ROC of the three key models (nested 10-fold pooled OOF).
圖四:三個關鍵模型的 ROC(nested 10-fold pooled OOF)。
Uses the SAME OOF arrays as final_stats.py so AUCs match Table 2 / 與 final_stats 同一份 OOF。
"""
import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

d0 = np.load("results/mil/instances_lobe21.npz", allow_pickle=True)
y = d0["y"].astype(int)
files = [str(f) for f in d0["files"]]
pos = {f: i for i, f in enumerate(files)}
N = len(y)

oof_feat = np.load("results/mil/oof_feat_mean.npz")["oof_feat"]
oof_hyb = np.array(json.load(open("results/mil/oof_hybrid_mask.json"))["oof"])
deep = np.full(N, np.nan)                                   # SwinUNETR, same loading as final_stats
for k in range(10):
    js = glob.glob(f"results/swinunetr_i/jobs/o{k}_refit_*.json")
    if js:
        dd = json.load(open(js[0]))
        for f_, s_ in zip(dd["test"]["filenames"], dd["test"]["y_score"]):
            if f_ in pos:
                deep[pos[f_]] = s_

MODELS = [
    ("Mask-guided hybrid", oof_hyb, "#1f77b4"),
    ("Per-lobe feature score", oof_feat, "#2ca02c"),
    ("SwinUNETR (deep)", deep, "#d62728"),
]

fig, ax = plt.subplots(figsize=(5.6, 5.4))
for name, oof, c in MODELS:
    m = np.isfinite(oof)
    fpr, tpr, _ = roc_curve(y[m], oof[m])
    auc = roc_auc_score(y[m], oof[m])
    ax.plot(fpr, tpr, color=c, lw=2.1, label=f"{name} (AUC {auc:.2f})")
ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect("equal")
ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity")
ax.set_title("ROC — nested 10-fold pooled OOF", fontsize=11.5)
ax.legend(loc="lower right", fontsize=9.5, frameon=False)
fig.savefig("results/figs/fig_roc.png", dpi=160, bbox_inches="tight")
print("saved results/figs/fig_roc.png")
for name, oof, c in MODELS:
    m = np.isfinite(oof)
    print(f"  {name}: AUC {roc_auc_score(y[m], oof[m]):.3f}  (n={int(m.sum())})")
