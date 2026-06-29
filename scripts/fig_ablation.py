"""Figure 3: feature-group ablation of the per-lobe feature model (mean-pooling MIL, 5 seeds, 150 ep).
圖三:逐肺葉特徵模型的特徵組消融(mean-MIL,與 headline 0.712 一致)。
(a) leave-one-group-out ΔAUC; (b) full vs no-hyperattenuation vs hyperattenuation-fraction-alone.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scripts.mil_train import load_bags, run_pooling

X, mask, y, files, names = load_bags("results/mil/instances_lobe21.npz")
folds = json.load(open("results/swinunetr_i/splits.json"))["folds"]
HP = dict(H=16, dropout=0.3, wd=1e-2, lr=1e-3, epochs=150, lam_ent=0.0)
SEEDS = 5
GROUPS = {
    "Ventilation/perfusion": [0, 1, 2],
    "HU statistics":         [3, 4, 5],
    "Volume":                [6, 7, 8, 9],
    "Mass / air":            [10, 11, 12, 13],
    "First-order":           [14, 15, 16, 17],
    "Hyperattenuation":      [18, 19, 20],
}
ALL = list(range(21))
HF = names.index("hyper_frac") if "hyper_frac" in [str(n) for n in names] else 20

def auc(idx):
    a, _, _, _, _ = run_pooling(X[:, :, idx], mask, y, folds, "mean", SEEDS, HP, "cpu")
    return a

full = auc(ALL)
loo = {nm: full - auc([i for i in ALL if i not in idx]) for nm, idx in GROUPS.items()}
no_hyper = auc([i for i in ALL if i not in GROUPS["Hyperattenuation"]])
hfrac = auc([HF])
print(f"full(21) {full:.3f}")
for nm, d in loo.items():
    print(f"  remove {nm:<24} dAUC {d:+.3f}")
print(f"no-hyperattenuation {no_hyper:.3f} | hyper_frac alone {hfrac:.3f}")

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.6))
items = sorted(loo.items(), key=lambda kv: kv[1])
labels = [k for k, _ in items]; vals = [v for _, v in items]
cols = ["#c0392b" if k == "Hyperattenuation" else "#9aa0a6" for k in labels]
ax0.barh(range(len(labels)), vals, color=cols, alpha=0.9)
for i, v in enumerate(vals):
    ax0.text(v + 0.0015, i, f"{v:+.3f}", va="center", fontsize=9)
ax0.axvline(0, c="0.4", lw=1)
ax0.set_yticks(range(len(labels))); ax0.set_yticklabels(labels, fontsize=9)
ax0.set_xlabel("ΔAUC when the group is removed (leave-one-group-out)")
ax0.set_title("(a) Only the hyperattenuation group matters", fontsize=10.5)

b_lab = ["Full\n(21 features)", "Without\nhyperattenuation", "Hyperattenuation\nfraction alone"]
b_val = [full, no_hyper, hfrac]; b_col = ["#2ca02c", "#9aa0a6", "#c0392b"]
ax1.bar(range(3), b_val, color=b_col, alpha=0.9, width=0.6)
for i, v in enumerate(b_val):
    ax1.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
ax1.axhline(0.5, ls="--", c="0.5", lw=1)
ax1.set_xticks(range(3)); ax1.set_xticklabels(b_lab, fontsize=9)
ax1.set_ylim(0.5, 0.80); ax1.set_ylabel("AUC (nested 10-fold pooled OOF)")
ax1.set_title("(b) The single feature ≈ the full model", fontsize=10.5)

fig.suptitle("Feature-group ablation of the per-lobe feature model", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("results/figs/fig_ablation.png", dpi=150, bbox_inches="tight")
print("saved results/figs/fig_ablation.png")
