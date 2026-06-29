"""Supplementary Figure S3: robustness of the per-lobe feature model to the clot-range HU window.
mean-MIL AUC heatmap;以 results/mil/threshold_auc.npz 快取 AUC grid(之後改顏色等不用重算)。
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scripts.mil_train import load_bags, run_pooling

X, mask, y, files, names = load_bags("results/mil/instances_lobe21.npz")
HF = [str(n) for n in names].index("hyper_frac")
folds = json.load(open("results/swinunetr_i/splits.json"))["folds"]
z = np.load("results/mil/lobe_hist.npz"); HST, TOT = z["HST"], z["TOT"]
EDGES = np.arange(-1000, 402, 2); CENT = (EDGES[:-1] + EDGES[1:]) / 2.0
HP = dict(H=16, dropout=0.3, wd=1e-2, lr=1e-3, epochs=150, lam_ent=0.0); SEEDS = 5
CACHE = "results/mil/threshold_auc.npz"


def frac(lo, hi):
    sel = (CENT >= lo) & (CENT < hi)
    f = HST[:, :, sel].sum(2) / np.maximum(TOT, 1); f[TOT < 50] = np.nan
    return f


if os.path.exists(CACHE):
    d = np.load(CACHE); A = d["A"]; LOS = list(d["LOS"]); HIS = list(d["HIS"])
    print("loaded cached AUC grid", CACHE)
else:
    LOS = [30, 40, 50, 60, 70, 80]; HIS = [100, 150, 200, 300]
    A = np.full((len(LOS), len(HIS)), np.nan)
    for i, lo in enumerate(LOS):
        for j, hi in enumerate(HIS):
            Xb = X.copy(); Xb[:, :, HF] = np.nan_to_num(frac(lo, hi), nan=0.0)
            A[i, j], *_ = run_pooling(Xb, mask, y, folds, "mean", SEEDS, HP, "cpu")
            print(f"  [{lo},{hi}] AUC {A[i,j]:.3f}", flush=True)
    np.savez(CACHE, A=A, LOS=np.array(LOS), HIS=np.array(HIS))

RED = "#ff0000"
fig, ax = plt.subplots(figsize=(6.4, 4.8))
im = ax.imshow(A, cmap="YlGnBu", vmin=0.66, vmax=0.74, aspect="auto")
for i in range(len(LOS)):
    for j in range(len(HIS)):
        ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=11,
                color="white" if A[i, j] > 0.715 else "black")
ax.set_xticks(range(len(HIS))); ax.set_xticklabels(HIS)
ax.set_yticks(range(len(LOS))); ax.set_yticklabels(LOS)
ax.set_xlabel("Upper bound of clot-range window (HU)")
ax.set_ylabel("Lower bound (HU)")
ax.set_title("Robustness of the 21-feature model to the clot-range window", fontsize=11)
cb = fig.colorbar(im, ax=ax); cb.set_label("Nested 10-fold pooled OOF AUC")
jx, iy = list(HIS).index(150), list(LOS).index(50)
ax.add_patch(plt.Rectangle((jx - 0.5, iy - 0.5), 1, 1, fill=False, edgecolor=RED, lw=3))
ax.text(jx, iy - 0.66, "a priori [50,150]", ha="center", fontsize=9, color=RED, fontweight="bold")
fig.tight_layout(); fig.savefig("results/figs/figS3_threshold.png", dpi=150, bbox_inches="tight")
print("saved figS3_threshold.png")
