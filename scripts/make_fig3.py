"""Figure 4 (main, single panel) + Supplementary Figure S1 (spatial localization, 2 panels):
characterization of the pulmonary hyperattenuation signal. Reads lobe_hist.npz + localize2.npz.
圖四(主):HU band 分層;補充圖 S1:距離區帶 + shell-vs-core。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

y = np.load("results/mil/instances_lobe21.npz", allow_pickle=True)["y"].astype(int)
z = np.load("results/mil/lobe_hist.npz"); HST, TOT = z["HST"], z["TOT"]
EDGES = np.arange(-1000, 402, 2); CENT = (EDGES[:-1] + EDGES[1:]) / 2.0
loc = np.load("results/mil/localize2.npz")
hf_q, hf_shell, hf_core, shellvol = loc["hf_q"], loc["hf_shell"], loc["hf_core"], loc["shellvol"]


def frac(lo, hi):
    s = (CENT >= lo) & (CENT < hi)
    f = HST[:, :, s].sum(2) / np.maximum(TOT, 1); f[TOT < 50] = np.nan
    return f


def maxlobe(a):
    return np.nanmax(np.where(np.isfinite(a), a, -np.inf), axis=1)


def Adir(s):
    m = np.isfinite(s); a = roc_auc_score(y[m], s[m]); return max(a, 1 - a)


def A(s):
    m = np.isfinite(s); return roc_auc_score(y[m], s[m])


# ===== Figure 4 (main): attenuation-band stratification (single panel) =====
BANDS = [("0–50 HU\n(consolidation/\neffusion)", 0, 50),
         ("50–100 HU\n(clot range)", 50, 100),
         ("100–200 HU\n(dense/\ncalcification)", 100, 200)]
ba = [Adir(maxlobe(frac(lo, hi))) for _, lo, hi in BANDS]
fig, ax = plt.subplots(figsize=(5.4, 4.7))
ax.bar(range(3), ba, color=["#bdbdbd", "#c0392b", "#bdbdbd"], alpha=0.9, width=0.62)
for i, a in enumerate(ba):
    ax.text(i, a + 0.007, f"{a:.2f}", ha="center", fontsize=12)
ax.axhline(0.5, ls="--", c="0.5", lw=1)
ax.set_xticks(range(3)); ax.set_xticklabels([b[0] for b in BANDS], fontsize=9.5)
ax.set_ylim(0.5, 0.78); ax.set_ylabel("AUC (single-feature, all 125)")
ax.set_title("Attenuation-band stratification of the\npulmonary hyperattenuation signal", fontsize=11.5)
fig.tight_layout(); fig.savefig("results/figs/fig4_band.png", dpi=150, bbox_inches="tight")
print("saved fig4_band.png  band AUCs (0-50/50-100/100-200):", [round(a, 3) for a in ba])

# ===== Supplementary Figure S1: spatial localization (zones + shell-vs-core) =====
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
zones = ["Outer third\n(subpleural)", "Middle third", "Inner third\n(central)"]
za = [A(hf_q[:, k]) for k in range(3)]
ax1.bar(range(3), za, color="#4c72b0", alpha=0.88, width=0.6)
for i, a in enumerate(za):
    ax1.text(i, a + 0.006, f"{a:.2f}", ha="center", fontsize=10)
ax1.axhline(0.5, ls="--", c="0.5", lw=1)
ax1.set_xticks(range(3)); ax1.set_xticklabels(zones, fontsize=8.5)
ax1.set_ylim(0.5, 0.80); ax1.set_ylabel("AUC (single-feature, all 125)")
ax1.set_title("(a) Equal-volume zones by lung depth\ndistributed (inner vs outer p = 0.34)", fontsize=10)
THR = [5, 10, 15]
sh = [A(hf_shell[:, j]) for j in range(3)]; co = [A(hf_core[:, j]) for j in range(3)]
vol = [float(np.nanmean(shellvol[:, j])) for j in range(3)]
x = np.arange(3); w = 0.38
ax2.bar(x - w / 2, sh, w, label="subpleural shell", color="#dd8452")
ax2.bar(x + w / 2, co, w, label="deeper core", color="#55a868")
for i in range(3):
    ax2.text(x[i] - w / 2, sh[i] + 0.006, f"{sh[i]:.2f}", ha="center", fontsize=8)
    ax2.text(x[i] + w / 2, co[i] + 0.006, f"{co[i]:.2f}", ha="center", fontsize=8)
ax2.axhline(0.5, ls="--", c="0.5", lw=1)
ax2.set_xticks(x); ax2.set_xticklabels([f"{t} mm\n({v:.0%})" for t, v in zip(THR, vol)], fontsize=8.5)
ax2.set_ylim(0.5, 0.80)
ax2.set_title("(b) Shell vs core, by shell thickness\ngap collapses with fair volume", fontsize=10)
ax2.legend(fontsize=8, loc="upper right")
fig2.suptitle("Spatial localization of the pulmonary hyperattenuation signal", fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.95]); fig2.savefig("results/figs/figS1_spatial.png", dpi=150, bbox_inches="tight")
print("saved figS1_spatial.png  zone AUCs (outer/mid/inner):", [round(a, 3) for a in za])
