"""Fig: spatial localization of the pulmonary hyperattenuation signal (supports Results 3.4).
定位圖:(a) 等體積距離區帶 AUC(顯示分布性、非中央)；(b) 多門檻 shell-vs-core(gap 隨殼變厚而崩)。
Reads results/mil/localize2.npz (no recompute of masks). 讀快取,不重算。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

d = np.load("results/mil/localize2.npz")
y = d["y"].astype(int); hf_q = d["hf_q"]; hf_shell = d["hf_shell"]; hf_core = d["hf_core"]; shellvol = d["shellvol"]
THR = [5, 10, 15]


def A(s):
    m = np.isfinite(s); return roc_auc_score(y[m], s[m])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

# (a) equal-volume tertiles by distance-from-pleura
zones = ["Outer third\n(subpleural)", "Middle third", "Inner third\n(central)"]
aucs = [A(hf_q[:, z]) for z in range(3)]
ax1.bar(range(3), aucs, color="#4c72b0", alpha=0.88, width=0.6)
for i, a in enumerate(aucs):
    ax1.text(i, a + 0.006, f"{a:.3f}", ha="center", fontsize=10)
ax1.axhline(0.5, ls="--", c="0.5", lw=1); ax1.text(2.35, 0.505, "chance", color="0.5", fontsize=8)
ax1.set_xticks(range(3)); ax1.set_xticklabels(zones, fontsize=9)
ax1.set_ylim(0.5, 0.80); ax1.set_ylabel("AUC (single-feature, all 125)")
ax1.set_title("(a) Equal-volume zones by lung depth\nsignal is distributed (inner vs outer p = 0.34)", fontsize=10)

# (b) multi-threshold subpleural shell vs deeper core
sh = [A(hf_shell[:, j]) for j in range(3)]; co = [A(hf_core[:, j]) for j in range(3)]
vol = [float(np.nanmean(shellvol[:, j])) for j in range(3)]
x = np.arange(3); w = 0.38
ax2.bar(x - w / 2, sh, w, label="subpleural shell", color="#dd8452")
ax2.bar(x + w / 2, co, w, label="deeper core", color="#55a868")
for i in range(3):
    ax2.text(x[i] - w / 2, sh[i] + 0.006, f"{sh[i]:.2f}", ha="center", fontsize=8)
    ax2.text(x[i] + w / 2, co[i] + 0.006, f"{co[i]:.2f}", ha="center", fontsize=8)
ax2.axhline(0.5, ls="--", c="0.5", lw=1)
ax2.set_xticks(x); ax2.set_xticklabels([f"{t} mm shell\n({v:.0%} of lung)" for t, v in zip(THR, vol)], fontsize=9)
ax2.set_ylim(0.5, 0.80); ax2.set_ylabel("AUC (single-feature, all 125)")
ax2.set_title("(b) Shell vs core, by shell thickness\ngap collapses as the shell is given fair volume", fontsize=10)
ax2.legend(fontsize=8, loc="upper right")

fig.suptitle("Spatial localization of the pulmonary hyperattenuation signal", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("results/figs/fig_localize.png", dpi=150, bbox_inches="tight")
print("saved results/figs/fig_localize.png")
