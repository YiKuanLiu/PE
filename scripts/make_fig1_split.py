"""Figure 1 (two standalone panels), reading ONLY results/mil/final_stats.json.
圖一兩張獨立圖,完全讀 final_stats.json(現為 mean-pooling 版),不再特例處理任何方法。
(1a) per-method AUC + 95% CI bars; (1b) paired-ΔAUC forest plot.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

S = json.load(open("results/mil/final_stats.json"))

COL = {"feat": "#2ca02c", "cnn": "#1f77b4", "dl": "#d62728"}
LEG = {"feat": "interpretable features", "cnn": "mask-guided / hybrid CNN", "dl": "image deep learning"}
META = [
    ("Hybrid + mask",        "Mask-guided hybrid CNN",                     "cnn"),
    ("Image CNN + mask",     "Mask-guided image CNN",                      "cnn"),
    ("Hybrid (no mask)",     "Hybrid CNN + features",                      "cnn"),
    ("Features (21)",        "Per-lobe features (V/Q+hyperattenuation)", "feat"),
    ("V/Q features",         "V/Q indices only",                           "feat"),
    ("Radiomics",            "Whole-lung radiomics",                       "feat"),
    ("SwinUNETR-I (DL)",     "SwinUNETR (deep, single-phase)",             "dl"),
    ("Image CNN (no mask)",  "Image-only CNN (no mask)",                   "dl"),
]
# methods that genuinely use multiple-instance learning (per-lobe mean pooling) / 真正用 MIL 的方法
MIL = {"Hybrid + mask", "Image CNN + mask", "Hybrid (no mask)", "Features (21)", "V/Q features", "Image CNN (no mask)"}
rows = [(lbl + (" [MIL]" if k in MIL else ""), cat, S[k]["metrics"]["AUC"], S[k]["ci"][0], S[k]["ci"][1])
        for k, lbl, cat in META]
rows.sort(key=lambda r: r[2])                       # ascending -> best at top / 由小到大,最佳在最上

# ===== Figure 1a: AUC + 95% CI =====
fig, ax = plt.subplots(figsize=(8.6, 4.9))
for i, (lbl, cat, a, lo, hi) in enumerate(rows):
    ax.barh(i, a, color=COL[cat], alpha=0.88, xerr=[[a - lo], [hi - a]], capsize=3,
            error_kw=dict(ecolor="0.3", lw=1.2))
    ax.text(hi + 0.006, i, f"{a:.3f}", va="center", fontsize=9)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
ax.axvline(0.5, ls="--", c="0.5", lw=1); ax.text(0.503, len(rows) - 0.42, "chance", color="0.5", fontsize=8)
ax.set_xlim(0.45, 0.88); ax.set_xlabel("AUC (nested 10-fold pooled OOF, 95% CI)")
ax.set_title("Pulmonary embolism on non-contrast CT: discrimination by method", fontsize=11.5)
ax.legend(handles=[Patch(color=COL[c], label=LEG[c]) for c in ("feat", "cnn", "dl")],
          loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8.5, frameon=False)
fig.savefig("results/figs/fig1a_auc.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ===== Figure 1b: paired ΔAUC forest plot =====
PAIR_ORDER = [
    (("Image CNN + mask", "Image CNN (no mask)"), "Mask image − Image (no mask)"),
    (("Hybrid + mask", "SwinUNETR-I (DL)"),       "Mask-hybrid − SwinUNETR"),
    (("Features (21)", "Radiomics"),              "Features − Radiomics"),
    (("Image CNN + mask", "SwinUNETR-I (DL)"),    "Mask image − SwinUNETR"),
    (("Features (21)", "SwinUNETR-I (DL)"),       "Features − SwinUNETR"),
    (("Hybrid + mask", "Features (21)"),          "Mask-hybrid − Features"),
    (("Hybrid + mask", "Hybrid (no mask)"),       "Mask-hybrid − Hybrid (no mask)"),
]
pmap = {(p["a"], p["b"]): p for p in S["pairs"]}
fr = [(lbl, pmap[k]) for k, lbl in PAIR_ORDER if k in pmap][::-1]   # reverse -> first on top
fig, ax2 = plt.subplots(figsize=(7.6, 4.3))
for i, (lbl, p) in enumerate(fr):
    d, lo, hi, pv = p["dAUC"], p["ci"][0], p["ci"][1], p["p"]
    sig = (lo > 0) or (hi < 0); c = "#d62728" if sig else "0.55"
    ax2.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=c, ecolor=c, capsize=3, ms=7,
                 mfc=c if sig else "white", mew=1.4, lw=1.4)
    ax2.text(hi + 0.012, i, f"p={pv:.3f}" + (" *" if sig else ""), va="center", fontsize=8.5,
             color=c, fontweight="bold" if sig else "normal")
ax2.axvline(0, c="0.3", lw=1)
ax2.set_yticks(range(len(fr))); ax2.set_yticklabels([f[0] for f in fr], fontsize=9.5)
ax2.set_xlim(-0.10, 0.42); ax2.set_xlabel("ΔAUC (A − B), paired bootstrap, 95% CI")
ax2.set_title("Paired between-method comparison", fontsize=11.5)
ax2.text(0.41, 0.10, "filled = significant\n(CI excludes 0)", ha="right", va="bottom", fontsize=8, color="#d62728")
fig.savefig("results/figs/fig1b_paired.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("saved fig1a_auc.png + fig1b_paired.png")
