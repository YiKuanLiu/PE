"""Figure 1 (final): per-method AUC+CI bar chart + paired-ΔAUC forest plot.
圖一(最終版):各方法 AUC+95%CI 長條圖 + 配對 ΔAUC 森林圖。
Reads the canonical results/mil/final_stats.json (no recompute). 讀已算好的 final_stats.json,不重算。
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

S = json.load(open("results/mil/final_stats.json"))

# --- panel (a): friendly label, category, for each method key ---
# 面板(a):每個方法 key 對應的顯示名稱與類別(用於上色)
META = [
    ("Hybrid + mask",        "Mask-guided hybrid CNN",            "cnn"),
    ("Image CNN + mask",     "Mask-guided image CNN",             "cnn"),
    ("Hybrid (no mask)",     "Hybrid CNN + features",             "cnn"),
    ("Features (21)",        "Per-lobe features (V/Q+hyperdense) ‡", "feat"),
    ("V/Q features",         "V/Q indices only",                  "feat"),
    ("Radiomics",            "Whole-lobe radiomics",              "feat"),
    ("SwinUNETR-I (DL)",     "SwinUNETR (deep, single-phase)",    "dl"),
    ("Image CNN (no mask)",  "Image-only CNN (no mask)",          "dl"),
]
COL = {"feat": "#2ca02c", "cnn": "#1f77b4", "dl": "#d62728"}
LEG = {"feat": "interpretable features", "cnn": "mask-guided / hybrid CNN", "dl": "image deep learning"}

rows = []
for key, lbl, cat in META:
    m = S[key]
    rows.append((lbl, cat, m["metrics"]["AUC"], m["ci"][0], m["ci"][1]))
rows.sort(key=lambda r: r[2])                       # ascending → best ends up at top / 由小到大,最佳在最上

# --- panel (b): paired comparisons in a story-driven order ---
# 面板(b):配對比較,依敘事順序排列(顯著的放上面)
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

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw=dict(width_ratios=[1.35, 1]))

# ===== (a) AUC + 95% CI =====
yp = np.arange(len(rows))
for i, (lbl, cat, a, lo, hi) in enumerate(rows):
    ax.barh(i, a, color=COL[cat], alpha=0.88,
            xerr=[[a - lo], [hi - a]], capsize=3, error_kw=dict(ecolor="0.3", lw=1.2))
    ax.text(hi + 0.006, i, f"{a:.3f}", va="center", fontsize=9)
ax.set_yticks(yp); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.axvline(0.5, ls="--", c="0.5", lw=1)
ax.text(0.503, len(rows) - 0.45, "chance", color="0.5", fontsize=8)
ax.axvline(0.722, ls=":", c="#d62728", lw=1.4)
ax.text(0.722, -0.95, "deep, non-nested\n(original, 0.722)", color="#d62728", fontsize=8, ha="center")
ax.set_xlim(0.45, 0.88); ax.set_xlabel("AUC (nested 10-fold pooled OOF, 95% CI)")
ax.set_title("(a) Discrimination by method", fontsize=11)
ax.legend(handles=[Patch(color=COL[c], label=LEG[c]) for c in ("feat", "cnn", "dl")],
          loc="lower right", fontsize=8, framealpha=0.9)

# ===== (b) paired ΔAUC forest plot =====
fr = [(lbl, pmap[k]) for k, lbl in PAIR_ORDER if k in pmap][::-1]   # reverse → first on top
for i, (lbl, p) in enumerate(fr):
    d, lo, hi, pv = p["dAUC"], p["ci"][0], p["ci"][1], p["p"]
    sig = (lo > 0) or (hi < 0)                       # CI excludes 0 → significant / CI 不跨 0 即顯著
    c = "#d62728" if sig else "0.55"
    ax2.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=c, ecolor=c,
                 capsize=3, ms=7, mfc=c if sig else "white", mew=1.4, lw=1.4)
    tag = f"p={pv:.3f}" + (" *" if sig else "")
    ax2.text(hi + 0.012, i, tag, va="center", fontsize=8,
             color=c, fontweight="bold" if sig else "normal")
ax2.axvline(0, c="0.3", lw=1)
ax2.set_yticks(range(len(fr))); ax2.set_yticklabels([f[0] for f in fr], fontsize=9)
ax2.set_xlim(-0.10, 0.42); ax2.set_xlabel("ΔAUC (A − B), 95% CI")
ax2.set_title("(b) Paired comparisons (paired bootstrap)", fontsize=11)
ax2.text(0.0, len(fr) - 0.4, "filled = significant (CI excludes 0)", fontsize=7.5, color="#d62728")

fig.suptitle("Pulmonary embolism classification on non-contrast CT — nested 10-fold cross-validation",
             fontsize=12.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("results/figs/fig1_auc_comparison.png", dpi=150)
print("saved results/figs/fig1_auc_comparison.png")
