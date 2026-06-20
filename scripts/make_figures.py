"""Generate manuscript figures + tables from cached results (nested 10-fold).
從快取結果產生論文圖表(nested 10 折)。輸出 results/figs/*.png + 印出 Table 1/2。
"""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from scripts.mil_train import load_bags, run_pooling

HP = dict(H=16, dropout=0.3, wd=1e-2, lr=1e-3, epochs=150, lam_ent=0.01)
splits = json.load(open("results/swinunetr_i/splits.json")); folds = splits["folds"]
X, mask, y, files, names = load_bags("results/mil/instances_lobe21.npz")
pos = {f: i for i, f in enumerate(files)}
import os; os.makedirs("results/figs", exist_ok=True)


def boot_ci(yv, pv, n=5000, seed=0):
    rng = np.random.default_rng(seed); out = []
    yv, pv = np.asarray(yv), np.asarray(pv)
    for _ in range(n):
        i = rng.integers(0, len(yv), len(yv))
        if len(set(yv[i])) > 1:
            out.append(roc_auc_score(yv[i], pv[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def metrics(yv, pv, thr=0.5):
    pred = (np.asarray(pv) > thr).astype(int); yv = np.asarray(yv)
    tp = int(((pred == 1) & (yv == 1)).sum()); fn = int(((pred == 0) & (yv == 1)).sum())
    tn = int(((pred == 0) & (yv == 0)).sum()); fp = int(((pred == 1) & (yv == 0)).sum())
    se = tp / (tp + fn) if tp + fn else np.nan; sp = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan; npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return se, sp, ppv, npv, f1


def rf_oof(cols):
    a, lo, hi, oof, _ = run_pooling(X[:, :, cols], mask, y, folds, "rf", 5, HP, "cpu")
    return a, lo, hi, oof


# ---- OOFs ----
A = list(range(21)); HYP = [18, 19, 20]; NOH = list(range(18)); VQ = [0, 1, 2]
auc_all, lo_all, hi_all, oof_all = rf_oof(A)
auc_hyp, lo_hyp, hi_hyp, oof_hyp = rf_oof(HYP)
auc_noh, lo_noh, hi_noh, _ = rf_oof(NOH)
auc_vq, lo_vq, hi_vq, _ = rf_oof(VQ)

# radiomics (89, raw-CT) nested RF OOF
dr = np.load("results/radiomics/features.npz", allow_pickle=True)
Xr = np.full((len(y), dr["X"].shape[1]), np.nan); rf_files = [str(f) for f in dr["files"]]
for i, f in enumerate(rf_files):
    if f in pos:
        Xr[pos[f]] = dr["X"][i]
oof_rad = np.full(len(y), np.nan)
for fdat in folds:
    tr, te = np.array(fdat["train_idx"]), np.array(fdat["test_idx"])
    Xt, Xe = Xr[tr].copy(), Xr[te].copy()
    med = np.nanmedian(Xt, 0); med = np.where(np.isfinite(med), med, 0)
    Xt = np.where(np.isfinite(Xt), Xt, med); Xe = np.where(np.isfinite(Xe), Xe, med)
    c = RandomForestClassifier(400, max_depth=3, random_state=0, n_jobs=-1).fit(Xt, y[tr])
    oof_rad[te] = c.predict_proba(Xe)[:, 1]
auc_rad = roc_auc_score(y, oof_rad); lo_rad, hi_rad = boot_ci(y, oof_rad)

# deep SwinUNETR-I OOF from refit JSONs
deep = np.full(len(y), np.nan)
for k in range(10):
    js = glob.glob(f"results/swinunetr_i/jobs/o{k}_refit_*.json")
    if js:
        d = json.load(open(js[0]))
        for fn, sc in zip(d["test"]["filenames"], d["test"]["y_score"]):
            if fn in pos:
                deep[pos[fn]] = sc
auc_deep = roc_auc_score(y, deep); lo_deep, hi_deep = boot_ci(y, deep)

# CNN point estimates (no OOF saved)
cnn = {k: json.load(open(f"results/mil/{k}.json"))["test_auc"]
       for k in ["cnn_hybrid_max", "cnn_image_max"]}

# ---- TABLE 1 (main) ----
print("=== TABLE 1: main results (nested 10-fold) ===")
print(f"{'Model':<34}{'AUC':>6}{'95% CI':>16}{'Sens':>7}{'Spec':>7}{'PPV':>7}{'NPV':>7}{'F1':>7}")
rows = [("Per-lobe features (21), RF", auc_all, lo_all, hi_all, oof_all),
        ("  hyperdense-only (3)", auc_hyp, lo_hyp, hi_hyp, oof_hyp),
        ("Radiomics whole-lobe (89)", auc_rad, lo_rad, hi_rad, oof_rad),
        ("SwinUNETR-I deep (single-phase)", auc_deep, lo_deep, hi_deep, deep)]
for nm, a, lo, hi, oof in rows:
    se, sp, ppv, npv, f1 = metrics(y, oof)
    print(f"{nm:<34}{a:>6.3f}{f'[{lo:.2f},{hi:.2f}]':>16}{se:>7.2f}{sp:>7.2f}{ppv:>7.2f}{npv:>7.2f}{f1:>7.2f}")
print(f"  (non-nested deep ref: 0.722; hybrid CNN {cnn['cnn_hybrid_max']:.3f}; image-only CNN {cnn['cnn_image_max']:.3f})")

# ---- TABLE 2 (ablation, leave-one-group-out, RF) ----
GROUPS = {"V/Q (M,V,R)": [0,1,2], "HU mean/chg": [3,4,5], "vol-ratio": [8,9], "vol-abs": [6,7],
          "mass/air-abs": [10,11,12,13], "first-order": [14,15,16,17], "HYPERDENSE": [18,19,20]}
print("\n=== TABLE 2: feature-group ablation (RF) ===")
print(f"FULL(21) AUC = {auc_all:.3f}")
print(f"{'group':<16}{'alone':>8}{'leave-out':>11}{'dAUC':>8}")
abl = {}
for nm, g in GROUPS.items():
    al = rf_oof(g)[0]; rest = [j for j in A if j not in g]; lo_ = rf_oof(rest)[0]
    abl[nm] = lo_ - auc_all
    print(f"{nm:<16}{al:>8.3f}{lo_:>11.3f}{lo_-auc_all:>+8.3f}")

# ============ FIGURES ============
# Fig 1: AUC comparison (spectrum), horizontal bars + CI
methods = [("Image-only CNN", cnn["cnn_image_max"], None, None, "DL"),
           ("SwinUNETR-I (deep)", auc_deep, lo_deep, hi_deep, "DL"),
           ("V/Q features", auc_vq, lo_vq, hi_vq, "feat"),
           ("Radiomics (whole-lobe)", auc_rad, lo_rad, hi_rad, "feat"),
           ("Hyperdense-only", auc_hyp, lo_hyp, hi_hyp, "feat"),
           ("All per-lobe features (21)", auc_all, lo_all, hi_all, "feat"),
           ("Hybrid CNN+features", cnn["cnn_hybrid_max"], None, None, "hybrid")]
col = {"DL": "#d62728", "feat": "#2ca02c", "hybrid": "#1f77b4"}
fig, ax = plt.subplots(figsize=(8, 4.5))
yp = np.arange(len(methods))
for i, (nm, a, lo, hi, cat) in enumerate(methods):
    err = [[a - lo], [hi - a]] if lo is not None else None
    ax.barh(i, a, color=col[cat], alpha=0.85, xerr=err, capsize=3,
            error_kw=dict(ecolor="0.3", lw=1.2))
    ax.text(a + 0.012, i, f"{a:.3f}", va="center", fontsize=9)
ax.set_yticks(yp); ax.set_yticklabels([m[0] for m in methods])
ax.axvline(0.5, ls="--", c="0.5", lw=1); ax.text(0.5, len(methods)-0.4, "chance", color="0.5", fontsize=8)
ax.axvline(0.722, ls=":", c="#d62728", lw=1.3)
ax.text(0.722, -0.9, "deep, non-nested (0.722)", color="#d62728", fontsize=8, ha="center")
ax.set_xlim(0.45, 0.85); ax.set_xlabel("AUC (nested 10-fold, 95% CI)")
ax.set_title("PE classification on NCCT: deep learning vs interpretable features")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=col["DL"], label="deep learning"),
                   Patch(color=col["feat"], label="hand-crafted features"),
                   Patch(color=col["hybrid"], label="hybrid")], loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig("results/figs/fig1_auc_comparison.png", dpi=150); plt.close(fig)

# Fig 2: ROC curves
fig, ax = plt.subplots(figsize=(5, 5))
for oof, nm, c in [(oof_all, f"All features (AUC {auc_all:.2f})", "#2ca02c"),
                   (oof_hyp, f"Hyperdense-only (AUC {auc_hyp:.2f})", "#9467bd"),
                   (deep, f"SwinUNETR-I deep (AUC {auc_deep:.2f})", "#d62728")]:
    fpr, tpr, _ = roc_curve(y, oof); ax.plot(fpr, tpr, c=c, lw=2, label=nm)
ax.plot([0, 1], [0, 1], "--", c="0.6", lw=1)
ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
ax.set_title("ROC (pooled nested 10-fold)"); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("results/figs/fig2_roc.png", dpi=150); plt.close(fig)

# Fig 3: feature-group ablation (leave-one-out dAUC)
fig, ax = plt.subplots(figsize=(7, 4))
gg = sorted(abl.items(), key=lambda kv: kv[1])
ax.barh([k for k, _ in gg], [v for _, v in gg],
        color=["#d62728" if k == "HYPERDENSE" else "#7f7f7f" for k, _ in gg])
ax.set_xlabel("ΔAUC when feature group removed (more negative = more important)")
ax.set_title("Feature-group importance (leave-one-group-out)")
ax.axvline(0, c="0.3", lw=1); fig.tight_layout()
fig.savefig("results/figs/fig3_ablation.png", dpi=150); plt.close(fig)

# Fig 4: confound — (a) HU-band AUC (from confound run), (b) per-lobe hyper_frac PE+/-
band_auc = {"[0,50]\nconsolidation": 0.594, "[50,100]\nclot (hyperdense)": 0.690, "[100,200]\ncalcif": 0.672}
hf = X[:, :, names.index("hyper_frac")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.bar(range(3), list(band_auc.values()),
       color=["#7f7f7f", "#d62728", "#7f7f7f"]); a1.set_xticks(range(3))
a1.set_xticklabels(list(band_auc.keys()), fontsize=8); a1.set_ylim(0.5, 0.75)
a1.set_ylabel("AUC (max-over-lobes fraction)"); a1.set_title("(a) Which HU band carries the PE signal")
for i, v in enumerate(band_auc.values()): a1.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
posm = [np.nanmean(hf[y == 1, l]) * 100 for l in range(5)]
negm = [np.nanmean(hf[y == 0, l]) * 100 for l in range(5)]
xl = np.arange(5)
a2.bar(xl - 0.2, posm, 0.4, label="PE+", color="#d62728")
a2.bar(xl + 0.2, negm, 0.4, label="PE−", color="#1f77b4")
a2.set_xticks(xl); a2.set_xticklabels([f"L{i+1}" for i in range(5)])
a2.set_ylabel("hyper_frac (% of lobar voxels)"); a2.legend(fontsize=8)
a2.set_title("(b) hyperdense fraction per lobe (L5=LLL lower-lobe)")
fig.tight_layout(); fig.savefig("results/figs/fig4_confound.png", dpi=150); plt.close(fig)

print("\nsaved figures -> results/figs/fig1_auc_comparison.png, fig2_roc.png, fig3_ablation.png, fig4_confound.png")
