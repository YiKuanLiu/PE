"""Radiomics sanity check: is there ANY learnable PE signal in NCCT lung ROI?
Radiomics 健全性檢查：NCCT 肺部 ROI 到底有沒有可學的 PE 訊號？

Hand-crafted radiomics (no PyRadiomics dependency — avoids numpy-downgrade risk to the
torch/monai env). Features are extracted from the RAW .mat in true HU within the lung-lobe
mask, then classified under the SAME outer 10-fold split as the deep models, so the AUC is
directly comparable to SwinUNETR-I (0.54) / SwinUNETR-IE (0.51).
自寫 radiomics（不依賴 PyRadiomics，避免把 numpy 降版弄壞 torch/monai 環境）。特徵取自原始 .mat 的
真實 HU、肺葉 mask 內，套用與深度模型相同的 outer 10 折 → AUC 可直接與 SwinUNETR-I/IE 對照。

Feature families / 特徵家族:
  - first-order (HU stats within lobe) for T00 and T50 / 一階強度統計（兩相位）
  - HU histogram bins (mosaic/air-trapping sensitive) / HU 直方圖（對 mosaic、air-trapping 敏感）
  - 3D texture (gradient / laplacian) / 3D 紋理（梯度 / 拉普拉斯）
  - shape (volume, per-lobe volume, sphericity) / 形狀（體積、各肺葉體積、球度）
  - DUAL-PHASE physiology (per-lobe T50/T00 volume ratio = deflation; HU shift) /
    雙相位生理（各肺葉 T50/T00 體積比＝收縮程度；HU 偏移）— registration-free / 免配準

    python -m scripts.radiomics_sanity --config configs/swinunetr_ie.yaml \
      --splits results/swinunetr_i/splits.json --out results/radiomics
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import yaml
from scipy import ndimage, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HU_LO, HU_HI = -1000.0, 400.0
HU_BINS = np.linspace(-1000.0, 0.0, 11)   # 10 lung-range bins / 肺範圍 10 格直方圖


# --------------------------------------------------------------------------- #
# Feature extraction / 特徵抽取
# --------------------------------------------------------------------------- #
def locate(raw_dir, fname):
    for sub in ("Positive_Anon", "Negative_Anon"):
        p = os.path.join(raw_dir, sub, fname)
        if os.path.exists(p):
            return p
    return None


def bbox(mask, margin=4):
    idx = np.where(mask)
    if len(idx[0]) == 0:
        return None
    sl = []
    for c, dim in zip(idx, mask.shape):
        sl.append(slice(max(0, int(c.min()) - margin), min(dim, int(c.max()) + margin + 1)))
    return tuple(sl)


def first_order(hu, mask, tag):
    """First-order HU stats inside mask / mask 內的一階 HU 統計."""
    v = hu[mask]
    if v.size < 50:
        keys = ["mean", "std", "min", "max", "median", "p10", "p25", "p75", "p90",
                "iqr", "rng", "skew", "kurt", "mad", "energy", "entropy", "uniformity"]
        return {f"{tag}_{k}": np.nan for k in keys}
    p10, p25, p50, p75, p90 = np.percentile(v, [10, 25, 50, 75, 90])
    h, _ = np.histogram(v, bins=64, range=(HU_LO, HU_HI))
    pmf = h / max(h.sum(), 1)
    nz = pmf[pmf > 0]
    return {
        f"{tag}_mean": float(v.mean()), f"{tag}_std": float(v.std()),
        f"{tag}_min": float(v.min()), f"{tag}_max": float(v.max()),
        f"{tag}_median": float(p50), f"{tag}_p10": float(p10), f"{tag}_p25": float(p25),
        f"{tag}_p75": float(p75), f"{tag}_p90": float(p90),
        f"{tag}_iqr": float(p75 - p25), f"{tag}_rng": float(v.max() - v.min()),
        f"{tag}_skew": float(stats.skew(v)), f"{tag}_kurt": float(stats.kurtosis(v)),
        f"{tag}_mad": float(np.mean(np.abs(v - v.mean()))),
        f"{tag}_energy": float(np.mean(v.astype(np.float64) ** 2)),
        f"{tag}_entropy": float(-(nz * np.log2(nz)).sum()),
        f"{tag}_uniformity": float((pmf ** 2).sum()),
    }


def histogram_feats(hu, mask, tag):
    """Fraction of lung voxels per HU bin / 各 HU 區間的肺體素比例."""
    v = hu[mask]
    if v.size < 50:
        return {f"{tag}_hbin{i}": np.nan for i in range(len(HU_BINS) - 1)}
    h, _ = np.histogram(v, bins=HU_BINS)
    h = h / max(h.sum(), 1)
    return {f"{tag}_hbin{i}": float(h[i]) for i in range(len(h))}


def texture_feats(hu, mask, tag):
    """Cheap 3D texture: gradient-magnitude & laplacian stats / 便宜的 3D 紋理（梯度、拉普拉斯）."""
    if mask.sum() < 50:
        return {f"{tag}_{k}": np.nan for k in ("grad_mean", "grad_std", "grad_p90",
                                               "lap_absmean", "lap_std")}
    gx = ndimage.sobel(hu, axis=0)
    gy = ndimage.sobel(hu, axis=1)
    gz = ndimage.sobel(hu, axis=2)
    gm = np.sqrt(gx * gx + gy * gy + gz * gz)
    lap = ndimage.laplace(hu)
    g, l = gm[mask], lap[mask]
    return {f"{tag}_grad_mean": float(g.mean()), f"{tag}_grad_std": float(g.std()),
            f"{tag}_grad_p90": float(np.percentile(g, 90)),
            f"{tag}_lap_absmean": float(np.abs(l).mean()), f"{tag}_lap_std": float(l.std())}


def shape_feats(lobe, spacing, tag):
    """Volume, per-lobe volume, sphericity / 體積、各肺葉體積、球度."""
    vox = float(np.prod(spacing))
    mask = lobe > 0
    out = {f"{tag}_volume": float(mask.sum() * vox)}
    for i in range(1, 6):
        out[f"{tag}_lobevol{i}"] = float((lobe == i).sum() * vox)
    out[f"{tag}_nlobes"] = float(len(set(np.unique(lobe)) - {0}))
    # sphericity via marching_cubes surface (downsampled for speed) / 球度（降採樣加速）
    out[f"{tag}_sphericity"] = np.nan
    try:
        from skimage import measure
        m2 = mask[::2, ::2, ::2].astype(np.float32)
        if m2.sum() > 50:
            verts, faces, _, _ = measure.marching_cubes(m2, level=0.5)
            area = measure.mesh_surface_area(verts, faces) * (np.prod(spacing) ** (2 / 3)) * 4
            vol = mask.sum() * vox
            out[f"{tag}_sphericity"] = float((np.pi ** (1 / 3)) * ((6 * vol) ** (2 / 3)) / max(area, 1e-6))
    except Exception:
        pass
    return out


def dual_phase_feats(lobe00, lobe50, hu00, hu50, spacing):
    """Registration-free dual-phase physiology / 免配準的雙相位生理特徵.

    Air-trapping (a PE correlate on NCCT) -> lobe fails to deflate T00->T50 -> ratio ~1.
    Air-trapping（NCCT 上的 PE 相關徵象）-> 肺葉吸->吐之間不收縮 -> 體積比接近 1。"""
    vox = float(np.prod(spacing))
    out = {}
    v00 = (lobe00 > 0).sum() * vox
    v50 = (lobe50 > 0).sum() * vox
    out["dp_volratio"] = float(v50 / v00) if v00 > 0 else np.nan
    for i in range(1, 6):
        a = (lobe00 == i).sum() * vox
        b = (lobe50 == i).sum() * vox
        out[f"dp_lobratio{i}"] = float(b / a) if a > 0 else np.nan
    m00, m50 = lobe00 > 0, lobe50 > 0
    if m00.sum() > 50 and m50.sum() > 50:
        out["dp_median_shift"] = float(np.median(hu50[m50]) - np.median(hu00[m00]))
        out["dp_mean_shift"] = float(hu50[m50].mean() - hu00[m00].mean())
        # histogram earth-mover-ish distance between phase HU distributions / 兩相位 HU 分布差異
        h0, _ = np.histogram(hu00[m00], bins=HU_BINS); h0 = h0 / max(h0.sum(), 1)
        h5, _ = np.histogram(hu50[m50], bins=HU_BINS); h5 = h5 / max(h5.sum(), 1)
        out["dp_hist_l1"] = float(np.abs(h0 - h5).sum())
    else:
        out["dp_median_shift"] = out["dp_mean_shift"] = out["dp_hist_l1"] = np.nan
    return out


def extract_one(path, spacing_default=(1.0, 1.0, 1.0)):
    m = sio.loadmat(path, variable_names=["T00", "T00_Lobe", "T50", "T50_Lobe", "xymm", "zmm"])
    xymm = float(np.ravel(m["xymm"])[0]) if "xymm" in m else spacing_default[0]
    zmm = float(np.ravel(m["zmm"])[0]) if "zmm" in m else spacing_default[2]
    spacing = (xymm, xymm, zmm)
    feats = {}
    cropped = {}
    for ph in ("T00", "T50"):
        hu = np.asarray(m[ph]).astype(np.float32) - 1024.0
        hu = np.clip(hu, HU_LO, HU_HI)
        lobe = np.asarray(m[ph + "_Lobe"]).astype(np.int16)
        mask = lobe > 0
        bb = bbox(mask, margin=4)
        if bb is not None:                       # crop to lobe bbox to speed texture / 裁切加速
            hu_c, lobe_c = hu[bb], lobe[bb]
        else:
            hu_c, lobe_c = hu, lobe
        mask_c = lobe_c > 0
        feats.update(first_order(hu_c, mask_c, ph))
        feats.update(histogram_feats(hu_c, mask_c, ph))
        feats.update(texture_feats(hu_c, mask_c, ph))
        feats.update(shape_feats(lobe, spacing, ph))   # shape on full-res lobe / 形狀用全解析度
        cropped[ph] = (lobe, hu)
    feats.update(dual_phase_feats(cropped["T00"][0], cropped["T50"][0],
                                  cropped["T00"][1], cropped["T50"][1], spacing))
    return feats


def build_features(cfg, cache_path):
    df = pd.read_csv(cfg["data"]["label_file"], header=None)
    raw_dir = cfg["data"]["raw_dir"]
    rows, names, missing = [], None, []
    for i, (fname, _) in enumerate(df.itertuples(index=False)):
        path = locate(raw_dir, fname)
        if path is None:
            missing.append(fname); rows.append(None); continue
        f = extract_one(path)
        if names is None:
            names = sorted(f.keys())
        rows.append([f.get(k, np.nan) for k in names])
        if (i + 1) % 25 == 0:
            print(f"  extracted {i+1}/{len(df)}", flush=True)
    F = len(names)
    X = np.full((len(df), F), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        if r is not None:
            X[i] = r
    y = df.iloc[:, 1].to_numpy().astype(int)
    files = df.iloc[:, 0].tolist()
    np.savez(cache_path, X=X, y=y, names=np.array(names), files=np.array(files))
    print(f"features: {X.shape} | missing raw: {missing}")
    return X, y, names, files


# --------------------------------------------------------------------------- #
# Nested-ish CV classification / 巢狀式 CV 分類
# --------------------------------------------------------------------------- #
def fit_predict_logistic(Xtr, ytr, Xte, seed):
    """L1-logistic; C tuned by inner 5-fold AUC / L1-logistic，C 用內層 5 折 AUC 選."""
    best_c, best_auc = 1.0, -1
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        aucs = []
        for tr, va in skf.split(Xtr, ytr):
            sc = StandardScaler().fit(Xtr[tr])
            clf = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000)
            clf.fit(sc.transform(Xtr[tr]), ytr[tr])
            p = clf.predict_proba(sc.transform(Xtr[va]))[:, 1]
            if len(set(ytr[va])) > 1:
                aucs.append(roc_auc_score(ytr[va], p))
        if aucs and np.mean(aucs) > best_auc:
            best_auc, best_c = np.mean(aucs), C
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(penalty="l1", solver="liblinear", C=best_c, max_iter=2000)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1], best_c


def fit_predict_rf(Xtr, ytr, Xte, seed):
    """RandomForest; max_depth tuned by inner 5-fold AUC / 隨機森林，max_depth 內層選."""
    best_d, best_auc = None, -1
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for d in (2, 3, 4, None):
        aucs = []
        for tr, va in skf.split(Xtr, ytr):
            clf = RandomForestClassifier(n_estimators=400, max_depth=d,
                                         random_state=seed, n_jobs=-1)
            clf.fit(Xtr[tr], ytr[tr])
            p = clf.predict_proba(Xtr[va])[:, 1]
            if len(set(ytr[va])) > 1:
                aucs.append(roc_auc_score(ytr[va], p))
        if aucs and np.mean(aucs) > best_auc:
            best_auc, best_d = np.mean(aucs), d
    clf = RandomForestClassifier(n_estimators=400, max_depth=best_d, random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1], best_d


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) > 1:
            out.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def run_cv(X, y, names, splits, seed=42):
    # median-impute using TRAIN stats per fold (done inside loop) / 折內以訓練中位數補值
    folds = splits["folds"]
    results = {}
    for mdl, fn in (("L1-logistic", fit_predict_logistic), ("RandomForest", fit_predict_rf)):
        pooled_y, pooled_p, per_fold, hps = [], [], [], []
        for f in folds:
            tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
            Xtr, Xte = X[tr].copy(), X[te].copy()
            med = np.nanmedian(Xtr, axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            Xtr = np.where(np.isfinite(Xtr), Xtr, med)
            Xte = np.where(np.isfinite(Xte), Xte, med)
            p, hp = fn(Xtr, y[tr], Xte, seed)
            pooled_y += y[te].tolist(); pooled_p += p.tolist()
            per_fold.append(roc_auc_score(y[te], p) if len(set(y[te])) > 1 else np.nan)
            hps.append(hp)
        lo, hi = boot_ci(pooled_y, pooled_p, seed=seed)
        results[mdl] = {
            "pooled_auc": float(roc_auc_score(pooled_y, pooled_p)),
            "pooled_ci": [lo, hi], "n": len(pooled_y),
            "per_fold_auc_mean": float(np.nanmean(per_fold)),
            "per_fold_auc_std": float(np.nanstd(per_fold)),
            "per_fold": [None if np.isnan(a) else round(float(a), 3) for a in per_fold],
            "chosen_hp": [str(h) for h in hps],
        }
    return results, np.array(pooled_y), np.array(pooled_p)


def univariate_top(X, y, names, k=12):
    """Quick univariate AUC per feature (impute median) / 各特徵單變量 AUC."""
    med = np.nanmedian(X, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isfinite(X), X, med)
    rows = []
    for j, nm in enumerate(names):
        col = Xi[:, j]
        if np.std(col) < 1e-9:
            continue
        a = roc_auc_score(y, col)
        rows.append((nm, max(a, 1 - a)))   # direction-agnostic / 不分方向
    rows.sort(key=lambda r: -r[1])
    return rows[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="results/radiomics")
    ap.add_argument("--force", action="store_true", help="re-extract features / 重新抽特徵")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "features.npz")

    if os.path.exists(cache) and not args.force:
        d = np.load(cache, allow_pickle=True)
        X, y, names = d["X"], d["y"], list(d["names"])
        print(f"loaded cached features: {X.shape}")
    else:
        print("extracting radiomics features (CPU)...")
        X, y, names, _ = build_features(cfg, cache)

    splits = json.load(open(args.splits))
    print(f"\nrunning nested-ish CV on {X.shape[0]} patients, {X.shape[1]} features, "
          f"{len(splits['folds'])} outer folds (pos={int(y.sum())})\n")
    results, py, pp = run_cv(X, y, names, splits, seed=cfg.get("seed", 42))

    print("=" * 64)
    print(f"{'model':<14} {'pooled AUC':>12} {'95% CI':>20} {'per-fold mean':>16}")
    print("-" * 64)
    for mdl, r in results.items():
        ci = f"[{r['pooled_ci'][0]:.3f}, {r['pooled_ci'][1]:.3f}]"
        print(f"{mdl:<14} {r['pooled_auc']:>12.3f} {ci:>20} "
              f"{r['per_fold_auc_mean']:>9.3f}±{r['per_fold_auc_std']:.3f}")
    print("=" * 64)
    print("reference (same splits): SwinUNETR-I pooled 0.593 / per-fold 0.542 ; "
          "SwinUNETR-IE(4-fold) 0.512")
    print("\ntop univariate features (direction-agnostic AUC):")
    for nm, a in univariate_top(X, y, names):
        print(f"  {nm:<22} {a:.3f}")

    json.dump({"results": results,
               "reference": {"swinunetr_i_pooled": 0.593, "swinunetr_i_perfold": 0.542,
                             "swinunetr_ie_4fold_pooled": 0.512}},
              open(os.path.join(args.out, "summary.json"), "w"), indent=2)
    print(f"\nsaved -> {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
