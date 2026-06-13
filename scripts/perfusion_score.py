"""Reproduce the published CT-P perfusion score (Kuo et al., npj Biomed Innov 2026).
重現已發表的 CT-P 灌注分數（Kuo et al.）。

Per-lobe blood-mass-change perfusion surrogate — NO registration. Uses the inhale/exhale
lobe segmentations already present in the raw .mat (T00_Lobe / T50_Lobe).
逐肺葉血液質量變化灌注替代量——不需配準。用 .mat 裡現成的吸/吐肺葉分割。

  rho = 1 + HU/1000                         (Eq 1)  density / 密度
  m(lobe) = sum( rho_i * voxel_vol )        (Eq 2)  lobe mass / 肺葉質量
  M_l = |m(inhale) - m(exhale)| / V(exhale) (Eq 3)  perfusion / 灌注 (PE+ -> lower)

Then: reproduce Fig.1 (PE+ lower per lobe) + honest AUC under the SAME nested 10-fold as the
deep models, for direct comparison (deep T00 0.593 / radiomics RF 0.642).
接著：重現 Fig.1，並在與深度模型相同的 nested 10 折下算誠實 AUC，直接對照。

    python -m scripts.perfusion_score --config configs/swinunetr_ie.yaml \
      --splits results/swinunetr_i/splits.json --out results/perfusion
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import yaml
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

LOBES = [1, 2, 3, 4, 5]


def locate(raw_dir, fname):
    for sub in ("Positive_Anon", "Negative_Anon"):
        p = os.path.join(raw_dir, sub, fname)
        if os.path.exists(p):
            return p
    return None


def perfusion_features(path):
    """Return [M1..M5] per-lobe blood-mass-change perfusion (Eq 1-3). / 逐肺葉灌注。"""
    m = sio.loadmat(path, variable_names=["T00", "T00_Lobe", "T50", "T50_Lobe", "xymm", "zmm"])
    xymm = float(np.ravel(m["xymm"])[0]) if "xymm" in m else 0.97
    zmm = float(np.ravel(m["zmm"])[0]) if "zmm" in m else 2.5
    vox = xymm * xymm * zmm                                  # voxel volume mm^3 / 體素體積
    hu_in = np.asarray(m["T00"]).astype(np.float64) - 1024.0  # true HU / 還原 HU
    hu_ex = np.asarray(m["T50"]).astype(np.float64) - 1024.0
    lob_in = np.asarray(m["T00_Lobe"]).astype(np.int16)
    lob_ex = np.asarray(m["T50_Lobe"]).astype(np.int16)
    rho_in = 1.0 + hu_in / 1000.0                            # Eq 1 density / 密度
    rho_ex = 1.0 + hu_ex / 1000.0
    out = []
    for l in LOBES:
        mi, me = lob_in == l, lob_ex == l
        if mi.sum() < 50 or me.sum() < 50:
            out.append(np.nan); continue
        m_in = rho_in[mi].sum() * vox                       # Eq 2 lobe mass / 肺葉質量
        m_ex = rho_ex[me].sum() * vox
        v_ex = me.sum() * vox                               # exhale lobe volume / 吐氣肺葉體積
        out.append(abs(m_in - m_ex) / v_ex)                 # Eq 3 perfusion / 灌注
    return out


def build(cfg, cache):
    df = pd.read_csv(cfg["data"]["label_file"], header=None)
    raw = cfg["data"]["raw_dir"]
    X = np.full((len(df), 5), np.nan)
    for i, (fn, _) in enumerate(df.itertuples(index=False)):
        p = locate(raw, fn)
        if p:
            X[i] = perfusion_features(p)
    y = df.iloc[:, 1].to_numpy().astype(int)
    np.savez(cache, X=X, y=y, files=df.iloc[:, 0].to_numpy())
    return X, y


def nested_auc(X, y, splits, model, seed=42):
    py, pp = [], []
    for f in splits["folds"]:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        Xtr, Xte = X[tr].copy(), X[te].copy()
        med = np.nanmedian(Xtr, axis=0); med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med)
        Xte = np.where(np.isfinite(Xte), Xte, med)
        if model == "logistic":
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), y[tr])
            p = clf.predict_proba(sc.transform(Xte))[:, 1]
        else:
            clf = RandomForestClassifier(400, max_depth=3, random_state=seed,
                                         n_jobs=-1).fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:, 1]
        py += y[te].tolist(); pp += p.tolist()
    return roc_auc_score(py, pp), np.array(py), np.array(pp)


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) > 1:
            out.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="results/perfusion")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "perf_features.npz")
    if os.path.exists(cache) and not args.force:
        d = np.load(cache, allow_pickle=True); X, y = d["X"], d["y"]
    else:
        print("computing per-lobe perfusion features..."); X, y = build(cfg, cache)
    print(f"X {X.shape}  pos {int(y.sum())}  neg {int((y==0).sum())}  "
          f"NaN lobes {int(np.isnan(X).sum())}\n")

    # --- reproduce Fig.1: PE+ lower perfusion per lobe / 重現 Fig.1 ---
    print("=== per-lobe perfusion: PE+ vs PE-  (paper: + lower in lobes 1,2,4,5) ===")
    print(f"{'lobe':<6}{'PE+ mean':>12}{'PE- mean':>12}{'t-test p':>12}{'AUC(-M)':>10}")
    for j, l in enumerate(LOBES):
        col = X[:, j]; ok = np.isfinite(col)
        pos = col[ok & (y == 1)]; neg = col[ok & (y == 0)]
        t, pv = stats.ttest_ind(pos, neg, equal_var=False)
        auc = roc_auc_score(y[ok], -col[ok])                # lower perfusion -> PE+ / 低灌注=陽性
        flag = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        print(f"L{l:<5}{pos.mean():>12.4e}{neg.mean():>12.4e}{pv:>12.4f} {flag:<3}{auc:>7.3f}")

    # --- honest AUC under our nested 10-fold / 我們 nested 10 折下的誠實 AUC ---
    splits = json.load(open(args.splits))
    print("\n=== combined 5-lobe perfusion, nested 10-fold AUC ===")
    for model in ("logistic", "RandomForest"):
        a, yy, ppv = nested_auc(X, y, splits, "logistic" if model == "logistic" else "rf")
        lo, hi = boot_ci(yy, ppv)
        print(f"  {model:<14} pooled AUC {a:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print("\nreferences (same splits): deep T00 0.593 | deep dual 0.512 | radiomics RF 0.642")
    json.dump({"note": "per-lobe blood-mass perfusion (Kuo et al. Eq1-3)"},
              open(os.path.join(args.out, "summary.json"), "w"))


if __name__ == "__main__":
    main()
