"""Per-lobe V/Q: perfusion + ventilation + mismatch, registration-free (same density model).
逐肺葉 V/Q：灌注 + 通氣 + 失配，免配準（同一密度模型）。

PE = perfusion DOWN but ventilation PRESERVED = V/Q MISMATCH. So per-lobe mismatch (V/Q ratio)
should be more specific for PE than perfusion alone. All from the inhale/exhale lobe masks.
PE = 灌注降、通氣保留 = V/Q 失配。逐肺葉失配（V/Q 比）應比單獨灌注更特異。全部用現成吸/吐肺葉遮罩。

  rho = 1 + HU/1000                                   density / 密度
  mass(lobe) = sum(rho*v)        air(lobe) = sum((1-rho)*v) = Vol - mass
  M_l = |mass_in - mass_ex| / Vol_ex                  perfusion (Kuo Eq3) / 灌注
  V_l = |air_in  - air_ex | / Vol_ex                  ventilation / 通氣
  R_l = log((V_l+e)/(M_l+e))                          V/Q mismatch / 失配 (PE+ -> higher)

    python -m scripts.perfusion_vq --config configs/swinunetr_ie.yaml \
      --splits results/swinunetr_i/splits.json --out results/perfusion
"""
import argparse
import glob
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
from sklearn.preprocessing import StandardScaler

LOBES = [1, 2, 3, 4, 5]
EPS = 1e-4


def locate(raw, fn):
    for sub in ("Positive_Anon", "Negative_Anon"):
        p = os.path.join(raw, sub, fn)
        if os.path.exists(p):
            return p
    return None


def vq_features(path):
    """Return 15 features: M1..5, V1..5, R1..5. / 回傳 15 個特徵。"""
    m = sio.loadmat(path, variable_names=["T00", "T00_Lobe", "T50", "T50_Lobe", "xymm", "zmm"])
    xymm = float(np.ravel(m["xymm"])[0]) if "xymm" in m else 0.97
    zmm = float(np.ravel(m["zmm"])[0]) if "zmm" in m else 2.5
    vox = xymm * xymm * zmm
    hu_in = np.asarray(m["T00"]).astype(np.float64) - 1024.0
    hu_ex = np.asarray(m["T50"]).astype(np.float64) - 1024.0
    li = np.asarray(m["T00_Lobe"]).astype(np.int16)
    le = np.asarray(m["T50_Lobe"]).astype(np.int16)
    rho_in = 1.0 + hu_in / 1000.0
    rho_ex = 1.0 + hu_ex / 1000.0
    M, V, R = [], [], []
    for l in LOBES:
        mi, me = li == l, le == l
        if mi.sum() < 50 or me.sum() < 50:
            M.append(np.nan); V.append(np.nan); R.append(np.nan); continue
        vol_in, vol_ex = mi.sum() * vox, me.sum() * vox
        mass_in = rho_in[mi].sum() * vox            # blood+tissue mass / 質量
        mass_ex = rho_ex[me].sum() * vox
        air_in = vol_in - mass_in                   # air volume / 空氣量 = Σ(1-rho)v
        air_ex = vol_ex - mass_ex
        Ml = abs(mass_in - mass_ex) / vol_ex        # perfusion / 灌注
        Vl = abs(air_in - air_ex) / vol_ex          # ventilation / 通氣
        M.append(Ml); V.append(Vl)
        R.append(np.log((Vl + EPS) / (Ml + EPS)))   # V/Q mismatch / 失配
    return M + V + R


def build(cfg, cache):
    df = pd.read_csv(cfg["data"]["label_file"], header=None)
    raw = cfg["data"]["raw_dir"]
    X = np.full((len(df), 15), np.nan)
    for i, (fn, _) in enumerate(df.itertuples(index=False)):
        p = locate(raw, fn)
        if p:
            X[i] = vq_features(p)
    y = df.iloc[:, 1].to_numpy().astype(int)
    np.savez(cache, X=X, y=y, files=df.iloc[:, 0].to_numpy(),
             names=np.array([f"{k}{l}" for k in ("M", "V", "R") for l in LOBES]))
    return X, y


def nested(X, y, folds, kind="logistic", seed=42):
    py, pp = [], []
    for f in folds:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        Xtr, Xte = X[tr].copy(), X[te].copy()
        med = np.nanmedian(Xtr, axis=0); med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med); Xte = np.where(np.isfinite(Xte), Xte, med)
        if kind == "logistic":
            sc = StandardScaler().fit(Xtr)
            c = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), y[tr])
            p = c.predict_proba(sc.transform(Xte))[:, 1]
        else:
            c = RandomForestClassifier(400, max_depth=3, random_state=seed,
                                       n_jobs=-1).fit(Xtr, y[tr])
            p = c.predict_proba(Xte)[:, 1]
        py += y[te].tolist(); pp += p.tolist()
    return roc_auc_score(py, pp)


def boot(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); out = []
    y, p = np.asarray(y), np.asarray(p)
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            out.append(roc_auc_score(y[i], p[i]))
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", default="results/perfusion")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "vq_features.npz")
    if os.path.exists(cache) and not args.force:
        d = np.load(cache, allow_pickle=True); X, y = d["X"], d["y"]
    else:
        print("computing per-lobe V/Q features..."); X, y = build(cfg, cache)
    names = [f"{k}{l}" for k in ("M", "V", "R") for l in LOBES]

    # per-lobe separation: which of M/V/R separates PE+- best? / 哪個指標最會分
    print("=== per-lobe PE+ vs PE-  (t-test p ; AUC, oriented) ===")
    print(f"{'lobe':<6}{'M perf p':>12}{'V vent p':>12}{'R mism p':>12}{'R AUC':>9}")
    for j, l in enumerate(LOBES):
        row = []
        for blk in range(3):
            col = X[:, blk * 5 + j]; ok = np.isfinite(col)
            _, pv = stats.ttest_ind(col[ok & (y == 1)], col[ok & (y == 0)], equal_var=False)
            row.append(pv)
        rcol = X[:, 10 + j]; ok = np.isfinite(rcol)
        rauc = roc_auc_score(y[ok], rcol[ok])           # mismatch high -> PE+ / 失配高=陽性
        print(f"L{l:<5}{row[0]:>12.4f}{row[1]:>12.4f}{row[2]:>12.4f}{max(rauc,1-rauc):>9.3f}")

    # load radiomics for combined / 載入 radiomics
    dr = np.load("results/radiomics/features.npz", allow_pickle=True)
    pos = {f: i for i, f in enumerate(np.load(cache, allow_pickle=True)["files"])}
    Xrad = np.full((len(y), dr["X"].shape[1]), np.nan)
    for i, f in enumerate(dr["files"]):
        if f in pos:
            Xrad[pos[f]] = dr["X"][i]

    folds = json.load(open(args.splits))["folds"]
    combos = {
        "perfusion M (5)": X[:, :5],
        "ventilation V (5)": X[:, 5:10],
        "mismatch R (5)": X[:, 10:],
        "M+V (10)": X[:, :10],
        "M+V+R (15)": X,
        "M+V+R + radiomics": np.column_stack([X, Xrad]),
    }
    print("\n=== nested 10-fold pooled AUC ===")
    for name, Xc in combos.items():
        a = nested(Xc, y, folds, "logistic"); ar = nested(Xc, y, folds, "rf")
        print(f"  {name:<22} logistic {a:.3f} | RF {ar:.3f}")
    print("\nreferences: perfusion-only 0.585 | radiomics 0.642 | perf+rad 0.623 | deep 0.593")


if __name__ == "__main__":
    main()
