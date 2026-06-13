"""Rich per-lobe feature bags for Attention-MIL (registration-free, integrated scalars).
Attention-MIL 的逐肺葉完整特徵(免配準、積分式純量)。

Per patient -> 5 lobe instances; each instance = 18 features computed from the inhale/exhale
lobe masks + HU (density model rho=1+HU/1000). All robust integrals (no voxel maps).
每病人 -> 5 個肺葉 instance;每 instance = 18 個特徵,取自吸/吐肺葉遮罩 + HU。全為穩健積分量。

    python -m scripts.mil_features --config configs/swinunetr_ie.yaml --out results/mil/instances_lobe.npz
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import yaml
from scipy import stats

LOBES = [1, 2, 3, 4, 5]
EPS = 1e-4
FEAT = ["M_perf", "V_vent", "R_mism", "HUin", "HUex", "HUchg", "volin", "volex",
        "volshrink", "volratio", "massin", "massex", "airin", "airex",
        "stdin", "stdex", "skewin", "skewex"]


def locate(raw, fn):
    for s in ("Positive_Anon", "Negative_Anon"):
        p = os.path.join(raw, s, fn)
        if os.path.exists(p):
            return p
    return None


def lobe_features(hu_in, hu_ex, rin, rex, mi, me, vox):
    """18 robust integrated features for one lobe. / 單一肺葉的 18 個積分特徵。"""
    vol_in, vol_ex = mi.sum() * vox, me.sum() * vox
    mass_in, mass_ex = rin[mi].sum() * vox, rex[me].sum() * vox
    air_in, air_ex = vol_in - mass_in, vol_ex - mass_ex
    M = abs(mass_in - mass_ex) / vol_ex                      # perfusion / 灌注
    V = abs(air_in - air_ex) / vol_ex                        # ventilation / 通氣
    R = np.log((V + EPS) / (M + EPS))                        # mismatch / 失配
    hi, he = hu_in[mi], hu_ex[me]
    return [M, V, R,
            float(hi.mean()), float(he.mean()), float(hi.mean() - he.mean()),
            vol_in, vol_ex, (vol_in - vol_ex) / vol_ex, vol_ex / vol_in,
            mass_in, mass_ex, air_in, air_ex,
            float(hi.std()), float(he.std()),
            float(stats.skew(hi)), float(stats.skew(he))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/mil/instances_lobe.npz")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    raw = cfg["data"]["raw_dir"]
    df = pd.read_csv(cfg["data"]["label_file"], header=None)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    N, K, F = len(df), 5, len(FEAT)
    bags = np.full((N, K, F), np.nan)
    mask = np.zeros((N, K))
    for i, (fn, _) in enumerate(df.itertuples(index=False)):
        p = locate(raw, fn)
        if p is None:
            continue
        m = sio.loadmat(p, variable_names=["T00", "T00_Lobe", "T50", "T50_Lobe", "xymm", "zmm"])
        xymm = float(np.ravel(m["xymm"])[0]); zmm = float(np.ravel(m["zmm"])[0])
        vox = xymm * xymm * zmm
        hu_in = np.asarray(m["T00"]).astype(np.float64) - 1024.0
        hu_ex = np.asarray(m["T50"]).astype(np.float64) - 1024.0
        rin, rex = 1 + hu_in / 1000.0, 1 + hu_ex / 1000.0
        li = np.asarray(m["T00_Lobe"]).astype(np.int16)
        le = np.asarray(m["T50_Lobe"]).astype(np.int16)
        for k, l in enumerate(LOBES):
            miL, meL = li == l, le == l
            if miL.sum() < 50 or meL.sum() < 50:
                continue
            bags[i, k] = lobe_features(hu_in, hu_ex, rin, rex, miL, meL, vox)
            mask[i, k] = 1.0
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{N}", flush=True)

    y = df.iloc[:, 1].to_numpy().astype(int)
    np.savez(args.out, bags=bags, mask=mask, y=y,
             files=df.iloc[:, 0].to_numpy(), names=np.array(FEAT))
    print(f"saved {args.out}  bags {bags.shape}  valid instances {int(mask.sum())}/{N*K}  "
          f"feat {FEAT}")


if __name__ == "__main__":
    main()
