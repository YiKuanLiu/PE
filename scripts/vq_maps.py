"""Voxel-wise V/Q maps via registration, + validation against the scalar perfusion.
經配準的 voxel-wise V/Q 圖，並對照純量 perfusion 驗證。

Register exhale(T50)->inhale(T00); on the inhale grid compute per-voxel:
  rho_in = 1+HU_in/1000 ; rho_ex = 1+HU_exWarped/1000 ; J = local volume ratio (exhale/inhale)
  perfusion_map = rho_in - rho_ex*J        (blood-mass change per unit inhale volume) / 灌注
  vent_map      = (1-rho_in) - (1-rho_ex)*J (air change) / 通氣
  mismatch_map  = log((|vent|+e)/(|perf|+e)) / 失配
Validation: per-lobe MEAN of perfusion_map should track the registration-free scalar M_l
(Kuo Eq3). Also test whether within-lobe MIN/p10 (focal defect) separates PE+/- better than MEAN.
驗證：voxel 圖的逐肺葉平均應與純量 M_l 吻合；並看肺葉內 min/p10（局部缺損）是否比平均更會分。

    python -m scripts.vq_maps --config configs/swinunetr_ie.yaml --n 6 [--all --out results/vq_maps]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import SimpleITK as sitk
import yaml
from scipy import stats

LOBES = [1, 2, 3, 4, 5]
EPS = 1e-4


def locate(raw, fn):
    for sub in ("Positive_Anon", "Negative_Anon"):
        p = os.path.join(raw, sub, fn)
        if os.path.exists(p):
            return p
    return None


def np2sitk(a, sp):
    img = sitk.GetImageFromArray(np.ascontiguousarray(a.transpose(2, 1, 0)).astype(np.float32))
    img.SetSpacing((float(sp[0]), float(sp[1]), float(sp[2])))
    return img


def sitk2np(img):
    return sitk.GetArrayFromImage(img).transpose(2, 1, 0)


def scalar_perf(hu_in, hu_ex, li, le, vox):
    """Registration-free scalar M_l (Kuo Eq3) for validation. / 純量灌注（驗證用）。"""
    rin, rex = 1 + hu_in / 1000.0, 1 + hu_ex / 1000.0
    out = []
    for l in LOBES:
        mi, me = li == l, le == l
        if mi.sum() < 50 or me.sum() < 50:
            out.append(np.nan); continue
        out.append(abs(rin[mi].sum() * vox - rex[me].sum() * vox) / (me.sum() * vox))
    return out


def process(path, shrink=2, iters=40):
    m = sio.loadmat(path, variable_names=["T00", "T00_Lobe", "T50", "T50_Lobe", "xymm", "zmm"])
    xymm = float(np.ravel(m["xymm"])[0]); zmm = float(np.ravel(m["zmm"])[0])
    s = shrink
    hu_in = (np.asarray(m["T00"]).astype(np.float32) - 1024.0)[::s, ::s, ::s]
    hu_ex = (np.asarray(m["T50"]).astype(np.float32) - 1024.0)[::s, ::s, ::s]
    li = np.asarray(m["T00_Lobe"]).astype(np.int16)[::s, ::s, ::s]
    le = np.asarray(m["T50_Lobe"]).astype(np.int16)[::s, ::s, ::s]
    sp = (xymm * s, xymm * s, zmm * s)
    vox = sp[0] * sp[1] * sp[2]

    fixed = np2sitk(np.clip(hu_in, -1000, 200), sp)
    moving = np2sitk(np.clip(hu_ex, -1000, 200), sp)
    mt = sitk.HistogramMatchingImageFilter(); mt.SetNumberOfHistogramLevels(256)
    mt.SetNumberOfMatchPoints(10); mt.ThresholdAtMeanIntensityOn()
    moving_m = mt.Execute(moving, fixed)
    dem = sitk.DiffeomorphicDemonsRegistrationFilter()
    dem.SetNumberOfIterations(iters); dem.SetStandardDeviations(1.2)
    disp = dem.Execute(fixed, moving_m)
    J = sitk2np(sitk.DisplacementFieldJacobianDeterminant(disp))
    tx = sitk.DisplacementFieldTransform(sitk.Cast(disp, sitk.sitkVectorFloat64))
    hu_ex_w = sitk2np(sitk.Resample(np2sitk(hu_ex, sp), fixed, tx, sitk.sitkLinear, -1000.0))

    rin = 1 + np.clip(hu_in, -1000, 200) / 1000.0
    rew = 1 + np.clip(hu_ex_w, -1000, 200) / 1000.0
    perf = rin - rew * J                       # blood-mass change / 灌注圖
    vent = (1 - rin) - (1 - rew) * J           # air change / 通氣圖
    mism = np.log((np.abs(vent) + EPS) / (np.abs(perf) + EPS))
    sc = scalar_perf(hu_in, hu_ex, li, le, vox)
    return perf, vent, mism, li, sc


def lobe_feats(perf, lobe):
    """Per-lobe MEAN, MIN, p10 of |perfusion| (focal-defect sensitive). / 逐肺葉平均/最差。"""
    f = {}
    ap = np.abs(perf)
    for l in LOBES:
        mk = lobe == l
        if mk.sum() < 50:
            f[l] = (np.nan, np.nan, np.nan); continue
        v = ap[mk]
        f[l] = (float(v.mean()), float(v.min()), float(np.percentile(v, 10)))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=6, help="cases per class for validation / 每類案例數")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    raw = cfg["data"]["raw_dir"]
    df = pd.read_csv(cfg["data"]["label_file"], header=None)
    pos = df[df.iloc[:, 1] == 1].iloc[:, 0].tolist()[:args.n]
    neg = df[df.iloc[:, 1] == 0].iloc[:, 0].tolist()[:args.n]
    cases = [(f, 1) for f in pos] + [(f, 0) for f in neg]

    scal_all, mapmean_all = [], []
    rows = []
    for fn, lab in cases:
        p = locate(raw, fn)
        if not p:
            continue
        perf, vent, mism, lobe, sc = process(p)
        lf = lobe_feats(perf, lobe)
        for l in LOBES:
            if np.isfinite(sc[l-1]) and np.isfinite(lf[l][0]):
                scal_all.append(sc[l-1]); mapmean_all.append(lf[l][0])
        means = [lf[l][0] for l in LOBES]
        mins = [lf[l][1] for l in LOBES]
        rows.append((fn, lab, np.nanmean(means), np.nanmean(mins)))
        print(f"  {fn:<14} lab{lab}  map-perf mean {np.nanmean(means):.4f}  min {np.nanmean(mins):.4f}",
              flush=True)

    r = np.corrcoef(scal_all, mapmean_all)[0, 1]
    print(f"\n[VALIDATION] corr(voxel-map per-lobe mean, scalar M_l) = {r:.3f}  "
          f"(want high -> voxel formula consistent with Kuo Eq3)")

    arr = np.array([(lab, mean, mn) for _, lab, mean, mn in rows])
    y = arr[:, 0]
    for name, col in (("map-perf MEAN", arr[:, 1]), ("map-perf MIN (focal)", arr[:, 2])):
        pos_v, neg_v = col[y == 1], col[y == 0]
        _, pv = stats.ttest_ind(pos_v, neg_v, equal_var=False)
        print(f"[SEPARATION] {name:<22} PE+ {pos_v.mean():.4f} vs PE- {neg_v.mean():.4f}  p={pv:.3f}")


if __name__ == "__main__":
    main()
