"""Confound check for hyper_frac: is its PE signal from CLOT (hyperdense sign) or COMORBIDITY?
hyper_frac 混淆檢查:訊號來自血栓(hyperdense sign)還是合併症(肺實變/積液)?

Splits the lobe HU into bands and asks which band carries the PE signal, checks whether
hyper_frac is just "denser lobe"/scanner, and its per-lobe (lower-lobe) distribution.
把肺葉 HU 切成帶,看哪一帶帶 PE 訊號;檢查是否只是「肺葉較密」/掃描儀;看逐肺葉分布。
"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio
import yaml
from sklearn.metrics import roc_auc_score

cfg = yaml.safe_load(open("configs/swinunetr_ie.yaml"))
raw = cfg["data"]["raw_dir"]
# label.csv on /mnt/hot was deleted; labels + filenames are preserved in the cached npz.
# /mnt/hot 的 label.csv 已被刪;標籤與檔名改從快取 npz 取得。
_d = np.load("results/mil/instances_lobe21.npz", allow_pickle=True)
y = _d["y"].astype(int)
files = [str(f) for f in _d["files"]]
LOB = [1, 2, 3, 4, 5]
BANDS = [("low [0,50] consolid/effusion", 0, 50),
         ("clot [50,100] hyperdense", 50, 100),
         ("dense [100,200] calcif", 100, 200)]


def loc(fn):
    for s in ("Positive_Anon", "Negative_Anon"):
        p = os.path.join(raw, s, fn)
        if os.path.exists(p):
            return p


N = len(files)
frac = np.full((N, 5, 3), np.nan)
cnt = np.full((N, 5), np.nan)            # absolute clot-band voxel count / 血栓帶絕對 voxel 數
xy, zz = [], []
for i, fn in enumerate(files):
    m = sio.loadmat(loc(fn), variable_names=["T00", "T00_Lobe", "xymm", "zmm"])
    xy.append(float(np.ravel(m["xymm"])[0])); zz.append(float(np.ravel(m["zmm"])[0]))
    hu = np.asarray(m["T00"]).astype(np.float64) - 1024.0
    li = np.asarray(m["T00_Lobe"]).astype(np.int16)
    for k, l in enumerate(LOB):
        mi = li == l
        if mi.sum() < 50:
            continue
        h = hu[mi]
        for bi, (_, lo, hi) in enumerate(BANDS):
            frac[i, k, bi] = ((h >= lo) & (h < hi)).mean()
        cnt[i, k] = int(((h >= 50) & (h < 100)).sum())
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{N}", flush=True)
xy, zz = np.array(xy), np.array(zz)

bags, names = _d["bags"], list(_d["names"])
hf = bags[:, :, names.index("hyper_frac")]
hu_in = bags[:, :, names.index("HUin")]


def maxlobe(a):
    return np.nanmax(np.where(np.isfinite(a), a, -np.inf), axis=1)


print("\n=== Q1: which HU band carries the PE signal? (max-over-lobes fraction) ===")
print("    (clot band >> low band => hyperdense/clot, not comorbidity / 血栓帶勝出=非合併症)")
for bi, (nm, _, _) in enumerate(BANDS):
    f = maxlobe(frac[:, :, bi]); ok = f > -1e8
    a = roc_auc_score(y[ok], f[ok])
    print(f"  {nm:<32} AUC {max(a, 1-a):.3f}")
cm = maxlobe(cnt)
print(f"  clot-band [50,100] voxel COUNT: median PE+ {np.nanmedian(cm[y==1]):.0f} vs "
      f"PE- {np.nanmedian(cm[y==0]):.0f}  (small+focal => clot, not large consolidation)")

print("\n=== Q2: is hyper_frac just 'denser lobe' or scanner? ===")
hfm = maxlobe(hf)
print(f"  corr(hyper_frac, mean lobe HU) = {np.corrcoef(hfm, maxlobe(hu_in))[0,1]:.2f}  "
      f"(low => NOT just overall density)")
print(f"  corr(hyper_frac, xymm) = {np.corrcoef(hfm, xy)[0,1]:.2f} | "
      f"corr(hyper_frac, zmm) = {np.corrcoef(hfm, zz)[0,1]:.2f}  (low => not scanner)")
av = roc_auc_score(y, xy * xy * zz)
print(f"  voxel-volume(spacing) alone AUC vs PE = {max(av, 1-av):.3f}")

print("\n=== Q3: per-lobe hyper_frac PE+ vs PE- (lower lobes L3=RLL L5=LLL are PE-typical) ===")
for k, l in enumerate(LOB):
    c = hf[:, k]; ok = np.isfinite(c)
    pos, neg = c[ok & (y == 1)], c[ok & (y == 0)]
    print(f"  L{l}: PE+ {pos.mean()*100:.2f}%  PE- {neg.mean()*100:.2f}%  "
          f"Δ {(pos.mean()-neg.mean())*100:+.2f}pp")
