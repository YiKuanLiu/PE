"""Extract per-patient voxel spacing (xymm, zmm) from raw mats -> cache + distribution.
從 raw mat 抽每位病人的體素間距(xymm,zmm),快取並報告分布(用於 leave-protocol-out 分組)。
"""
import glob
import os

import numpy as np
import scipy.io as sio
import yaml

cfg = yaml.safe_load(open("configs/swinunetr_ie.yaml")); raw = cfg["data"].get("raw_dir", "")
d0 = np.load("results/mil/instances_lobe21.npz", allow_pickle=True)
files = [str(f) for f in d0["files"]]; y = d0["y"].astype(int)

# build basename -> path index over candidate roots (robust to deleted /mnt/hot)
idx = {}
for r in [raw, "/home/yikuan/PE"]:
    if r and os.path.isdir(r):
        for p in glob.glob(os.path.join(r, "**", "*.mat"), recursive=True):
            idx.setdefault(os.path.basename(p), p)
print(f"raw_dir(cfg)={raw} | indexed {len(idx)} mats", flush=True)

xy = np.full(len(files), np.nan); zz = np.full(len(files), np.nan); miss = 0
for i, fn in enumerate(files):
    p = idx.get(fn) or idx.get(os.path.basename(fn))
    if not p:
        miss += 1; continue
    try:
        m = sio.loadmat(p, variable_names=["xymm", "zmm"])
        xy[i] = float(np.ravel(m["xymm"])[0]); zz[i] = float(np.ravel(m["zmm"])[0])
    except Exception:
        miss += 1
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(files)}", flush=True)

np.savez("results/mil/spacing.npz", xymm=xy, zmm=zz, files=np.array(files, dtype=object))
import collections
fz = np.isfinite(zz)
print(f"missing {miss}/{len(files)}", flush=True)
print("zmm  counts:", sorted(collections.Counter(np.round(zz[fz], 2)).items()))
print("xymm counts:", sorted(collections.Counter(np.round(xy[fz], 2)).items()))
if fz.any():
    print("zmm  min/med/max: %.2f/%.2f/%.2f" % (np.nanmin(zz), np.nanmedian(zz), np.nanmax(zz)))
    print("xymm min/med/max: %.2f/%.2f/%.2f" % (np.nanmin(xy), np.nanmedian(xy), np.nanmax(xy)))
    zmed = np.nanmedian(zz)
    a = (zz <= zmed) & fz; b = (zz > zmed) & fz
    print(f"split zmm@{zmed:.2f}: low n={a.sum()} PE+={y[a].sum()} | high n={b.sum()} PE+={y[b].sum()}")
