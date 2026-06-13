"""Pre-cache lung-ROI-cropped, dual-phase volumes from the RAW .mat files.
從原始 .mat 預先快取「肺部 ROI 裁切」的雙相位體積。

Per volume / 每個體積:
  HU = stored - 1024  ->  clip [-1000,400]  ->  crop to lung-lobe bbox (+margin)
  ->  normalise [0,1]  ->  resize 256x256x96  ->  save .npy
A QC check flags volumes whose lung-region median HU is outside ~[-900,-600].
QC：肺區中位數 HU 不在 [-900,-600] 會被標記（抓出強度校正異常）。

    python -m scripts.precache_roi --config configs/swinunetr_ie.yaml [--limit N]
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

TARGET = (256, 256, 96)


def locate(raw_dir, fname):
    for sub in ("Positive_Anon", "Negative_Anon"):
        p = os.path.join(raw_dir, sub, fname)
        if os.path.exists(p):
            return p
    return None


def process(ct, lobe, margin=8):
    """Returns (roi_norm 256^3 float32, lung_median_HU, crop_dims, warn)."""
    hu = ct.astype(np.float32) - 1024.0      # de-offset to true HU / 還原 HU
    hu = np.clip(hu, -1000.0, 400.0)
    mask = lobe > 0
    lung_med = float(np.median(hu[mask])) if mask.any() else float("nan")
    warn = ""
    if mask.sum() < 1000:                     # segmentation failed -> use whole volume
        warn = "tiny/empty lobe mask"
        bb = [(0, s) for s in hu.shape]
    else:
        bb = []
        for c, dim in zip(np.where(mask), hu.shape):
            lo, hi = int(c.min()), int(c.max())
            bb.append((max(0, lo - margin), min(dim, hi + margin + 1)))
    crop = hu[bb[0][0]:bb[0][1], bb[1][0]:bb[1][1], bb[2][0]:bb[2][1]]
    norm = (crop + 1000.0) / 1400.0          # [-1000,400] -> [0,1]
    t = torch.from_numpy(np.ascontiguousarray(norm)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=TARGET, mode="trilinear")
    arr = t.squeeze(0).squeeze(0).numpy().astype(np.float32)
    if not (-900 < lung_med < -600):
        warn = (warn + "; " if warn else "") + f"lung HU {lung_med:.0f} out of range"
    return arr, lung_med, [b[1] - b[0] for b in bb], warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="process only first N (smoke test)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    raw_dir = cfg["data"]["raw_dir"]
    cache = cfg["data"]["cache_dir"]
    phases = cfg["data"]["phases"]
    df = pd.read_csv(cfg["data"]["label_file"], header=None)
    if args.limit:
        df = df.head(args.limit)
    for ph in phases:
        os.makedirs(os.path.join(cache, ph), exist_ok=True)

    warns, meds = [], []
    for fname, _ in tqdm(list(df.itertuples(index=False))):
        if all(os.path.exists(os.path.join(cache, ph, fname + ".npy")) for ph in phases):
            continue
        path = locate(raw_dir, fname)
        if path is None:
            warns.append((fname, "raw .mat not found")); continue
        m = sio.loadmat(path)
        for ph in phases:
            out = os.path.join(cache, ph, fname + ".npy")
            if os.path.exists(out):
                continue
            if ph not in m or (ph + "_Lobe") not in m:
                warns.append((fname, f"{ph}/{ph}_Lobe missing")); continue
            arr, med, dims, w = process(np.asarray(m[ph]), np.asarray(m[ph + "_Lobe"]))
            np.save(out, arr)
            meds.append(med)
            if w:
                warns.append((fname, f"{ph}: {w} (crop {dims})"))

    if meds:
        print(f"\nlung HU median: mean={np.nanmean(meds):.0f} "
              f"range=[{np.nanmin(meds):.0f}, {np.nanmax(meds):.0f}]")
    print(f"warnings: {len(warns)}")
    for fn, w in warns[:40]:
        print("  ", fn, w)


if __name__ == "__main__":
    main()
