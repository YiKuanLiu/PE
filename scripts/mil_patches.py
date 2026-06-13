"""One mat-read pass -> per-lobe 96^3 T00 image patch + 21 hand-crafted features (incl hyperdense).
一次讀 mat -> 逐肺葉 96³ T00 影像 patch + 21 個手算特徵(含 hyperdense)。

Image patch: T00 HU, lobe bbox (+margin, NOT masked -> keeps proximal vessels), clip[-1000,400],
normalize [0,1], resize 96^3 (float16). Hyperdense feats target the acute-clot bright sign on the
ORIGINAL resolution (not washed out by resize). Reuses lobe_features (18) from mil_features.
影像 patch:T00、肺葉 bbox(不遮罩、保留近端血管)、正規化、96³。hyperdense 在原解析度上算(不被 resize 抹掉)。

    python -m scripts.mil_patches --config configs/swinunetr_ie.yaml [--limit N]
"""
import argparse
import os

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn.functional as F
import yaml

from scripts.mil_features import FEAT, LOBES, lobe_features

HYPER = ["hyper_max", "hyper_p99", "hyper_frac"]   # 急性血栓亮點 / acute-clot bright sign
SZ = 96


def locate(raw, fn):
    for s in ("Positive_Anon", "Negative_Anon"):
        p = os.path.join(raw, s, fn)
        if os.path.exists(p):
            return p
    return None


def bbox(mask, margin=6):
    idx = np.where(mask)
    sl = []
    for c, dim in zip(idx, mask.shape):
        sl.append(slice(max(0, int(c.min()) - margin), min(dim, int(c.max()) + margin + 1)))
    return tuple(sl)


def patch96(hu, sl):
    """Crop lobe bbox -> clip -> normalize [0,1] -> resize 96^3 (float16). / 裁切→正規化→96³。"""
    crop = np.clip(hu[sl], -1000.0, 400.0)
    t = torch.from_numpy(np.ascontiguousarray(crop)).float()[None, None]
    t = F.interpolate(t, size=(SZ, SZ, SZ), mode="trilinear", align_corners=False)
    return ((t.squeeze().numpy() + 1000.0) / 1400.0).astype(np.float16)


def hyperdense(hi):
    """max HU (capped 300, avoid calcium), p99 HU, fraction HU in [50,150]. / 高密度血栓特徵。"""
    return [float(min(hi.max(), 300.0)), float(np.percentile(hi, 99)),
            float(((hi >= 50) & (hi <= 150)).mean())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out_img", default="results/mil/patches_t00_96.npz")
    ap.add_argument("--out_feat", default="results/mil/instances_lobe21.npz")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    raw = cfg["data"]["raw_dir"]
    df = pd.read_csv(cfg["data"]["label_file"], header=None)
    if args.limit:
        df = df.head(args.limit)
    os.makedirs("results/mil", exist_ok=True)

    N, K = len(df), 5
    names = list(FEAT) + HYPER
    feats = np.full((N, K, len(names)), np.nan, dtype=np.float64)
    imgs = np.zeros((N, K, SZ, SZ, SZ), dtype=np.float16)
    mask = np.zeros((N, K), dtype=np.float64)

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
            mi, me = li == l, le == l
            if mi.sum() < 50 or me.sum() < 50:
                continue
            feats[i, k] = lobe_features(hu_in, hu_ex, rin, rex, mi, me, vox) + hyperdense(hu_in[mi])
            imgs[i, k] = patch96(hu_in, bbox(mi, margin=6))
            mask[i, k] = 1.0
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N}", flush=True)

    y = df.iloc[:, 1].to_numpy().astype(int)
    files = df.iloc[:, 0].to_numpy()
    np.savez(args.out_feat, bags=feats, mask=mask, y=y, files=files, names=np.array(names))
    np.savez(args.out_img, bags_img=imgs, mask=mask, y=y, files=files)
    print(f"saved {args.out_feat} ({feats.shape}) + {args.out_img} ({imgs.shape}) "
          f"valid {int(mask.sum())}/{N*K}  feats {names}")


if __name__ == "__main__":
    main()
