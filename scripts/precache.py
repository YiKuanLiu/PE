"""Pre-cache volumes at the model's working resolution for fast training I/O.

Reads each phase from the large ``.mat`` files, resamples to 256x256x96 with the
same trilinear interpolation the model uses, and saves float32 ``.npy`` files to
``data.cache_dir/<phase>/<filename>.npy``.  This shrinks each volume from ~200 MB
(512x512x96 float64) to ~24 MB and removes the per-step interpolation.

    python -m scripts.precache --config configs/swinunetr_i.yaml
    python -m scripts.precache --config configs/swinunetr_i.yaml --phases T00 T50
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from src.data import read_volume

TARGET = (256, 256, 96)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phases", nargs="+", default=None,
                    help="phases to cache (default: the config's data.phase)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_dir = cfg["data"]["data_dir"]
    cache_dir = cfg["data"]["cache_dir"]
    phases = args.phases or [cfg["data"]["phase"]]

    import pandas as pd
    df = pd.read_csv(cfg["data"]["label_file"], header=None)
    filenames = df.iloc[:, 0].tolist()

    for phase in phases:
        out_dir = os.path.join(cache_dir, phase)
        os.makedirs(out_dir, exist_ok=True)
        print(f"caching phase {phase} -> {out_dir}")
        for fname in tqdm(filenames):
            out_path = os.path.join(out_dir, fname + ".npy")
            if os.path.exists(out_path):
                continue
            vol = read_volume(os.path.join(data_dir, fname), phase)  # (1, X, Y, Z)
            vol = vol.unsqueeze(0)  # (1, 1, X, Y, Z)
            vol = F.interpolate(vol, size=TARGET, mode="trilinear")
            arr = vol.squeeze(0).squeeze(0).numpy().astype(np.float32)  # (256,256,96)
            np.save(out_path, arr)
    print("done.")


if __name__ == "__main__":
    main()
