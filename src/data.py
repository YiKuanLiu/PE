"""Datasets for PE classification.
PE 分類的資料集。

PEDataset    -- single-phase (inhalation / T00), baseline / 單相位 baseline。
PEDatasetIE  -- dual-phase lung-ROI volumes (T00 + T50) + optional 3D augmentation /
                雙相位肺部 ROI（吸氣+吐氣）+ 可選 3D 增強。

Each preprocessed ``.mat`` holds T00 (inhale) and T50 (exhale); see scripts/precache*.
"""
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


def read_volume(path, phase="T00"):
    """Load one phase from a ``.mat`` as a float32 tensor of shape (1, X, Y, Z).
    從 ``.mat`` 載入指定相位，回傳形狀 (1, X, Y, Z) 的 float32 tensor。"""
    mat = loadmat(path)
    if phase in mat:
        cube = mat[phase]
    else:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        cube = mat[keys[0]]
    cube = np.asarray(cube, dtype=np.float32)
    return torch.from_numpy(cube).unsqueeze(0)


class PEDataset(Dataset):
    """Single-phase: reads ``label.csv`` (filename,label); returns (volume, label[, filename]).
    單相位：讀 ``label.csv``，回傳 (volume, label[, filename])。從 ``cache_dir`` 載入 .npy（若提供）。"""

    def __init__(self, label_file, img_dir, phase="T00", return_filename=False, cache_dir=None):
        self.df = pd.read_csv(label_file, header=None)
        self.img_dir = img_dir
        self.phase = phase
        self.return_filename = return_filename
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.df)

    @property
    def labels(self):
        return self.df.iloc[:, 1].to_numpy().astype(int)

    @property
    def filenames(self):
        return self.df.iloc[:, 0].tolist()

    def __getitem__(self, idx):
        fname = self.df.iloc[idx, 0]
        if self.cache_dir:
            arr = np.load(os.path.join(self.cache_dir, self.phase, fname + ".npy"))
            volume = torch.from_numpy(arr).unsqueeze(0)
        else:
            volume = read_volume(os.path.join(self.img_dir, fname), self.phase)
        label = torch.tensor(float(self.df.iloc[idx, 1]), dtype=torch.float32)
        if self.return_filename:
            return volume, label, fname
        return volume, label


def _rand_aug_params():
    """Sample one set of augmentation params (shared across both phases of a sample).
    抽一組增強參數（同一樣本的兩個相位共用，保持一致）。"""
    return dict(fx=random.random() < 0.5,                 # flip L-R / 左右翻轉
                fy=random.random() < 0.5,                 # flip A-P / 前後翻轉
                scale=random.uniform(0.9, 1.1),           # intensity scale / 強度縮放
                shift=random.uniform(-0.05, 0.05))        # intensity shift / 強度平移


def _apply_aug(vol, p):
    """vol: (1, X, Y, Z) tensor in [0,1]."""
    if p["fx"]:
        vol = torch.flip(vol, dims=[1])
    if p["fy"]:
        vol = torch.flip(vol, dims=[2])
    return torch.clamp(vol * p["scale"] + p["shift"], 0.0, 1.0)


class PEDatasetIE(Dataset):
    """Dual-phase lung-ROI dataset; returns (vol_T00, vol_T50, label[, filename]).
    雙相位肺部 ROI 資料集；回傳 (vol_T00, vol_T50, label[, filename])。

    Loads 256x256x96 float32 ROI volumes from ``cache_dir/<phase>/<filename>.npy``
    (built by scripts/precache_roi.py). With ``augment=True`` the SAME random spatial
    transform is applied to both phases (kept consistent).
    從 ``cache_dir/<phase>/<filename>.npy`` 載入 ROI 體積；augment=True 時兩相位套用相同的隨機變換。
    """

    def __init__(self, label_file, cache_dir, phases=("T00", "T50"),
                 augment=False, return_filename=False):
        self.df = pd.read_csv(label_file, header=None)
        self.cache_dir = cache_dir
        self.phases = list(phases)
        self.augment = augment
        self.return_filename = return_filename

    def __len__(self):
        return len(self.df)

    @property
    def labels(self):
        return self.df.iloc[:, 1].to_numpy().astype(int)

    @property
    def filenames(self):
        return self.df.iloc[:, 0].tolist()

    def __getitem__(self, idx):
        fname = self.df.iloc[idx, 0]
        vols = []
        for ph in self.phases:
            arr = np.load(os.path.join(self.cache_dir, ph, fname + ".npy"))
            vols.append(torch.from_numpy(arr).unsqueeze(0))  # (1, X, Y, Z)
        if self.augment:
            p = _rand_aug_params()
            vols = [_apply_aug(v, p) for v in vols]
        label = torch.tensor(float(self.df.iloc[idx, 1]), dtype=torch.float32)
        out = (*vols, label)
        if self.return_filename:
            out = (*out, fname)
        return out
