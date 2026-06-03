"""單相位（吸氣 / T00）PE 分類的資料集。

每個預處理過的 ``.mat`` 檔含兩個變數：``T00``（吸氣）與 ``T50``（吐氣），
皆已裁切成 512x512x96、HU 值截斷到 [-1000, 400] 並正規化到 [0, 1]。
單相位模型只用 ``T00``。
"""
import os

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


def read_volume(path, phase="T00"):
    """從 ``.mat`` 載入指定相位，回傳形狀為 (1, X, Y, Z) 的 float32 tensor。"""
    mat = loadmat(path)
    if phase in mat:
        cube = mat[phase]
    else:
        # 找不到指定相位名稱時，退而用位置順序取（略過 __header__/__version__/__globals__）。
        keys = [k for k in mat.keys() if not k.startswith("__")]
        cube = mat[keys[0]]
    cube = np.asarray(cube, dtype=np.float32)
    return torch.from_numpy(cube).unsqueeze(0)  # 加上 channel 維度


class PEDataset(Dataset):
    """讀取 ``label.csv``（filename,label），回傳 (volume, label[, filename])。

    若提供 ``cache_dir``，則改從預先快取好的 ``.npy``
    （``cache_dir/<phase>/<filename>.npy``，256x256x96 float32）載入，
    而非讀取龐大的 ``.mat`` —— I/O 快很多。見 ``scripts/precache.py``。
    """

    def __init__(self, label_file, img_dir, phase="T00", return_filename=False,
                 cache_dir=None):
        self.df = pd.read_csv(label_file, header=None)
        self.img_dir = img_dir
        self.phase = phase
        self.return_filename = return_filename
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.df)

    @property
    def labels(self):
        """整數標籤陣列 —— 供分層 CV 切分器使用。"""
        return self.df.iloc[:, 1].to_numpy().astype(int)

    @property
    def filenames(self):
        return self.df.iloc[:, 0].tolist()

    def __getitem__(self, idx):
        fname = self.df.iloc[idx, 0]
        if self.cache_dir:
            # 走快取：直接讀 256x256x96 的 .npy，省去 .mat 解壓與降採樣
            arr = np.load(os.path.join(self.cache_dir, self.phase, fname + ".npy"))
            volume = torch.from_numpy(arr).unsqueeze(0)  # (1, X, Y, Z) float32
        else:
            volume = read_volume(os.path.join(self.img_dir, fname), self.phase)
        label = torch.tensor(float(self.df.iloc[idx, 1]), dtype=torch.float32)
        if self.return_filename:
            return volume, label, fname
        return volume, label
