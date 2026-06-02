"""Dataset for single-phase (inhalation / T00) PE classification.

Each pre-processed ``.mat`` file holds two variables, ``T00`` (inhalation) and
``T50`` (exhalation), both already cropped to 512x512x96, HU-clipped to
[-1000, 400] and normalised to [0, 1].  The single-phase model uses ``T00``.
"""
import os

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


def read_volume(path, phase="T00"):
    """Load one phase from a ``.mat`` file as a float32 tensor of shape (1, X, Y, Z)."""
    mat = loadmat(path)
    if phase in mat:
        cube = mat[phase]
    else:
        # Fall back to positional order (skip __header__/__version__/__globals__).
        keys = [k for k in mat.keys() if not k.startswith("__")]
        cube = mat[keys[0]]
    cube = np.asarray(cube, dtype=np.float32)
    return torch.from_numpy(cube).unsqueeze(0)  # add channel dim


class PEDataset(Dataset):
    """Reads ``label.csv`` (filename,label) and returns (volume, label[, filename]).

    If ``cache_dir`` is given, volumes are loaded from pre-cached ``.npy`` files
    at ``cache_dir/<phase>/<filename>.npy`` (256x256x96 float32) instead of the
    large ``.mat`` files -- far faster I/O. See ``scripts/precache.py``.
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
        """Integer label array -- used by the stratified CV splitters."""
        return self.df.iloc[:, 1].to_numpy().astype(int)

    @property
    def filenames(self):
        return self.df.iloc[:, 0].tolist()

    def __getitem__(self, idx):
        fname = self.df.iloc[idx, 0]
        if self.cache_dir:
            arr = np.load(os.path.join(self.cache_dir, self.phase, fname + ".npy"))
            volume = torch.from_numpy(arr).unsqueeze(0)  # (1, X, Y, Z) float32
        else:
            volume = read_volume(os.path.join(self.img_dir, fname), self.phase)
        label = torch.tensor(float(self.df.iloc[idx, 1]), dtype=torch.float32)
        if self.return_filename:
            return volume, label, fname
        return volume, label
