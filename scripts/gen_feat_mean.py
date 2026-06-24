"""Regenerate mean-pooling MIL OOF for the V/Q and 21-feature models with an
explicit, documented configuration (5 seeds, 150 epochs) so the result is fully
reproducible. Overwrites results/mil/oof_feat_mean.npz {oof_feat, oof_vq, y}.
以明確的 5 seeds、150 epochs、mean pooling 重新產生 V/Q 與 21-特徵的 MIL OOF(可重現),
取代先前互動式產生、無法核對 seed 數的同名檔案。

    CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m scripts.gen_feat_mean
"""
import json

import numpy as np
import torch

from scripts.mil_train import load_bags, run_pooling

FEATS = "results/mil/instances_lobe21.npz"
SPLITS = "results/swinunetr_i/splits.json"
SEEDS, EPOCHS = 5, 150
HP = dict(H=16, dropout=0.3, wd=1e-2, lr=1e-3, epochs=EPOCHS, lam_ent=0.0)

X, mask, y, files, names = load_bags(FEATS)            # X [N,5,21]; first 3 instance feats = M,V,R
folds = json.load(open(SPLITS))["folds"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"X {X.shape}  pos {int(y.sum())}  device {dev}  seeds {SEEDS}  epochs {EPOCHS}")

af, lof, hif, oof_feat, _ = run_pooling(X, mask, y, folds, "mean", SEEDS, HP, dev)            # 21 feats
av, lov, hiv, oof_vq, _ = run_pooling(X[:, :, :3], mask, y, folds, "mean", SEEDS, HP, dev)    # M,V,R only

np.savez("results/mil/oof_feat_mean.npz", oof_feat=oof_feat, oof_vq=oof_vq, y=y)
print(f"Features(21) mean-MIL  AUC {af:.3f} [{lof:.3f}, {hif:.3f}]")
print(f"V/Q (M,V,R)  mean-MIL  AUC {av:.3f} [{lov:.3f}, {hiv:.3f}]")
print("saved results/mil/oof_feat_mean.npz  (seeds=5, epochs=150, mean pooling)")
