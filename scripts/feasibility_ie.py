"""Feasibility test for dual-phase + lung-ROI + augmentation (SwinUNETR-IE).
雙相位 + 肺ROI + 增強 的可行性測試。

Trains the dual-phase model on one outer fold's training pool for a fixed number of
epochs (no noisy early-stopping), tests on the held-out fold, and compares to the
single-phase baseline test AUC for the same fold. Reuses the swinunetr_i CV splits
so the comparison is apples-to-apples.
在某外層折的訓練池上以固定 epoch 訓練雙相位模型，測該折，並與單相位 baseline 對照。沿用 swinunetr_i 的切分。

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      python -m scripts.feasibility_ie --config configs/swinunetr_ie.yaml \
      --splits results/swinunetr_i/splits.json --outer-fold <k> --epochs 50 --out <path>
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from src.data import PEDatasetIE
from src.metrics import point_metrics
from src.models_ie import SwinClassifierIE, load_pretrained_ie
from src.splits import load_splits
from src.train_fold import AMP_DTYPE, set_seed


@torch.no_grad()
def evaluate_ie(model, loader, device):
    model.eval()
    ys, ps = [], []
    for v00, v50, label in loader:
        v00, v50 = v00.to(device), v50.to(device)
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            logit = model(v00, v50).squeeze(1)
        ys.append(label.numpy())
        ps.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--outer-fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    splits = load_splits(args.splits)
    fold = splits["folds"][args.outer_fold]
    set_seed(cfg["seed"])
    device = torch.device("cuda:0")

    lf, cd, ph = cfg["data"]["label_file"], cfg["data"]["cache_dir"], cfg["data"]["phases"]
    ds_tr = PEDatasetIE(lf, cd, phases=ph, augment=cfg["training"].get("augment", True))
    ds_ev = PEDatasetIE(lf, cd, phases=ph, augment=False)
    bs, nw = cfg["training"]["batch_size"], cfg["training"]["num_workers"]
    accum = cfg["training"].get("accum_steps", 1)
    tr = DataLoader(Subset(ds_tr, fold["train_idx"]), batch_size=bs, shuffle=True,
                    num_workers=nw, pin_memory=True)
    te = DataLoader(Subset(ds_ev, fold["test_idx"]), batch_size=bs, shuffle=False, num_workers=nw)

    model = SwinClassifierIE(feature_size=cfg["model"]["feature_size"],
                             dropout=cfg["training"]["dropout"], use_checkpoint=True).to(device)
    load_pretrained_ie(model, cfg["model"]["pretrained_path"], verbose=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    crit = torch.nn.BCEWithLogitsLoss()

    print(f"fold {args.outer_fold}: train {len(fold['train_idx'])} test {len(fold['test_idx'])} "
          f"epochs {args.epochs} aug {cfg['training'].get('augment', True)}", flush=True)
    nb = len(tr)
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        run = 0.0
        for step, (v00, v50, label) in enumerate(tr):
            v00, v50, label = v00.to(device), v50.to(device), label.to(device)
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                loss = crit(model(v00, v50).squeeze(1), label) / accum
            loss.backward()
            run += loss.item() * accum
            if (step + 1) % accum == 0 or (step + 1) == nb:
                opt.step(); opt.zero_grad()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep+1}/{args.epochs} loss={run/nb:.4f}", flush=True)

    y, p = evaluate_ie(model, te, device)
    m = point_metrics(y, p)
    orig = None
    for f in glob.glob(os.path.join("results/swinunetr_i/jobs", f"o{args.outer_fold}_refit_*.json")):
        orig = json.load(open(f)).get("test_metrics", {}).get("AUC")

    res = {"outer_fold": args.outer_fold, "epochs": args.epochs,
           "model": "SwinUNETR-IE + lung-ROI + aug", "test_metrics": m,
           "baseline_test_auc": orig,
           "test": {"y_true": y.astype(int).tolist(), "y_score": p.astype(float).tolist()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    oa = f"{orig:.3f}" if orig is not None else "?"
    print(f"fold {args.outer_fold}: baseline AUC={oa} -> IE+ROI AUC={m['AUC']:.3f} "
          f"(Sens {m['Sensitivity']:.3f} Spec {m['Specificity']:.3f})", flush=True)


if __name__ == "__main__":
    main()
