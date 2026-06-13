"""Phase-isolation feasibility: single-phase deep model on T00 (inhale) vs T50 (exhale).
相位隔離可行性：單相位深度模型 T00（吸氣）vs T50（吐氣）。

Radiomics found the PE signal concentrates in EXHALE (T50), yet the original single-phase
baseline used INHALE (T00). This trains the SAME SwinUNETR-I on lung-ROI volumes of one
phase only, fixed epochs, on the same outer fold, so T00-ROI vs T50-ROI differ ONLY in phase.
Radiomics 顯示 PE 訊號集中在吐氣（T50），但原 baseline 用吸氣（T00）。本腳本用相同的 SwinUNETR-I、
相同折、固定 epoch，只吃單一相位的肺 ROI → T00-ROI 與 T50-ROI 唯一差別就是相位。

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      python -m scripts.feasibility_phase --config configs/swinunetr_ie.yaml \
      --splits results/swinunetr_i/splits.json --cache /home/yikuan/PE_project/cache_roi \
      --phase T50 --outer-fold <k> --epochs 40 --out <path>
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data import PEDataset
from src.metrics import point_metrics
from src.models import SwinClassifierI, load_pretrained
from src.splits import load_splits
from src.train_fold import AMP_DTYPE, set_seed


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for vol, label in loader:
        vol = vol.to(device)
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            logit = model(vol).squeeze(1)
        ys.append(label.numpy())
        ps.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--cache", required=True, help="ROI cache dir (cache_roi) / ROI 快取目錄")
    ap.add_argument("--phase", required=True, choices=["T00", "T50"])
    ap.add_argument("--outer-fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    splits = load_splits(args.splits)
    fold = splits["folds"][args.outer_fold]
    set_seed(cfg["seed"])
    device = torch.device("cuda:0")

    lf = cfg["data"]["label_file"]
    ds = PEDataset(lf, img_dir=None, phase=args.phase, cache_dir=args.cache)  # ROI single-phase
    bs, nw = cfg["training"]["batch_size"], cfg["training"]["num_workers"]
    accum = cfg["training"].get("accum_steps", 1)
    tr = DataLoader(Subset(ds, fold["train_idx"]), batch_size=bs, shuffle=True,
                    num_workers=nw, pin_memory=True)
    te = DataLoader(Subset(ds, fold["test_idx"]), batch_size=bs, shuffle=False, num_workers=nw)

    model = SwinClassifierI(feature_size=cfg["model"]["feature_size"],
                            dropout=cfg["training"]["dropout"], use_checkpoint=False).to(device)
    load_pretrained(model, cfg["model"]["pretrained_path"], verbose=False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                            weight_decay=cfg["training"]["weight_decay"])
    crit = torch.nn.BCEWithLogitsLoss()

    print(f"phase {args.phase} fold {args.outer_fold}: train {len(fold['train_idx'])} "
          f"test {len(fold['test_idx'])} epochs {args.epochs}", flush=True)
    nb = len(tr)
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        run = 0.0
        for step, (vol, label) in enumerate(tr):
            vol, label = vol.to(device), label.to(device)
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                loss = crit(model(vol).squeeze(1), label) / accum
            loss.backward()
            run += loss.item() * accum
            if (step + 1) % accum == 0 or (step + 1) == nb:
                opt.step(); opt.zero_grad()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep+1}/{args.epochs} loss={run/nb:.4f}", flush=True)

    y, p = evaluate(model, te, device)
    m = point_metrics(y, p)
    orig = None
    for f in glob.glob(os.path.join("results/swinunetr_i/jobs", f"o{args.outer_fold}_refit_*.json")):
        orig = json.load(open(f)).get("test_metrics", {}).get("AUC")

    res = {"outer_fold": args.outer_fold, "phase": args.phase, "epochs": args.epochs,
           "model": f"SwinUNETR-I {args.phase} lung-ROI", "test_metrics": m,
           "baseline_test_auc": orig,
           "test": {"y_true": y.astype(int).tolist(), "y_score": p.astype(float).tolist()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    oa = f"{orig:.3f}" if orig is not None else "?"
    print(f"phase {args.phase} fold {args.outer_fold}: baseline(T00 noROI) AUC={oa} -> "
          f"{args.phase}+ROI AUC={m['AUC']:.3f} (Sens {m['Sensitivity']:.3f} "
          f"Spec {m['Specificity']:.3f})", flush=True)


if __name__ == "__main__":
    main()
