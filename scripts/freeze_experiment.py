"""小實驗：在「同一個固定資料分割」上比較各種凍結策略。

對給定的凍結策略，在 外層折 / 內層折 上訓練 SwinUNETR-I 模型
（所有策略看到完全相同的資料），記錄驗證 AUC 的逐 epoch 軌跡
與 train/val 落差（過擬合）。一張 GPU 跑一個策略：

    for s in all freeze_swinvit freeze_heavy head_only; do
      CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
        python -m scripts.freeze_experiment --config configs/swinunetr_i.yaml \
        --strategy $s --out results/freeze_exp/$s.json &
    done

再比較各 JSON（最佳驗證 AUC、達標 epoch 數、train loss）。
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from src.data import PEDataset
from src.models import SwinClassifierI, apply_freeze, load_pretrained
from src.splits import load_splits
from src.train_fold import evaluate, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outer-fold", type=int, default=0)
    ap.add_argument("--inner-fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # 取用既有的巢狀切分中，指定外層/內層折的 train/val 索引
    splits = load_splits(os.path.join(cfg["output"]["root"], cfg["experiment_name"], "splits.json"))
    inner = splits["folds"][args.outer_fold]["inner"][args.inner_fold]
    train_idx, val_idx = inner["train_idx"], inner["val_idx"]

    set_seed(cfg["seed"])
    device = torch.device("cuda:0")
    ds = PEDataset(cfg["data"]["label_file"], cfg["data"]["data_dir"],
                   phase=cfg["data"]["phase"], cache_dir=cfg["data"].get("cache_dir"))
    bs, nw = cfg["training"]["batch_size"], cfg["training"]["num_workers"]
    accum = cfg["training"].get("accum_steps", 1)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=bs, shuffle=False, num_workers=nw)

    # 建模 → 載入預訓練 → 套用凍結策略，並印出可訓練參數比例
    model = SwinClassifierI(feature_size=cfg["model"]["feature_size"],
                            use_checkpoint=cfg["model"].get("use_checkpoint", False)).to(device)
    load_pretrained(model, cfg["model"]["pretrained_path"], verbose=False)
    n_train, n_total = apply_freeze(model, args.strategy)
    print(f"[{args.strategy}] trainable {n_train/1e6:.3f}M / {n_total/1e6:.1f}M "
          f"({100*n_train/n_total:.2f}%)", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    crit = torch.nn.BCEWithLogitsLoss()

    traj, best_auc, best_epoch, trigger = [], -1.0, -1, 0
    n_batches = len(train_loader)
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        running = 0.0
        for step, (vol, label) in enumerate(train_loader):
            vol, label = vol.to(device), label.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(vol).squeeze(1), label) / accum  # 梯度累積
            loss.backward()
            running += loss.item() * accum
            if (step + 1) % accum == 0 or (step + 1) == n_batches:
                opt.step(); opt.zero_grad()
        # 記錄這個 epoch 的 train loss 與驗證 AUC（整條軌跡）
        y, p = evaluate(model, val_loader, device)
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        tl = running / n_batches
        traj.append({"epoch": epoch, "train_loss": tl, "val_auc": float(auc)})
        print(f"[{args.strategy}] ep{epoch+1}/{args.epochs} loss={tl:.4f} val_auc={auc:.4f} "
              f"best={max(best_auc,0):.4f}", flush=True)
        if not np.isnan(auc) and auc > best_auc:
            best_auc, best_epoch, trigger = auc, epoch, 0
        else:
            trigger += 1
            if trigger >= args.patience:
                print(f"[{args.strategy}] early stop @ {epoch+1}", flush=True)
                break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"strategy": args.strategy, "outer_fold": args.outer_fold,
                   "inner_fold": args.inner_fold, "n_trainable": n_train, "n_total": n_total,
                   "best_val_auc": float(best_auc), "best_epoch": best_epoch,
                   "trajectory": traj}, f, indent=2)
    print(f"[{args.strategy}] WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
