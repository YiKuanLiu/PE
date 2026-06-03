"""在「一個資料分割」上訓練「一個模型」—— 巢狀 CV 的最小執行單元。

由 ``scripts/run_nested_cv.py`` 以子程序方式呼叫（一個 job 一張 GPU），
也可單獨執行除錯。兩種模式：

  * ``--mode inner --inner-fold K``：在內層訓練集上訓練、在內層驗證集上驗證，
    記錄最佳驗證 AUC（用來排序超參數）。不存模型到磁碟。

  * ``--mode refit``               ：在外層池的 refit-train 上訓練、用 refit-val
    做 early stopping，然後預測被隔離的外層測試折。存下每筆測試機率與指標。

所用的 GPU 由呼叫端透過 ``CUDA_VISIBLE_DEVICES`` 指定（故本程式一律用 ``cuda:0``）。
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from .data import PEDataset
from .metrics import point_metrics
from .models import SwinClassifierI, apply_freeze, load_pretrained


def set_seed(seed):
    """固定所有亂數來源，確保可重現。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(dataset, indices, batch_size, num_workers, shuffle):
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


# bfloat16 autocast：A100 原生支援 bf16，指數範圍同 fp32，數值穩定
#（純 fp16 會讓此 SwinUNETR 溢位 → logits 變 NaN），且不需 GradScaler。
AMP_DTYPE = torch.bfloat16


@torch.no_grad()
def evaluate(model, loader, device):
    """對一個 loader 回傳 (標籤, 機率)。"""
    model.eval()
    ys, ps = [], []
    for vol, label in loader:
        vol = vol.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            logit = model(vol).squeeze(1)
        prob = torch.sigmoid(logit.float())  # logits → 機率
        ys.append(label.numpy())
        ps.append(prob.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_one(dataset, train_idx, val_idx, hp, device, *, epochs, patience,
              batch_size, num_workers, pretrained_path, feature_size, seed,
              accum_steps=1, use_checkpoint=False, freeze="all", keep_best_model=False,
              tb_dir=None):
    """以「驗證 AUC 的 early stopping」訓練，回傳結果字典。

    ``batch_size`` 是單卡的每步批次大小；透過 ``accum_steps`` 做梯度累積，
    有效批次 = ``batch_size * accum_steps``
    （論文用有效批次 4，是以 4 卡各 1 筆達成）。
    """
    set_seed(seed)
    train_loader = _loader(dataset, train_idx, batch_size, num_workers, shuffle=True)
    val_loader = _loader(dataset, val_idx, batch_size, num_workers, shuffle=False)

    # 建模 → 載入預訓練權重 → 套用凍結策略
    model = SwinClassifierI(in_channels=1, n_class=1, feature_size=feature_size,
                            dropout=hp["dropout"], use_checkpoint=use_checkpoint).to(device)
    load_pretrained(model, pretrained_path, verbose=False)
    apply_freeze(model, freeze)

    # 只把「可訓練」的參數交給 optimizer（凍結的不更新）
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=hp["lr"], weight_decay=hp["weight_decay"])
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc, best_epoch, trigger = -1.0, -1, 0
    best_state = None
    n_batches = len(train_loader)
    writer = SummaryWriter(tb_dir) if tb_dir else None  # TensorBoard：記錄 loss / AUC 曲線

    for epoch in range(epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        for step, (vol, label) in enumerate(train_loader):
            vol = vol.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                logit = model(vol).squeeze(1)
                loss = criterion(logit, label) / accum_steps  # 除以累積步數
            loss.backward()
            running += loss.item() * accum_steps
            # 每累積 accum_steps 步、或到最後一個 batch 才更新權重
            if (step + 1) % accum_steps == 0 or (step + 1) == n_batches:
                optimizer.step()
                optimizer.zero_grad()

        # 每個 epoch 結束算一次驗證 AUC
        y, p = evaluate(model, val_loader, device)
        val_auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        tl = running / len(train_loader)
        print(f"epoch {epoch+1}/{epochs}  train_loss={tl:.4f}  "
              f"val_auc={val_auc:.4f}  best={max(best_auc,0):.4f}", flush=True)
        if writer is not None:
            writer.add_scalar("train/loss", tl, epoch)        # 訓練損失曲線
            if not np.isnan(val_auc):
                writer.add_scalar("val/auc", val_auc, epoch)  # 驗證 AUC 曲線

        # AUC 有進步就更新最佳（並視需要保存最佳權重）；否則累加耐心計數
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc, best_epoch, trigger = val_auc, epoch, 0
            if keep_best_model:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            trigger += 1
            if trigger >= patience:
                print(f"early stopping at epoch {epoch+1}", flush=True)
                break

    if writer is not None:
        writer.close()

    result = {"best_val_auc": float(best_auc), "best_epoch": int(best_epoch)}
    if keep_best_model and best_state is not None:
        model.load_state_dict(best_state)  # 還原到最佳權重供後續測試
    return result, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--outer-fold", type=int, required=True)
    ap.add_argument("--mode", choices=["inner", "refit"], required=True)
    ap.add_argument("--inner-fold", type=int, default=-1)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--weight-decay", type=float, required=True)
    ap.add_argument("--dropout", type=float, required=True)
    ap.add_argument("--out", required=True, help="結果 JSON 的輸出路徑")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from .splits import load_splits
    splits = load_splits(args.splits)

    device = torch.device("cuda:0")
    hp = {"lr": args.lr, "weight_decay": args.weight_decay, "dropout": args.dropout}
    fold = splits["folds"][args.outer_fold]

    # TensorBoard 記錄目錄：results/<exp>/tb/<mode>/<job 名稱>（每個 job 一個 run）
    job_name = os.path.splitext(os.path.basename(args.out))[0]
    exp_dir = os.path.dirname(os.path.dirname(args.out))  # results/<exp>
    tb_dir = os.path.join(exp_dir, "tb", args.mode, job_name)

    dataset = PEDataset(cfg["data"]["label_file"], cfg["data"]["data_dir"],
                        phase=cfg["data"]["phase"], cache_dir=cfg["data"].get("cache_dir"))

    # train_one 共用的參數（兩種模式都一樣）
    common = dict(batch_size=cfg["training"]["batch_size"],
                  num_workers=cfg["training"]["num_workers"],
                  accum_steps=cfg["training"].get("accum_steps", 1),
                  use_checkpoint=cfg["model"].get("use_checkpoint", False),
                  freeze=cfg["model"].get("freeze", "all"),
                  pretrained_path=cfg["model"]["pretrained_path"],
                  feature_size=cfg["model"]["feature_size"], seed=cfg["seed"])

    if args.mode == "inner":
        # 內層：訓練+驗證，只回報最佳驗證 AUC（用較少 epoch 預算）
        inner = fold["inner"][args.inner_fold]
        res, _ = train_one(dataset, inner["train_idx"], inner["val_idx"], hp, device,
                           epochs=cfg["hardware"]["inner_epochs"],
                           patience=cfg["hardware"]["inner_patience"],
                           keep_best_model=False, tb_dir=tb_dir, **common)
        res.update({"outer_fold": args.outer_fold, "inner_fold": args.inner_fold,
                    "mode": "inner", "hp": hp})
    else:  # refit
        # refit：用完整 epoch 預算訓練，再在被隔離的外層測試折上評估
        res, model = train_one(dataset, fold["refit_train_idx"], fold["refit_val_idx"],
                               hp, device, epochs=cfg["training"]["epochs"],
                               patience=cfg["training"]["patience"],
                               keep_best_model=True, tb_dir=tb_dir, **common)
        test_loader = _loader(dataset, fold["test_idx"], cfg["training"]["batch_size"],
                              cfg["training"]["num_workers"], shuffle=False)
        y, p = evaluate(model, test_loader, device)
        names = [dataset.filenames[i] for i in fold["test_idx"]]
        test_metrics = point_metrics(y, p)
        res.update({"outer_fold": args.outer_fold, "mode": "refit", "hp": hp,
                    "test": {"filenames": names, "y_true": y.astype(int).tolist(),
                             "y_score": p.astype(float).tolist()},
                    "test_metrics": test_metrics})
        # 把該折的測試指標也寫進 TensorBoard（與訓練曲線同一個 run）
        w = SummaryWriter(tb_dir)
        for k, v in test_metrics.items():
            if v == v:  # 非 nan
                w.add_scalar(f"test/{k}", v, 0)
        w.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
