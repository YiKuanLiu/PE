"""Diagnostic refit: does the low nested-CV AUC come from early-stopping on the tiny
11-sample refit_val? Train a FIXED number of epochs (= median inner-CV best_epoch for
the selected HP) on the FULL outer-training pool, with NO early stopping, then test.
診斷用 refit：低 AUC 是否來自「用 11 例驗證集早停」？改用「內層 CV best_epoch 中位數」
固定 epoch、在完整外層訓練池上訓練、不早停，再測 test 折。

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      python -m scripts.refit_diag --config configs/swinunetr_i.yaml --outer-fold <k> --out <path>
"""
import argparse
import glob
import json
import os
import statistics

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data import PEDataset
from src.metrics import point_metrics
from src.models import SwinClassifierI, apply_freeze, load_pretrained
from src.splits import load_splits
from src.train_fold import AMP_DTYPE, evaluate, set_seed


def hp_tag(h):
    g = lambda x: f"{x:g}"
    return f"lr{g(h['lr'])}_wd{g(h['weight_decay'])}_do{g(h['dropout'])}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--outer-fold", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    exp_dir = os.path.join(cfg["output"]["root"], cfg["experiment_name"])
    splits = load_splits(os.path.join(exp_dir, "splits.json"))
    sel = json.load(open(os.path.join(exp_dir, "selected_hp.json")))
    o = args.outer_fold
    hp = sel[str(o)]
    fold = splits["folds"][o]

    # fixed epochs = median of the 5 inner best_epochs for the selected combo (leak-free)
    # 固定 epoch = 該選定組合在 5 個內層折的 best_epoch 中位數（無洩漏）
    best_epochs = []
    for k in range(cfg["cv"]["inner_folds"]):
        p = os.path.join(exp_dir, "jobs", f"o{o}_inner{k}_{hp_tag(hp)}.json")
        if os.path.exists(p):
            be = json.load(open(p)).get("best_epoch", -1)
            if be >= 0:
                best_epochs.append(be + 1)  # index -> count
    fixed_epochs = max(int(statistics.median(best_epochs)), 5) if best_epochs else 50

    set_seed(cfg["seed"])
    device = torch.device("cuda:0")
    ds = PEDataset(cfg["data"]["label_file"], cfg["data"]["data_dir"],
                   phase=cfg["data"]["phase"], cache_dir=cfg["data"].get("cache_dir"))
    bs, nw = cfg["training"]["batch_size"], cfg["training"]["num_workers"]
    accum = cfg["training"].get("accum_steps", 1)
    train_loader = DataLoader(Subset(ds, fold["train_idx"]), batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True)
    test_loader = DataLoader(Subset(ds, fold["test_idx"]), batch_size=bs, shuffle=False, num_workers=nw)

    model = SwinClassifierI(feature_size=cfg["model"]["feature_size"], dropout=hp["dropout"],
                            use_checkpoint=cfg["model"].get("use_checkpoint", False)).to(device)
    load_pretrained(model, cfg["model"]["pretrained_path"], verbose=False)
    apply_freeze(model, cfg["model"].get("freeze", "all"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=hp["lr"], weight_decay=hp["weight_decay"])
    crit = torch.nn.BCEWithLogitsLoss()

    print(f"fold {o} hp={hp} fixed_epochs={fixed_epochs} (inner best_epochs={best_epochs}) "
          f"train={len(fold['train_idx'])} test={len(fold['test_idx'])}", flush=True)
    nb = len(train_loader)
    for ep in range(fixed_epochs):
        model.train()
        opt.zero_grad()
        for step, (vol, label) in enumerate(train_loader):
            vol, label = vol.to(device), label.to(device)
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                loss = crit(model(vol).squeeze(1), label) / accum
            loss.backward()
            if (step + 1) % accum == 0 or (step + 1) == nb:
                opt.step(); opt.zero_grad()

    y, p = evaluate(model, test_loader, device)
    m = point_metrics(y, p)

    # original refit test AUC for comparison / 原本 refit 的 test AUC 供對照
    orig = None
    for f in glob.glob(os.path.join(exp_dir, "jobs", f"o{o}_refit_*.json")):
        orig = json.load(open(f)).get("test_metrics", {}).get("AUC")

    res = {"outer_fold": o, "hp": hp, "fixed_epochs": fixed_epochs,
           "inner_best_epochs": best_epochs, "new_test_metrics": m, "orig_test_auc": orig,
           "test": {"y_true": y.astype(int).tolist(), "y_score": p.astype(float).tolist()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    os_auc = f"{orig:.3f}" if orig is not None else "?"
    print(f"fold {o}: orig AUC={os_auc} -> NEW AUC={m['AUC']:.3f} "
          f"(Sens {m['Sensitivity']:.3f} Spec {m['Specificity']:.3f})", flush=True)


if __name__ == "__main__":
    main()
