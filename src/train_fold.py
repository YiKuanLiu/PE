"""Train one model on one data partition -- the atomic unit of the nested CV.

Run as a subprocess by ``scripts/run_nested_cv.py`` (one GPU per job), or
standalone for debugging.  Two modes:

  * ``--mode inner --inner-fold K`` : train on the inner training split, validate
    on the inner validation split, and record the best validation AUC (used to
    rank hyper-parameters).  No model is written to disk.

  * ``--mode refit``               : train on the outer pool's refit-train split
    with early stopping on the refit-val split, then predict the held-out outer
    test fold.  Per-sample test probabilities and metrics are saved.

The chosen GPU is selected by the caller via ``CUDA_VISIBLE_DEVICES`` (so this
script always uses ``cuda:0``).
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from .data import PEDataset
from .metrics import point_metrics
from .models import SwinClassifierI, apply_freeze, load_pretrained


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(dataset, indices, batch_size, num_workers, shuffle):
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=False)


# bfloat16 autocast: A100 has native bf16 with fp32 exponent range, so it is
# numerically stable here (plain fp16 overflows this SwinUNETR -> NaN logits)
# and needs no GradScaler.
AMP_DTYPE = torch.bfloat16


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (labels, probabilities) over a loader."""
    model.eval()
    ys, ps = [], []
    for vol, label in loader:
        vol = vol.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
            logit = model(vol).squeeze(1)
        prob = torch.sigmoid(logit.float())
        ys.append(label.numpy())
        ps.append(prob.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_one(dataset, train_idx, val_idx, hp, device, *, epochs, patience,
              batch_size, num_workers, pretrained_path, feature_size, seed,
              accum_steps=1, use_checkpoint=False, freeze="all", keep_best_model=False):
    """Train with early stopping on validation AUC. Returns a result dict.

    ``batch_size`` is the per-step (single-GPU) batch; gradient accumulation over
    ``accum_steps`` gives an effective batch of ``batch_size * accum_steps``
    (the paper used effective batch 4, achieved as 1/GPU across 4 GPUs).
    """
    set_seed(seed)
    train_loader = _loader(dataset, train_idx, batch_size, num_workers, shuffle=True)
    val_loader = _loader(dataset, val_idx, batch_size, num_workers, shuffle=False)

    model = SwinClassifierI(in_channels=1, n_class=1, feature_size=feature_size,
                            dropout=hp["dropout"], use_checkpoint=use_checkpoint).to(device)
    load_pretrained(model, pretrained_path, verbose=False)
    apply_freeze(model, freeze)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=hp["lr"], weight_decay=hp["weight_decay"])
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc, best_epoch, trigger = -1.0, -1, 0
    best_state = None
    n_batches = len(train_loader)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        for step, (vol, label) in enumerate(train_loader):
            vol = vol.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE):
                logit = model(vol).squeeze(1)
                loss = criterion(logit, label) / accum_steps
            loss.backward()
            running += loss.item() * accum_steps
            if (step + 1) % accum_steps == 0 or (step + 1) == n_batches:
                optimizer.step()
                optimizer.zero_grad()

        y, p = evaluate(model, val_loader, device)
        val_auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        print(f"epoch {epoch+1}/{epochs}  train_loss={running/len(train_loader):.4f}  "
              f"val_auc={val_auc:.4f}  best={max(best_auc,0):.4f}", flush=True)

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc, best_epoch, trigger = val_auc, epoch, 0
            if keep_best_model:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            trigger += 1
            if trigger >= patience:
                print(f"early stopping at epoch {epoch+1}", flush=True)
                break

    result = {"best_val_auc": float(best_auc), "best_epoch": int(best_epoch)}
    if keep_best_model and best_state is not None:
        model.load_state_dict(best_state)
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
    ap.add_argument("--out", required=True, help="path to write the result JSON")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    from .splits import load_splits
    splits = load_splits(args.splits)

    device = torch.device("cuda:0")
    hp = {"lr": args.lr, "weight_decay": args.weight_decay, "dropout": args.dropout}
    fold = splits["folds"][args.outer_fold]

    dataset = PEDataset(cfg["data"]["label_file"], cfg["data"]["data_dir"],
                        phase=cfg["data"]["phase"], cache_dir=cfg["data"].get("cache_dir"))

    common = dict(batch_size=cfg["training"]["batch_size"],
                  num_workers=cfg["training"]["num_workers"],
                  accum_steps=cfg["training"].get("accum_steps", 1),
                  use_checkpoint=cfg["model"].get("use_checkpoint", False),
                  freeze=cfg["model"].get("freeze", "all"),
                  pretrained_path=cfg["model"]["pretrained_path"],
                  feature_size=cfg["model"]["feature_size"], seed=cfg["seed"])

    if args.mode == "inner":
        inner = fold["inner"][args.inner_fold]
        res, _ = train_one(dataset, inner["train_idx"], inner["val_idx"], hp, device,
                           epochs=cfg["hardware"]["inner_epochs"],
                           patience=cfg["hardware"]["inner_patience"],
                           keep_best_model=False, **common)
        res.update({"outer_fold": args.outer_fold, "inner_fold": args.inner_fold,
                    "mode": "inner", "hp": hp})
    else:  # refit
        res, model = train_one(dataset, fold["refit_train_idx"], fold["refit_val_idx"],
                               hp, device, epochs=cfg["training"]["epochs"],
                               patience=cfg["training"]["patience"],
                               keep_best_model=True, **common)
        test_loader = _loader(dataset, fold["test_idx"], cfg["training"]["batch_size"],
                              cfg["training"]["num_workers"], shuffle=False)
        y, p = evaluate(model, test_loader, device)
        names = [dataset.filenames[i] for i in fold["test_idx"]]
        res.update({"outer_fold": args.outer_fold, "mode": "refit", "hp": hp,
                    "test": {"filenames": names, "y_true": y.astype(int).tolist(),
                             "y_score": p.astype(float).tolist()},
                    "test_metrics": point_metrics(y, p)})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
