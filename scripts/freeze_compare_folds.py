"""Aggregate the multi-fold freeze comparison (results/freeze_exp_folds/*.json).
彙整「多折」凍結比較（讀取 results/freeze_exp_folds/*.json）。

Per-fold best validation AUC for each strategy, plus the mean +/- std across folds,
so the gap between strategies can be judged against fold-to-fold noise.
逐策略列出各折的最佳驗證 AUC，加上跨折的 平均 ± 標準差，好讓策略之間的差距與「折間雜訊」做對照判斷。
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def main(exp_dir="results/freeze_exp_folds"):
    by_strat = defaultdict(dict)        # strategy -> {outer_fold: best_auc}
    epoch_by_strat = defaultdict(dict)
    for f in glob.glob(os.path.join(exp_dir, "*.json")):
        d = json.load(open(f))
        by_strat[d["strategy"]][d["outer_fold"]] = d["best_val_auc"]
        epoch_by_strat[d["strategy"]][d["outer_fold"]] = d["best_epoch"] + 1

    folds = sorted({o for v in by_strat.values() for o in v})
    hdr = "strategy".ljust(16) + "".join(f"  fold{o:<2d}" for o in folds) + "    mean +/- std"
    print(hdr)
    print("-" * len(hdr))
    for s in ["all", "freeze_heavy", "freeze_swinvit", "head_only"]:
        if s not in by_strat:
            continue
        vals = [by_strat[s].get(o, float("nan")) for o in folds]
        arr = np.array([v for v in vals if v == v])  # drop nan / 濾掉 nan
        cells = "".join(f"  {v:5.3f}" if v == v else "    -- " for v in vals)
        msd = f"  {arr.mean():.3f} +/- {arr.std():.3f}" if len(arr) else ""
        print(f"{s:16s}{cells}{msd}")

    # Head-to-head: all vs freeze_heavy, per-fold difference.
    # 兩兩對比：all vs freeze_heavy，逐折計算差值。
    if "all" in by_strat and "freeze_heavy" in by_strat:
        print("\nper-fold (all - freeze_heavy):")
        diffs = []
        for o in folds:
            a, h = by_strat["all"].get(o), by_strat["freeze_heavy"].get(o)
            if a == a and h == h:
                diffs.append(a - h)
                print(f"  fold {o}: all={a:.3f}  heavy={h:.3f}  diff={a-h:+.3f}")
        if diffs:
            d = np.array(diffs)
            print(f"  mean diff = {d.mean():+.3f} +/- {d.std():.3f}  "
                  f"(all wins {int((d>0).sum())}/{len(d)} folds)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/freeze_exp_folds")
