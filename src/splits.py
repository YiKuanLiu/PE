"""巢狀交叉驗證（nested CV）的切分產生。

外層：分層 10-fold，用於「無偏的效能估計」。
內層：在「每個外層的訓練池內」再做分層 5-fold，用於「選超參數」。

切分只產生一次、固定亂數種子並存成 JSON（記錄樣本索引與檔名），
讓每組超參數都在完全相同的分割上評估 —— 既公平又可完整重現。
"""
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold


def build_nested_splits(labels, filenames, outer_folds=10, inner_folds=5, seed=42):
    labels = np.asarray(labels).astype(int)
    idx_all = np.arange(len(labels))

    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    splits = {"seed": seed, "outer_folds": outer_folds, "inner_folds": inner_folds,
              "n_samples": int(len(labels)), "folds": []}

    for o, (train_idx, test_idx) in enumerate(outer.split(idx_all, labels)):
        # 內層分層 k-fold —— 只在「外層訓練池」上切，絕不碰外層測試折（避免洩漏）
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + o)
        inner_splits = []
        pool_labels = labels[train_idx]
        for i, (tr, va) in enumerate(inner.split(train_idx, pool_labels)):
            inner_splits.append({
                "inner_fold": i,
                "train_idx": train_idx[tr].tolist(),
                "val_idx": train_idx[va].tolist(),
            })

        # 從外層訓練池另切一份固定的分層 10% 驗證集，
        # 供「用選定超參數 refit 時」做 early stopping 用。
        refit_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed + 100 + o)
        rt, rv = next(iter(refit_inner.split(train_idx, pool_labels)))

        splits["folds"].append({
            "outer_fold": o,
            "train_idx": train_idx.tolist(),     # 外層訓練池（9 折）
            "test_idx": test_idx.tolist(),       # 外層測試折（1 折，全程隔離）
            "refit_train_idx": train_idx[rt].tolist(),
            "refit_val_idx": train_idx[rv].tolist(),
            "inner": inner_splits,
        })

    splits["filenames"] = list(filenames)
    return splits


def save_splits(splits, path):
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)


def load_splits(path):
    with open(path) as f:
        return json.load(f)
