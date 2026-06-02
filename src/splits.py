"""Nested cross-validation split generation.

Outer: stratified 10-fold for unbiased performance estimation.
Inner: stratified 5-fold *within each outer training pool* for HP selection.

Splits are generated once, seeded, and saved to JSON (as sample indices and
filenames) so that every hyper-parameter configuration is evaluated on exactly
the same partitions -- a fair comparison and fully reproducible.
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
        # Inner stratified k-fold on the outer training pool only.
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + o)
        inner_splits = []
        pool_labels = labels[train_idx]
        for i, (tr, va) in enumerate(inner.split(train_idx, pool_labels)):
            inner_splits.append({
                "inner_fold": i,
                "train_idx": train_idx[tr].tolist(),
                "val_idx": train_idx[va].tolist(),
            })

        # A fixed stratified 10% validation slice of the outer pool, used for
        # early stopping when refitting with the selected hyper-parameters.
        refit_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed + 100 + o)
        rt, rv = next(iter(refit_inner.split(train_idx, pool_labels)))

        splits["folds"].append({
            "outer_fold": o,
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
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
