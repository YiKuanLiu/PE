"""Generate and save the nested CV splits for an experiment.

    python -m scripts.make_splits --config configs/swinunetr_i.yaml

Splits depend only on labels + filenames (no volume loading), so this is fast.
"""
import argparse
import os

import yaml

from src.data import PEDataset
from src.splits import build_nested_splits, save_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds = PEDataset(cfg["data"]["label_file"], cfg["data"]["data_dir"],
                   phase=cfg["data"]["phase"])
    splits = build_nested_splits(
        labels=ds.labels, filenames=ds.filenames,
        outer_folds=cfg["cv"]["outer_folds"], inner_folds=cfg["cv"]["inner_folds"],
        seed=cfg["seed"],
    )

    out_dir = os.path.join(cfg["output"]["root"], cfg["experiment_name"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "splits.json")
    save_splits(splits, out_path)

    n = splits["n_samples"]
    print(f"{n} samples | {splits['outer_folds']} outer x {splits['inner_folds']} inner")
    print("saved:", out_path)


if __name__ == "__main__":
    main()
