"""為一個實驗產生並儲存巢狀 CV 切分。

    python -m scripts.make_splits --config configs/swinunetr_i.yaml

切分只依賴 標籤 + 檔名（不載入影像），所以很快。
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

    # 只需要標籤與檔名，故不傳 cache_dir、不會載入體積資料
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
