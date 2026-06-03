"""隨時查看 nested CV 的進度與「目前為止」的效能（讀 results/<exp>/）。

    python -m scripts.status --config configs/swinunetr_i.yaml

不需等全部跑完 —— 已完成的折會即時納入彙總。互動式的 loss/AUC 曲線請用 TensorBoard
（見 RUN.md）；本工具是終端機快速快照。
"""
import argparse
import glob
import json
import os

import yaml

from src.metrics import summarise_across_folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    exp = cfg["experiment_name"]
    jobs_dir = os.path.join(cfg["output"]["root"], exp, "jobs")

    # 總 job 數（與 dry-run 一致）：外層 x 候選值總數 x 內層 + refit
    outer = cfg["cv"]["outer_folds"]
    inner = cfg["cv"]["inner_folds"]
    n_cand = sum(len(s["values"]) for s in cfg["hpsearch"]["stages"])
    total_inner = outer * n_cand * inner
    total = total_inner + outer

    done_inner = len(glob.glob(os.path.join(jobs_dir, "*_inner*.json")))
    done_refit = len(glob.glob(os.path.join(jobs_dir, "*_refit_*.json")))
    done = done_inner + done_refit

    print(f"experiment: {exp}")
    print(f"progress  : {done}/{total} jobs   "
          f"(inner {done_inner}/{total_inner}, refit {done_refit}/{outer})")

    # 已完成的 refit/測試結果（逐折）
    per_fold = []
    refits = sorted(glob.glob(os.path.join(jobs_dir, "*_refit_*.json")),
                    key=lambda f: json.load(open(f))["outer_fold"])
    if refits:
        print("\nper-fold test results:")
        for f in refits:
            d = json.load(open(f))
            m = d.get("test_metrics", {})
            per_fold.append(m)
            hp = d.get("hp", {})
            print(f"  fold {d['outer_fold']:>2}: AUC={m.get('AUC', float('nan')):.3f} "
                  f"Sens={m.get('Sensitivity', float('nan')):.3f} "
                  f"Spec={m.get('Specificity', float('nan')):.3f} "
                  f"F1={m.get('F1', float('nan')):.3f}  "
                  f"| lr={hp.get('lr')} wd={hp.get('weight_decay')} do={hp.get('dropout')}")

    if per_fold:
        agg = summarise_across_folds(per_fold)
        print(f"\nmean +/- 95% CI across {len(per_fold)} completed fold(s):")
        for k, v in agg.items():
            print(f"  {k:12s}: {v['mean']:.3f}  [{v['ci_low']:.3f}, {v['ci_high']:.3f}]")
    else:
        print("\n(still in the inner hyper-parameter search; no refit/test results yet)")


if __name__ == "__main__":
    main()
