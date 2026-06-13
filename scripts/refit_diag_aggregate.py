"""Aggregate the 10-fold fixed-epoch refit diagnostic and compare to the original.
彙整「10 折固定-epoch refit」診斷，並與原本的 refit 結果對照。

    python -m scripts.refit_diag_aggregate --config configs/swinunetr_i.yaml
"""
import argparse
import glob
import json
import os

import yaml

from src.metrics import metrics_with_bootstrap_ci, point_metrics, summarise_across_folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    exp_dir = os.path.join(cfg["output"]["root"], cfg["experiment_name"])
    diag_dir = os.path.join(exp_dir, "diag")

    per_fold, pooled_true, pooled_score = [], [], []
    rows = []
    for f in sorted(glob.glob(os.path.join(diag_dir, "o*.json")),
                    key=lambda p: json.load(open(p))["outer_fold"]):
        d = json.load(open(f))
        per_fold.append(d["new_test_metrics"])
        pooled_true += d["test"]["y_true"]
        pooled_score += d["test"]["y_score"]
        rows.append((d["outer_fold"], d.get("orig_test_auc"),
                     d["new_test_metrics"]["AUC"], d["fixed_epochs"]))

    print(f"folds aggregated: {len(rows)}\n")
    print("fold | orig AUC | new AUC | fixed_epochs")
    print("-" * 42)
    for o, oa, na, fe in rows:
        oas = f"{oa:.3f}" if isinstance(oa, (int, float)) else "?"
        print(f"  {o:>2} |  {oas}   |  {na:.3f}  | {fe}")

    print("\n=== NEW (fixed-epoch refit) — mean +/- 95% CI across folds ===")
    for k, v in summarise_across_folds(per_fold).items():
        print(f"  {k:12s}: {v['mean']:.3f}  [{v['ci_low']:.3f}, {v['ci_high']:.3f}]")

    print("\n=== NEW — pooled bootstrap 95% CI (all held-out predictions) ===")
    pb = metrics_with_bootstrap_ci(pooled_true, pooled_score)
    for k in ["AUC", "Sensitivity", "Specificity", "PPV", "NPV", "F1"]:
        v = pb[k]
        print(f"  {k:12s}: {v['value']:.3f}  [{v['ci_low']:.3f}, {v['ci_high']:.3f}]")
    print(f"\npooled-AUC lower bound = {pb['AUC']['ci_low']:.3f} -> "
          f"{'ABOVE 0.5' if pb['AUC']['ci_low'] > 0.5 else 'still INCLUDES 0.5 (not vs chance)'}")


if __name__ == "__main__":
    main()
