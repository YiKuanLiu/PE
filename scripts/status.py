"""Snapshot of nested-CV progress and the performance so far (reads results/<exp>/).
隨時查看 nested CV 的進度與「目前為止」的效能（讀 results/<exp>/）。

    python -m scripts.status --config configs/swinunetr_i.yaml

No need to wait for completion -- finished folds are aggregated live. For interactive
loss/AUC curves use TensorBoard (see RUN.md); this is a quick terminal snapshot.
不需等全部跑完 —— 已完成的折會即時納入彙總。互動式的 loss/AUC 曲線請用 TensorBoard
（見 RUN.md）；本工具是終端機快速快照。
"""
import argparse
import glob
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import yaml

from src.metrics import summarise_across_folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    exp = cfg["experiment_name"]
    jobs_dir = os.path.join(cfg["output"]["root"], exp, "jobs")

    # Total job count (matches dry-run): outer x #candidates x inner + refit.
    # 總 job 數（與 dry-run 一致）：外層 x 候選值總數 x 內層 + refit。
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

    # --- ETA from the pace of completed jobs / 依完成速度推算剩餘時間 ---
    job_files = glob.glob(os.path.join(jobs_dir, "*.json"))
    if len(job_files) >= 2 and done < total:
        mtimes = sorted(os.path.getmtime(f) for f in job_files)
        span = mtimes[-1] - mtimes[0]  # active working span / 實際工作時間跨度
        if span > 0:
            rate = (len(mtimes) - 1) / span        # jobs/sec / 每秒完成的 job 數
            remaining = total - done
            eta_sec = remaining / rate
            finish = time.time() + eta_sec
            print(f"pace      : {rate*3600:.1f} jobs/hr | remaining {remaining} jobs "
                  f"~ {eta_sec/3600:.0f} h ({eta_sec/86400:.1f} d)")
            print(f"est.finish: {datetime.fromtimestamp(finish):%Y-%m-%d %H:%M} "
                  f"(rough; the final {outer} refit jobs run longer than inner ones)")

    # --- Best hyper-parameter combo so far (inner search) / 目前最佳超參數組合 ---
    combo = defaultdict(list)
    for f in glob.glob(os.path.join(jobs_dir, "*_inner*.json")):
        d = json.load(open(f))
        hp = d.get("hp", {})
        v = d.get("best_val_auc")
        if v is not None and v == v and v >= 0:  # exclude nan / -1 sentinel
            combo[(hp.get("lr"), hp.get("weight_decay"), hp.get("dropout"))].append(v)
    if combo:
        ranked = sorted(((sum(vs) / len(vs), len(vs), k) for k, vs in combo.items()),
                        reverse=True)
        print("\ntop hyper-parameter combos so far (mean inner-val AUC):")
        for mean_auc, n, (lr, wd, do) in ranked[:3]:
            print(f"  val AUC {mean_auc:.3f} (n={n} folds)  lr={lr} wd={wd} dropout={do}")

    # Completed refit/test results, per fold. / 已完成的 refit/測試結果（逐折）。
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
