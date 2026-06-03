"""Summarise the freeze-strategy experiment (reads results/freeze_exp/*.json).
彙整凍結策略實驗（讀取 results/freeze_exp/*.json）。

Per strategy: trainable %, best validation AUC, the epoch it was reached (convergence
speed), and the final train loss (lower with more capacity -> a big train/val gap
signals overfitting).
逐策略列出：可訓練參數比例、最佳驗證 AUC、達到最佳的 epoch（收斂速度），以及最終
train loss（容量越大通常越低 —— 與驗證 AUC 的落差大代表過擬合）。
"""
import glob
import json
import os
import sys

# print order, most trainable params first / 報表列出的順序（可訓練參數由多到少）
ORDER = ["all", "freeze_swinvit", "freeze_heavy", "head_only"]


def main(exp_dir="results/freeze_exp"):
    files = glob.glob(os.path.join(exp_dir, "*.json"))
    runs = {}
    for f in files:
        d = json.load(open(f))
        runs[d["strategy"]] = d

    print(f"{'strategy':16s} {'trainable':>11s} {'best_AUC':>9s} {'@epoch':>7s} "
          f"{'final_loss':>11s} {'epochs_run':>11s}")
    print("-" * 70)
    rows = []
    for s in ORDER + [k for k in runs if k not in ORDER]:
        if s not in runs:
            continue
        d = runs[s]
        traj = d["trajectory"]
        final_loss = traj[-1]["train_loss"] if traj else float("nan")
        pct = 100 * d["n_trainable"] / d["n_total"]
        print(f"{s:16s} {d['n_trainable']/1e6:8.3f}M  {d['best_val_auc']:9.4f} "
              f"{d['best_epoch']+1:7d} {final_loss:11.4f} {len(traj):11d}")
        rows.append((s, d["best_val_auc"], d["best_epoch"], final_loss, pct))

    if rows:
        # highest validation AUC (nan treated as lowest) / 找出最高驗證 AUC 的策略（nan 視為最低）
        best = max(rows, key=lambda r: (r[1] if r[1] == r[1] else -1))
        print("-" * 70)
        print(f"highest val AUC: '{best[0]}' = {best[1]:.4f} "
              f"(reached @ epoch {best[2]+1}, {best[4]:.1f}% trainable)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/freeze_exp")
