"""巢狀交叉驗證的協調器（orchestrator），含「分階段超參數搜尋」。

設計
----
* 外層 10-fold 估計效能；內層 5-fold 選超參數。
* 超參數以「階段」方式調（lr -> weight_decay -> dropout）：每階段只調一個參數、
  並把前面階段的勝出值固定下來。
* 各階段在所有外層折之間「同步」進行。同一階段內，每個內層 job
  （外層折 x 候選值 x 內層折）彼此獨立，因此會並行派發 —— 一個 job 一張 GPU。
* 每個 job 都會寫出結果 JSON；協調器「完全可續跑」
  （結果檔已存在的 job 會被略過）。

用法
----
    python -m scripts.make_splits   --config configs/swinunetr_i.yaml
    python -m scripts.run_nested_cv --config configs/swinunetr_i.yaml
    python -m scripts.run_nested_cv --config configs/swinunetr_i.yaml --dry-run
    python -m scripts.run_nested_cv --config configs/swinunetr_i.yaml --only-aggregate
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import yaml

from src.metrics import metrics_with_bootstrap_ci, point_metrics, summarise_across_folds
from src.splits import load_splits

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fmt(x):
    return f"{x:g}"


def hp_tag(hp):
    """把一組超參數轉成檔名安全的字串標籤。"""
    return f"lr{fmt(hp['lr'])}_wd{fmt(hp['weight_decay'])}_do{fmt(hp['dropout'])}"


def job_out_path(jobs_dir, outer, mode, hp, inner=None):
    """依 (外層折, 模式, 超參數) 推導「確定性」的結果檔路徑 —— 這是可續跑的關鍵。"""
    if mode == "inner":
        return os.path.join(jobs_dir, f"o{outer}_inner{inner}_{hp_tag(hp)}.json")
    return os.path.join(jobs_dir, f"o{outer}_refit_{hp_tag(hp)}.json")


def is_done(path):
    """結果檔存在且能成功解析 JSON，才算完成。"""
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, OSError):
        return False


def make_job(cfg_path, splits_path, outer, mode, hp, out_path, inner=None, log_path=None):
    """組出一個「呼叫 src.train_fold 子程序」的工作描述。"""
    cmd = [sys.executable, "-m", "src.train_fold",
           "--config", cfg_path, "--splits", splits_path,
           "--outer-fold", str(outer), "--mode", mode,
           "--lr", repr(hp["lr"]), "--weight-decay", repr(hp["weight_decay"]),
           "--dropout", repr(hp["dropout"]), "--out", out_path]
    if mode == "inner":
        cmd += ["--inner-fold", str(inner)]
    return {"cmd": cmd, "out": out_path, "log": log_path,
            "desc": os.path.basename(out_path)}


def run_jobs(jobs, gpus, poll=5.0):
    """並行執行一批 job（一張 GPU 一個），並略過已完成的。"""
    pending = [j for j in jobs if not is_done(j["out"])]
    skipped = len(jobs) - len(pending)
    if skipped:
        print(f"  [resume] {skipped}/{len(jobs)} already done, {len(pending)} to run")
    if not pending:
        return

    running = {}            # gpu -> (Popen, job, log_fh)
    queue = list(pending)
    done = 0
    total = len(pending)
    while queue or running:
        # 把待辦 job 派發到空閒的 GPU
        for gpu in gpus:
            if gpu in running or not queue:
                continue
            job = queue.pop(0)
            # CUDA_DEVICE_ORDER=PCI_BUS_ID 讓 CUDA 的裝置編號與 nvidia-smi 一致，
            # 這樣 config 裡的 GPU id 才會選到正確的卡（否則 CUDA 預設的
            # FASTEST_FIRST 順序可能把某個 id 對應到小顯示卡而 OOM）。
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       CUDA_DEVICE_ORDER="PCI_BUS_ID")
            log_fh = open(job["log"], "w") if job["log"] else None
            p = subprocess.Popen(job["cmd"], cwd=REPO_ROOT, env=env,
                                 stdout=log_fh, stderr=subprocess.STDOUT)
            running[gpu] = (p, job, log_fh)
            print(f"  -> GPU{gpu}: {job['desc']}", flush=True)
        # 輪詢已結束的 job，釋放其 GPU 供下一個使用
        time.sleep(poll)
        for gpu, (p, job, log_fh) in list(running.items()):
            if p.poll() is None:
                continue
            if log_fh:
                log_fh.close()
            done += 1
            ok = is_done(job["out"])
            status = "ok" if (p.returncode == 0 and ok) else f"FAIL(rc={p.returncode})"
            print(f"  <- GPU{gpu}: {job['desc']} [{status}] ({done}/{total})", flush=True)
            if not ok:
                print(f"     !! see log: {job['log']}", flush=True)
            del running[gpu]


def read_result(path):
    with open(path) as f:
        return json.load(f)


def mean_inner_auc(jobs_dir, outer, hp, inner_folds):
    """某外層折、某組超參數，在所有內層折上的平均驗證 AUC（用來排序超參數）。"""
    aucs = []
    for k in range(inner_folds):
        path = job_out_path(jobs_dir, outer, "inner", hp, inner=k)
        if is_done(path):
            v = read_result(path).get("best_val_auc", float("nan"))
            if not np.isnan(v):
                aucs.append(v)
    return float(np.mean(aucs)) if aucs else float("nan")


def aggregate(cfg, splits, jobs_dir, out_dir):
    """把各折的 refit/測試結果彙整成 逐折 + 彙總 的摘要。"""
    selected, per_fold, pooled_true, pooled_score = [], [], [], []
    for fold in splits["folds"]:
        o = fold["outer_fold"]
        # 找這一折的 refit 結果（正常情況下恰好一個）
        cand = [f for f in os.listdir(jobs_dir) if f.startswith(f"o{o}_refit_") and f.endswith(".json")]
        if not cand:
            print(f"  [warn] outer fold {o}: no refit result yet")
            continue
        res = read_result(os.path.join(jobs_dir, cand[0]))
        selected.append({"outer_fold": o, "hp": res["hp"], **res.get("test_metrics", {})})
        per_fold.append(res["test_metrics"])
        pooled_true += res["test"]["y_true"]      # 彙總所有折的測試預測
        pooled_score += res["test"]["y_score"]

    summary = {"experiment": cfg["experiment_name"], "n_outer_folds": len(per_fold),
               "selected_per_fold": selected}
    if per_fold:
        summary["across_folds_mean_ci"] = summarise_across_folds(per_fold)        # 跨折平均±CI（論文用）
        summary["pooled_bootstrap_ci"] = metrics_with_bootstrap_ci(pooled_true, pooled_score)
        summary["pooled_point"] = point_metrics(pooled_true, pooled_score)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 美化輸出
    print("\n" + "=" * 64)
    print(f"  Nested CV summary: {cfg['experiment_name']}  ({len(per_fold)} outer folds)")
    print("=" * 64)
    if per_fold:
        print("  Mean +/- 95% CI across outer folds:")
        for k, v in summary["across_folds_mean_ci"].items():
            print(f"    {k:12s}: {v['mean']:.3f}  [{v['ci_low']:.3f}, {v['ci_high']:.3f}]")
    print("  saved:", os.path.join(out_dir, "summary.json"))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="只印出計畫、不實際執行")
    ap.add_argument("--only-aggregate", action="store_true", help="只重建 summary.json")
    ap.add_argument("--max-parallel", type=int, default=None, help="覆寫使用的 GPU 數")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = os.path.join(cfg["output"]["root"], cfg["experiment_name"])
    jobs_dir = os.path.join(out_dir, "jobs")
    logs_dir = os.path.join(out_dir, "logs")
    for d in (jobs_dir, logs_dir):
        os.makedirs(d, exist_ok=True)

    splits_path = os.path.join(out_dir, "splits.json")
    if not os.path.exists(splits_path):
        sys.exit(f"splits not found: {splits_path}\nRun: python -m scripts.make_splits --config {args.config}")
    splits = load_splits(splits_path)

    if args.only_aggregate:
        aggregate(cfg, splits, jobs_dir, out_dir)
        return

    gpus = cfg["hardware"]["gpus"]
    if args.max_parallel:
        gpus = gpus[:args.max_parallel]
    inner_folds = cfg["cv"]["inner_folds"]
    stages = cfg["hpsearch"]["stages"]
    default_hp = {"lr": cfg["training"]["lr"],
                  "weight_decay": cfg["training"]["weight_decay"],
                  "dropout": cfg["training"]["dropout"]}

    # 每個外層折各自維護一份「目前最佳超參數」，隨階段逐步精修
    best = {fold["outer_fold"]: dict(default_hp) for fold in splits["folds"]}
    n_outer = len(splits["folds"])

    if args.dry_run:
        n_inner = n_outer * sum(len(s["values"]) for s in stages) * inner_folds
        print(f"plan: {n_outer} outer folds, inner {inner_folds}-fold")
        print(f"stages: {[ (s['name'], s['values']) for s in stages ]}")
        print(f"inner jobs: {n_inner} ; refit jobs: {n_outer} ; total: {n_inner + n_outer}")
        print(f"GPUs: {gpus}")
        return

    # ---- 分階段內層搜尋，所有折同步進行 ----
    for si, stage in enumerate(stages):
        param, values = stage["name"], stage["values"]
        print(f"\n### Stage {si+1}/{len(stages)}: tuning '{param}' over {values}")
        batch = []
        # 收集這一階段「所有折 x 所有候選值 x 所有內層折」的內層 job
        for fold in splits["folds"]:
            o = fold["outer_fold"]
            for val in values:
                hp = dict(best[o]); hp[param] = val      # 在該折目前最佳上，替換正在調的參數
                for k in range(inner_folds):
                    out_p = job_out_path(jobs_dir, o, "inner", hp, inner=k)
                    log_p = os.path.join(logs_dir, os.path.basename(out_p).replace(".json", ".log"))
                    batch.append(make_job(args.config, splits_path, o, "inner", hp, out_p, inner=k, log_path=log_p))
        run_jobs(batch, gpus)

        # 逐折挑出這一階段的勝出值（平均內層 AUC 最高者）
        for fold in splits["folds"]:
            o = fold["outer_fold"]
            scored = []
            for val in values:
                hp = dict(best[o]); hp[param] = val
                scored.append((val, mean_inner_auc(jobs_dir, o, hp, inner_folds)))
            scored = [(v, a) for v, a in scored if not np.isnan(a)]
            if scored:
                win = max(scored, key=lambda t: t[1])
                best[o][param] = win[0]
                print(f"  fold {o}: best {param}={fmt(win[0])} (inner AUC={win[1]:.3f})")

    # ---- 以選定的超參數 refit，並在外層測試折上評估 ----
    print(f"\n### Refit + test on {n_outer} outer folds")
    with open(os.path.join(out_dir, "selected_hp.json"), "w") as f:
        json.dump(best, f, indent=2)
    batch = []
    for fold in splits["folds"]:
        o = fold["outer_fold"]
        hp = best[o]
        out_p = job_out_path(jobs_dir, o, "refit", hp)
        log_p = os.path.join(logs_dir, os.path.basename(out_p).replace(".json", ".log"))
        batch.append(make_job(args.config, splits_path, o, "refit", hp, out_p, log_path=log_p))
    run_jobs(batch, gpus)

    aggregate(cfg, splits, jobs_dir, out_dir)


if __name__ == "__main__":
    main()
