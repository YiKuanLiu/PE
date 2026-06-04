#!/bin/bash
# One-shot status report for the PE nested-CV run (run on the server; the Mac's
# daily launchd job calls this over SSH and saves/notifies the output).
# 產生 PE 實驗的當下狀態報告（在 server 上跑；Mac 的每日 launchd 透過 SSH 呼叫它）。
source /home/yikuan/miniconda3/etc/profile.d/conda.sh
conda activate PE
cd /home/yikuan/PE_project || exit 1
CFG=configs/swinunetr_i.yaml

echo "=== PE nested CV status @ $(date '+%Y-%m-%d %H:%M %Z') ==="
python -m scripts.status --config "$CFG" 2>/dev/null

echo
echo "--- run alive? / 是否在跑 ---"
if pgrep -f run_nested_cv >/dev/null; then echo "orchestrator: RUNNING"; else echo "orchestrator: NOT running"; fi
echo "train_fold procs: $(pgrep -f src.train_fold | wc -l)"
echo "tmux sessions: $(tmux ls 2>/dev/null | cut -d: -f1 | paste -sd, - || echo none)"

echo
echo "--- GPUs (idx, mem.used, util) ---"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null

echo
echo "--- jobs whose logs contain errors / 有錯誤的 job log ---"
errs=$(grep -lriE "Traceback|out of memory|Error" results/swinunetr_i/logs/ 2>/dev/null | head -5)
if [ -n "$errs" ]; then echo "$errs"; else echo "none"; fi

echo
echo "--- recent orchestrator log (run.log tail) ---"
tail -n 8 results/swinunetr_i/run.log 2>/dev/null || echo "(no run.log yet / 尚未開始)"
