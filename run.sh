#!/usr/bin/env bash
# 一鍵執行 nested CV（建議在 tmux 內執行，斷線不中斷）。
# 用法：  bash run.sh [config路徑]
#   預設 config = configs/swinunetr_i.yaml
# 特性：可中斷續跑 —— 重新執行會自動跳過已完成的 job。
set -eo pipefail

CONFIG="${1:-configs/swinunetr_i.yaml}"

# 啟用 conda 環境 PE（若已啟用亦無妨），並切到此腳本所在的 repo 根目錄
source /home/yikuan/miniconda3/etc/profile.d/conda.sh
conda activate PE
cd "$(dirname "$0")"

# 從 config 讀出 實驗名稱 / 輸出根目錄，組出 log 路徑
EXP=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['experiment_name'])" "$CONFIG")
ROOT=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output']['root'])" "$CONFIG")
OUT="$ROOT/$EXP"
mkdir -p "$OUT"

echo "=== config: $CONFIG  |  output: $OUT ==="
python -m scripts.make_splits --config "$CONFIG"          # 產生巢狀切分（快、不吃 GPU）
echo "=== plan ==="
python -m scripts.run_nested_cv --config "$CONFIG" --dry-run   # 先列出 job 計畫
echo "=== running (可 Ctrl-C / 卸離，再跑會續跑) ==="
python -u -m scripts.run_nested_cv --config "$CONFIG" 2>&1 | tee -a "$OUT/run.log"
echo "=== done. summary: $OUT/summary.json ==="
