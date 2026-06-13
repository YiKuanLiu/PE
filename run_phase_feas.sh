#!/usr/bin/env bash
# Phase feasibility launcher: T00-ROI then T50-ROI, folds 0-3, GPUs 0/1/2/4.
# 相位可行性啟動器：先 T00-ROI 再 T50-ROI，folds 0-3，GPU 0/1/2/4。
set -u
cd /home/yikuan/PE_project
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PE

GPUS=(0 1 2 4)
OUT=results/swinunetr_phase/feas
CACHE=/home/yikuan/PE_project/cache_roi
mkdir -p "$OUT"

run_wave () {
  phase=$1
  echo "=== wave $phase start $(date +%H:%M) ==="
  pids=()
  for i in 0 1 2 3; do
    gpu=${GPUS[$i]}
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu \
      python -u -m scripts.feasibility_phase \
        --config configs/swinunetr_ie.yaml \
        --splits results/swinunetr_i/splits.json \
        --cache "$CACHE" --phase "$phase" --outer-fold "$i" --epochs 40 \
        --out "$OUT/${phase}_o${i}.json" \
        > "$OUT/${phase}_o${i}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  echo "=== wave $phase done $(date +%H:%M) ==="
}

run_wave T00
run_wave T50
echo "ALL PHASE FEASIBILITY DONE $(date +%H:%M)"
grep -h "+ROI AUC" "$OUT"/*.log | sort
