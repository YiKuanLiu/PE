#!/usr/bin/env bash
# Image-only MIL with mean & attention pooling (max already done = 0.556). / 純影像 MIL 補 mean/attention。
set -u
cd /home/yikuan/PE_project
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PE
export PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
IMG=results/mil/patches_t00_96.npz
FEAT=results/mil/instances_lobe21.npz
SP=results/swinunetr_i/splits.json
run () {
  echo "=== image / $1  start $(date +%H:%M) ==="
  python -m scripts.mil_cnn_train --img $IMG --feat $FEAT --splits $SP \
    --mode image --pooling "$1" --seeds 5 --bs 32 --out "results/mil/cnn_image_$1.json" 2>&1 | tail -3
}
run mean
run attention
echo "ALL IMG-MIL DONE $(date +%H:%M)"
