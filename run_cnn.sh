#!/usr/bin/env bash
# Run essential CNN-MIL configs (nohup'd, survives ssh drop). / 跑必要的 CNN 設定。
set -u
cd /home/yikuan/PE_project
source ~/miniconda3/etc/profile.d/conda.sh
conda activate PE
export PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
IMG=results/mil/patches_t00_96.npz
FEAT=results/mil/instances_lobe21.npz
SP=results/swinunetr_i/splits.json
run () {  # mode pooling outfile
  echo "=== $1 / $2  start $(date +%H:%M) ==="
  python -m scripts.mil_cnn_train --img $IMG --feat $FEAT --splits $SP \
    --mode "$1" --pooling "$2" --seeds 5 --bs 32 --out "results/mil/$3" 2>&1 | tail -3
}
run hybrid max cnn_hybrid_max.json   # does image+features beat features-only 0.71? / 影像有沒有加值
run image  max cnn_image_max.json    # image alone / 單影像
echo "ALL CNN DONE $(date +%H:%M)"
