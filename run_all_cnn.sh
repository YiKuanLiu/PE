#!/usr/bin/env bash
set -u
cd /home/yikuan/PE_project
source ~/miniconda3/etc/profile.d/conda.sh; conda activate PE
export PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
A="--img results/mil/patches_t00_96.npz --feat results/mil/instances_lobe21.npz --splits results/swinunetr_i/splits.json --pooling max --seeds 5 --bs 32"
B="results/mil/tuned_bands.npz"
echo "=== image no-mask $(date +%H:%M) ===";  python -m scripts.mil_cnn_mask $A --no_mask --mode image  --out results/mil/oof_image_nomask.json  2>&1 | tail -2
echo "=== hybrid no-mask $(date +%H:%M) ==="; python -m scripts.mil_cnn_mask $A --no_mask --mode hybrid --out results/mil/oof_hybrid_nomask.json 2>&1 | tail -2
echo "=== image mask $(date +%H:%M) ===";     python -m scripts.mil_cnn_mask $A --bands_npz $B --mode image  --out results/mil/oof_image_mask.json   2>&1 | tail -2
echo "=== hybrid mask $(date +%H:%M) ===";    python -m scripts.mil_cnn_mask $A --bands_npz $B --mode hybrid --out results/mil/oof_hybrid_mask.json  2>&1 | tail -2
echo "ALL CNN OOF DONE $(date +%H:%M)"
