#!/usr/bin/env bash
set -u
cd /home/yikuan/PE_project
source ~/miniconda3/etc/profile.d/conda.sh; conda activate PE
export PYTHONPATH=. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
A="--img results/mil/patches_t00_96.npz --feat results/mil/instances_lobe21.npz --splits results/swinunetr_i/splits.json --pooling max --seeds 5 --bs 32 --bands_npz results/mil/tuned_bands.npz"
echo "=== image-mask-tuned $(date +%H:%M) ==="; python -m scripts.mil_cnn_mask $A --mode image  --out results/mil/cnn_image_mask_tuned.json  2>&1 | tail -3
echo "=== hybrid-mask-tuned $(date +%H:%M) ==="; python -m scripts.mil_cnn_mask $A --mode hybrid --out results/mil/cnn_hybrid_mask_tuned.json 2>&1 | tail -3
echo "ALL MASK-TUNED DONE $(date +%H:%M)"
