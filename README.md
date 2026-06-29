# PE — Pulmonary Embolism Classification from Non-contrast CT Images
Implementation of the NCCT-based pulmonary embolism (PE) classification study
with a rigorous nested cross-validation protocol for evaluation and hyper-parameter tuning (using inner fold).

This repo builds up the scripts from image processing using Matlab, generating features, 
and finally train 8 different models for the classification task. 
The models can be divided in 3 main categories:
1. Pure imaged-based deep learning models
2. Feature based MLP models
3. Hybrid models that combine deep learning and meaningful features

In this study, we applied multi-instance learning (MIL) for pooling the embeddings from each lobe (if applicable).
Also, we evaluated the model performances by averaging the output probabilities from 5 different random seeds.

## Methods

Models:
* Pretrained SwinUNETR (VoComni pretrained weights needed) with whole CT image
* Simple CNN+MIL with 5 image patch per-lobe 
* Whole-lung Radiomics features with MLP
* Ventilation/Perfusion/Mismatch (V/Q/M) features with MLP+MIL
* 21 features (V/Q/M) + Volumes + HUs + hyperattentuation with MLP+MIL
* Hybrid: simple CNN + 21 features
* Masked CNN
* Hybrid Masked CNN + 21 features


## Layout

```
configs/swinunetr_i.yaml   # all paths + CV / HP-search settings
src/                       # SwinUNETR (deep) pipeline, used by run_nested_cv
  data.py        # PEDataset (loads T00 from .mat)
  models.py      # SwinClassifierI + load_pretrained
  metrics.py     # point metrics + 95% CI (bootstrap / across-fold)
  splits.py      # nested CV split generation (seeded, saved to JSON)
  train_fold.py  # train ONE model on ONE partition (single-GPU unit)
scripts/
  # --- shared inputs (run once for all models) ---
  make_splits.py      # one nested-CV split -> results/<exp>/splits.json
  mil_patches.py      # per-lobe 96^3 image patches + 21 per-lobe features
  mil_features.py     # per-lobe V/Q + density features only (18-feature variant)
  # --- the 8 models ---
  run_nested_cv.py    # SwinUNETR (deep): staged search, multi-GPU, resumable
  gen_feat_mean.py    # feature MIL (21-feature + V/Q), mean-pooling, 5-seed -> OOF
  radiomics_sanity.py # whole-lung radiomics + random forest
  mil_cnn_mask.py     # CNN / mask-guided / hybrid MIL (image, +/-mask, +/-features)
  mil_train.py        # feature-MIL engine (MLP encoder; mean/max/attention pooling)
  # --- aggregate + figures ---
  final_stats.py      # pool all 8 models' OOF -> AUC/CI + paired-bootstrap dAUC
  make_fig1_split.py  fig_ablation.py  make_fig3.py  fig_roc.py
  make_fig_lc.py  make_fig_threshold.py     # main + supplementary figures
  hires_export.py     # publication formats (PDF / 600-dpi TIFF / 300-dpi PNG)
results/             # splits, per-job JSON, OOF, figures, summary  (git-ignored)
```

## Setup

```bash
conda create -y -n PE python=3.10
conda activate PE
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Run

All eight models share **one** nested-CV split and two cached input files
(per-lobe image patches and per-lobe features); generate those once, then run
each model family. Commands assume `PYTHONPATH=.` and the `PE` conda env.

### 0. Shared inputs (run once)

```bash
conda activate PE
export PYTHONPATH=.

# nested-CV split (fast, no GPU) -> results/swinunetr_i/splits.json
python -m scripts.make_splits --config configs/swinunetr_i.yaml

# cached per-lobe 96^3 patches + 21 per-lobe features (shared by feature & CNN models)
#   -> results/mil/{patches_t00_96.npz, instances_lobe21.npz}
python -m scripts.mil_patches --config configs/swinunetr_i.yaml
```

### 1. Image-only deep learning — SwinUNETR (multi-GPU, resumable)

```bash
python -m scripts.run_nested_cv --config configs/swinunetr_i.yaml --dry-run         # show the plan
python -m scripts.run_nested_cv --config configs/swinunetr_i.yaml                    # run (resumable)
python -m scripts.run_nested_cv --config configs/swinunetr_i.yaml --only-aggregate   # (re)build summary
```

Scheduling notes:
* One training per GPU, dispatched across `hardware.gpus` (default `[0, 1, 2, 4]` — GPU 3 is the display card).
* **Resumable**: any job whose result JSON already exists is skipped, so it is safe to stop/restart (or run under `nohup` / `tmux`).
* Inner-search uses the reduced `hardware.inner_epochs` / `inner_patience`; refits use the full `training.epochs` / `patience`.

### 2. Interpretable feature models (CPU, seconds)

```bash
# 21-feature + V/Q mean-pooling MIL, 5-seed averaged -> results/mil/oof_feat_mean.npz
python -m scripts.gen_feat_mean

# whole-lung radiomics + random forest -> results/radiomics/features.npz
python -m scripts.radiomics_sanity --config configs/swinunetr_i.yaml \
        --splits results/swinunetr_i/splits.json
```

### 3. Domain-knowledge-guided CNN / hybrid (GPU; mean-pooling MIL, 5 seeds)

```bash
IMG=results/mil/patches_t00_96.npz
FEAT=results/mil/instances_lobe21.npz
SP=results/swinunetr_i/splits.json
common="--img $IMG --feat $FEAT --splits $SP --pooling mean --seeds 5"

python -m scripts.mil_cnn_mask $common --mode image  --no_mask --out results/mil/oof_image_nomask.json   # image-only CNN
python -m scripts.mil_cnn_mask $common --mode image            --out results/mil/oof_image_mask.json     # + hyperdense mask
python -m scripts.mil_cnn_mask $common --mode hybrid --no_mask --out results/mil/oof_hybrid_nomask.json  # CNN + features
python -m scripts.mil_cnn_mask $common --mode hybrid           --out results/mil/oof_hybrid_mask.json    # + mask (best)
```

The masked runs use the a-priori clot-range HU window by default; add
`--bands_npz results/mil/tuned_bands.npz` to use per-fold inner-CV-selected
bands instead (the manuscript reports both; they are equivalent).

### 4. Aggregate + figures

```bash
# pool all 8 models' OOF predictions -> results/mil/final_stats.json (AUC/CI + paired bootstrap)
python -m scripts.final_stats

# regenerate every manuscript figure, then export publication formats
for f in make_fig1_split fig_ablation make_fig3 fig_roc make_fig_lc make_fig_threshold; do
  python -m scripts.$f
done
python -m scripts.hires_export        # -> figs/hires/*.{pdf,tiff,png}
```

