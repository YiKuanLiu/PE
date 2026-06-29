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
src/
  data.py        # PEDataset (loads T00 from .mat)
  models.py      # SwinClassifierI + load_pretrained
  metrics.py     # point metrics + 95% CI (bootstrap / across-fold)
  splits.py      # nested CV split generation (seeded, saved to JSON)
  train_fold.py  # train ONE model on ONE partition (single-GPU unit)
scripts/
  make_splits.py     # generate results/<exp>/splits.json
  run_nested_cv.py   # orchestrator: staged search, multi-GPU, resumable
results/             # splits, per-job JSON, logs, summary  (git-ignored)
```

## Setup

```bash
conda create -y -n PE python=3.10
conda activate PE
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Run

```bash
# 1. generate the nested CV splits (fast, no GPU)
python -m scripts.make_splits --config configs/swinunetr_i.yaml

# 2. see the plan without running anything
python -m scripts.run_nested_cv --config configs/swinunetr_i.yaml --dry-run

# 3. run the full nested CV (resumable; one job per GPU across hardware.gpus)
python -m scripts.run_nested_cv --config configs/swinunetr_i.yaml

# 4. (re)build the summary from finished jobs at any time
python -m scripts.run_nested_cv --config configs/swinunetr_i.yaml --only-aggregate
```

Scheduling notes:
* One training runs per GPU; jobs are dispatched concurrently across
  `hardware.gpus` (defaults to `[0, 1, 2, 4]` — GPU 3 is the display card).
* The run is **resumable**: any job whose result JSON already exists is skipped,
  so it is safe to stop and restart (or run under `nohup` / `tmux`).
* Inner-search trainings use the reduced `hardware.inner_epochs` / `inner_patience`;
  refits use the full `training.epochs` / `patience`.

