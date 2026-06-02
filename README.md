# PE — Pulmonary Embolism Classification from Single-Phase NCCT

Re-implementation of the NCCT-based pulmonary embolism (PE) classification study
with a **rigorous nested cross-validation** protocol for hyper-parameter tuning.

This repo rebuilds the training pipeline from the (rejected) draft so experiments
can be re-run cleanly. The headline methodological change is **nested CV**
(outer 10-fold for performance estimation, inner 5-fold for HP selection),
replacing the previous single internal validation split.

## Method (first model: SwinUNETR-I)

* **Backbone**: VoComni-pretrained SwinUNETR (Large, feature size 96).
  157/159 tensors load from the pretrained checkpoint; only the new linear
  classification head is randomly initialised.
* **Input**: single-phase inhalation NCCT (`T00`), 1×512×512×96, internally
  resampled to 256×256×96. Data are pre-cropped, HU-clipped to [-1000, 400]
  and normalised to [0, 1].
* **Loss / opt**: `BCEWithLogitsLoss`, AdamW, mixed-precision (AMP).
* **Selection**: early stopping on validation AUC.
* **Metrics**: AUC, sensitivity, specificity, PPV, NPV, F1 — reported as
  mean ± 95% CI across the 10 outer folds, plus pooled bootstrap CIs.

### Nested CV + staged hyper-parameter search

Hyper-parameters are tuned in **stages** (one group at a time, freezing the
previous winner), as in the draft:

1. learning rate ∈ {1e-3, 1e-4, 1e-5}
2. weight decay ∈ {1e-3, 1e-4, 1e-5}
3. dropout      ∈ {0.0, 0.1, 0.2}

For each outer fold the selected config is refit on the full outer-training pool
and evaluated on the held-out test fold. Total ≈ 460 model trainings
(10 outer × (3+3+3 candidates × 5 inner) + 10 refits).

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

## Data

Pre-processed volumes (125 patients: 65 PE-positive, 60 PE-negative) live at
`/mnt/hot/public/Yi-Kuan/PE/original_size_crop51251296` with `label.csv`
(`filename,label`). Patient data and model weights are **never** committed
(see `.gitignore`).
