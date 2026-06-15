# MATLAB feature extraction (per-lobe V/Q + hyperdense)

MATLAB reimplementation of the image-processing / feature-generation steps used by the
best classifier in this project (honest nested-10-fold **AUC ≈ 0.71**, driven by the
**hyperdense lumen-sign** feature). Mirrors the Python `scripts/mil_patches.py`.

對應本專案最佳分類器(誠實 nested 10 折 **AUC ≈ 0.71**,由 **hyperdense lumen sign** 特徵主導)的
影像處理 / 特徵生成步驟之 MATLAB 版,對應 Python 的 `scripts/mil_patches.py`。

## Files
- `extract_lobe_features.m` — core function: one `.mat` → `5 × 21` per-lobe feature matrix.
- `run_all_features.m` — driver: loops over `Positive_Anon` / `Negative_Anon`, saves `lobe_features_matlab.mat`.

## Pipeline (per lobe, registration-free integrals)
1. `HU = stored − 1024`; density `rho = 1 + HU/1000` (air −1000HU→0, water 0HU→1).
2. Inhale mask = `T00_Lobe==l`, exhale mask = `T50_Lobe==l` (independent DenseNet lobe segs).
3. mass = Σ rho·vox, air = Σ(1−rho)·vox = vol − mass, over each lobe & phase.
4. **perfusion** `M=|mass_in−mass_ex|/vol_ex`, **ventilation** `V=|air_in−air_ex|/vol_ex`,
   **mismatch** `R=log(V/M)`; plus HU mean/Δ, volumes & ratios, mass/air, HU std/skew, and
   **hyperdense**: `hyper_max` (max HU capped 300), `hyper_p99`, **`hyper_frac` = fraction of
   voxels with HU∈[50,150]** (the dominant feature, ~0.70 alone).

## Numerical notes (match Python)
- `std` uses population (÷N) to match `numpy.std`; `skewness` is the biased g1 (`scipy.stats.skew`
  default); percentile uses linear interpolation (`numpy.percentile` default). All implemented
  inline so **no Statistics Toolbox is needed** and values match the Python features.

## What feeds the classifier
The `5 × 21` features per patient → MIL (mean/max/attention pooling) or RandomForest.
Best: ~0.71 AUC. A pure-image 3D-CNN (no hand features) only reached ~0.55 — the hand-crafted
hyperdense feature encodes domain knowledge the CNN cannot learn from 125 cases.
