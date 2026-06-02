"""Classification metrics with 95% confidence intervals.

Reports AUC, sensitivity, specificity, PPV, NPV and F1, matching the paper.
Two CI flavours are provided:
  * ``metrics_with_bootstrap_ci`` -- bootstrap CI on a pooled prediction set;
  * ``summarise_across_folds``    -- mean +/- 95% CI across the 10 outer folds
    (this is what the paper reports).
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def point_metrics(y_true, y_score, threshold=0.5):
    """Compute all metrics at a fixed probability threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    def safe(num, den):
        return float(num) / den if den > 0 else float("nan")

    sens = safe(tp, tp + fn)
    spec = safe(tn, tn + fp)
    ppv = safe(tp, tp + fp)
    npv = safe(tn, tn + fn)
    f1 = safe(2 * tp, 2 * tp + fp + fn)
    try:
        auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")

    return {"AUC": auc, "Sensitivity": sens, "Specificity": spec,
            "PPV": ppv, "NPV": npv, "F1": f1}


def metrics_with_bootstrap_ci(y_true, y_score, threshold=0.5, n_boot=2000, seed=42):
    """Pooled point metrics + percentile bootstrap 95% CIs."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    base = point_metrics(y_true, y_score, threshold)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot = {k: [] for k in base}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = point_metrics(y_true[idx], y_score[idx], threshold)
        for k, v in m.items():
            if not np.isnan(v):
                boot[k].append(v)

    out = {}
    for k, v in base.items():
        arr = np.asarray(boot[k])
        if len(arr):
            lo, hi = np.percentile(arr, [2.5, 97.5])
        else:
            lo = hi = float("nan")
        out[k] = {"value": v, "ci_low": float(lo), "ci_high": float(hi)}
    return out


def summarise_across_folds(per_fold_metrics):
    """Mean +/- 95% CI (normal approx) across folds, as reported in the paper.

    ``per_fold_metrics``: list of dicts from ``point_metrics`` (one per outer fold).
    """
    keys = per_fold_metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.asarray([m[k] for m in per_fold_metrics], dtype=float)
        vals = vals[~np.isnan(vals)]
        mean = float(np.mean(vals)) if len(vals) else float("nan")
        if len(vals) > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            half = 1.96 * sem
        else:
            half = float("nan")
        out[k] = {"mean": mean, "ci_low": mean - half, "ci_high": mean + half}
    return out
