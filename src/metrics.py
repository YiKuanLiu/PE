"""分類指標與 95% 信賴區間（CI）。

回報 AUC、敏感度、特異度、PPV、NPV、F1，與論文一致。提供兩種 CI：
  * ``metrics_with_bootstrap_ci`` —— 對「彙總後的預測集合」做 bootstrap CI；
  * ``summarise_across_folds``    —— 跨 10 個外層折取 平均 ± 95% CI
    （這是論文採用的回報方式）。
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def point_metrics(y_true, y_score, threshold=0.5):
    """在固定機率門檻下計算所有指標。"""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    # 混淆矩陣四格
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    def safe(num, den):
        # 分母為 0 時回傳 nan，避免除零
        return float(num) / den if den > 0 else float("nan")

    sens = safe(tp, tp + fn)          # 敏感度 / 召回率
    spec = safe(tn, tn + fp)          # 特異度
    ppv = safe(tp, tp + fp)           # 陽性預測值 / 精確率
    npv = safe(tn, tn + fn)           # 陰性預測值
    f1 = safe(2 * tp, 2 * tp + fp + fn)
    try:
        # 單一類別時 AUC 無定義
        auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")

    return {"AUC": auc, "Sensitivity": sens, "Specificity": spec,
            "PPV": ppv, "NPV": npv, "F1": f1}


def metrics_with_bootstrap_ci(y_true, y_score, threshold=0.5, n_boot=2000, seed=42):
    """彙總後的點估計 + 百分位 bootstrap 95% CI。"""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    base = point_metrics(y_true, y_score, threshold)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot = {k: [] for k in base}
    for _ in range(n_boot):
        # 有放回重抽
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # 重抽到單一類別則略過該次
        m = point_metrics(y_true[idx], y_score[idx], threshold)
        for k, v in m.items():
            if not np.isnan(v):
                boot[k].append(v)

    out = {}
    for k, v in base.items():
        arr = np.asarray(boot[k])
        if len(arr):
            lo, hi = np.percentile(arr, [2.5, 97.5])  # 取 2.5 / 97.5 百分位
        else:
            lo = hi = float("nan")
        out[k] = {"value": v, "ci_low": float(lo), "ci_high": float(hi)}
    return out


def summarise_across_folds(per_fold_metrics):
    """跨折取 平均 ± 95% CI（常態近似），即論文的回報方式。

    ``per_fold_metrics``：每個外層折一個 ``point_metrics`` 字典所組成的 list。
    """
    keys = per_fold_metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.asarray([m[k] for m in per_fold_metrics], dtype=float)
        vals = vals[~np.isnan(vals)]
        mean = float(np.mean(vals)) if len(vals) else float("nan")
        if len(vals) > 1:
            # 標準誤 = 樣本標準差 / sqrt(n)；95% CI 半寬 = 1.96 * SEM
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            half = 1.96 * sem
        else:
            half = float("nan")
        out[k] = {"mean": mean, "ci_low": mean - half, "ci_high": mean + half}
    return out
