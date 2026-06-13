"""Fusion (deep + perfusion + radiomics) + abstention, under the SAME nested 10-fold.
融合（深度 + 灌注 + radiomics）+ 棄答，套同一份 nested 10 折。

C: stack per-case signals -> meta-classifier -> pooled AUC (does fusion beat 0.59?).
B: accuracy-rejection curve on the fused OOF probabilities (match the paper's abstention;
   does accuracy at ~17% rejection reach the paper's 0.72?).
C：把每病人的訊號疊起來 -> meta 分類器 -> pooled AUC。B：對融合機率畫 accuracy-棄答曲線。

Deep OOF predictions come from the existing single-phase nested-CV refit JSONs (each case was
predicted by a model that did not train on its fold). Mild stacking optimism noted.
深度 OOF 來自既有單相位 nested-CV 的逐折 test 預測。stacking 輕微樂觀已註明。

    python -m scripts.fusion_abstain --config configs/swinunetr_ie.yaml \
      --splits results/swinunetr_i/splits.json
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); out = []
    y, p = np.asarray(y), np.asarray(p)
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) > 1:
            out.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def nested_oof(X, y, folds, seed=42):
    """Logistic meta-model, return pooled OOF probs aligned to sample index. / 回傳對齊的 OOF 機率。"""
    oof = np.full(len(y), np.nan)
    for f in folds:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        Xtr, Xte = X[tr].copy(), X[te].copy()
        med = np.nanmedian(Xtr, axis=0); med = np.where(np.isfinite(med), med, 0.0)
        Xtr = np.where(np.isfinite(Xtr), Xtr, med)
        Xte = np.where(np.isfinite(Xte), Xte, med)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), y[tr])
        oof[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
    return oof


def acc_rejection(y, p, rejects=(0.0, 0.10, 0.17, 0.25, 0.33)):
    """Accuracy / sens / spec at each rejection rate (drop least-confident first). / 棄答曲線。"""
    y = np.asarray(y); p = np.asarray(p)
    conf = np.abs(p - 0.5)                          # confidence = distance from 0.5 / 信心
    order = np.argsort(conf)                         # least confident first / 最不確定先棄
    rows = []
    for r in rejects:
        ndrop = int(round(r * len(y)))
        keep = np.sort(order[ndrop:])               # keep most-confident / 留下較確定者
        yk, pk = y[keep], p[keep]
        pred = (pk > 0.5).astype(int)
        acc = float((pred == yk).mean())
        tp = int(((pred == 1) & (yk == 1)).sum()); fn = int(((pred == 0) & (yk == 1)).sum())
        tn = int(((pred == 0) & (yk == 0)).sum()); fp = int(((pred == 1) & (yk == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")
        rows.append((r, len(keep), acc, sens, spec))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    splits = json.load(open(args.splits))
    folds = splits["folds"]
    files = list(splits["filenames"])
    pos = {f: i for i, f in enumerate(files)}
    n = len(files)

    # labels aligned to splits order / 標籤對齊
    df = pd.read_csv(cfg["data"]["label_file"], header=None)
    lab = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    y = np.array([int(lab[f]) for f in files])

    def align(npz_path, key_files="files", key_X="X"):
        d = np.load(npz_path, allow_pickle=True)
        src = list(d[key_files]); X = d[key_X]
        out = np.full((n, X.shape[1]), np.nan)
        for i, f in enumerate(src):
            if f in pos:
                out[pos[f]] = X[i]
        return out

    Xperf = align("results/perfusion/perf_features.npz")        # 5 perfusion / 灌注
    Xrad = align("results/radiomics/features.npz")              # 89 radiomics

    # deep OOF predictions from refit JSONs / 深度 OOF
    deep = np.full(n, np.nan)
    for k in range(len(folds)):
        js = glob.glob(f"results/swinunetr_i/jobs/o{k}_refit_*.json")
        if not js:
            continue
        d = json.load(open(js[0]))
        for fn, sc in zip(d["test"]["filenames"], d["test"]["y_score"]):
            if fn in pos:
                deep[pos[fn]] = sc
    print(f"aligned: deep OOF {np.isfinite(deep).sum()}/{n} | perf {Xperf.shape} | rad {Xrad.shape}")
    print(f"deep-alone AUC {roc_auc_score(y, np.nan_to_num(deep, nan=np.nanmean(deep))):.3f} | "
          f"corr(deep,perfL2) {np.corrcoef(np.nan_to_num(deep), np.nan_to_num(Xperf[:,1]))[0,1]:.2f}\n")

    combos = {
        "deep only (1)": deep[:, None],
        "perfusion only (5)": Xperf,
        "radiomics only (89)": Xrad,
        "deep + perfusion (6)": np.column_stack([deep, Xperf]),
        "deep + perf + radiomics": np.column_stack([deep, Xperf, Xrad]),
        "perf + radiomics": np.column_stack([Xperf, Xrad]),
    }
    print("=== fusion: nested 10-fold pooled AUC ===")
    best_name, best_auc, best_oof = None, -1, None
    for name, X in combos.items():
        oof = nested_oof(X, y, folds)
        auc = roc_auc_score(y, oof)
        lo, hi = boot_ci(y, oof)
        print(f"  {name:<26} AUC {auc:.3f}  [{lo:.3f}, {hi:.3f}]")
        if auc > best_auc:
            best_name, best_auc, best_oof = name, auc, oof
    print(f"\nreferences: deep 0.593 | perfusion 0.585 | radiomics 0.642 | paper 0.72=ACC(not AUC)")

    print(f"\n=== abstention (B): accuracy-rejection on BEST = '{best_name}' (AUC {best_auc:.3f}) ===")
    print(f"{'reject%':>8}{'n kept':>8}{'accuracy':>10}{'sens':>8}{'spec':>8}   (paper: 17% rej -> 0.72 acc)")
    for r, nk, acc, sens, spec in acc_rejection(y, best_oof):
        print(f"{int(r*100):>7}%{nk:>8}{acc:>10.3f}{sens:>8.3f}{spec:>8.3f}")


if __name__ == "__main__":
    main()
