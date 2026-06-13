"""Feature ablation for the lobe MIL: which feature groups actually help?
肺葉 MIL 的特徵消融:哪些特徵群真的有用?

Reuses mil_train (model + nested run). Tests each feature group alone and cumulatively, under
max (best) and attention pooling. Flags spacing-dependent absolute features (possible confound).
重用 mil_train。各特徵群單獨/累加測試,用 max 與 attention 聚合。標記 spacing 相依的絕對特徵(可能混淆)。
"""
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.mil_train import load_bags, run_pooling

# feature index groups (matching mil_features.FEAT order) / 特徵分組
GROUPS = {
    "VQ (M,V,R)":      [0, 1, 2],
    "HU (in,ex,chg)":  [3, 4, 5],
    "vol-ratio":       [8, 9],            # volshrink, volratio (spacing-robust) / 體積比(抗 spacing)
    "vol-abs":         [6, 7],            # volin, volex (spacing-dependent!) / 絕對體積(spacing 相依!)
    "mass/air-abs":    [10, 11, 12, 13],  # spacing-dependent! / spacing 相依!
    "first-order":     [14, 15, 16, 17],  # std/skew HU
}
CUMULATIVE = [
    ("VQ", [0, 1, 2]),
    ("VQ+HU", [0, 1, 2, 3, 4, 5]),
    ("VQ+HU+volratio", [0, 1, 2, 3, 4, 5, 8, 9]),
    ("VQ+HU+volratio+firstorder", [0, 1, 2, 3, 4, 5, 8, 9, 14, 15, 16, 17]),
    ("ALL-robust (no abs vol/mass)", [0, 1, 2, 3, 4, 5, 8, 9, 14, 15, 16, 17]),
    ("ALL (18, incl abs)", list(range(18))),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    X, mask, y, files, names = load_bags(args.features)
    folds = json.load(open(args.splits))["folds"]
    hp = dict(H=16, dropout=0.3, wd=1e-2, lr=1e-3, epochs=150, lam_ent=0.01)

    def auc(idx, pooling):
        a, _, _, _, _ = run_pooling(X[:, :, idx], mask, y, folds, pooling, args.seeds, hp, "cpu")
        return a

    print("=== each feature group ALONE ===")
    print(f"{'group':<30}{'max':>8}{'attention':>11}")
    for nm, idx in GROUPS.items():
        print(f"{nm:<30}{auc(idx,'max'):>8.3f}{auc(idx,'attention'):>11.3f}")

    print("\n=== cumulative ===")
    print(f"{'feature set':<32}{'max':>8}{'attention':>11}")
    for nm, idx in CUMULATIVE:
        print(f"{nm:<32}{auc(idx,'max'):>8.3f}{auc(idx,'attention'):>11.3f}")
    print("\nref: radiomics 0.642 | all-18 max 0.659 | mismatch 0.61")
    print("note: abs vol/mass features are spacing-dependent -> compare ALL-robust vs ALL(incl abs)")


if __name__ == "__main__":
    main()
