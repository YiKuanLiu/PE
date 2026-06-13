"""Compare single-phase T00-ROI vs T50-ROI feasibility (4 folds). / 比較單相位 T00-ROI vs T50-ROI."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score


def load_phase(phase):
    ys, ps, perfold = [], [], []
    for k in range(4):
        d = json.load(open(f"results/swinunetr_phase/feas/{phase}_o{k}.json"))
        yt = [int(v) for v in d["test"]["y_true"]]
        pp = [float(v) for v in d["test"]["y_score"]]
        ys += yt; ps += pp
        perfold.append(roc_auc_score(yt, pp) if len(set(yt)) > 1 else float("nan"))
    return np.array(ys), np.array(ps), perfold


def boot(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) > 1:
            out.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


y0, p0, pf0 = load_phase("T00")
y5, p5, pf5 = load_phase("T50")
a0, a5 = roc_auc_score(y0, p0), roc_auc_score(y5, p5)
c0, c5 = boot(y0, p0), boot(y5, p5)

print("=" * 72)
print(f"{'model':<26}{'pooled AUC':>12}{'95% CI':>22}{'per-fold mean':>12}")
print("-" * 72)
print(f"{'T00-ROI (inhale/吸氣)':<26}{a0:>12.3f}{f'[{c0[0]:.3f}, {c0[1]:.3f}]':>22}{np.nanmean(pf0):>12.3f}")
print(f"{'T50-ROI (exhale/吐氣)':<26}{a5:>12.3f}{f'[{c5[0]:.3f}, {c5[1]:.3f}]':>22}{np.nanmean(pf5):>12.3f}")
print("=" * 72)
print("per-fold:  fold   T00-ROI   T50-ROI    delta")
for k in range(4):
    print(f"           {k}      {pf0[k]:.3f}     {pf5[k]:.3f}    {pf5[k]-pf0[k]:+.3f}")
wins = sum(pf5[k] > pf0[k] for k in range(4))
print(f"\nT50 > T00 in {wins}/4 folds | pooled delta {a5-a0:+.3f}")
print("references (pooled): baseline T00 no-ROI=0.593 | dual IE+ROI(4f)=0.512 | radiomics RF=0.639")
json.dump({"T00_pooled": a0, "T50_pooled": a5, "T00_ci": list(c0), "T50_ci": list(c5),
           "T00_perfold": pf0, "T50_perfold": pf5, "T50_wins": wins},
          open("results/swinunetr_phase/feas/compare.json", "w"), indent=2)
print("\nsaved -> results/swinunetr_phase/feas/compare.json")
