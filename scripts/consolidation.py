"""Generalizability-without-external-data package, on the feature model (RF, 21 feats, fixed [50,150]).
無外部資料下的泛化性鞏固包(特徵模型 RF):permutation / 重複 nested CV / learning curve /
optimism-corrected bootstrap / leave-protocol-out(跨 spacing 的準外部驗證)。
"""
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

d0 = np.load("results/mil/instances_lobe21.npz", allow_pickle=True)
y = d0["y"].astype(int); N = len(y)
X = d0["bags"].astype(float).reshape(N, -1)                  # (125, 5*21)
folds = json.load(open("results/swinunetr_i/splits.json"))["folds"]


def imp(Xt, Xe):                                            # train-median impute / 用 train 中位數補值
    med = np.nanmedian(Xt, 0); med = np.where(np.isfinite(med), med, 0)
    return np.where(np.isfinite(Xt), Xt, med), np.where(np.isfinite(Xe), Xe, med)


def RF(seed=0):
    return RandomForestClassifier(400, max_depth=3, random_state=seed, n_jobs=-1)


def nested_oof(Xf, yy, fl, seed=0):
    oof = np.full(len(yy), np.nan)
    for f in fl:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        Xt, Xe = imp(Xf[tr], Xf[te])
        oof[te] = RF(seed).fit(Xt, yy[tr]).predict_proba(Xe)[:, 1]
    return oof


print("=== feature-model robustness (RF on 21 features, fixed [50,150]) ===", flush=True)
oof0 = nested_oof(X, y, folds); auc0 = roc_auc_score(y, oof0)
rng = np.random.default_rng(0)
bs = [roc_auc_score(y[i], oof0[i]) for i in (rng.integers(0, N, N) for _ in range(5000)) if len(set(y[i])) > 1]
print(f"[baseline] nested-OOF AUC {auc0:.3f} [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]", flush=True)

# 1) permutation test
NP = 500; pr = np.random.default_rng(1); null = np.empty(NP)
for k in range(NP):
    yp = pr.permutation(y); null[k] = roc_auc_score(yp, nested_oof(X, yp, folds))
    if (k + 1) % 100 == 0:
        print(f"  perm {k+1}/{NP}", flush=True)
p_perm = (1 + np.sum(null >= auc0)) / (NP + 1)
print(f"[1 permutation] null {null.mean():.3f}±{null.std():.3f} (95th {np.percentile(null,95):.3f}) | observed {auc0:.3f} | p={p_perm:.4f}", flush=True)

# 2) repeated nested CV (different fold partitions)
M = 20; reps = np.empty(M)
for s in range(M):
    fl = [{"train_idx": tr.tolist(), "test_idx": te.tolist()}
          for tr, te in StratifiedKFold(10, shuffle=True, random_state=s).split(X, y)]
    reps[s] = roc_auc_score(y, nested_oof(X, y, fl))
print(f"[2 repeated nested CV x{M}] {reps.mean():.3f} ± {reps.std():.3f} (min {reps.min():.3f}, max {reps.max():.3f})", flush=True)

# 3) learning curve
print("[3 learning curve] train-fraction -> pooled OOF AUC", flush=True)
for frac in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    a = []
    for ss in range(5):
        rr = np.random.default_rng(100 + ss); oof = np.full(N, np.nan)
        for f in folds:
            tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
            sub = rr.choice(tr, max(12, int(round(len(tr) * frac))), replace=False)
            if len(set(y[sub])) < 2:
                continue
            Xt, Xe = imp(X[sub], X[te]); oof[te] = RF().fit(Xt, y[sub]).predict_proba(Xe)[:, 1]
        m = np.isfinite(oof); a.append(roc_auc_score(y[m], oof[m]))
    print(f"   {frac:.1f} (~{int(round(112*frac))} train): {np.mean(a):.3f} ± {np.std(a):.3f}", flush=True)

# 4) optimism-corrected bootstrap (Harrell)
Xi, _ = imp(X, X); app_full = roc_auc_score(y, RF().fit(Xi, y).predict_proba(Xi)[:, 1])
rb = np.random.default_rng(7); opt = []
for _ in range(200):
    ix = rb.integers(0, N, N)
    if len(set(y[ix])) < 2:
        continue
    cb = RF().fit(Xi[ix], y[ix])
    opt.append(roc_auc_score(y[ix], cb.predict_proba(Xi[ix])[:, 1]) - roc_auc_score(y, cb.predict_proba(Xi)[:, 1]))
print(f"[4 optimism bootstrap] apparent {app_full:.3f} - optimism {np.mean(opt):.3f} = corrected {app_full-np.mean(opt):.3f}", flush=True)

# 5) leave-protocol-out (quasi-external across spacing)
try:
    sp = np.load("results/mil/spacing.npz", allow_pickle=True)
    for key in ["zmm", "xymm"]:
        v = sp[key].astype(float); fin = np.isfinite(v); med = np.nanmedian(v[fin]); grp = (v > med)
        oof = np.full(N, np.nan); dirinfo = []
        for hi_test in [True, False]:
            te = (grp == hi_test) & fin; tr = (grp != hi_test) & fin
            if not (8 <= te.sum() and 0 < y[te].sum() < te.sum() and 0 < y[tr].sum() < tr.sum()):
                continue
            Xt, Xe = imp(X[tr], X[te]); pe = RF().fit(Xt, y[tr]).predict_proba(Xe)[:, 1]
            oof[te] = pe; dirinfo.append((round(roc_auc_score(y[te], pe), 3), int(te.sum())))
        m = np.isfinite(oof)
        print(f"[5 leave-{key}-out @median {med:.2f}] cross-protocol pooled AUC {roc_auc_score(y[m],oof[m]):.3f} "
              f"(n={m.sum()}); per-direction(auc,n) {dirinfo}", flush=True)
except FileNotFoundError:
    print("[5 leave-protocol-out] spacing.npz missing -> skipped", flush=True)

print("ALLDONE", flush=True)
