"""Attention-MIL (deep learning) for PE diagnosis from per-lobe V/Q features.
逐肺葉 V/Q 特徵的 Attention-MIL(深度學習)PE 診斷。

bag = patient, instance = lung lobe (5), instance feature = [perfusion, ventilation, mismatch,
HU, volume-change, ...]. Gated-attention MIL pools the lobes; attention is masked for missing
lobes (e.g. neg_10 has no lobe 3). mean/max pooling + RF are ablation baselines.
bag=病人,instance=肺葉(5),特徵=逐肺葉 V/Q 等。Gated-attention 聚合肺葉,缺葉以 mask 處理。

    python -m scripts.mil_train --features results/perfusion/vq_features.npz \
      --splits results/swinunetr_i/splits.json --pooling all --seeds 5
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------- #
# Data: load bags [N, K, F] + mask [N, K] / 載入 bag
# --------------------------------------------------------------------------- #
def load_bags(path):
    d = np.load(path, allow_pickle=True)
    keys = set(d.keys())
    if "bags" in keys:                                   # rich features npz / 完整特徵
        X = d["bags"].astype(np.float64); mask = d["mask"].astype(np.float64)
        names = list(d["names"]) if "names" in keys else [f"f{i}" for i in range(X.shape[2])]
    else:                                                # vq_features.npz (125,15)->reshape / 重塑
        flat = d["X"].astype(np.float64); n = flat.shape[0]
        X = np.stack([flat[:, [l, 5 + l, 10 + l]] for l in range(5)], axis=1)  # [N,5,3] M,V,R
        mask = np.isfinite(X).all(axis=2).astype(np.float64)
        names = ["M", "V", "R"]
    y = d["y"].astype(int); files = list(d["files"])
    X = np.nan_to_num(X, nan=0.0)        # masked instances -> finite (0); ignored via mask / 缺葉填 0,靠 mask 忽略
    return X, mask, y, files, names


# --------------------------------------------------------------------------- #
# Gated-Attention MIL (Ilse et al. 2018), tiny + masked / 小型、可遮罩
# --------------------------------------------------------------------------- #
class MIL(nn.Module):
    def __init__(self, F, H=16, D=8, dropout=0.3, pooling="attention"):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(F, H), nn.ReLU(), nn.Dropout(dropout))
        self.att_V = nn.Linear(H, D); self.att_U = nn.Linear(H, D); self.att_w = nn.Linear(D, 1)
        self.head = nn.Linear(H, 1)
        self.pooling = pooling
        self.last_att = None

    def forward(self, x, mask):                          # x [B,K,F], mask [B,K]
        h = self.enc(x)                                  # [B,K,H]
        if self.pooling == "attention":
            a = self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h))).squeeze(-1)
            a = a.masked_fill(mask == 0, -1e9)
            a = torch.softmax(a, dim=1)                   # [B,K] attention weights / 注意力
            self.last_att = a.detach()
            z = (a.unsqueeze(-1) * h).sum(1)              # [B,H]
            ent = -(a.clamp_min(1e-9) * a.clamp_min(1e-9).log()).sum(1).mean()  # attention entropy
        elif self.pooling == "mean":
            z = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            ent = torch.tensor(0.0, device=x.device)
        elif self.pooling == "max":
            z = h.masked_fill(mask.unsqueeze(-1) == 0, -1e9).max(1).values
            ent = torch.tensor(0.0, device=x.device)
        return self.head(z).squeeze(-1), ent             # logit [B], attention entropy


# --------------------------------------------------------------------------- #
# Train one model (fixed epochs, heavy reg, multi-seed avg outside) / 訓練單一模型
# --------------------------------------------------------------------------- #
def standardize(Xtr, mask_tr, Xte):
    """Standardize each feature over VALID train instances. / 用訓練集有效 instance 標準化。"""
    v = mask_tr.astype(bool)
    flat = Xtr[v]                                        # [n_valid, F]
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def train_eval(Xtr, mtr, ytr, Xte, mte, pooling, seed, H=16, dropout=0.3,
               wd=1e-2, lr=1e-3, epochs=150, lam_ent=0.0, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    F = Xtr.shape[2]
    m = MIL(F, H=H, dropout=dropout, pooling=pooling).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    crit = nn.BCEWithLogitsLoss()
    xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    mt = torch.tensor(mtr, dtype=torch.float32, device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        logit, ent = m(xt, mt)
        loss = crit(logit, yt) - lam_ent * ent           # maximize attention entropy / 鼓勵分散
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        xe = torch.tensor(Xte, dtype=torch.float32, device=device)
        me = torch.tensor(mte, dtype=torch.float32, device=device)
        p = torch.sigmoid(m(xe, me)[0]).cpu().numpy()
        att = m.last_att.cpu().numpy() if m.last_att is not None else None
    return p, att


def rf_baseline(Xtr, mtr, ytr, Xte, mte, seed):
    """Non-DL baseline: flatten lobes (impute), RandomForest. / 非-DL 基準。"""
    def flat(X, M):
        Xf = X.copy(); Xf[M == 0] = np.nan
        return Xf.reshape(X.shape[0], -1)
    ftr, fte = flat(Xtr, mtr), flat(Xte, mte)
    med = np.nanmedian(ftr, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    ftr = np.where(np.isfinite(ftr), ftr, med); fte = np.where(np.isfinite(fte), fte, med)
    c = RandomForestClassifier(400, max_depth=3, random_state=seed, n_jobs=-1).fit(ftr, ytr)
    return c.predict_proba(fte)[:, 1]


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed); out = []
    y, p = np.asarray(y), np.asarray(p)
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            out.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def run_pooling(X, mask, y, folds, pooling, seeds, hp, device):
    """Nested 10-fold; seed-averaged test probs pooled -> AUC. / 多種子平均 → pooled AUC。"""
    oof = np.full(len(y), np.nan)
    att_store = {}
    for f in folds:
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        Xtr, Xte = standardize(X[tr], mask[tr], X[te])
        if pooling == "rf":
            ps = [rf_baseline(Xtr, mask[tr], y[tr], Xte, mask[te], s) for s in range(seeds)]
        else:
            res = [train_eval(Xtr, mask[tr], y[tr], Xte, mask[te], pooling, s, device=device, **hp)
                   for s in range(seeds)]
            ps = [r[0] for r in res]
            if pooling == "attention" and res[0][1] is not None:
                for j, idx in enumerate(te):
                    att_store[idx] = np.mean([r[1][j] for r in res], axis=0)
        oof[te] = np.mean(ps, axis=0)
    auc = roc_auc_score(y, oof)
    lo, hi = boot_ci(y, oof)
    return auc, lo, hi, oof, att_store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--pooling", default="all", help="attention|mean|max|rf|all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lam_ent", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    X, mask, y, files, names = load_bags(args.features)
    folds = json.load(open(args.splits))["folds"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hp = dict(H=args.H, dropout=args.dropout, wd=args.wd, lr=args.lr,
              epochs=args.epochs, lam_ent=args.lam_ent)
    print(f"bags {X.shape}  instances/feat names {names}  pos {int(y.sum())}  device {device}")
    print(f"HP {hp}  seeds {args.seeds}\n")

    pools = ["attention", "mean", "max", "rf"] if args.pooling == "all" else [args.pooling]
    print(f"{'pooling':<12}{'pooled AUC':>12}{'95% CI':>20}")
    print("-" * 44)
    results, best_att = {}, None
    for pl in pools:
        auc, lo, hi, oof, att = run_pooling(X, mask, y, folds, pl, args.seeds, hp, device)
        results[pl] = {"auc": auc, "ci": [lo, hi]}
        print(f"{pl:<12}{auc:>12.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}")
        if pl == "attention":
            best_att = att
    print("-" * 44)
    print("ref: deep 0.593 | radiomics 0.642 | mismatch 0.61 | perfusion 0.585")

    if best_att and len(names) <= 6:                     # attention by lobe, PE+ vs PE- / 注意力解讀
        A = np.full((len(y), 5), np.nan)
        for i, a in best_att.items():
            A[i, :len(a)] = a
        print("\nmean attention per lobe (PE+ / PE-):")
        for l in range(5):
            col = A[:, l]
            print(f"  L{l+1}: {np.nanmean(col[y==1]):.3f} / {np.nanmean(col[y==0]):.3f}")
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
