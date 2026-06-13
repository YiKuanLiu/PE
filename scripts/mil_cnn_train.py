"""Hybrid CNN-MIL: per-lobe 96^3 image patch (small 3D CNN) + 21 hand-crafted features.
Hybrid CNN-MIL:逐肺葉 96³ 影像 patch(小型 3D CNN)+ 21 個手算特徵。

Per lobe: e_l = CNN(patch); x_l = features; modality-dropout on e_l; fuse [e_l,x_l] -> h_l;
pool over 5 lobes (max/mean/attention) -> bag -> head. Anchored by hand-crafted features.
PERF: the whole patch tensor is preloaded to GPU once (1.1GB f16 on 80GB A100) so per-batch is
on-GPU indexing (no host->device transfer per step). Nested 10-fold + multi-seed.
效能:整個 patch 張量一次預載到 GPU(避免每 batch 搬資料)。對照 features-only MIL 0.659 / hyperdense 0.71。

    python -m scripts.mil_cnn_train --img results/mil/patches_t00_96.npz \
      --feat results/mil/instances_lobe21.npz --splits results/swinunetr_i/splits.json \
      --mode hybrid --pooling max --seeds 5 [--quick]
"""
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

SZ = 96


class CNN3D(nn.Module):
    """Tiny 3D CNN: 96^3 -> 32-dim embedding (~120K params). / 小型 3D CNN。"""
    def __init__(self, emb=32, p=0.2):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv3d(i, o, 3, 2, 1), nn.BatchNorm3d(o), nn.ReLU(),
                                 nn.Dropout3d(p))
        self.net = nn.Sequential(blk(1, 8), blk(8, 16), blk(16, 32), blk(32, 32),
                                 nn.AdaptiveAvgPool3d(1))
        self.fc = nn.Linear(32, emb)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class HybridMIL(nn.Module):
    def __init__(self, n_feat=21, emb=32, H=32, dropout=0.3, p_img=0.3, pooling="max",
                 use_img=True, use_feat=True):
        super().__init__()
        self.use_img, self.use_feat, self.p_img, self.pooling = use_img, use_feat, p_img, pooling
        self.cnn = CNN3D(emb) if use_img else None
        fin = (emb if use_img else 0) + (n_feat if use_feat else 0)
        self.fuse = nn.Sequential(nn.Linear(fin, H), nn.ReLU(), nn.Dropout(dropout))
        self.att_V = nn.Linear(H, 8); self.att_U = nn.Linear(H, 8); self.att_w = nn.Linear(8, 1)
        self.head = nn.Linear(H, 1)
        self.last_att = None

    def forward(self, img, feat, mask):          # img[B,K,1,96^3] feat[B,K,F] mask[B,K]
        B, K = mask.shape
        parts = []
        if self.use_img:
            e = self.cnn(img.reshape(B * K, 1, SZ, SZ, SZ)).reshape(B, K, -1)
            if self.training and self.p_img > 0:
                e = e * (torch.rand(B, K, 1, device=e.device) > self.p_img).float()
            parts.append(e)
        if self.use_feat:
            parts.append(feat)
        h = self.fuse(torch.cat(parts, dim=2))
        if self.pooling == "max":
            z = h.masked_fill(mask.unsqueeze(-1) == 0, -1e9).max(1).values
        elif self.pooling == "mean":
            z = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            a = self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h))).squeeze(-1)
            a = torch.softmax(a.masked_fill(mask == 0, -1e9), 1)
            self.last_att = a.detach()
            z = (a.unsqueeze(-1) * h).sum(1)
        return self.head(z).squeeze(-1)


def augment(img):                                # img [B,K,1,96,96,96] on GPU
    for d in (3, 4, 5):
        if random.random() < 0.5:
            img = torch.flip(img, dims=[d])
    sc = torch.empty(img.size(0), 1, 1, 1, 1, 1, device=img.device).uniform_(0.9, 1.1)
    sh = torch.empty(img.size(0), 1, 1, 1, 1, 1, device=img.device).uniform_(-0.05, 0.05)
    return (img * sc + sh).clamp_(0, 1)


def standardize(Xtr, mtr, Xte):
    v = mtr.astype(bool)
    mu, sd = Xtr[v].mean(0), Xtr[v].std(0) + 1e-8
    return np.nan_to_num((Xtr - mu) / sd), np.nan_to_num((Xte - mu) / sd)


def run_fold(IMG_t, FEAT, MASK, y, tr, te, mode, pooling, seed, dev, epochs, bs):
    """IMG_t: patches preloaded on device (torch f16 [N,5,96,96,96]). / patch 已在 GPU。"""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    Xtr, Xte = standardize(FEAT[tr], MASK[tr], FEAT[te])
    use_img = mode in ("hybrid", "image"); use_feat = mode in ("hybrid", "feature")
    m = HybridMIL(n_feat=FEAT.shape[2], pooling=pooling, use_img=use_img, use_feat=use_feat).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    crit = nn.BCEWithLogitsLoss()
    feat_tr = torch.tensor(Xtr, dtype=torch.float32, device=dev)
    mask_tr = torch.tensor(MASK[tr], dtype=torch.float32, device=dev)
    y_tr = torch.tensor(y[tr], dtype=torch.float32, device=dev)
    tr_t = torch.as_tensor(tr, device=dev)
    idx = np.arange(len(tr))

    def img_for(global_idx):                     # on-GPU index + f16->f32 + add channel
        return IMG_t[global_idx].float().unsqueeze(2) if use_img \
            else torch.zeros(len(global_idx), 5, 1, 1, 1, 1, device=dev)

    for ep in range(epochs):
        m.train(); np.random.shuffle(idx)
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            img = img_for(tr_t[b])
            if use_img:
                img = augment(img)
            opt.zero_grad()
            crit(m(img, feat_tr[b], mask_tr[b]), y_tr[b]).backward(); opt.step()
    m.eval()

    def predict(ix_global):
        Xs = standardize(FEAT[tr], MASK[tr], FEAT[ix_global])[1]
        ix_t = torch.as_tensor(ix_global, device=dev)
        ps = []
        with torch.no_grad():
            for s in range(0, len(ix_global), bs):
                sl = slice(s, s + bs)
                fe = torch.tensor(Xs[sl], dtype=torch.float32, device=dev)
                mk = torch.tensor(MASK[ix_global[sl]], dtype=torch.float32, device=dev)
                ps.append(torch.sigmoid(m(img_for(ix_t[sl]), fe, mk)).cpu().numpy())
        return np.concatenate(ps)
    return predict(te), predict(tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True); ap.add_argument("--feat", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--mode", default="hybrid"); ap.add_argument("--pooling", default="max")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--out", default=None)
    args = ap.parse_args()

    di = np.load(args.img); IMG = di["bags_img"]
    dfe = np.load(args.feat, allow_pickle=True)
    FEAT = dfe["bags"].astype(np.float64); MASK = dfe["mask"].astype(np.float64)
    y = dfe["y"].astype(int)
    folds = json.load(open(args.splits))["folds"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.quick:
        folds = folds[:1]; args.seeds = 1; args.epochs = 5
    IMG_t = torch.from_numpy(IMG).to(dev)        # preload patches to GPU once / 一次預載
    print(f"IMG {tuple(IMG_t.shape)} on {dev} FEAT {FEAT.shape} mode {args.mode} "
          f"pool {args.pooling} seeds {args.seeds}", flush=True)

    oof = np.full(len(y), np.nan); tr_aucs = []
    for fi, f in enumerate(folds):
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"])
        pte, ptr = [], []
        for s in range(args.seeds):
            a, b = run_fold(IMG_t, FEAT, MASK, y, tr, te, args.mode, args.pooling, s, dev,
                            args.epochs, args.bs)
            pte.append(a); ptr.append(b)
        oof[te] = np.mean(pte, axis=0)
        if len(set(y[tr])) > 1:
            tr_aucs.append(roc_auc_score(y[tr], np.mean(ptr, axis=0)))
        print(f"  fold {fi} done", flush=True)
    valid = np.isfinite(oof)
    auc = roc_auc_score(y[valid], oof[valid])
    print(f"\n{args.mode}/{args.pooling}: pooled test AUC {auc:.3f} | mean train AUC "
          f"{np.mean(tr_aucs):.3f} (gap {np.mean(tr_aucs)-auc:+.3f} overfit check)", flush=True)
    print("ref: features-only MIL 0.659 | +hyperdense 0.71 | radiomics 0.642")
    if args.out:
        json.dump({"mode": args.mode, "pooling": args.pooling, "test_auc": auc,
                   "train_auc": float(np.mean(tr_aucs))}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
