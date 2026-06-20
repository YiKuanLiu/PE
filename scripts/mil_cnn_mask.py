"""Mask-guided hybrid CNN-MIL: feed [CT, hyperdense-binary-mask] (2 channels) to the encoder.
mask-guided CNN:把 [CT, hyperdense 二值遮罩] 雙通道餵給 image encoder,強迫它注意血栓區。

The mask is derived on-the-fly from the cached 96^3 CT patches (patch = (HU+1000)/1400, so
HU = patch*1400-1000), so NO mat re-read is needed. Decisive ablation: does telling the CNN
WHERE the clot is let it beat the simple count? Compares image-only(mask) and hybrid(mask).
遮罩由快取 patch 直接推出(不必讀 mat)。決定性 ablation:把血栓位置直接給 CNN,它能否贏過簡單計數?

設計概觀 / Design at a glance:
  • MIL 框架:一個病人 = 一個 bag,5 個肺葉 = 5 個 instance。
    MIL: one patient = one bag, its 5 lung lobes = 5 instances.
  • 每個肺葉:96³ 影像 patch 經 3D-CNN → 32 維嵌入,(hybrid 時)再接上 21 個手算特徵。
    Per lobe: a 96³ patch through a 3D-CNN → 32-d embedding, optionally concatenated with the 21 hand-crafted features.
  • 5 個肺葉用 pooling(max / mean / attention)聚合成病人向量 → 線性分類頭。
    The 5 lobes are pooled (max / mean / attention) into a patient vector → linear classification head.
  • 「mask」= 把 hyperattenuation(血栓密度)位置當成第 2 個輸入通道,引導 CNN。
    The "mask" = the hyperattenuation (clot-density) location as a 2nd input channel, guiding the CNN.
  • 評估:外層 nested 10-fold(此檔只跑各折,splits 由 --splits 提供),每折 5 seeds 平均。
    Evaluation: outer nested 10-fold (folds supplied via --splits), 5 seeds averaged per fold.
"""
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

SZ = 96                                    # patch edge length (96^3 voxels) / patch 邊長(96³ 體素)
MLO, MHI = 50.0, 200.0                     # default hyperdense band for the mask / 遮罩的預設 HU 範圍
CH = 2                                     # input channels: 2=[CT,mask], 1=CT-only / 輸入通道數(main 會依 --no_mask 改)


class CNN3D(nn.Module):
    """Tiny 3D conv encoder: one 96^3 (multi-channel) patch -> emb-dim vector.
    精簡 3D 卷積編碼器:把一個 96³(多通道)patch 壓成 emb 維向量。
    Four stride-2 blocks 96->48->24->12->6, then global average pool -> 32 channels -> Linear(emb).
    四個 stride-2 區塊把空間 96 逐步降到 6,再全域平均池化成 32 通道,最後線性投影到 emb 維。
    """
    def __init__(self, in_ch=2, emb=32, p=0.2):
        super().__init__()
        # one downsampling block = Conv3d(stride 2) + BatchNorm + ReLU + 3D Dropout
        # 一個降採樣區塊 = 3D 卷積(stride 2)+ 批次正規化 + ReLU + 3D Dropout
        def blk(i, o):
            return nn.Sequential(nn.Conv3d(i, o, 3, 2, 1), nn.BatchNorm3d(o), nn.ReLU(), nn.Dropout3d(p))
        # channels in_ch->8->16->32->32, spatial 96->6, then AdaptiveAvgPool3d(1) -> [N,32,1,1,1]
        # 通道 in_ch->8->16->32->32,空間 96->6,再 AdaptiveAvgPool3d(1) 壓成單點
        self.net = nn.Sequential(blk(in_ch, 8), blk(8, 16), blk(16, 32), blk(32, 32),
                                 nn.AdaptiveAvgPool3d(1))
        self.fc = nn.Linear(32, emb)             # 32 -> emb embedding / 投影到 emb 維嵌入

    def forward(self, x):
        # net(x) -> [N,32,1,1,1]; flatten(1) -> [N,32]; fc -> [N,emb]
        # net(x) 出 [N,32,1,1,1];flatten 成 [N,32];fc 投影成 [N,emb]
        return self.fc(self.net(x).flatten(1))


class HybridMIL(nn.Module):
    """Hybrid multiple-instance model over the 5 lobes of one patient.
    對單一病人的 5 個肺葉做的混合式 MIL 模型。

    Each lobe (instance) -> [optional CNN image embedding] (+) [optional 21 features] -> fuse(H) ;
    the H-dim lobe vectors are then pooled (max/mean/gated-attention) -> patient vector -> head(1 logit).
    每個肺葉 instance -> [可選的 CNN 影像嵌入](+)[可選的 21 特徵] -> fuse 成 H 維;
    再把 5 個 H 維肺葉向量 pool 成病人向量 -> 分類頭輸出 1 個 logit。

    mode 由 use_img / use_feat 決定:hybrid=兩者、image=只影像、feature=只特徵。
    """
    def __init__(self, n_feat=21, emb=32, H=32, dropout=0.3, p_img=0.3, pooling="max",
                 use_img=True, use_feat=True, in_ch=2):
        super().__init__()
        self.use_img, self.use_feat, self.p_img, self.pooling = use_img, use_feat, p_img, pooling
        self.cnn = CNN3D(in_ch, emb) if use_img else None     # image branch / 影像分支
        fin = (emb if use_img else 0) + (n_feat if use_feat else 0)   # fused input dim / 融合後輸入維度
        self.fuse = nn.Sequential(nn.Linear(fin, H), nn.ReLU(), nn.Dropout(dropout))  # per-lobe projection / 逐肺葉投影
        # gated-attention pooling params (Ilse et al. 2018) / 閘控注意力 pooling 的參數
        self.att_V = nn.Linear(H, 8); self.att_U = nn.Linear(H, 8); self.att_w = nn.Linear(8, 1)
        self.head = nn.Linear(H, 1)              # patient-level classifier / 病人層級分類頭
        self.in_ch = in_ch

    def forward(self, img, feat, mask):          # img [B,K,C,96^3]; feat [B,K,n_feat]; mask [B,K] (1=valid lobe)
        B, K = mask.shape                        # B=batch(病人數), K=5 lobes / 肺葉數
        parts = []
        if self.use_img:
            # encode every lobe patch: reshape B*K patches -> CNN -> back to [B,K,emb]
            # 把 B*K 個肺葉 patch 攤平丟進 CNN,再 reshape 回 [B,K,emb]
            e = self.cnn(img.reshape(B * K, self.in_ch, SZ, SZ, SZ)).reshape(B, K, -1)
            if self.training and self.p_img > 0:
                # modality dropout: randomly zero a whole lobe's image embedding so the model
                # 不會過度依賴影像分支(只在訓練時、機率 p_img)
                # modality dropout: randomly null out an entire lobe's image embedding during training
                e = e * (torch.rand(B, K, 1, device=e.device) > self.p_img).float()
            parts.append(e)
        if self.use_feat:
            parts.append(feat)                   # append the 21 hand-crafted features / 接上 21 個手算特徵
        h = self.fuse(torch.cat(parts, dim=2))   # per-lobe fused vector h: [B,K,H] / 逐肺葉融合向量
        # ---- aggregate the K lobes into one patient vector z, ignoring missing lobes (mask==0) ----
        # ---- 把 K 個肺葉聚合成病人向量 z,並忽略缺葉(mask==0) ----
        if self.pooling == "max":
            # worst/strongest-lobe-decides: set missing lobes to -inf then take max over lobes
            # 「最強肺葉決定」:把缺葉填 -inf,再沿肺葉維取最大
            z = h.masked_fill(mask.unsqueeze(-1) == 0, -1e9).max(1).values
        elif self.pooling == "mean":
            # masked average over valid lobes / 對有效肺葉做遮罩平均
            z = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            # gated attention: weight a = softmax( w·(tanh(V h) ⊙ sigmoid(U h)) ), missing lobes -inf
            # 閘控注意力:算出每葉權重 a(缺葉設 -inf 後 softmax),再加權求和
            a = torch.softmax(self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h))
                                          ).squeeze(-1).masked_fill(mask == 0, -1e9), 1)
            z = (a.unsqueeze(-1) * h).sum(1)
        return self.head(z).squeeze(-1)          # logit per patient / 每位病人一個 logit


def augment(img):                                # img [B,K,2,96,96,96]; intensity on CT channel only / 強度增強只動 CT 通道
    """3D augmentation: random spatial flips (both channels together) + intensity jitter on CT only.
    3D 資料增強:隨機空間翻轉(兩通道一起翻)+ 只對 CT 通道做亮度縮放/平移。"""
    for d in (3, 4, 5):                          # spatial axes (x,y,z) / 三個空間軸
        if random.random() < 0.5:
            img = torch.flip(img, dims=[d])
    # per-sample random scale in [0.9,1.1] and shift in [-0.05,0.05], applied to CT channel (index 0) only
    # 每筆樣本隨機縮放 [0.9,1.1]、平移 [-0.05,0.05],只作用在 CT 通道(第 0 通道);mask 是二值不動
    sc = torch.empty(img.size(0), 1, 1, 1, 1, 1, device=img.device).uniform_(0.9, 1.1)
    sh = torch.empty(img.size(0), 1, 1, 1, 1, 1, device=img.device).uniform_(-0.05, 0.05)
    img[:, :, 0:1] = (img[:, :, 0:1] * sc + sh).clamp(0, 1)
    return img


def standardize(Xtr, mtr, Xte):
    """Z-score the features using TRAIN statistics only (computed over valid lobes); NaN->0.
    只用「訓練集的有效肺葉」算 mean/std 來標準化特徵(避免洩漏);NaN 補 0。"""
    v = mtr.astype(bool)                         # valid-lobe mask of the train set / 訓練集的有效肺葉遮罩
    mu, sd = Xtr[v].mean(0), Xtr[v].std(0) + 1e-8
    return np.nan_to_num((Xtr - mu) / sd), np.nan_to_num((Xte - mu) / sd)


def run_fold(IMG_t, FEAT, MASK, y, tr, te, mode, pooling, seed, dev, epochs, bs, band=(MLO, MHI)):
    """Train one outer fold with one random seed; return (test OOF probs, train probs).
    用一個 seed 訓練一個外層折,回傳(測試集 OOF 機率, 訓練集機率)。
    Key args / 主要參數:
      tr, te = this fold's train / test patient indices / 本折的訓練、測試病人索引
      seed   = random seed (results are averaged over several seeds) / 隨機種子(多 seed 取平均)
      epochs = number of training passes / 訓練輪數
      bs     = mini-batch size in PATIENTS (bags) per step; from --bs, default 32 / 每步處理幾個病人,預設 32
      dev    = "cuda" or "cpu" / 運算裝置
      band   = this fold's hyperattenuation HU window (lo, hi) for the mask / 本折遮罩的 HU 範圍 (下界, 上界)"""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    Xtr, Xte = standardize(FEAT[tr], MASK[tr], FEAT[te])      # standardize feats with train stats / 用 train 標準化
    use_img = mode in ("hybrid", "image"); use_feat = mode in ("hybrid", "feature")
    m = HybridMIL(n_feat=FEAT.shape[2], pooling=pooling, use_img=use_img, use_feat=use_feat, in_ch=CH).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2); crit = nn.BCEWithLogitsLoss()
    feat_tr = torch.tensor(Xtr, dtype=torch.float32, device=dev)          # train features / 訓練特徵
    mask_tr = torch.tensor(MASK[tr], dtype=torch.float32, device=dev)     # train lobe mask / 訓練有效肺葉
    y_tr = torch.tensor(y[tr], dtype=torch.float32, device=dev)           # train labels / 訓練標籤
    tr_t = torch.as_tensor(tr, device=dev); idx = np.arange(len(tr))      # global indices + shuffle order / 全域索引

    def two_ch(gidx):
        """Build the CNN input for the given patients: CT-only or [CT, hyperattenuation mask].
        為指定病人組裝 CNN 輸入:純 CT,或 [CT, hyperattenuation 遮罩] 雙通道。"""
        ct = IMG_t[gidx].float()                 # cached normalized patches [n,5,96,96,96] / 快取的正規化 patch
        if CH == 1:
            return ct.unsqueeze(2)               # [n,5,1,96,96,96] CT-only / 純 CT(加一個通道維)
        # invert the patch normalization to recover HU. mil_patches.patch96 caches
        # patch = (HU + 1000) / 1400 over the clipped window HU∈[-1000,400]
        # (1400 = window width 400−(−1000); 1000 = offset), so HU = patch*1400 − 1000.
        # 反正規化還原 HU:mil_patches 存的是 patch=(HU+1000)/1400(HU 先裁到 [-1000,400]),故 HU=patch*1400−1000
        hu = ct * 1400.0 - 1000.0
        msk = ((hu >= band[0]) & (hu < band[1])).float()     # binary hyperattenuation mask / 落在 HU 範圍=1
        return torch.stack([ct, msk], dim=2)     # stack as 2 channels [n,5,2,96,96,96] / 疊成 2 通道

    def img_for(gidx):
        # if no image branch, return a dummy tensor (the model ignores it) / 無影像分支時回傳佔位張量
        return two_ch(gidx) if use_img else torch.zeros(len(gidx), 5, CH, 1, 1, 1, device=dev)

    for ep in range(epochs):                     # training loop / 訓練迴圈
        m.train(); np.random.shuffle(idx)        # shuffle samples each epoch / 每個 epoch 打亂順序
        for s in range(0, len(idx), bs):         # mini-batches / 小批次
            b = idx[s:s + bs]; img = img_for(tr_t[b])
            if use_img:
                img = augment(img)               # augment only when an image branch is used / 有影像才增強
            # standard step: zero grad -> BCE loss -> backprop -> update / 標準一步
            opt.zero_grad(); crit(m(img, feat_tr[b], mask_tr[b]), y_tr[b]).backward(); opt.step()
    m.eval()

    def predict(ix):
        """Predict probabilities for patient indices ix (features re-standardized with TRAIN stats).
        對病人索引 ix 預測機率(特徵一律用 train 統計重新標準化)。"""
        Xs = standardize(FEAT[tr], MASK[tr], FEAT[ix])[1]; ix_t = torch.as_tensor(ix, device=dev)
        ps = []
        with torch.no_grad():                    # no gradients at inference / 推論不需梯度
            for s in range(0, len(ix), bs):
                sl = slice(s, s + bs)
                fe = torch.tensor(Xs[sl], dtype=torch.float32, device=dev)
                mk = torch.tensor(MASK[ix[sl]], dtype=torch.float32, device=dev)
                ps.append(torch.sigmoid(m(img_for(ix_t[sl]), fe, mk)).cpu().numpy())   # logit -> prob / 轉機率
        return np.concatenate(ps)
    return predict(te), predict(tr)              # test OOF + train preds / 測試 OOF + 訓練預測


def main():
    # ---- command-line arguments / 命令列參數 ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True); ap.add_argument("--feat", required=True)   # cached patches / features npz
    ap.add_argument("--splits", required=True); ap.add_argument("--mode", default="hybrid")  # hybrid/image/feature
    ap.add_argument("--pooling", default="max"); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--bands_npz", default=None, help="per-fold train-selected bands (tuned_bands.npz)")  # 每折 HU band
    ap.add_argument("--no_mask", action="store_true", help="CT-only (1 channel), no hyperdense mask")  # 純 CT 對照
    ap.add_argument("--out", default=None)       # if omitted, nothing is written / 不給就不寫任何檔
    args = ap.parse_args()
    global CH; CH = 1 if args.no_mask else 2     # set channel count from --no_mask / 依 --no_mask 設定通道數
    # ---- load cached inputs (read-only) / 載入快取輸入(唯讀)----
    IMG = np.load(args.img)["bags_img"]; dfe = np.load(args.feat, allow_pickle=True)
    FEAT = dfe["bags"].astype(np.float64); MASK = dfe["mask"].astype(np.float64); y = dfe["y"].astype(int)
    folds = json.load(open(args.splits))["folds"]                # outer 10-fold split / 外層 10 折切分
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    IMG_t = torch.from_numpy(IMG).to(dev)        # preload all patches to GPU once (fast) / 一次性預載 patch 到 GPU
    print(f"mask-CNN [{MLO:.0f},{MHI:.0f}] mode {args.mode} pool {args.pooling} seeds {args.seeds} dev {dev}", flush=True)
    # per-fold train-selected HU bands (from tuned_bands.npz); else use the fixed default / 每折 train 選的 band
    sel = [tuple(int(v) for v in x) for x in np.load(args.bands_npz)["sel"]] if args.bands_npz else None
    if sel:
        print(f"per-fold train-selected bands: {sel}", flush=True)
    # ---- nested-CV outer loop: pooled out-of-fold (OOF) predictions / nested 外層迴圈:彙整 OOF 預測 ----
    oof = np.full(len(y), np.nan); tr_aucs = []
    for fi, f in enumerate(folds):
        tr, te = np.array(f["train_idx"]), np.array(f["test_idx"]); pte, ptr = [], []
        band = sel[fi] if sel else (MLO, MHI)    # this fold's HU band / 本折的 HU band
        for s in range(args.seeds):              # average over seeds to reduce noise / 多 seed 平均降噪
            a, b = run_fold(IMG_t, FEAT, MASK, y, tr, te, args.mode, args.pooling, s, dev, args.epochs, args.bs, band=band)
            pte.append(a); ptr.append(b)
        oof[te] = np.mean(pte, axis=0)           # store this fold's averaged test predictions / 存該折平均測試預測
        if len(set(y[tr])) > 1:
            tr_aucs.append(roc_auc_score(y[tr], np.mean(ptr, axis=0)))   # train AUC (overfit check) / 訓練 AUC(看過擬合)
        print(f"  fold {fi} done", flush=True)
    # ---- pooled OOF AUC across all patients / 全病人彙整的 OOF AUC ----
    auc = roc_auc_score(y[np.isfinite(oof)], oof[np.isfinite(oof)])
    print(f"\n{args.mode}/{args.pooling} mask[{MLO:.0f},{MHI:.0f}]: test AUC {auc:.3f} | train {np.mean(tr_aucs):.3f} "
          f"(gap {np.mean(tr_aucs)-auc:+.3f})", flush=True)
    print("ref: features-only 0.71 | hybrid(no mask) 0.715 | image-only(no mask) 0.55", flush=True)
    # ---- save OOF + AUCs only if --out given (used later by final_stats.py) / 只有給 --out 才寫檔 ----
    if args.out:
        json.dump({"mode": args.mode, "pooling": args.pooling, "no_mask": bool(args.no_mask),
                   "test_auc": auc, "train_auc": float(np.mean(tr_aucs)),
                   "oof": oof.tolist(), "y": y.tolist()}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
