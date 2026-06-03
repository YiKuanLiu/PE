"""SwinUNETR-I: VoComni-pretrained SwinUNETR backbone + MLP classification head.
SwinUNETR-I：VoComni 預訓練的 SwinUNETR 主幹 + MLP 分類頭。

Ported from the original ``model.py`` and cleaned up:
  * the network outputs **logits** (train with ``BCEWithLogitsLoss``; apply ``sigmoid``
    for probabilities) -- numerically safer than the old sigmoid + ``BCELoss``;
  * a configurable dropout before the head exposes the tuned hyper-parameter.
由原始 ``model.py`` 移植並整理：
  * 網路輸出 **logits**（用 ``BCEWithLogitsLoss`` 訓練、用 ``sigmoid`` 取機率）——
    數值上比舊的 sigmoid + ``BCELoss`` 組合更安全；
  * 在分類頭前加一個可設定的 dropout，作為可調的超參數。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep


class SwinClassifierI(nn.Module):
    """Single-phase classifier: full SwinUNETR encoder-decoder + avg-pool + linear head.
    單相位分類器：完整 SwinUNETR 編碼-解碼 + 全域平均池化 + 線性頭。"""

    INPUT_SIZE = (256, 256, 96)  # working resolution / 模型實際運作的解析度

    def __init__(self, in_channels=1, n_class=1, feature_size=96, dropout=0.0,
                 use_checkpoint=False):
        super().__init__()
        patch_size = ensure_tuple_rep(2, 3)
        window_size = ensure_tuple_rep(7, 3)
        spatial_dims = 3
        norm_name = "instance"

        # Swin Transformer encoder (the main body of the pretrained backbone)
        # Swin Transformer 編碼器（預訓練主幹的主體）
        self.swinViT = SwinViT(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,  # gradient checkpointing: memory<->compute / 梯度檢查點：省記憶體換算力
            spatial_dims=spatial_dims,
        )

        # Two small factories for the UNETR encoder/decoder blocks.
        # 兩個小工廠函式，簡化 UNETR 編碼/解碼區塊的建立。
        def basic(in_c, out_c):
            return UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=in_c,
                                   out_channels=out_c, kernel_size=3, stride=1,
                                   norm_name=norm_name, res_block=True)

        def up(in_c, out_c):
            return UnetrUpBlock(spatial_dims=spatial_dims, in_channels=in_c,
                                out_channels=out_c, kernel_size=3,
                                upsample_kernel_size=2, norm_name=norm_name,
                                res_block=True)

        # Multi-scale CNN encoders (encoder10 is the bottleneck, ~127M params)
        # 多尺度 CNN 編碼器（encoder10 是瓶頸層，參數最多 ~127M）
        self.encoder1 = basic(in_channels, feature_size)
        self.encoder2 = basic(feature_size, feature_size)
        self.encoder3 = basic(2 * feature_size, 2 * feature_size)
        self.encoder4 = basic(4 * feature_size, 4 * feature_size)
        self.encoder10 = basic(16 * feature_size, 16 * feature_size)

        # Decoders (decoder5 is the first, second-largest ~58M params)
        # 解碼器（decoder5 是首個、參數第二多 ~58M）
        self.decoder5 = up(16 * feature_size, 8 * feature_size)
        self.decoder4 = up(8 * feature_size, 4 * feature_size)
        self.decoder3 = up(4 * feature_size, 2 * feature_size)
        self.decoder2 = up(2 * feature_size, feature_size)
        self.decoder1 = up(feature_size, feature_size)

        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(feature_size, n_class)  # new classification head (random init) / 新增的分類頭（隨機初始化）

    def forward(self, x_in):
        b = x_in.size(0)
        # Resample to the working resolution (no-op if pre-cached at it).
        # 重採樣到工作解析度（若已預快取成此尺寸，則為 no-op）。
        if tuple(x_in.shape[2:]) != self.INPUT_SIZE:
            x_in = F.interpolate(x_in, size=self.INPUT_SIZE, mode="trilinear")
        hs = self.swinViT(x_in)  # per-stage hidden states / 各層級的隱藏狀態

        # Encoding path: x and each Swin stage output go through a CNN encoder.
        # 編碼路徑：x 與 Swin 各層輸出分別過 CNN 編碼器。
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hs[0])
        enc2 = self.encoder3(hs[1])
        enc3 = self.encoder4(hs[2])
        dec4 = self.encoder10(hs[4])

        # Decoding path: upsample and fuse with the matching encoder feature (U-Net skips).
        # 解碼路徑：逐層上採樣並與對應編碼特徵融合（U-Net 跳接）。
        dec3 = self.decoder5(dec4, hs[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        # Global average pool -> dropout -> linear head, returning logits.
        # 全域平均池化 → dropout → 線性頭，輸出 logits。
        out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(b, -1)
        out = self.dropout(out)
        return self.head(out)  # logits (no sigmoid) / logits（不做 sigmoid）


def apply_freeze(model, strategy):
    """Freeze parameter groups for transfer learning. Returns (n_trainable, n_total).
    為遷移學習凍結參數群組，回傳 (可訓練參數量, 總參數量)。

    (The heavy pretrained blocks are encoder10 ~127M and decoder5 ~58M, not the
    transformer ~32M.)
    （重塊是 encoder10 ~127M 與 decoder5 ~58M，不是 transformer ~32M）：
      * ``all``           : full fine-tune / 全微調（不凍結）。
      * ``freeze_swinvit``: freeze the Swin transformer encoder only / 只凍結 Swin transformer 編碼器。
      * ``freeze_heavy``  : freeze swinViT + encoder10 + decoder5 (~217M); train the
                            lighter UNETR blocks + head (~31M) / 凍結重塊、只訓練輕量層 + head。
      * ``head_only``     : linear probe -- train only the head / 線性探測，只訓練分類頭。
    """
    # First mark everything trainable, then turn off groups per the strategy.
    # 先全部設為可訓練，再依策略關閉部分。
    for p in model.parameters():
        p.requires_grad = True

    if strategy == "all":
        frozen = []
    elif strategy == "freeze_swinvit":
        frozen = [model.swinViT]
    elif strategy == "freeze_heavy":
        frozen = [model.swinViT, model.encoder10, model.decoder5]
    elif strategy == "head_only":
        # freeze everything except head and dropout / 除了 head 與 dropout 之外全部凍結
        frozen = [m for n, m in model.named_children() if n not in ("head", "dropout")]
    else:
        raise ValueError(f"unknown freeze strategy: {strategy}")

    for mod in frozen:
        for p in mod.parameters():
            p.requires_grad = False

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_train, n_total


def load_pretrained(model, pretrained_path, verbose=True):
    """Load shape-matching weights from a VoComni checkpoint (head stays random-init).
    從 VoComni checkpoint 載入「形狀相符」的權重（head 維持隨機初始化）。"""
    ckpt = torch.load(pretrained_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cur = model.state_dict()
    # Keep only keys that exist with matching shape; otherwise keep the model's value.
    # 只挑「存在且形狀相同」的鍵載入，其餘維持模型現值。
    new_state = {
        k: state[k] if (k in state and state[k].size() == cur[k].size()) else cur[k]
        for k in cur.keys()
    }
    model.load_state_dict(new_state, strict=True)
    if verbose:
        loaded = sum(1 for k in cur if k in state and state[k].size() == cur[k].size())
        print(f"[load_pretrained] loaded {loaded}/{len(cur)} tensors from {pretrained_path}")
    return model
