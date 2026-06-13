"""SwinUNETR-IE: dual-phase (inhale + exhale) classifier.
SwinUNETR-IE：雙相位（吸氣 + 吐氣）分類器。

Two VoComni-pretrained SwinUNETR encoder branches process T00 and T50 independently;
their multi-scale features are fused (1x1 conv) and decoded by a shared decoder + head.
Outputs logits. Ported/cleaned from the original ``Swin_2E1D_Cls``.
兩個 VoComni 預訓練的編碼分支分別處理 T00/T50，多尺度特徵以 1x1 conv 融合，再經共用解碼器 + head。輸出 logits。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep

from .models import load_pretrained


class SwinEncoder(nn.Module):
    """One VoComni SwinUNETR encoder branch -> multi-scale features (no decoder).
    一個 VoComni SwinUNETR 編碼分支 -> 多尺度特徵（不含解碼器）。"""

    INPUT_SIZE = (256, 256, 96)

    def __init__(self, in_channels=1, feature_size=96, use_checkpoint=False):
        super().__init__()
        patch_size = ensure_tuple_rep(2, 3)
        window_size = ensure_tuple_rep(7, 3)
        sd, nm = 3, "instance"
        self.swinViT = SwinViT(in_chans=in_channels, embed_dim=feature_size,
                               window_size=window_size, patch_size=patch_size,
                               depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24], mlp_ratio=4.0,
                               qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                               drop_path_rate=0.0, norm_layer=nn.LayerNorm,
                               use_checkpoint=use_checkpoint, spatial_dims=sd)

        def basic(ic, oc):
            return UnetrBasicBlock(spatial_dims=sd, in_channels=ic, out_channels=oc,
                                   kernel_size=3, stride=1, norm_name=nm, res_block=True)

        self.encoder1 = basic(in_channels, feature_size)
        self.encoder2 = basic(feature_size, feature_size)
        self.encoder3 = basic(2 * feature_size, 2 * feature_size)
        self.encoder4 = basic(4 * feature_size, 4 * feature_size)
        self.encoder10 = basic(16 * feature_size, 16 * feature_size)

    def forward(self, x):
        if tuple(x.shape[2:]) != self.INPUT_SIZE:
            x = F.interpolate(x, size=self.INPUT_SIZE, mode="trilinear")
        hs = self.swinViT(x)
        enc0 = self.encoder1(x)
        enc1 = self.encoder2(hs[0])
        enc2 = self.encoder3(hs[1])
        enc3 = self.encoder4(hs[2])
        dec4 = self.encoder10(hs[4])
        h3 = hs[3]
        return enc0, enc1, enc2, enc3, dec4, h3


class SwinClassifierIE(nn.Module):
    """Dual-phase classifier: two encoder branches fused into a shared decoder + head.
    雙相位分類器：兩個編碼分支融合 -> 共用解碼器 + head。輸出 logits。"""

    def __init__(self, in_channels=1, n_class=1, feature_size=96, dropout=0.0,
                 use_checkpoint=False):
        super().__init__()
        fs, sd, nm = feature_size, 3, "instance"
        self.enc_in = SwinEncoder(in_channels, fs, use_checkpoint)  # inhale / 吸氣分支
        self.enc_ex = SwinEncoder(in_channels, fs, use_checkpoint)  # exhale / 吐氣分支

        # fuse the two branches at each scale: concat (2c) -> 1x1 conv -> c
        # 各尺度融合：兩分支 concat(2c) -> 1x1 conv -> c
        def fuse(c):
            return nn.Conv3d(2 * c, c, kernel_size=1)

        self.fuse_enc0 = fuse(fs)
        self.fuse_enc1 = fuse(fs)
        self.fuse_enc2 = fuse(2 * fs)
        self.fuse_enc3 = fuse(4 * fs)
        self.fuse_dec4 = fuse(16 * fs)
        self.fuse_h3 = fuse(8 * fs)

        def up(ic, oc):
            return UnetrUpBlock(spatial_dims=sd, in_channels=ic, out_channels=oc,
                                kernel_size=3, upsample_kernel_size=2, norm_name=nm,
                                res_block=True)

        self.decoder5 = up(16 * fs, 8 * fs)
        self.decoder4 = up(8 * fs, 4 * fs)
        self.decoder3 = up(4 * fs, 2 * fs)
        self.decoder2 = up(2 * fs, fs)
        self.decoder1 = up(fs, fs)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(fs, n_class)

    def forward(self, x_in, x_ex):
        b = x_in.size(0)
        e0i, e1i, e2i, e3i, d4i, h3i = self.enc_in(x_in)
        e0e, e1e, e2e, e3e, d4e, h3e = self.enc_ex(x_ex)
        c = lambda a, bb: torch.cat([a, bb], dim=1)
        enc0 = self.fuse_enc0(c(e0i, e0e))
        enc1 = self.fuse_enc1(c(e1i, e1e))
        enc2 = self.fuse_enc2(c(e2i, e2e))
        enc3 = self.fuse_enc3(c(e3i, e3e))
        dec4 = self.fuse_dec4(c(d4i, d4e))
        h3 = self.fuse_h3(c(h3i, h3e))

        dec3 = self.decoder5(dec4, h3)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(b, -1)
        out = self.dropout(out)
        return self.head(out)  # logits


def load_pretrained_ie(model, pretrained_path, verbose=True):
    """Load VoComni weights into both encoder branches AND the shared decoders.
    把 VoComni 權重載入兩個編碼分支，以及共用的解碼器。"""
    load_pretrained(model, pretrained_path, verbose=False)          # top-level: decoders / 解碼器
    for branch in (model.enc_in, model.enc_ex):
        load_pretrained(branch, pretrained_path, verbose=False)     # each branch: encoders / 各分支編碼器
    if verbose:
        print("[load_pretrained_ie] loaded VoComni into both encoder branches + decoders")
    return model
