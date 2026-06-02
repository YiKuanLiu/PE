"""SwinUNETR-I : VoComni-pretrained SwinUNETR backbone + MLP classification head.

Ported from the original ``model.py`` and cleaned up:
  * the network outputs **logits** (train with ``BCEWithLogitsLoss``; apply
    ``sigmoid`` for probabilities) -- numerically safer than the old
    sigmoid + ``BCELoss`` combination;
  * a configurable dropout before the head exposes the tuned hyper-parameter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep


class SwinClassifierI(nn.Module):
    """Single-phase classifier: full SwinUNETR encoder-decoder + avg-pool + linear head."""

    INPUT_SIZE = (256, 256, 96)

    def __init__(self, in_channels=1, n_class=1, feature_size=96, dropout=0.0,
                 use_checkpoint=False):
        super().__init__()
        patch_size = ensure_tuple_rep(2, 3)
        window_size = ensure_tuple_rep(7, 3)
        spatial_dims = 3
        norm_name = "instance"

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
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
        )

        def basic(in_c, out_c):
            return UnetrBasicBlock(spatial_dims=spatial_dims, in_channels=in_c,
                                   out_channels=out_c, kernel_size=3, stride=1,
                                   norm_name=norm_name, res_block=True)

        def up(in_c, out_c):
            return UnetrUpBlock(spatial_dims=spatial_dims, in_channels=in_c,
                                out_channels=out_c, kernel_size=3,
                                upsample_kernel_size=2, norm_name=norm_name,
                                res_block=True)

        self.encoder1 = basic(in_channels, feature_size)
        self.encoder2 = basic(feature_size, feature_size)
        self.encoder3 = basic(2 * feature_size, 2 * feature_size)
        self.encoder4 = basic(4 * feature_size, 4 * feature_size)
        self.encoder10 = basic(16 * feature_size, 16 * feature_size)

        self.decoder5 = up(16 * feature_size, 8 * feature_size)
        self.decoder4 = up(8 * feature_size, 4 * feature_size)
        self.decoder3 = up(4 * feature_size, 2 * feature_size)
        self.decoder2 = up(2 * feature_size, feature_size)
        self.decoder1 = up(feature_size, feature_size)

        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(feature_size, n_class)

    def forward(self, x_in):
        b = x_in.size(0)
        # Resample to the working resolution (no-op if volumes are pre-cached at it).
        if tuple(x_in.shape[2:]) != self.INPUT_SIZE:
            x_in = F.interpolate(x_in, size=self.INPUT_SIZE, mode="trilinear")
        hs = self.swinViT(x_in)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hs[0])
        enc2 = self.encoder3(hs[1])
        enc3 = self.encoder4(hs[2])
        dec4 = self.encoder10(hs[4])

        dec3 = self.decoder5(dec4, hs[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        out = F.adaptive_avg_pool3d(out, (1, 1, 1)).view(b, -1)
        out = self.dropout(out)
        return self.head(out)  # logits


def load_pretrained(model, pretrained_path, verbose=True):
    """Load matching weights from a VoComni checkpoint (head stays random-init)."""
    ckpt = torch.load(pretrained_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    cur = model.state_dict()
    new_state = {
        k: state[k] if (k in state and state[k].size() == cur[k].size()) else cur[k]
        for k in cur.keys()
    }
    model.load_state_dict(new_state, strict=True)
    if verbose:
        loaded = sum(1 for k in cur if k in state and state[k].size() == cur[k].size())
        print(f"[load_pretrained] loaded {loaded}/{len(cur)} tensors from {pretrained_path}")
    return model
