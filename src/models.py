"""
Model architectures for 3D brain-age regression.

Two families are used in this project, both built from small configurable
building blocks so architecture variants can be defined as plain dicts:

  ModularCNN3D              - a plain 3D CNN regressor (used for models F-J).
  ModularHybridCNNTransformer - a 3D CNN feature extractor feeding a small
                                 transformer encoder (used for models A-E).
                                 The CNN's output grid controls how many
                                 spatial tokens the transformer attends over,
                                 which is the main variable this project
                                 studies (`token_grid`).
"""

import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared transformer building blocks
# ---------------------------------------------------------------------------

class SimpleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.dropout(self.proj(x))


class SimpleMLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimpleAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SimpleMLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _build_regression_head(in_features, head_config, default_dropout):
    """Normalize a head config (ints or dicts) into Linear/GELU/Dropout layers."""
    if head_config is None:
        head_config = [{"units": in_features // 2, "dropout": default_dropout}]
    normalized, layers = [], []
    for hc in head_config:
        if isinstance(hc, int):
            hc = {"units": hc, "dropout": default_dropout}
        normalized.append(hc)
        layers += [nn.Linear(in_features, hc["units"]), nn.GELU(),
                   nn.Dropout(hc.get("dropout", default_dropout))]
        in_features = hc["units"]
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers), normalized


def _init_transformer_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv3d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# ---------------------------------------------------------------------------
# Plain 3D CNN
# ---------------------------------------------------------------------------

class ConvBlock3D(nn.Module):
    """Conv3d -> BatchNorm3d -> ReLU -> optional MaxPool3d -> optional Dropout3d."""

    def __init__(self, in_channels, out_channels, dropout=0.0, kernel_size=3, pool_size=2):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool_size is not None:
            layers.append(nn.MaxPool3d(pool_size))
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ModularCNN3D(nn.Module):
    """A 3D CNN regressor built from a list of conv-block configs."""

    def __init__(self, layer_config: List[Dict], fc_config=None, input_channels=1,
                 use_adaptive_pool=True, adaptive_pool_size=(1, 1, 1), default_fc_dropout=0.2):
        super().__init__()
        self.layer_config = layer_config
        self.use_adaptive_pool = use_adaptive_pool
        self.adaptive_pool_size = adaptive_pool_size

        conv_blocks, in_ch = [], input_channels
        for i, cfg in enumerate(layer_config):
            pool_size = cfg.get("pool_size", 2)
            if use_adaptive_pool and i == len(layer_config) - 1:
                pool_size = None  # adaptive pooling handles the final reduction
            conv_blocks.append(ConvBlock3D(
                in_ch, cfg["filters"], dropout=cfg.get("dropout", 0.0),
                kernel_size=cfg.get("kernel_size", 3), pool_size=pool_size,
            ))
            in_ch = cfg["filters"]
        self.features = nn.Sequential(*conv_blocks)
        self.adaptive_pool = nn.AdaptiveAvgPool3d(adaptive_pool_size) if use_adaptive_pool else None

        head, self.fc_config_normalized = _build_regression_head(
            layer_config[-1]["filters"], fc_config, default_fc_dropout
        )
        self.regressor = nn.Sequential(nn.Flatten(), *head)

    def forward(self, x):
        x = self.features(x)
        if self.adaptive_pool is not None:
            x = self.adaptive_pool(x)
        return self.regressor(x)

    def get_model_info(self):
        return {
            "model_type": "ModularCNN3D",
            "num_conv_blocks": len(self.layer_config),
            "layer_config": self.layer_config,
            "fc_config": self.fc_config_normalized,
            "n_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


# ---------------------------------------------------------------------------
# Hybrid CNN + Transformer, with explicit control over token count
# ---------------------------------------------------------------------------

class CNNBlock3D(nn.Module):
    """Double-conv block used by the hybrid CNN feature extractor."""

    def __init__(self, in_channels, out_channels, dropout=0.1, use_pool=True, pool_size=2):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels), nn.ReLU(inplace=True),
        ]
        if use_pool:
            layers.append(nn.MaxPool3d(pool_size))
        if dropout > 0:
            layers.append(nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ModularHybridCNNTransformer(nn.Module):
    """
    CNN feature extractor -> transformer encoder -> regression head.

    The CNN's native output grid (set by how many blocks pool) determines how
    many spatial tokens reach the transformer. `token_grid` can pool that grid
    down further; it cannot be set larger than the native grid, since that
    would require upsampling tokens that carry no additional information.
    """

    def __init__(self, cnn_config: List[Dict], transformer_config: List[Dict],
                 head_config=None, input_shape: Tuple[int, int, int] = (57, 57, 71),
                 in_channels=1, embed_dim=256, use_cls_token=True,
                 default_head_dropout=0.1, token_grid: Optional[Tuple[int, int, int]] = None,
                 max_tokens=1024):
        super().__init__()
        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        self.cnn_config = cnn_config
        self.transformer_config = transformer_config

        cnn_blocks, in_ch = [], in_channels
        for cfg in cnn_config:
            cnn_blocks.append(CNNBlock3D(
                in_ch, cfg["filters"], dropout=cfg.get("dropout", 0.1),
                use_pool=cfg.get("use_pool", True), pool_size=cfg.get("pool_size", 2),
            ))
            in_ch = cfg["filters"]
        self.cnn_extractor = nn.Sequential(*cnn_blocks)
        self.cnn_out_channels = cnn_config[-1]["filters"]

        self.native_grid = self._cnn_output_shape(input_shape, cnn_config)
        self.native_tokens = self.native_grid[0] * self.native_grid[1] * self.native_grid[2]

        if token_grid is None:
            self.token_grid = self.native_grid
            self.token_pool = nn.Identity()
        else:
            for axis, (want, have) in enumerate(zip(token_grid, self.native_grid)):
                if want > have:
                    raise ValueError(
                        f"token_grid {tuple(token_grid)} exceeds the CNN's native grid "
                        f"{tuple(self.native_grid)} on axis {axis} ({want} > {have})."
                    )
            self.token_grid = tuple(token_grid)
            self.token_pool = nn.AdaptiveAvgPool3d(self.token_grid)

        self.num_tokens = self.token_grid[0] * self.token_grid[1] * self.token_grid[2]
        if self.num_tokens > max_tokens:
            raise ValueError(f"{self.num_tokens} tokens exceeds max_tokens={max_tokens}; "
                              f"attention cost is O(N^2).")
        if self.num_tokens < 8:
            warnings.warn(f"Only {self.num_tokens} tokens ({self.token_grid}); self-attention "
                           f"over this few positions is close to a weighted average.")

        self.feature_projection = nn.Sequential(
            nn.Conv3d(self.cnn_out_channels, embed_dim, kernel_size=1),
            nn.BatchNorm3d(embed_dim), nn.ReLU(inplace=True),
        )

        num_positions = self.num_tokens + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.randn(1, num_positions, embed_dim) * 0.02)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(embed_dim, cfg["num_heads"], cfg.get("mlp_ratio", 4),
                                    cfg.get("dropout", 0.1))
            for cfg in transformer_config
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head, self.head_config_normalized = _build_regression_head(
            embed_dim, head_config, default_head_dropout
        )

        self.apply(_init_transformer_weights)

    @staticmethod
    def _cnn_output_shape(input_shape, cnn_config):
        D, H, W = input_shape
        for cfg in cnn_config:
            if cfg.get("use_pool", True):
                p = cfg.get("pool_size", 2)
                D, H, W = D // p, H // p, W // p
        return (D, H, W)

    def forward(self, x):
        B = x.shape[0]
        f = self.cnn_extractor(x)
        f = self.token_pool(f)
        f = self.feature_projection(f)
        f = f.reshape(B, self.embed_dim, -1).transpose(1, 2)

        if self.use_cls_token:
            f = torch.cat([self.cls_token.expand(B, -1, -1), f], dim=1)
        f = f + self.pos_embed

        for block in self.blocks:
            f = block(f)
        f = self.norm(f)
        f = f[:, 0] if self.use_cls_token else f.mean(dim=1)
        return self.head(f)

    def get_model_info(self):
        return {
            "model_type": "ModularHybridCNNTransformer",
            "native_grid": self.native_grid,
            "native_tokens": self.native_tokens,
            "token_grid": self.token_grid,
            "num_tokens": self.num_tokens,
            "embed_dim": self.embed_dim,
            "transformer_depth": len(self.blocks),
            "n_params": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_modular_cnn(layer_config, fc_config=None, default_fc_dropout=0.2,
                        use_adaptive_pool=True, adaptive_pool_size=(1, 1, 1)):
    return ModularCNN3D(layer_config, fc_config, default_fc_dropout=default_fc_dropout,
                         use_adaptive_pool=use_adaptive_pool, adaptive_pool_size=adaptive_pool_size)


def create_modular_hybrid(cnn_config, transformer_config, head_config=None,
                           input_shape=(57, 57, 71), embed_dim=256, use_cls_token=True,
                           default_head_dropout=0.1, token_grid=None, max_tokens=1024):
    return ModularHybridCNNTransformer(
        cnn_config=cnn_config, transformer_config=transformer_config, head_config=head_config,
        input_shape=input_shape, in_channels=1, embed_dim=embed_dim, use_cls_token=use_cls_token,
        default_head_dropout=default_head_dropout, token_grid=token_grid, max_tokens=max_tokens,
    )
