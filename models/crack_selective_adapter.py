from __future__ import annotations

import torch
import torch.nn as nn

from models.texture_suppression import _make_gn


class CrackSelectiveAdapter(nn.Module):
    """Lightweight residual adapter for crack-selective representation adaptation.

    Applied after each SegFormer encoder stage to reshape backbone features
    toward crack-specific evidence. Uses only local operations:

    - Depthwise 3x3 conv for local line/edge patterns;
    - Pointwise projection for channel mixing;
    - Squeeze-style channel gate for crack-relevant channel selection;
    - Spatial gate for suppressing broad texture responses.

    Tensor shape and channel count are preserved. The residual scale starts
    at *init_scale* (default 0.0) to keep pretrained behaviour intact early
    in training.

    Args:
        channels: Input feature channels.
        reduction: Channel reduction ratio for bottleneck projections.
        init_scale: Initial value for the learnable residual scale.
    """

    def __init__(
        self, channels: int, reduction: int = 4, init_scale: float = 0.0,
    ):
        super().__init__()
        hidden = max(channels // reduction, 8)

        # Local line/edge evidence
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False,
        )
        self.dw_norm = _make_gn(channels)
        self.dw_act = nn.GELU()

        # Pointwise projection for channel mixing
        self.pw_proj = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

        # Squeeze-style channel gate
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

        # Spatial gate for suppressing broad texture responses
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), init_scale))

        self._init_weights()

    def _init_weights(self):
        for conv in [self.pw_proj[-1], self.channel_gate[-1]]:
            nn.init.normal_(conv.weight, std=0.01)
            nn.init.constant_(conv.bias, 0.0)
        nn.init.normal_(self.dw_conv.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local edge evidence
        feat = self.dw_conv(x)
        feat = self.dw_norm(feat)
        feat = self.dw_act(feat)

        # Channel mixing
        feat = self.pw_proj(feat)

        # Channel gate
        ch_gate = torch.sigmoid(self.channel_gate(self.channel_pool(feat)))
        feat = feat * ch_gate

        # Spatial gate
        sp_gate = self.spatial_gate(feat)
        feat = feat * sp_gate

        return x + self.gamma * feat
