from __future__ import annotations

import torch
import torch.nn as nn


def _make_gn(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class TextureSuppressionGate(nn.Module):
    """Lightweight spatial gate that learns to suppress pavement texture.

    Placed between SegFormer Stage 1 and Stage 2. A local depthwise branch
    captures fine spatial structure; a global pooling branch injects
    image-level context. The sigmoid gate starts near identity so the
    pretrained backbone is undisturbed at initialization.

    Args:
        channels: Stage 1 output channels (32 for SegFormer-B0).
        reduction: channel reduction ratio for the global MLP bottleneck.
    """

    def __init__(self, channels: int = 32, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)

        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            _make_gn(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        # Gate starts near 1.0 (identity) so the pretrained backbone is
        # undisturbed early in training.
        for conv in [self.local[-1], self.global_mlp[-1]]:
            nn.init.normal_(conv.weight, std=0.01)
            nn.init.constant_(conv.bias, 2.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (gated_features, gate_map).

        gate_map is exposed for visualisation and debugging only.
        """
        local_feat = self.local(x)
        global_feat = self.global_mlp(self.global_pool(x))
        gate = torch.sigmoid(local_feat + global_feat)
        return x * gate, gate
