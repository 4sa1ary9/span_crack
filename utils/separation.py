"""Local gap suppression: an exploratory proxy, not a topology guarantee."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def gap_background_mask(mask: torch.Tensor, radius: int = 4) -> torch.Tensor:
    """Select GT background between opposing nearby foreground evidence.

    Search four axes, excluding the center, at distances 1..radius. Pixels on
    opposite sides need not belong to distinct GT components: a false shortcut
    across a curved crack should also be penalized. Ground truth alone selects
    the support; predictions cannot turn off their own penalty.
    """
    if radius < 1:
        raise ValueError("gap radius must be positive")
    foreground = (mask.detach() > 0.5)
    h, w = foreground.shape[-2:]
    padded = F.pad(foreground, (radius, radius, radius, radius), value=False)
    opposing = torch.zeros_like(foreground)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        positive = torch.zeros_like(foreground)
        negative = torch.zeros_like(foreground)
        for distance in range(1, radius + 1):
            y, x = radius + dy * distance, radius + dx * distance
            positive |= padded[..., y:y + h, x:x + w]
            y, x = radius - dy * distance, radius - dx * distance
            negative |= padded[..., y:y + h, x:x + w]
        opposing |= positive & negative
    return opposing & ~foreground


def background_suppression_loss(logits, mask, *, mode="gap", radius=4):
    if mode == "gap":
        support = gap_background_mask(mask, radius)
    elif mode == "all_background":
        support = mask.detach() <= 0.5
    else:
        raise ValueError(f"Unknown separation mode: {mode}")
    # Explicit denominator retains a differentiable zero for empty support.
    weights = support.to(logits.dtype)
    return (F.softplus(logits) * weights).sum() / weights.sum().clamp_min(1)
