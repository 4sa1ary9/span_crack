"""Binary adaptations of published connectivity losses; see sources/baselines_v4.

clDice uses the author's cross erosion, ten iterations, batch reduction and
smooth=1. Skeleton Recall uses a twice cross-dilated Zhang skeleton intersected
with foreground, and per-image negative recall (smooth=1e-5).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize, dilation


def author_soft_skeleton(image: torch.Tensor, iterations: int = 10):
    def erode(x):
        vertical = -F.max_pool2d(-x, (3, 1), 1, (1, 0))
        horizontal = -F.max_pool2d(-x, (1, 3), 1, (0, 1))
        return torch.minimum(vertical, horizontal)

    def opening(x):
        return F.max_pool2d(erode(x), 3, 1, 1)

    skeleton = F.relu(image - opening(image))
    for _ in range(iterations):
        image = erode(image)
        delta = F.relu(image - opening(image))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def author_cldice_loss(logits, masks):
    probs = logits.sigmoid()
    ps, gs = author_soft_skeleton(probs), author_soft_skeleton(masks)
    precision = ((ps * masks).sum() + 1.) / (ps.sum() + 1.)
    recall = ((gs * probs).sum() + 1.) / (gs.sum() + 1.)
    return 1. - 2. * precision * recall / (precision + recall)


def skeleton_recall_target(mask):
    """Compute after augmentation on CPU workers, matching the author transform."""
    foreground = np.asarray(mask) > .5
    skeleton = skeletonize(foreground, method="zhang")
    # skimage's default 2D footprint is the connectivity-one cross.
    tube = dilation(dilation(skeleton)) & foreground
    return np.asarray(tube, dtype=np.float32)


def skeleton_recall_loss(logits, tube):
    dims = (2, 3)
    overlap = (logits.sigmoid() * tube).sum(dims)
    mass = tube.sum(dims)
    return -((overlap + 1e-5) / (mass + 1e-5).clamp_min(1e-8)).mean()
