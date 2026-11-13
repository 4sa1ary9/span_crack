from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.topology import binary_skeletonize


# ---------------------------------------------------------------------------
# Width label generation (CPU, called once per dataset before training)
# ---------------------------------------------------------------------------

# Default bin edges on the 512 x 512 mask grid. Keep in sync with the
# ``width_bin_thin_max`` / ``width_bin_medium_max`` loss config keys so that
# the sensitivity runs (H5/H6) can shift the binning without touching code.
DEFAULT_WIDTH_THIN_MAX = 2.0
DEFAULT_WIDTH_MEDIUM_MAX = 5.0
BACKGROUND_LABEL = 255


def _nearest_skeleton_value(
    skeleton: np.ndarray, thickness: np.ndarray
) -> np.ndarray:
    """Propagate skeleton thickness to every crack pixel via nearest skeleton.

    For each pixel, the returned value is the thickness measured at the
    closest skeleton pixel. This is the "width estimate convention" used by
    the paper: a connected crack region gets the local centreline thickness
    rather than the per-pixel distance-to-edge value, so the edge and the
    centre of a constant-width straight crack receive the *same* label instead
    of being split into thin/wide.

    Uses ``cv2.distanceTransformWithLabels`` with a non-zero mask so only
    skeleton pixels are label seeds. Pixels equidistant to several skeleton
    points get an implementation-defined pick; that ambiguity is confined to a
    thin zero-measure set and is documented, not hidden.
    """
    # distanceTransformWithLabels expects background (0) pixels as seeds.
    seeds = (skeleton == 0).astype(np.uint8)
    _dist, labels = cv2.distanceTransformWithLabels(
        seeds, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL
    )
    # labels 1..K map to the K-th zero pixel of ``seeds`` in raster order,
    # which are exactly the skeleton pixels (plus the implicit border row).
    skel_coords = np.argwhere(skeleton > 0)
    # Build index from label id -> thickness. Label ids are assigned to every
    # zero pixel of ``seeds`` in row-major order; skeleton mask is row-major so
    # the ordering matches when we walk the zero pixels in the same order.
    zero_positions = np.argwhere(seeds == 0)
    id_to_thickness = np.zeros(labels.max() + 1, dtype=np.float32)
    if len(zero_positions) == labels.max():
        id_to_thickness[1:] = thickness[
            zero_positions[:, 0], zero_positions[:, 1]
        ]
    else:
        # Fallback: leave unmatched ids at 0 (treated as thin below).
        n = min(len(zero_positions), labels.max())
        id_to_thickness[1:n + 1] = thickness[
            zero_positions[:n, 0], zero_positions[:n, 1]
        ]
    del skel_coords
    return id_to_thickness[labels]


def generate_width_labels(
    mask: np.ndarray,
    thin_max: float = DEFAULT_WIDTH_THIN_MAX,
    medium_max: float = DEFAULT_WIDTH_MEDIUM_MAX,
) -> np.ndarray:
    """Generate per-pixel crack-width class labels from a binary GT mask.

    Width convention (documented digital-image convention, not a physical
    measurement): a crack's local thickness is ``2 * distance_to_background``
    *measured at the crack centreline (skeleton)*. Every crack pixel then
    inherits the thickness of its nearest skeleton pixel. This fixes the
    original bug where an edge pixel of a 2-px-wide straight crack and its
    centre pixel were systematically labelled thin and wide respectively.

    Returns an int32 array with the same spatial shape as *mask*:
        - 0: thin   (estimated width ``<= thin_max``)
        - 1: medium (``thin_max < width <= medium_max``)
        - 2: wide   (``width > medium_max``)
        - 255: background / ignore

    Args:
        mask: binary crack mask (H, W), crack=1, background=0.
        thin_max: inclusive upper edge (grid px) of the thin class.
        medium_max: inclusive upper edge (grid px) of the medium class.
    """
    binary = (mask > 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return np.full(binary.shape, BACKGROUND_LABEL, dtype=np.int32)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    skeleton = binary_skeletonize(binary)
    if skeleton.sum() == 0:
        # Degenerate case (e.g. a single isolated pixel): treat the whole
        # foreground as its own skeleton so every crack pixel is still labelled.
        skeleton = binary

    thickness_at_skeleton = (dist * 2.0).astype(np.float32)
    width = _nearest_skeleton_value(skeleton, thickness_at_skeleton)

    labels = np.full(binary.shape, BACKGROUND_LABEL, dtype=np.int32)
    crack = binary == 1
    labels[crack & (width <= float(thin_max))] = 0
    labels[crack & (width > float(thin_max)) & (width <= float(medium_max))] = 1
    labels[crack & (width > float(medium_max))] = 2
    return labels


# ---------------------------------------------------------------------------
# Path Continuity Loss (differentiable, GPU)
# ---------------------------------------------------------------------------

def _sample_crack_pairs(
    crack_coords: torch.Tensor,
    num_paths: int,
    min_dist: float,
    max_dist: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Sample (anchor, target) pairs of crack pixels within a distance band.

    Args:
        crack_coords: (N, 2) tensor of (y, x) crack pixel coordinates.
        num_paths: number of paths to sample.
        min_dist, max_dist: distance band for target selection.

    Returns:
        (anchors, targets) each (num_paths, 2), or None if too few pixels.
    """
    n = crack_coords.shape[0]
    if n < 2:
        return None

    idx = torch.randperm(n, device=crack_coords.device)[:num_paths * 3]
    anchors = crack_coords[idx[:num_paths * 3]]

    # Pairwise distances within the pool
    diffs = anchors.unsqueeze(0) - anchors.unsqueeze(1)
    dists = torch.norm(diffs.float(), dim=2)

    mask = (dists >= min_dist) & (dists <= max_dist)
    # For each anchor pick the first valid target (if any)
    valid = mask.any(dim=1)
    if valid.sum() < num_paths // 2:
        return None

    anchors_out = []
    targets_out = []
    for i in range(min(num_paths * 3, anchors.shape[0])):
        if not valid[i]:
            continue
        candidates = torch.where(mask[i])[0]
        tgt = candidates[torch.randint(0, len(candidates), (1,), device=candidates.device)]
        anchors_out.append(anchors[i])
        targets_out.append(anchors[tgt[0]])
        if len(anchors_out) >= num_paths:
            break

    if len(anchors_out) < 2:
        return None
    return (torch.stack(anchors_out), torch.stack(targets_out))


def path_continuity_loss(
    pred_probs: torch.Tensor,
    gt_mask: torch.Tensor,
    num_paths: int = 50,
    path_length: int = 16,
    min_dist: float = 5.0,
    max_dist: float = 30.0,
) -> torch.Tensor:
    """Penalise crack fragmentation by enforcing high probability along
    straight-line paths between nearby crack pixels.

    For each image in the batch:
    1. Sample pairs of crack pixels separated by [min_dist, max_dist] px.
    2. Linearly interpolate *path_length* points between them.
    3. The predicted probability at every interpolated point should be 1.

    Args:
        pred_probs: (B, 1, H, W) sigmoid probabilities.
        gt_mask: (B, 1, H, W) binary ground-truth mask.
        num_paths: number of anchor-target pairs per image.
        path_length: number of interpolation points per path.
        min_dist, max_dist: distance band for pair sampling.

    Returns:
        Scalar BCE loss averaged over all path points.
    """
    device = pred_probs.device
    batch_size, _, height, width = pred_probs.shape
    total_loss = torch.tensor(0.0, device=device)
    total_points = 0

    for b in range(batch_size):
        mask = gt_mask[b, 0]
        crack_yx = torch.where(mask > 0.5)

        if crack_yx[0].numel() < 2:
            continue

        coords = torch.stack([crack_yx[0].float(), crack_yx[1].float()], dim=1)

        pairs = _sample_crack_pairs(coords, num_paths, min_dist, max_dist)
        if pairs is None:
            continue

        anchors, targets = pairs  # (K, 2), (K, 2) in (y, x)

        # Interpolate: alphas [0..1] spaced evenly
        alphas = torch.linspace(0.0, 1.0, path_length, device=device)
        # (1, path_length, 1) * (K, 1, 2) broadcast
        interp_yx = (anchors.unsqueeze(1) +
                     alphas.view(1, path_length, 1) *
                     (targets - anchors).unsqueeze(1))  # (K, path_length, 2)

        # Normalise to [-1, 1] for grid_sample
        grid_y = 2.0 * interp_yx[..., 0] / (height - 1) - 1.0
        grid_x = 2.0 * interp_yx[..., 1] / (width - 1) - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (K, path_length, 2)

        # grid_sample expects (N, C, H_out, W_out), grid (N, H_out, W_out, 2)
        prob_map = pred_probs[b:b + 1]  # (1, 1, H, W)
        grid_batch = grid.unsqueeze(0)  # (1, K, path_length, 2)

        sampled = F.grid_sample(prob_map, grid_batch, mode='bilinear',
                                padding_mode='border', align_corners=True)
        # sampled: (1, 1, K, path_length)
        sampled = torch.nan_to_num(sampled, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        targets_ones = torch.ones_like(sampled)
        total_loss += F.binary_cross_entropy(sampled, targets_ones, reduction='sum')
        total_points += sampled.numel()

    if total_points == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / total_points
