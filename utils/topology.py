from __future__ import annotations

import cv2
import numpy as np


def binary_skeletonize(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return binary

    skeleton = np.zeros_like(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(binary, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(binary, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        binary = eroded
        if cv2.countNonZero(binary) == 0:
            break

    return (skeleton > 0).astype(np.uint8)


def topology_scores_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> dict[str, float]:
    pred_mask = (pred_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    pred_skeleton = binary_skeletonize(pred_mask)
    gt_skeleton = binary_skeletonize(gt_mask)

    topo_precision = float((pred_skeleton * gt_mask).sum()) / (float(pred_skeleton.sum()) + eps)
    topo_sensitivity = float((gt_skeleton * pred_mask).sum()) / (float(gt_skeleton.sum()) + eps)
    cldice = (2.0 * topo_precision * topo_sensitivity) / (topo_precision + topo_sensitivity + eps)

    return {
        "cldice": float(cldice),
    }
