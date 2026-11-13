"""Thinning clDice and explicitly defined exploratory component diagnostics."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize


def structure_scores(pred, gt):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    ps, gs = skeletonize(pred), skeletonize(gt)
    if not pred.any() and not gt.any():
        precision = recall = cldice = 1.0
    else:
        precision = float((ps & gt).sum() / max(ps.sum(), 1))
        recall = float((gs & pred).sum() / max(gs.sum(), 1))
        cldice = 2 * precision * recall / max(precision + recall, 1e-12)
    ng, gl = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
    npred, pl = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
    # Background 4-connectivity complements foreground 8-connectivity.
    def holes(mask):
        padded = np.pad(mask, 1, constant_values=False)
        return cv2.connectedComponents((~padded).astype(np.uint8), connectivity=4)[0] - 2
    def overlap_excess(source_labels, target_labels, skeleton, nsource, ntarget):
        # Unique component pairs reproduce the component-by-component loops
        # without scanning the entire image for every isolated prediction.
        valid = skeleton & (target_labels > 0)
        pairs = np.unique(source_labels[valid].astype(np.int64) * ntarget + target_labels[valid])
        counts = np.bincount(pairs // ntarget, minlength=nsource)
        return int(np.maximum(counts[1:] - 1, 0).sum())
    split = overlap_excess(gl, pl, gs, ng, npred)
    merge = overlap_excess(pl, gl, ps, npred, ng)
    return {
        "cldice": float(cldice), "skeleton_precision": precision,
        "skeleton_recall": recall, "betti0_error": abs(ng - npred),
        "betti1_error": abs(holes(gt) - holes(pred)),
        "split_excess": split, "merge_excess": merge,
        "gt_components": ng - 1, "pred_components": npred - 1,
        "gt_skeleton_pixels": int(gs.sum()), "gt_empty": int(not gt.any()),
        "pred_empty": int(not pred.any()),
    }
