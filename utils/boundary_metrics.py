"""Symmetric boundary F1 at a fixed Euclidean pixel tolerance (512px canvas).

Inner one-pixel boundaries use 8-neighbor erosion and zero outside the canvas.
Matches are proximity matches, not a one-to-one contour assignment. Aggregate
matched and total boundary pixels over images before computing micro F1.
"""
import cv2
import numpy as np


def boundary_counts(pred, gt, tolerance=2.):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Boundary tolerance must be finite and nonnegative")
    def boundary(mask):
        mask = np.asarray(mask, np.uint8)
        inner = cv2.erode(mask, np.ones((3, 3), np.uint8),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
        return (mask - inner).astype(bool)
    p, g = boundary(pred), boundary(gt)
    def matched(a, b):
        if not a.any() or not b.any():
            return 0
        distance = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return int((a & (distance <= tolerance)).sum())
    return dict(boundary_pred_pixels=int(p.sum()), boundary_gt_pixels=int(g.sum()),
                boundary_pred_matched=matched(p, g), boundary_gt_matched=matched(g, p))


def summarize_boundary(counts):
    pn, gn = counts["boundary_pred_pixels"], counts["boundary_gt_pixels"]
    if pn == gn == 0:
        p = r = f = 1.
    else:
        p = counts["boundary_pred_matched"] / max(pn, 1)
        r = counts["boundary_gt_matched"] / max(gn, 1)
        f = 2 * p * r / max(p + r, 1e-12)
    return dict(boundary_precision=p, boundary_recall=r, boundary_f1=f)
