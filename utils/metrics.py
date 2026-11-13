from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from utils.topology import topology_scores_from_masks

EPS = 1e-6


@dataclass
class Confusion:
    """Binary confusion counts for foreground crack segmentation.

    Positive class: crack / foreground = 1
    Negative class: background = 0
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, other: "Confusion") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn


def unpack_batch(batch: Any):
    if isinstance(batch, dict):
        return batch["image"], batch["mask"]
    return batch


def _ensure_bhw(array: np.ndarray, name: str) -> np.ndarray:
    """Convert common mask/probability shapes to B x H x W.

    Supported inputs:
      H x W
      B x H x W
      B x 1 x H x W
    """
    array = np.asarray(array)

    if array.ndim == 2:
        return array[None, ...]

    if array.ndim == 3:
        return array

    if array.ndim == 4 and array.shape[1] == 1:
        return array[:, 0, ...]

    raise ValueError(f"Unsupported {name} shape. Expected HxW, BxHxW, or Bx1xHxW, got {array.shape}.")


def _prepare_prob_maps(probs: np.ndarray) -> np.ndarray:
    """Return probability maps as float32 B x H x W."""
    probs = _ensure_bhw(probs, "probability map").astype(np.float32, copy=False)
    return np.clip(probs, 0.0, 1.0)


def _prepare_binary_masks(masks: np.ndarray) -> np.ndarray:
    """Return ground-truth masks as uint8 B x H x W with crack=1, background=0.

    This avoids the common bug where 0/255 masks are cast with astype(np.int8),
    causing 255 to become -1 and breaking comparisons against 1.
    """
    masks = _ensure_bhw(masks, "ground-truth mask")
    return (masks > 0.5).astype(np.uint8)


def _binary_confusion_from_masks(preds: np.ndarray, masks: np.ndarray) -> Confusion:
    preds = preds.astype(bool, copy=False)
    masks = masks.astype(bool, copy=False)

    return Confusion(
        tp=int(np.logical_and(preds, masks).sum()),
        fp=int(np.logical_and(preds, np.logical_not(masks)).sum()),
        fn=int(np.logical_and(np.logical_not(preds), masks).sum()),
        tn=int(np.logical_and(np.logical_not(preds), np.logical_not(masks)).sum()),
    )


def confusion_from_probs(probs: np.ndarray, masks: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    """Compute foreground-crack confusion counts at one threshold.

    This function is kept for compatibility with existing imports.
    The metric is foreground-class binary segmentation, not macro-F1 or micro-F1.
    """
    probs_bhw = _prepare_prob_maps(probs)
    masks_bhw = _prepare_binary_masks(masks)

    if probs_bhw.shape != masks_bhw.shape:
        raise ValueError(f"Probability/mask shape mismatch: {probs_bhw.shape} vs {masks_bhw.shape}.")

    preds = (probs_bhw >= float(threshold)).astype(np.uint8)
    conf = _binary_confusion_from_masks(preds, masks_bhw)
    return conf.tp, conf.fp, conf.fn, conf.tn


def summarize_binary_metrics(tp: int, fp: int, fn: int, tn: int, threshold: float) -> dict[str, float]:
    """Summarize foreground crack metrics from global TP/FP/FN/TN counts.

    F1 here is foreground-class crack F1:
        F1 = 2TP / (2TP + FP + FN)

    mIoU is class-macro IoU over background and crack:
        mIoU = (IoU_crack + IoU_background) / 2
    """
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2 * tp) / (2 * tp + fp + fn + EPS)
    crack_iou = tp / (tp + fp + fn + EPS)
    bg_iou = tn / (tn + fp + fn + EPS)
    miou = (crack_iou + bg_iou) / 2

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "miou": float(miou),
        "crack_iou": float(crack_iou),
        "bg_iou": float(bg_iou),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def summarize_threshold_sweep_metrics(
    total_tps: np.ndarray,
    total_fps: np.ndarray,
    total_fns: np.ndarray,
    image_f1s: list[float],
    thresholds: np.ndarray,
) -> dict[str, float]:
    """Summarize ODS/OIS from a threshold sweep.

    ODS: best dataset-level foreground F1 using one threshold for the whole test set.
    OIS: mean of the best foreground F1 selected separately for each image.
    """
    dataset_precisions = total_tps / (total_tps + total_fps + EPS)
    dataset_recalls = total_tps / (total_tps + total_fns + EPS)
    dataset_f1s = (2 * total_tps) / (2 * total_tps + total_fps + total_fns + EPS)

    best_index = int(np.argmax(dataset_f1s))
    ods = float(dataset_f1s[best_index])

    return {
        "ods": ods,
        "best_f1": ods,
        "best_threshold": float(thresholds[best_index]),
        "best_precision": float(dataset_precisions[best_index]),
        "best_recall": float(dataset_recalls[best_index]),
        "ois": float(np.mean(image_f1s)) if image_f1s else 0.0,
        "threshold_min": float(thresholds[0]) if len(thresholds) else 0.0,
        "threshold_max": float(thresholds[-1]) if len(thresholds) else 0.0,
        "threshold_steps": int(len(thresholds)),
    }


def _squeeze_single_channel_masks(batch_masks: np.ndarray) -> np.ndarray:
    return _ensure_bhw(batch_masks, "mask for topology metrics")


def _model_prob_maps(model, images: torch.Tensor) -> np.ndarray:
    logits = model(images)
    return _prepare_prob_maps(torch.sigmoid(logits).detach().cpu().numpy())


def threshold_confusions(prob_map, gt_mask, thresholds):
    """Exact >= sweep counts via histograms, without 99 whole-image scans."""
    # Original sweep iterates np.float64 scalars, promoting float32 maps for
    # exact comparisons. Do not round the threshold edges to float32.
    edges = np.asarray(thresholds, dtype=np.float64)
    if len(edges) == 0 or np.any(edges[1:] < edges[:-1]):
        raise ValueError("Thresholds must be nonempty and sorted")
    bins = np.searchsorted(edges, np.asarray(prob_map, dtype=np.float32), side="right")
    gt = np.asarray(gt_mask, dtype=bool)
    positive = np.bincount(bins[gt].ravel(), minlength=len(edges) + 1)
    negative = np.bincount(bins[~gt].ravel(), minlength=len(edges) + 1)
    tp = np.cumsum(positive[::-1])[::-1][1:].astype(np.float64)
    fp = np.cumsum(negative[::-1])[::-1][1:].astype(np.float64)
    return tp, fp, float(gt.sum()) - tp


def evaluate_fixed_threshold(
    model,
    data_loader,
    device,
    threshold: float,
    desc: str | None = None,
    include_topology_metrics: bool = False,
) -> dict[str, float]:
    """Evaluate foreground crack metrics at one fixed threshold.

    This is a dataset-level/global aggregation of TP/FP/FN/TN over all pixels.
    It is not sklearn-style macro-F1 or micro-F1.
    """
    was_training = model.training
    model.eval()

    total = Confusion()
    topology_sum = 0.0
    topology_count = 0
    iterator = tqdm(data_loader, desc=desc, leave=False) if desc else data_loader

    with torch.no_grad():
        for batch in iterator:
            images, masks = unpack_batch(batch)
            images = images.to(device)

            batch_probs = _model_prob_maps(model, images)
            batch_masks = _prepare_binary_masks(masks.detach().cpu().numpy() if torch.is_tensor(masks) else masks)

            if batch_probs.shape != batch_masks.shape:
                raise ValueError(f"Probability/mask shape mismatch: {batch_probs.shape} vs {batch_masks.shape}.")

            batch_preds = (batch_probs >= float(threshold)).astype(np.uint8)
            total.add(_binary_confusion_from_masks(batch_preds, batch_masks))

            if include_topology_metrics:
                for pred_mask, gt_mask in zip(batch_preds, batch_masks):
                    topology_sum += topology_scores_from_masks(pred_mask, gt_mask)["cldice"]
                    topology_count += 1

    if was_training:
        model.train()

    metrics = summarize_binary_metrics(total.tp, total.fp, total.fn, total.tn, threshold)
    if include_topology_metrics:
        metrics["cldice"] = float(topology_sum / topology_count) if topology_count > 0 else 0.0
    return metrics


def evaluate_threshold_sweep(
    model,
    data_loader,
    device,
    threshold_min: float = 0.01,
    threshold_max: float = 0.99,
    threshold_steps: int = 99,
    desc: str | None = None,
) -> dict[str, float]:
    """Compute ODS and OIS using foreground crack F1.

    ODS uses one threshold for the whole test dataset and aggregates TP/FP/FN
    globally before computing F1.

    OIS selects the best threshold per image, computes foreground F1 per image,
    and then averages those per-image best F1 scores.
    """
    if threshold_steps <= 0:
        raise ValueError(f"threshold_steps must be positive, got {threshold_steps}.")
    if threshold_min > threshold_max:
        raise ValueError(f"threshold_min must be <= threshold_max, got {threshold_min} > {threshold_max}.")

    was_training = model.training
    model.eval()

    thresholds = np.linspace(float(threshold_min), float(threshold_max), int(threshold_steps))
    total_tps = np.zeros(len(thresholds), dtype=np.float64)
    total_fps = np.zeros(len(thresholds), dtype=np.float64)
    total_fns = np.zeros(len(thresholds), dtype=np.float64)
    image_f1s: list[float] = []
    iterator = tqdm(data_loader, desc=desc, leave=False) if desc else data_loader

    with torch.no_grad():
        for batch in iterator:
            images, masks = unpack_batch(batch)
            images = images.to(device)

            batch_probs = _model_prob_maps(model, images)
            batch_masks = _prepare_binary_masks(masks.detach().cpu().numpy() if torch.is_tensor(masks) else masks)

            if batch_probs.shape != batch_masks.shape:
                raise ValueError(f"Probability/mask shape mismatch: {batch_probs.shape} vs {batch_masks.shape}.")

            # Dataset-level counts for ODS.
            # Per-image best F1 for OIS. Do this image-by-image, not batch-by-batch.
            for prob_map, gt_mask in zip(batch_probs, batch_masks):
                tp, fp, fn = threshold_confusions(prob_map, gt_mask, thresholds)
                total_tps += tp
                total_fps += fp
                total_fns += fn
                image_f1s.append(float(np.max(2 * tp / (2 * tp + fp + fn + EPS))))

    if was_training:
        model.train()

    return summarize_threshold_sweep_metrics(total_tps, total_fps, total_fns, image_f1s, thresholds)
