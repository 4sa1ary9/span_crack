from __future__ import annotations

import copy
import functools
import hashlib
import math
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from models.factory import build_model
from utils.config import (
    CONFIG_ROOT,
    load_yaml_config,
    parse_config_args,
    require_section,
    resolve_path,
    validate_model_variant,
    validate_training_loss,
)
from utils.connectivity import generate_width_labels, path_continuity_loss
from utils.dataloader import CrackDataset, DataLoader, get_transforms
from utils.metrics import evaluate_fixed_threshold, evaluate_threshold_sweep
from utils.output_manager import (
    build_archive_checkpoint_path,
    build_best_checkpoint_path,
    build_latest_checkpoint_path,
    build_log_path,
    get_experiment_root,
    save_json_report,
)

DEFAULT_CONFIG_PATH = CONFIG_ROOT / "tcs_seg" / "train" / "deepcrack.yaml"
DEFAULT_FIXED_THRESHOLD = 0.5


def bce_dice_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    bce = nn.BCEWithLogitsLoss()(logits, masks)
    probs = torch.sigmoid(logits)
    smooth = 1e-5
    intersection = (probs * masks).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    return bce + dice.mean()


def tversky_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    smooth = 1e-5
    dims = (2, 3)
    true_positive = (probs * masks).sum(dim=dims)
    false_positive = (probs * (1.0 - masks)).sum(dim=dims)
    false_negative = ((1.0 - probs) * masks).sum(dim=dims)
    score = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return 1.0 - score.mean()


def fixed_threshold_margin_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    *,
    positive_margin: float = 1.0,
    negative_margin: float = 1.0,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
) -> torch.Tensor:
    positive_mask = masks > 0.5
    negative_mask = ~positive_mask
    positive_penalty = F.relu(float(positive_margin) - logits).square()
    negative_penalty = F.relu(logits + float(negative_margin)).square()

    zero = logits.sum() * 0.0
    positive_term = positive_penalty[positive_mask].mean() if positive_mask.any() else zero
    negative_term = negative_penalty[negative_mask].mean() if negative_mask.any() else zero
    return float(positive_weight) * positive_term + float(negative_weight) * negative_term


SCNP_KAPPA = 1.0e4


def scnp_replace_logits(logits: torch.Tensor, masks: torch.Tensor, window: int = 3) -> torch.Tensor:
    """SCNP (Same-Class Neighbor Penalization) logit replacement.

    Foreground pixels take the minimum same-class logit in their w x w
    neighborhood (the worst foreground prediction); background pixels take
    the maximum same-class logit. Off-class positions are excluded via the
    kappa mask constant. Gradients route to the worst (selected) pixel,
    which is the mechanism under study.
    """
    fg_vals = logits * masks + SCNP_KAPPA * (1.0 - masks)
    bg_vals = logits * (1.0 - masks) - SCNP_KAPPA * masks
    v1 = -F.max_pool2d(-fg_vals, window, stride=1, padding=window // 2)
    v2 = F.max_pool2d(bg_vals, window, stride=1, padding=window // 2)
    return v1 * masks + v2 * (1.0 - masks)


def hard_neighbor_bce(
    logits: torch.Tensor,
    masks: torch.Tensor,
    window: int = 3,
    cap: float | None = None,
) -> torch.Tensor:
    """Hard-neighborhood BCE with optional propagation-count capping.

    For every position i, let j(i) be the worst same-class pixel in its
    w x w neighborhood. The term adds BCE(z_{j(i)}, y_i). With cap=None
    every selection counts with weight 1 (uncapped arm D). With cap=c the
    cumulative weight of a source pixel j selected by n_j neighborhoods is
    capped: w_j = min(1, c / max(1, n_j)) (arm E). Indices and counts are
    treated as constants (no gradient).
    """
    fg_vals = logits * masks + SCNP_KAPPA * (1.0 - masks)
    bg_vals = logits * (1.0 - masks) - SCNP_KAPPA * masks

    neg_pooled, idx_fg = F.max_pool2d_with_indices(
        -fg_vals, window, stride=1, padding=window // 2,
    )
    v1 = -neg_pooled
    v2, idx_bg = F.max_pool2d_with_indices(bg_vals, window, stride=1, padding=window // 2)
    worst = v1 * masks + v2 * (1.0 - masks)

    # BCE(worst logit, own label): foreground wants worst logit up,
    # background wants worst logit down.
    per_pos = F.softplus(-worst) * masks + F.softplus(worst) * (1.0 - masks)

    if cap is None:
        return per_pos.mean()

    # Count how many neighborhoods select each pixel as their worst
    # neighbor (no gradient), then cap the per-selection cumulative weight.
    # max_pool2d_with_indices returns flattened (y * W + x) input coordinates.
    b, _, h, w = logits.shape
    counts = torch.zeros((b, h * w), device=logits.device, dtype=logits.dtype)
    fg_flat = idx_fg[:, 0].reshape(b, -1)
    bg_flat = idx_bg[:, 0].reshape(b, -1)
    batch_offsets = torch.arange(b, device=logits.device).view(b, 1) * (h * w)
    counts.view(-1).index_add_(
        0, (fg_flat + batch_offsets).view(-1), masks.view(b, -1).detach().reshape(-1),
    )
    counts.view(-1).index_add_(
        0, (bg_flat + batch_offsets).view(-1), (1.0 - masks).view(b, -1).detach().reshape(-1),
    )
    # w_j = min(1, c / max(1, n_j)): a source pixel selected by many
    # neighborhoods gets its cumulative weight capped at c.
    weights_map = torch.minimum(
        torch.ones_like(counts), counts.clamp_min(1.0).reciprocal() * float(cap),
    )
    # Weight of each position's term is the cap weight of its SOURCE pixel.
    src_weight_fg = torch.gather(weights_map, 1, fg_flat).view(b, 1, h, w)
    src_weight_bg = torch.gather(weights_map, 1, bg_flat).view(b, 1, h, w)
    src_weight = src_weight_fg * masks + src_weight_bg * (1.0 - masks)
    return (per_pos * src_weight.detach()).sum() / per_pos.numel()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int, base_seed: int) -> None:
    # PyTorch assigns a fresh worker seed for each DataLoader iterator. Using it
    # keeps runs reproducible without repeating identical augmentations each epoch.
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    info = torch.utils.data.get_worker_info()
    if info is not None and getattr(info.dataset, "seed_augmentation", False):
        # Albumentations 2 owns RNG state independent of numpy/random.
        for name in ("transform", "weak_transform"):
            transform = getattr(info.dataset, name, None)
            if hasattr(transform, "set_random_seed"):
                transform.set_random_seed(worker_seed)


def resolve_fixed_threshold(config_section, *, required: bool = False, context: str = "config") -> float | None:
    fixed_threshold = config_section.get("fixed_threshold")
    if fixed_threshold is None:
        if required:
            raise KeyError(f"Missing 'fixed_threshold' in {context}.")
        return DEFAULT_FIXED_THRESHOLD
    return float(fixed_threshold)


def soft_erode(mask: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)


def soft_dilate(mask: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)


def soft_open(mask: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(mask))


def soft_skeletonize(mask: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    mask = mask.clamp(0.0, 1.0)
    opened = soft_open(mask)
    skeleton = F.relu(mask - opened)
    eroded = mask
    for _ in range(max(iterations, 1)):
        eroded = soft_erode(eroded)
        opened = soft_open(eroded)
        delta = F.relu(eroded - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0.0, 1.0)


def soft_cldice_loss(logits: torch.Tensor, masks: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred_skeleton = soft_skeletonize(probs, iterations=iterations)
    target_skeleton = soft_skeletonize(masks, iterations=iterations)
    smooth = 1e-6
    topo_precision = (pred_skeleton * masks).sum(dim=(2, 3)) / (pred_skeleton.sum(dim=(2, 3)) + smooth)
    topo_sensitivity = (target_skeleton * probs).sum(dim=(2, 3)) / (target_skeleton.sum(dim=(2, 3)) + smooth)
    cldice = (2.0 * topo_precision * topo_sensitivity + smooth) / (
        topo_precision + topo_sensitivity + smooth
    )
    return 1.0 - cldice.mean()


def boundary_target_from_mask(masks: torch.Tensor) -> torch.Tensor:
    dilated = soft_dilate(masks)
    eroded = soft_erode(masks)
    return (dilated - eroded).clamp(0.0, 1.0)


def _loss_warmup_multiplier(
    current_epoch: int, loss_config: dict,
) -> float:
    """Linear ramp from 0 to 1 for new-loss warmup."""
    start = int(loss_config.get("loss_warmup_start_epoch", 20))
    end = int(loss_config.get("loss_warmup_end_epoch", 40))
    if current_epoch < start:
        return 0.0
    if current_epoch >= end or end <= start:
        return 1.0
    return (current_epoch - start) / (end - start)


def build_total_loss(
    outputs: dict[str, torch.Tensor],
    batch,
    loss_config: dict | None = None,
    current_epoch: int = 0,
    teacher_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_config = loss_config or {}
    mask_logits = outputs["mask_logits"]
    masks = batch["mask"]
    main_term = bce_dice_loss(mask_logits, masks)
    total_loss = float(loss_config.get("main_weight", 1.0)) * main_term
    loss_terms = {
        "main": float(main_term.detach().item()),
    }

    author_weight = float(loss_config.get("author_cldice_weight", 0.0))
    recall_weight = float(loss_config.get("skeleton_recall_weight", 0.0))
    if author_weight > 0:
        from utils.publication_losses import author_cldice_loss
        term = author_cldice_loss(mask_logits, masks)
        total_loss = total_loss + author_weight * term
        loss_terms["author_cldice"] = float(term.detach())
    if recall_weight > 0:
        from utils.publication_losses import skeleton_recall_loss
        term = skeleton_recall_loss(mask_logits, batch["skeleton_recall_target"])
        total_loss = total_loss + recall_weight * term
        loss_terms["skeleton_recall"] = float(term.detach())

    separation_weight = float(loss_config.get("separation_weight", 0.0))
    if separation_weight > 0:
        from utils.separation import background_suppression_loss
        separation = background_suppression_loss(
            mask_logits, masks,
            mode=loss_config.get("separation_mode", "gap"),
            radius=int(loss_config.get("separation_radius", 4)),
        )
        total_loss = total_loss + separation_weight * separation
        loss_terms["separation"] = float(separation.detach())

    # --- Mean-teacher consistency: student(strong view) vs EMA teacher(weak view).
    consistency_weight = float(loss_config.get("consistency_weight", 0.0))
    if teacher_logits is not None and consistency_weight > 0.0:
        ramp_epochs = max(int(loss_config.get("consistency_ramp_epochs", 10)), 0)
        ramp = 1.0 if ramp_epochs == 0 or current_epoch >= ramp_epochs else current_epoch / ramp_epochs
        teacher_prob = torch.sigmoid(teacher_logits).detach()
        student_prob = torch.sigmoid(mask_logits)
        consistency_term = F.mse_loss(student_prob, teacher_prob)
        total_loss = total_loss + consistency_weight * ramp * consistency_term
        loss_terms["consistency"] = float(consistency_term.detach().item())

    # SCNP: main_weight=0 is replacement-only; main_weight=1 is joint loss.
    scnp_weight = float(loss_config.get("scnp_weight", 0.0))
    if scnp_weight > 0.0:
        scnp_window = int(loss_config.get("scnp_window", 3))
        scnp_logits = scnp_replace_logits(mask_logits, masks, window=scnp_window)
        scnp_term = bce_dice_loss(scnp_logits, masks)
        total_loss = total_loss + scnp_weight * scnp_term
        loss_terms["scnp"] = float(scnp_term.detach().item())

    # --- Hard-neighborhood BCE (arms D/E of the propagation-control study).
    hard_bce_weight = float(loss_config.get("hard_neighbor_bce_weight", 0.0))
    if hard_bce_weight > 0.0:
        hard_window = int(loss_config.get("hard_neighbor_window", 3))
        hard_cap = loss_config.get("hard_neighbor_cap", None)
        hard_cap = None if hard_cap is None else float(hard_cap)
        hard_term = hard_neighbor_bce(
            mask_logits, masks, window=hard_window, cap=hard_cap,
        )
        total_loss = total_loss + hard_bce_weight * hard_term
        loss_terms["hard_neighbor_bce"] = float(hard_term.detach().item())

    fixed_margin_weight = float(loss_config.get("fixed_margin_weight", 0.0))
    if fixed_margin_weight > 0.0:
        fixed_margin_term = fixed_threshold_margin_loss(
            mask_logits,
            masks,
            positive_margin=float(loss_config.get("fixed_margin_positive", 1.0)),
            negative_margin=float(loss_config.get("fixed_margin_negative", 1.0)),
            positive_weight=float(loss_config.get("fixed_margin_positive_weight", 1.0)),
            negative_weight=float(loss_config.get("fixed_margin_negative_weight", 1.0)),
        )
        total_loss = total_loss + fixed_margin_weight * fixed_margin_term
        loss_terms["fixed_margin"] = float(fixed_margin_term.detach().item())

    # --- Precision-aware Tversky loss ---
    pt_weight = float(loss_config.get("precision_tversky_weight", 0.0))
    if pt_weight > 0.0:
        pt_term = tversky_loss(
            mask_logits,
            masks,
            alpha=float(loss_config.get("tversky_alpha", 0.75)),
            beta=float(loss_config.get("tversky_beta", 0.25)),
        )
        total_loss = total_loss + pt_weight * pt_term
        loss_terms["precision_tversky"] = float(pt_term.detach().item())

    # --- Far-background margin loss ---
    fb_weight = float(loss_config.get("far_background_margin_weight", 0.0))
    if fb_weight > 0.0:
        fb_radius = int(loss_config.get("far_background_radius", 3))
        fb_margin = float(loss_config.get("far_background_margin", 0.75))
        # Dilate crack region so only far-away background is penalized
        dilated = F.max_pool2d(
            masks, kernel_size=2 * fb_radius + 1, stride=1, padding=fb_radius,
        )
        far_bg = (dilated < 0.5)
        if far_bg.any():
            fb_penalty = F.relu(mask_logits[far_bg] + fb_margin).square().mean()
            total_loss = total_loss + fb_weight * fb_penalty
            loss_terms["far_background_margin"] = float(fb_penalty.detach().item())

    warmup = _loss_warmup_multiplier(current_epoch, loss_config)

    # --- Topology losses (no warmup — proven effective from epoch 0) ---

    cldice_weight = float(loss_config.get("soft_cldice_weight", 0.0))
    if cldice_weight > 0.0:
        cldice_term = soft_cldice_loss(
            mask_logits, masks,
            iterations=int(loss_config.get("soft_skeleton_iterations", 10)),
        )
        total_loss = total_loss + cldice_weight * cldice_term
        loss_terms["soft_cldice"] = float(cldice_term.detach().item())

    skeleton_weight = float(loss_config.get("skeleton_aux_weight", 0.0))
    if skeleton_weight > 0.0 and "skeleton_logits" in outputs:
        skeleton_target = soft_skeletonize(
            masks,
            iterations=int(loss_config.get("target_skeleton_iterations", 10)),
        ).detach()
        skeleton_term = nn.BCEWithLogitsLoss()(
            outputs["skeleton_logits"], skeleton_target,
        )
        total_loss = total_loss + skeleton_weight * skeleton_term
        loss_terms["skeleton_aux"] = float(skeleton_term.detach().item())

    boundary_weight = float(loss_config.get("boundary_aux_weight", 0.0))
    if boundary_weight > 0.0 and "boundary_logits" in outputs:
        boundary_target = boundary_target_from_mask(masks).detach()
        boundary_term = nn.BCEWithLogitsLoss()(
            outputs["boundary_logits"], boundary_target,
        )
        total_loss = total_loss + boundary_weight * boundary_term
        loss_terms["boundary_aux"] = float(boundary_term.detach().item())

    # --- New losses (warmup) ---

    # Width prior CE loss
    width_prior_weight = float(loss_config.get("width_prior_weight", 0.0))
    if width_prior_weight > 0.0 and warmup > 0.0 and (
        "width_logits" in outputs or "width_prior" in outputs
    ):
        width_logits = outputs.get("width_logits")
        if width_logits is None:
            # Backward-compatibility: if only a probability map is available,
            # recover logits so cross-entropy still sees unnormalised scores.
            width_logits = torch.log(
                outputs["width_prior"].clamp_min(1e-6)
            )
        thin_max = float(loss_config.get("width_bin_thin_max", 2.0))
        medium_max = float(loss_config.get("width_bin_medium_max", 5.0))
        width_term = torch.tensor(0.0, device=masks.device)
        width_count = 0
        for b in range(masks.shape[0]):
            mask_np = masks[b, 0].detach().cpu().numpy()
            width_labels = generate_width_labels(
                mask_np, thin_max=thin_max, medium_max=medium_max,
            )
            # Skip sample when all labels are ignored — no crack pixels in crop
            valid_pixels = (width_labels != 255).sum()
            if valid_pixels == 0:
                continue
            width_labels_t = torch.from_numpy(width_labels).long().to(masks.device)
            wl_up = F.interpolate(
                width_logits[b:b + 1],
                size=width_labels_t.shape,
                mode="bilinear",
                align_corners=False,
            )
            width_term += F.cross_entropy(
                wl_up,                          # (1, 3, H, W) raw logits
                width_labels_t.unsqueeze(0),    # (1, H, W)
                ignore_index=255,
            )
            width_count += 1
        if width_count > 0:
            width_term = width_term / width_count
            total_loss = total_loss + width_prior_weight * warmup * width_term
            loss_terms["width_prior"] = float(width_term.detach().item())

    # Path continuity loss
    pcl_weight = float(loss_config.get("path_continuity_weight", 0.0))
    if pcl_weight > 0.0 and warmup > 0.0:
        probs = torch.sigmoid(mask_logits)
        pcl_term = path_continuity_loss(
            probs, masks,
            num_paths=int(loss_config.get("pcl_num_paths", 50)),
            path_length=int(loss_config.get("pcl_path_length", 16)),
            min_dist=float(loss_config.get("pcl_min_dist", 5.0)),
            max_dist=float(loss_config.get("pcl_max_dist", 30.0)),
        )
        if pcl_term.requires_grad:
            total_loss = total_loss + pcl_weight * warmup * pcl_term
            loss_terms["path_continuity"] = float(pcl_term.detach().item())

    loss_terms["total"] = float(total_loss.detach().item())
    return total_loss, loss_terms


def build_dataset(
    data_config,
    image_key: str,
    mask_key: str | None = None,
    mode: str = "train",
    augmentation_profile: str = "standard",
    copy_paste: dict | None = None,
    weak_transform=None,
    include_skeleton_recall: bool = False,
) -> CrackDataset:
    return CrackDataset(
        image_dir=resolve_path(data_config[image_key]),
        mask_dir=resolve_path(data_config[mask_key]) if mask_key else None,
        transform=get_transforms(mode=mode, augmentation_profile=augmentation_profile),
        mode=mode,
        mask_suffix=data_config.get("mask_suffix", ".png"),
        image_extensions=data_config.get("image_extensions"),
        copy_paste=copy_paste,
        weak_transform=weak_transform,
        include_skeleton_recall=include_skeleton_recall,
    )


def build_train_loader(data_config, training_config) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(training_config.get("seed", 3407)))
    worker_init_fn = functools.partial(seed_worker, base_seed=int(training_config.get("seed", 3407)))
    use_mean_teacher = bool(training_config.get("use_mean_teacher", False))
    dataset = build_dataset(
        data_config,
        image_key="image_dir",
        mask_key="mask_dir",
        mode="train",
        augmentation_profile=training_config.get("augmentation_profile", "standard"),
        copy_paste=training_config.get("copy_paste"),
        include_skeleton_recall=float(training_config.get("loss", {}).get("skeleton_recall_weight", 0)) > 0,
        weak_transform=(
            get_transforms(
                mode="train",
                augmentation_profile=training_config.get("weak_augmentation_profile", "standard"),
            )
            if use_mean_teacher
            else None
        ),
    )
    dataset.seed_augmentation = bool(training_config.get("seed_augmentation", False))
    if dataset.seed_augmentation and hasattr(dataset.transform, "set_random_seed"):
        dataset.transform.set_random_seed(int(training_config.get("seed", 3407)))
    return DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["num_workers"],
        generator=generator,
        worker_init_fn=worker_init_fn,
        drop_last=bool(training_config.get("drop_last", False)),
    )


def resolve_validation_dir(data_config, key: str, fallback_source_key: str, train_token: str, val_token: str) -> Path:
    if key in data_config:
        return resolve_path(data_config[key])

    source_path = resolve_path(data_config[fallback_source_key])
    source_name = source_path.name
    if train_token not in source_name:
        raise KeyError(
            f"Missing '{key}' in config, and cannot infer it from '{fallback_source_key}={source_path}'."
        )
    return source_path.with_name(source_name.replace(train_token, val_token, 1))


def build_val_loader(data_config, training_config) -> DataLoader:
    dataset = CrackDataset(
        image_dir=resolve_validation_dir(data_config, "val_image_dir", "image_dir", "train_img", "val_img"),
        mask_dir=resolve_validation_dir(data_config, "val_mask_dir", "mask_dir", "train_lab", "val_lab"),
        transform=get_transforms(mode="eval"),
        mode="eval",
        mask_suffix=data_config.get("mask_suffix", ".png"),
        image_extensions=data_config.get("image_extensions"),
    )
    return DataLoader(
        dataset,
        batch_size=training_config.get("val_batch_size", 1),
        shuffle=False,
        num_workers=training_config["num_workers"],
    )


def save_training_artifacts(
    history_path: Path,
    config_snapshot_path: Path,
    raw_config_path: Path,
    experiment_name: str,
    config,
    history,
    run_best_record,
    checkpoint_paths,
) -> None:
    config_snapshot_path.write_text(raw_config_path.read_text(encoding="utf-8"), encoding="utf-8")
    save_json_report(
        history_path,
        {
            "experiment_name": experiment_name,
            "config_path": str(config["_meta"]["config_path"]),
            "history": history,
            "run_best_record": run_best_record,
            "checkpoints": checkpoint_paths,
        },
    )


def clone_state_dict_to_cpu(state_dict) -> dict:
    cloned = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def build_optimizer(model: nn.Module, training_config) -> AdamW:
    backbone_lr = float(training_config.get("backbone_lr", training_config.get("learning_rate", 2e-4)))
    head_lr = float(training_config.get("head_lr", training_config.get("learning_rate", 2e-4)))
    weight_decay = float(training_config.get("weight_decay", 0.0))

    backbone_module = None
    if hasattr(model, "base_model") and hasattr(model.base_model, "segformer"):
        backbone_module = model.base_model.segformer
    elif hasattr(model, "backbone"):
        backbone_module = model.backbone
    backbone_param_ids = (
        {id(param) for param in backbone_module.parameters() if param.requires_grad}
        if backbone_module is not None
        else set()
    )
    backbone_params = []
    head_params = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in backbone_param_ids:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})
    return AdamW(param_groups, weight_decay=weight_decay)


def build_scheduler(optimizer: AdamW, training_config, total_epochs: int) -> LambdaLR:
    warmup_epochs = max(int(training_config.get("warmup_epochs", 0)), 0)

    def lr_lambda(epoch_index: int) -> float:
        if total_epochs <= 0:
            return 1.0
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / warmup_epochs
        if total_epochs <= warmup_epochs:
            return 1.0

        progress = (epoch_index - warmup_epochs + 1) / max(total_epochs - warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def apply_initial_scheduler_lr(optimizer: AdamW, scheduler: LambdaLR, training_config) -> None:
    warmup_epochs = max(int(training_config.get("warmup_epochs", 0)), 0)
    if warmup_epochs <= 0:
        return

    for group_index, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = scheduler.base_lrs[group_index] * scheduler.lr_lambdas[group_index](0)


def sha256_file(path: Path) -> str | None:
    """Return the SHA256 of a file, or None if it does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pretrained_dir(model_config) -> Path | None:
    """Best-effort resolution of the pretrained weight directory for hashing."""
    if model_config.get("pretrained", True) is False:
        return None
    model_type = model_config.get("type", "b0")
    try:
        from models.segformer_base import MODEL_DIRS

        return MODEL_DIRS.get(model_type)
    except Exception:
        return None


def build_run_manifest(
    *,
    run_id: str,
    attempt: str,
    seed,
    model_config,
    experiment_name: str,
    dataset: str,
    config_path: Path,
    best_checkpoint_path: Path,
    latest_checkpoint_path: Path,
    run_best_record,
    history,
    started_at: str,
    finished_at: str,
    device,
) -> dict:
    """Assemble the per-run traceability manifest.

    A run's evaluation must be traceable to its own config, pretrained weights,
    checkpoint SHA256 and validation-best epoch. No historical run is consulted,
    so results from different seeds cannot contaminate each other.
    """
    pretrained_dir = resolve_pretrained_dir(model_config)
    pretrained_hash = None
    if pretrained_dir is not None:
        pretrained_hash = sha256_file(Path(pretrained_dir) / "pytorch_model.bin")
    try:
        code_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        code_revision = None
    return {
        "run_id": run_id,
        "attempt": attempt,
        "seed": None if seed is None else int(seed),
        "experiment_name": experiment_name,
        "dataset": dataset,
        "variant": model_config.get("variant"),
        "model_type": model_config.get("type"),
        "config_path": str(config_path),
        "config_sha256": sha256_file(Path(config_path)),
        "pretrained_dir": None if pretrained_dir is None else str(pretrained_dir),
        "pretrained_sha256": pretrained_hash,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "code_revision": code_revision,
        "started_at": started_at,
        "finished_at": finished_at,
        "epochs": len(history),
        "best_epoch": None if run_best_record is None else run_best_record.get("epoch"),
        "selection_metric": "val.fixed.f1",
        "val_fixed_f1_at_best": (
            None if run_best_record is None else run_best_record.get("fixed_f1_at_selection")
        ),
        "checkpoint_best_path": str(best_checkpoint_path),
        "checkpoint_best_sha256": sha256_file(best_checkpoint_path),
        "checkpoint_latest_path": str(latest_checkpoint_path),
        "checkpoint_latest_sha256": sha256_file(latest_checkpoint_path),
        "selection_rule": "highest val fixed@0.5 foreground F1; earliest epoch on ties",
    }


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


@torch.no_grad()
def update_ema_teacher(teacher: nn.Module, student: nn.Module, decay: float) -> None:
    """EMA-update teacher params and float buffers toward the student."""
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.mul_(decay).add_(s_param.detach(), alpha=1.0 - decay)
    for t_buffer, s_buffer in zip(teacher.buffers(), student.buffers()):
        if torch.is_floating_point(t_buffer):
            t_buffer.mul_(decay).add_(s_buffer.detach(), alpha=1.0 - decay)
        else:
            t_buffer.copy_(s_buffer)


def run(config_path) -> None:
    config = load_yaml_config(config_path)
    experiment_config = config.get("experiment", {})
    model_config = require_section(config, "model")
    data_config = require_section(config, "data")
    training_config = require_section(config, "training")
    validate_training_loss(training_config.get("loss", {}))
    output_config = require_section(config, "output")
    variant = model_config.get("variant", "tcs_seg")
    config_family = validate_model_variant(config, model_config)
    if "seed" in training_config:
        set_random_seed(int(training_config["seed"]))
    fixed_threshold = resolve_fixed_threshold(
        training_config,
        context=f"training config for variant '{variant}'",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = experiment_config.get("name", "train_experiment")
    dataset_name = experiment_config.get(
        "dataset", experiment_name.split("_")[0] if experiment_name else "unknown",
    )
    # Per-run identity. When the YAML provides experiment.run_id (as the
    # generated paper_v2 configs do) it is recorded verbatim; otherwise fall
    # back to the experiment name so the manifest is still traceable.
    run_id = str(experiment_config.get("run_id", experiment_name))
    attempt = str(experiment_config.get("attempt", "attempt_01"))
    warning_if_not_cuda = (
        "WARNING: CUDA is not available; training will run on CPU. Long runs "
        "should not be started on CPU."
    )
    print(f"Experiment: {experiment_name}")
    print(f"Run ID: {run_id} ({attempt})")
    print(f"Using device: {device}")
    print(f"Using config: {config['_meta']['config_path']}")
    if config_family:
        print(f"Config family: {config_family} -> model.variant={variant}")
    print(f"Selection metric: val.fixed.f1 @ threshold {fixed_threshold:.4f}")
    print(f"Train augmentation profile: {training_config.get('augmentation_profile', 'standard')}")
    if device.type != "cuda":
        print(warning_if_not_cuda)

    started_at = datetime.now(timezone.utc).isoformat()

    experiment_root = get_experiment_root(output_config, experiment_name)
    history_path = build_log_path(experiment_root, "train_history.json")
    config_snapshot_path = build_log_path(experiment_root, "train_config.yaml")
    manifest_path = build_log_path(experiment_root, "run_manifest.json")
    best_checkpoint_path = build_best_checkpoint_path(experiment_root)
    latest_checkpoint_path = build_latest_checkpoint_path(experiment_root)
    archive_checkpoint_path = build_archive_checkpoint_path(experiment_root)

    train_loader = build_train_loader(data_config, training_config)
    val_loader = build_val_loader(data_config, training_config)

    model = build_model(model_config).to(device)
    optimizer = build_optimizer(model, training_config)
    total_epochs = training_config["epochs"]
    scheduler = build_scheduler(optimizer, training_config, total_epochs)
    apply_initial_scheduler_lr(optimizer, scheduler, training_config)

    use_mean_teacher = bool(training_config.get("use_mean_teacher", False))
    ema_decay = float(training_config.get("ema_decay", 0.99))
    teacher = None
    if use_mean_teacher:
        teacher = copy.deepcopy(model)
        for param in teacher.parameters():
            param.requires_grad_(False)
        teacher.eval()

    model.train()
    run_best_score = float("-inf")
    run_best_record = None
    run_best_state_dict = None
    run_best_teacher_state_dict = None
    history = []

    for epoch in range(total_epochs):
        epoch_loss = 0.0
        epoch_main_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}")

        for batch in loop:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["image"], return_aux=True)
            teacher_logits = None
            if teacher is not None and "image_weak" in batch:
                teacher_logits = teacher(batch["image_weak"], return_aux=False)
            loss, loss_terms = build_total_loss(
                outputs, batch, training_config.get("loss", {}), current_epoch=epoch,
                teacher_logits=teacher_logits,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if teacher is not None:
                update_ema_teacher(teacher, model, ema_decay)

            epoch_loss += loss.item()
            epoch_main_loss += loss_terms["main"]
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                main=f"{loss_terms['main']:.4f}",
            )

        mean_loss = epoch_loss / len(train_loader)
        mean_main_loss = epoch_main_loss / len(train_loader)
        val_sweep_metrics = evaluate_threshold_sweep(
            model,
            val_loader,
            device,
            threshold_min=training_config.get("threshold_min", 0.01),
            threshold_max=training_config.get("threshold_max", 0.99),
            threshold_steps=training_config.get("threshold_steps", 99),
            desc=f"Val {epoch + 1}/{total_epochs}",
        )
        val_fixed_metrics = evaluate_fixed_threshold(
            model,
            val_loader,
            device,
            threshold=fixed_threshold,
            desc=f"ValFixed {epoch + 1}/{total_epochs}",
            include_topology_metrics=False,
        )

        current_score = val_fixed_metrics["f1"]
        teacher_val_record = None
        if teacher is not None:
            teacher_fixed = evaluate_fixed_threshold(
                teacher,
                val_loader,
                device,
                threshold=fixed_threshold,
                desc=f"ValTeacher {epoch + 1}/{total_epochs}",
                include_topology_metrics=False,
            )
            teacher_val_record = {
                "f1": float(teacher_fixed["f1"]),
                "precision": float(teacher_fixed["precision"]),
                "recall": float(teacher_fixed["recall"]),
            }
        is_run_best = current_score > run_best_score
        if is_run_best:
            run_best_score = current_score
            run_best_record = {
                "scope": "current_run_best",
                "epoch": epoch + 1,
                "selection_metric": "val.fixed.f1",
                "selection_score": float(current_score),
                "best_threshold": float(val_sweep_metrics["best_threshold"]),
                "ods": float(val_sweep_metrics["ods"]),
                "ois": float(val_sweep_metrics["ois"]),
                "fixed_threshold": float(val_fixed_metrics["threshold"]),
                "fixed_f1_at_selection": float(val_fixed_metrics["f1"]),
                "fixed_cldice_at_selection": val_fixed_metrics.get("cldice"),
                "checkpoint_path": str(latest_checkpoint_path),
            }
            run_best_state_dict = clone_state_dict_to_cpu(model.state_dict())
            if teacher is not None:
                run_best_teacher_state_dict = clone_state_dict_to_cpu(teacher.state_dict())

        epoch_record = {
            "epoch": epoch + 1,
            "train": {
                "loss": float(mean_loss),
                "main_loss": float(mean_main_loss),
            },
            "val": {
                "sweep": val_sweep_metrics,
                "fixed": val_fixed_metrics,
            },
            "val_teacher": teacher_val_record,
            "is_run_best": is_run_best,
        }
        history.append(epoch_record)
        save_training_artifacts(
            history_path=history_path,
            config_snapshot_path=config_snapshot_path,
            raw_config_path=config["_meta"]["config_path"],
            experiment_name=experiment_name,
            config=config,
            history=history,
            run_best_record=run_best_record,
            checkpoint_paths={
                "best": str(best_checkpoint_path),
                "latest": str(latest_checkpoint_path),
                "archive": str(archive_checkpoint_path),
            },
        )
        print(
            f"Epoch {epoch + 1} completed | "
            f"train_loss={mean_loss:.4f} | "
            f"train_main_loss={mean_main_loss:.4f} | "
            f"val_fixed_f1={val_fixed_metrics['f1']:.4f} | "
            f"val_sweep_best_f1={val_sweep_metrics['best_f1']:.4f}"
        )
        scheduler.step()

    if run_best_state_dict is None:
        run_best_state_dict = clone_state_dict_to_cpu(model.state_dict())
        run_best_record = {
            "scope": "current_run_best",
            "epoch": total_epochs,
            "selection_metric": "val.fixed.f1",
            "selection_score": float("-inf"),
            "best_threshold": float("nan"),
            "ods": 0.0,
            "ois": 0.0,
            "fixed_threshold": fixed_threshold,
            "fixed_f1_at_selection": None,
            "fixed_cldice_at_selection": None,
            "checkpoint_path": str(latest_checkpoint_path),
        }

    # Within one isolated run directory, best.pth is simply this run's
    # validation-best checkpoint. There is no cross-run "historical best".
    torch.save(run_best_state_dict, latest_checkpoint_path)
    torch.save(run_best_state_dict, archive_checkpoint_path)
    torch.save(run_best_state_dict, best_checkpoint_path)
    if run_best_teacher_state_dict is not None:
        torch.save(
            run_best_teacher_state_dict,
            best_checkpoint_path.parent / "best_teacher.pth",
        )

    finished_at = datetime.now(timezone.utc).isoformat()
    manifest = build_run_manifest(
        run_id=run_id,
        attempt=attempt,
        seed=training_config.get("seed"),
        model_config=model_config,
        experiment_name=experiment_name,
        dataset=dataset_name,
        config_path=Path(config["_meta"]["config_path"]),
        best_checkpoint_path=best_checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        run_best_record=run_best_record,
        history=history,
        started_at=started_at,
        finished_at=finished_at,
        device=device,
    )

    save_training_artifacts(
        history_path=history_path,
        config_snapshot_path=config_snapshot_path,
        raw_config_path=config["_meta"]["config_path"],
        experiment_name=experiment_name,
        config=config,
        history=history,
        run_best_record=run_best_record,
        checkpoint_paths={
            "best": str(best_checkpoint_path),
            "latest": str(latest_checkpoint_path),
            "archive": str(archive_checkpoint_path),
        },
    )
    save_json_report(manifest_path, manifest)

    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Latest checkpoint: {latest_checkpoint_path}")
    print(f"Run archive checkpoint: {archive_checkpoint_path}")
    print(f"Run manifest: {manifest_path}")
    print(
        f"Run best (this run only): val fixed-F1 "
        f"{run_best_record['selection_score']:.4f} @ epoch {run_best_record['epoch']}"
    )


def main() -> None:
    args = parse_config_args(
        "Train a crack segmentation model from a train-specific YAML config.",
        default_config_path=DEFAULT_CONFIG_PATH,
    )
    run(args.config)


if __name__ == "__main__":
    main()
