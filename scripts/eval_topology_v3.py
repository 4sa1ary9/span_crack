"""Validation-selected thresholds with pixel and structural test metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.factory import build_model
from utils.config import load_yaml_config, resolve_path
from utils.dataloader import CrackDataset, get_transforms
from utils.metrics import threshold_confusions, summarize_binary_metrics
from utils.structure_metrics import structure_scores

THRESHOLDS = np.linspace(0.01, 0.99, 99)


def sweep_arrays(probs, masks):
    tp, fp, fn = (np.zeros(len(THRESHOLDS)) for _ in range(3))
    for p, g in zip(probs, masks):
        a, b, c = threshold_confusions(p, g, THRESHOLDS)
        tp += a; fp += b; fn += c
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def select_threshold(val_probs, val_masks):
    scores = sweep_arrays(val_probs, val_masks)
    # Deterministic earliest threshold on exact ties.
    index = int(np.argmax(scores))
    return float(THRESHOLDS[index]), float(scores[index])


def infer(model, config, split):
    ds = CrackDataset(resolve_path(config["data"][f"{split}_image_dir"]),
                      resolve_path(config["data"][f"{split}_mask_dir"]),
                      transform=get_transforms("eval"), mode="eval")
    loader = torch.utils.data.DataLoader(ds, batch_size=config["structure_evaluation"]["batch_size"],
                                        shuffle=False, num_workers=2)
    probs, masks, names = [], [], []
    with torch.inference_mode():
        for batch in loader:
            probs.extend(torch.sigmoid(model(batch["image"].cuda())).cpu().numpy()[:, 0])
            masks.extend(batch["mask"].numpy()[:, 0] > 0.5)
            names.extend(batch["image_name"])
    return probs, masks, names


def measure(probs, masks, names, threshold, postprocess=None, boundary_tolerance=None):
    rows = []
    for p, g, name in zip(probs, masks, names):
        pred = p >= np.float64(threshold)
        if postprocess:
            op = cv2.MORPH_DILATE if postprocess == "dilate1" else cv2.MORPH_CLOSE
            pred = cv2.morphologyEx(pred.astype(np.uint8), op, np.ones((3, 3), np.uint8)).astype(bool)
        tp = int((pred & g).sum()); fp = int((pred & ~g).sum())
        fn = int((~pred & g).sum()); tn = int((~pred & ~g).sum())
        row = summarize_binary_metrics(tp, fp, fn, tn, threshold)
        row.update(structure_scores(pred, g)); row["image"] = name
        if boundary_tolerance is not None:
            from utils.boundary_metrics import boundary_counts, summarize_boundary
            counts = boundary_counts(pred, g, boundary_tolerance)
            row.update(counts); row.update(summarize_boundary(counts))
        rows.append(row)
    totals = [sum(r[k] for r in rows) for k in ("tp", "fp", "fn", "tn")]
    summary = summarize_binary_metrics(*totals, threshold)
    for k in structure_scores(np.zeros((3, 3), bool), np.zeros((3, 3), bool)):
        summary[k] = float(np.mean([r[k] for r in rows]))
    summary["n"] = len(rows)
    summary["empty_gt_images"] = sum(r["gt_empty"] for r in rows)
    if boundary_tolerance is not None:
        counts = {k: sum(r[k] for r in rows) for k in (
            "boundary_pred_pixels", "boundary_gt_pixels", "boundary_pred_matched", "boundary_gt_matched")}
        summary.update(counts); summary.update(summarize_boundary(counts))
        summary["boundary_f1_image_mean"] = float(np.mean([r["boundary_f1"] for r in rows]))
    return summary, rows


def run(path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1", "2", "3", "4", "5", "6", "7"}:
        raise RuntimeError("Explicitly select one authorized physical GPU (0–7)")
    torch.set_num_threads(4); cv2.setNumThreads(1)
    config = load_yaml_config(path)
    boundary_tolerance = config["structure_evaluation"].get("boundary_tolerance_px")
    checkpoint = resolve_path(config["output"]["root_dir"]) / "checkpoints/best.pth"
    mc = dict(config["model"]); mc["pretrained"] = False
    model = build_model(mc).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    val = infer(model, config, "val")
    threshold, val_f1 = select_threshold(*val[:2])
    # Commit threshold to the report before reading test predictions/labels.
    target = resolve_path(config["structure_evaluation"]["output_dir"])
    target.mkdir(parents=True, exist_ok=True)
    (target / "validation_threshold.json").write_text(json.dumps({"threshold": threshold,
        "val_f1": val_f1, "selection": "validation_only"}, indent=2))
    test = infer(model, config, "test")
    summary = dict(experiment=config["experiment"], checkpoint=str(checkpoint),
                   checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   config_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                   physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"],
                   val_selected_threshold=threshold, val_selected_f1=val_f1,
                   cldice_definition="skimage Zhang thinning, image mean, both-empty=1", metrics={})
    if boundary_tolerance is not None:
        summary["boundary_definition"] = dict(tolerance_px=boundary_tolerance,
            canvas=[512, 512], boundary="inner 8-neighbor; zero outside canvas",
            distance="Euclidean, proximity matching", aggregation="micro counts across images")
    all_rows = []
    for protocol, th, post in (("fixed05", 0.5, None), ("val_f1", threshold, None),
                               ("dilate1_control", threshold, "dilate1"),
                               ("close1_control", threshold, "close1")):
        metrics, rows = measure(*test, th, post, boundary_tolerance=boundary_tolerance)
        summary["metrics"][protocol] = metrics
        all_rows.extend(dict(protocol=protocol, **r) for r in rows)
    val_metrics, _ = measure(*val, threshold, boundary_tolerance=boundary_tolerance)
    summary["validation_metrics"] = val_metrics
    summary["diagnostic_test_ods"] = float(np.max(sweep_arrays(*test[:2])))
    (target / "summary.json").write_text(json.dumps(summary, indent=2))
    with (target / "per_image.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    print(json.dumps({"run_id": config["experiment"]["run_id"], **summary["metrics"]["val_f1"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    run(Path(parser.parse_args().config).resolve())
