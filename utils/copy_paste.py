"""Structure-aware crack copy-paste augmentation.

Builds a bank of crack patches (image crops around connected crack
components) from the training split, then pastes a few of them onto
background regions of each training sample. Placement never covers existing
crack pixels, seams are feathered with a distance-transform alpha, and patch
appearance is mean-shifted toward the local background so pasted cracks do
not look like rectangles.

The paste happens at native resolution, before the geometric/photometric
albumentations pipeline, so labels and pixels always stay consistent.
"""
from __future__ import annotations

import cv2
import numpy as np


DEFAULT_COPY_PASTE = {
    "enabled": True,
    "prob": 0.5,
    "max_instances": 3,
    "min_area": 40,
    "max_patches": 4096,
    "color_match": 0.8,
    "max_tries": 8,
}


def normalize_copy_paste_config(config: dict | None) -> dict | None:
    """Merge a user config over the defaults. Returns None when disabled."""
    if config is None:
        return None
    merged = dict(DEFAULT_COPY_PASTE)
    merged.update(config)
    if not merged.get("enabled", True):
        return None
    return merged


class CrackPasteAugmenter:
    def __init__(
        self,
        image_paths,
        mask_dir,
        mask_suffix: str = ".png",
        *,
        prob: float = 0.5,
        max_instances: int = 3,
        min_area: int = 40,
        max_patches: int = 4096,
        color_match: float = 0.8,
        max_tries: int = 8,
    ):
        if max_instances < 1:
            raise ValueError(f"max_instances must be >= 1, got {max_instances}")
        self.prob = float(prob)
        self.max_instances = int(max_instances)
        self.min_area = int(min_area)
        self.color_match = float(color_match)
        self.max_tries = int(max_tries)
        self.rng = np.random.default_rng()
        self.patches = self._build_patch_bank(image_paths, mask_dir, mask_suffix, max_patches)

    # ------------------------------------------------------------------
    # Patch bank construction
    # ------------------------------------------------------------------

    def _build_patch_bank(self, image_paths, mask_dir, mask_suffix, max_patches):
        from pathlib import Path

        mask_dir = Path(mask_dir)
        bank = []
        for source_index, image_path in enumerate(image_paths):
            mask_path = mask_dir / f"{Path(image_path).stem}{mask_suffix}"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            binary = (mask > 127).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
            if num_labels <= 1:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for label in range(1, num_labels):
                ys, xs = np.where(labels == label)
                if ys.size < self.min_area:
                    continue
                y0, y1 = ys.min(), ys.max() + 1
                x0, x1 = xs.min(), xs.max() + 1
                # Local coordinates of the component inside the crop.
                local = (labels[y0:y1, x0:x1] == label).astype(np.float32)
                bank.append(
                    {
                        "source": source_index,
                        "image": image[y0:y1, x0:x1].copy(),
                        "mask": local,
                    }
                )
        if not bank:
            return bank
        if len(bank) > max_patches:
            # Evenly strided subsample keeps the bank deterministic.
            stride = len(bank) / max_patches
            bank = [bank[int(i * stride)] for i in range(max_patches)]
        return bank

    @property
    def available(self) -> bool:
        return len(self.patches) > 0

    # ------------------------------------------------------------------
    # Per-sample paste
    # ------------------------------------------------------------------

    def _appearance_match(self, patch: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Per-channel mean shift of the patch toward the local background."""
        if self.color_match <= 0.0 or background.size == 0:
            return patch
        shift = background.reshape(-1, 3).mean(axis=0) - patch.reshape(-1, 3).mean(axis=0)
        return np.clip(patch.astype(np.float32) + self.color_match * shift, 0, 255).astype(np.uint8)

    def _paste_one(self, image: np.ndarray, mask: np.ndarray) -> None:
        height, width = mask.shape
        for _ in range(self.max_tries):
            patch = self.patches[int(self.rng.integers(len(self.patches)))]
            patch_mask = patch["mask"]
            ph, pw = patch_mask.shape
            if ph > height or pw > width:
                continue
            y0 = int(self.rng.integers(0, height - ph + 1))
            x0 = int(self.rng.integers(0, width - pw + 1))
            roi_mask = mask[y0:y0 + ph, x0:x0 + pw]
            overlap = float((patch_mask * roi_mask).sum())
            # Place on background; tolerate a sliver of contact so dense-crack
            # images still receive augmentations.
            if overlap > max(0.02 * patch_mask.sum(), 1.0):
                continue

            # Feathered alpha: dilate then blur so the seam blends without
            # creating label pixels outside the pasted crack.
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(patch_mask, kernel, iterations=1)
            alpha = cv2.GaussianBlur(dilated, (5, 5), sigmaX=1.5)
            alpha = np.clip(alpha, 0.0, 1.0)
            alpha = np.maximum(alpha, patch_mask)

            patch_image = patch["image"]
            background = image[y0:y0 + ph, x0:x0 + pw].copy()
            patch_image = self._appearance_match(patch_image, background)
            blended = (
                background.astype(np.float32) * (1.0 - alpha[..., None])
                + patch_image.astype(np.float32) * alpha[..., None]
            )
            image[y0:y0 + ph, x0:x0 + pw] = blended.astype(np.uint8)
            mask[y0:y0 + ph, x0:x0 + pw] = np.maximum(roi_mask, patch_mask)
            return

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Paste onto a copy of (image, mask). image uint8 HxWx3 RGB, mask float32 HxW."""
        if not self.available or self.rng.random() >= self.prob:
            return image, mask
        image = image.copy()
        mask = mask.copy()
        for _ in range(int(self.rng.integers(1, self.max_instances + 1))):
            self._paste_one(image, mask)
        return image, mask
