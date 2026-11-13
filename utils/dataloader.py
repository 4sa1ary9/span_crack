from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.copy_paste import CrackPasteAugmenter, normalize_copy_paste_config


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".png")
DEFAULT_IMAGE_SIZE = 512


class CrackDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir=None,
        transform=None,
        mode: str = "train",
        mask_suffix: str = ".png",
        image_extensions=None,
        copy_paste: dict | None = None,
        weak_transform=None,
        include_skeleton_recall: bool = False,
    ):
        if mode not in {"train", "eval", "infer"}:
            raise ValueError(f"Unsupported dataset mode: {mode}")

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.transform = transform
        self.mode = mode
        self.include_skeleton_recall = include_skeleton_recall and mode == "train"
        self.mask_suffix = mask_suffix
        self.image_extensions = tuple(
            extension.lower() for extension in (image_extensions or SUPPORTED_IMAGE_EXTENSIONS)
        )
        self.images = [
            image_name
            for image_name in sorted(self.image_dir.iterdir())
            if image_name.is_file() and image_name.suffix.lower() in self.image_extensions
        ]

        if self.mode == "train":
            if self.mask_dir is None:
                raise ValueError("Training dataset requires mask_dir.")
        elif self.mode == "eval" and self.mask_dir is None:
            raise ValueError("Evaluation dataset requires mask_dir.")

        # Copy-paste only makes sense in training; the patch bank is built once
        # here, before DataLoader workers fork, so it is shared copy-on-write.
        copy_paste_config = normalize_copy_paste_config(copy_paste) if mode == "train" else None
        self.paste_augmenter = None
        if copy_paste_config is not None:
            self.paste_augmenter = CrackPasteAugmenter(
                [self.image_dir / name for name in self.images],
                self.mask_dir,
                mask_suffix=mask_suffix,
                prob=copy_paste_config["prob"],
                max_instances=copy_paste_config["max_instances"],
                min_area=copy_paste_config["min_area"],
                max_patches=copy_paste_config["max_patches"],
                color_match=copy_paste_config["color_match"],
                max_tries=copy_paste_config["max_tries"],
            )
        # Mean-teacher two-view mode: `transform` is the strong view, this is
        # the weak view. Both see the same (possibly pasted) raw image.
        self.weak_transform = weak_transform if mode == "train" else None

    def __len__(self):
        return len(self.images)

    def _attach_skeleton_recall(self, sample):
        if self.include_skeleton_recall:
            import torch
            from utils.publication_losses import skeleton_recall_target
            tube = skeleton_recall_target(sample["mask"][0].numpy())
            sample["skeleton_recall_target"] = torch.from_numpy(tube).unsqueeze(0)
        return sample

    def _read_binary_mask(self, path: Path):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read binary target: {path}")
        return (mask > 127).astype("float32")

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        sample = {
            "image_name": image_path.name,
        }

        if self.mode == "infer":
            if self.transform:
                image = self.transform(image=image)["image"]
            sample["image"] = image
            return sample

        mask_path = self.mask_dir / f"{image_path.stem}{self.mask_suffix}"
        mask = self._read_binary_mask(mask_path)

        if self.mode == "train":
            if self.paste_augmenter is not None:
                image, mask = self.paste_augmenter(image, mask)
            if self.weak_transform is not None:
                # Shared-seed two views: both pipelines draw identical
                # geometric randomness so weak/strong stay pixel-aligned;
                # only image-only (photometric) extras may differ.
                view_seed = int(np.random.randint(0, 2**31 - 1))
                state = random.getstate()
                np_state = np.random.get_state()
                random.seed(view_seed)
                np.random.seed(view_seed)
                weak = self.weak_transform(image=image, mask=mask)
                random.seed(view_seed)
                np.random.seed(view_seed)
                strong = self.transform(image=image, mask=mask)
                random.setstate(state)
                np.random.set_state(np_state)
                sample["image"] = strong["image"]
                sample["mask"] = strong["mask"].unsqueeze(0)
                sample["image_weak"] = weak["image"]
                return self._attach_skeleton_recall(sample)
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            sample["image"] = image
            sample["mask"] = mask.unsqueeze(0)
            return self._attach_skeleton_recall(sample)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        sample["image"] = image
        sample["mask"] = mask.unsqueeze(0)
        return sample


def get_transforms(mode: str = "eval", augmentation_profile: str = "standard"):
    transforms = []
    if mode == "train":
        if augmentation_profile not in {
            "standard", "crackmap_strong", "strong", "medium", "std_photometric",
        }:
            raise ValueError(f"Unsupported augmentation_profile: {augmentation_profile}")
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            ]
        )
        if augmentation_profile == "std_photometric":
            # Identical geometry to `standard` (same transforms, same draw
            # order); adds image-only photometric jitter afterwards. Used as
            # the strong view for mean-teacher consistency so weak/strong
            # stay pixel-aligned.
            transforms.extend(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
                ]
            )
        if augmentation_profile in {"crackmap_strong", "strong"}:
            transforms.extend(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.15,
                        rotate_limit=10,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=0.5,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=0.5,
                    ),
                ]
            )
        if augmentation_profile == "medium":
            # Mild multi-scale crop only. Photometric/blur transforms are
            # deliberately excluded: strong-profile scouts showed they destroy
            # thin-crack evidence (DeepCrack best val -6.5 vs baseline).
            transforms.extend(
                [
                    A.RandomScale(
                        scale_limit=(0.85, 1.15),
                        interpolation=cv2.INTER_LINEAR,
                        p=0.5,
                    ),
                    A.PadIfNeeded(
                        min_height=DEFAULT_IMAGE_SIZE,
                        min_width=DEFAULT_IMAGE_SIZE,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=1.0,
                    ),
                    A.RandomCrop(height=DEFAULT_IMAGE_SIZE, width=DEFAULT_IMAGE_SIZE, p=1.0),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.15,
                        contrast_limit=0.15,
                        p=0.3,
                    ),
                ]
            )
        if augmentation_profile == "strong":
            # Geometric: multi-scale training with a random 512x512 crop. The
            # final Resize is a no-op for this branch but keeps the two paths
            # in one Compose.
            transforms.extend(
                [
                    A.RandomScale(
                        scale_limit=(0.65, 1.35),
                        interpolation=cv2.INTER_LINEAR,
                        p=0.5,
                    ),
                    A.PadIfNeeded(
                        min_height=DEFAULT_IMAGE_SIZE,
                        min_width=DEFAULT_IMAGE_SIZE,
                        border_mode=cv2.BORDER_REFLECT_101,
                        p=1.0,
                    ),
                    A.RandomCrop(height=DEFAULT_IMAGE_SIZE, width=DEFAULT_IMAGE_SIZE, p=1.0),
                    # Photometric: pavement illumination variability.
                    A.RandomBrightnessContrast(
                        brightness_limit=0.25,
                        contrast_limit=0.25,
                        p=0.5,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=20,
                        val_shift_limit=20,
                        p=0.3,
                    ),
                    A.OneOf(
                        [
                            A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
                            A.GaussianBlur(sigma_limit=(0.5, 2.0), p=1.0),
                            A.MotionBlur(blur_limit=(3, 7), p=1.0),
                        ],
                        p=0.3,
                    ),
                ]
            )
    transforms.extend(
        [
            A.Resize(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)
