from __future__ import annotations

import random
from copy import copy, deepcopy

import numpy as np

from ultralytics.data.augment import LetterBox, RandomPerspective
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import colorstr


class SCRFDSquareCrop:
    """Apply the random padded square crop from the official SCRFD training pipeline."""

    choices = (0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)

    def __call__(self, labels):
        image, instances = labels["img"], deepcopy(labels["instances"])
        height, width = image.shape[:2]
        instances.convert_bbox("xyxy")
        instances.denormalize(width, height)
        centers = (instances.bboxes[:, :2] + instances.bboxes[:, 2:]) / 2
        if not len(centers):
            return labels

        for _ in range(2500):
            size = int(random.choice(self.choices) * min(width, height))
            left = random.randint(min(0, width - size), max(0, width - size))
            top = random.randint(min(0, height - size), max(0, height - size))
            keep = (
                (centers[:, 0] > left)
                & (centers[:, 1] > top)
                & (centers[:, 0] < left + size)
                & (centers[:, 1] < top + size)
            )
            if keep.any():
                break
        else:
            return labels

        cropped = np.full((size, size, image.shape[2]), 128, dtype=image.dtype)
        src_x1, src_y1 = max(left, 0), max(top, 0)
        src_x2, src_y2 = min(left + size, width), min(top + size, height)
        dst_x1, dst_y1 = src_x1 - left, src_y1 - top
        cropped[dst_y1 : dst_y1 + src_y2 - src_y1, dst_x1 : dst_x1 + src_x2 - src_x1] = image[
            src_y1:src_y2, src_x1:src_x2
        ]
        instances = instances[keep]
        instances.add_padding(-left, -top)
        instances.clip(size, size)
        labels["img"] = cropped
        labels["instances"] = instances
        labels["cls"] = labels["cls"][keep]
        labels["resized_shape"] = cropped.shape[:2]
        return labels


class SCRFDDataset(YOLODataset):
    """Ultralytics pose dataset with SCRFD-compatible spatial preprocessing."""

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if self.augment:
            spatial = transforms[0]
            spatial.insert(0, SCRFDSquareCrop())
            affine_index = next(i for i, transform in enumerate(spatial) if isinstance(transform, RandomPerspective))
            spatial.insert(affine_index, LetterBox((self.imgsz, self.imgsz), scale_fill=True))
        else:
            transforms[0].padding_value = 0
            transforms[0].center = False
        return transforms


def build_scrfd_dataset(cfg, img_path, batch, data, mode="train", rect=False, stride=32):
    """Build an SCRFD train or validation dataset."""
    return SCRFDDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=copy(cfg),
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )
