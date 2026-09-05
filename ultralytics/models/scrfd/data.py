from __future__ import annotations

import random
from copy import copy, deepcopy

import cv2
import numpy as np

from ultralytics.data.augment import LetterBox, RandomPerspective
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import colorstr
from ultralytics.utils.instance import Instances


def rotate_face_sample(
    image,
    bboxes,
    keypoints=None,
    angle=0.0,
    output_shape=None,
    canvas_mode="fixed",
    border_mode="constant",
    border_value=114,
    drop_visible=0.3,
    keep_visible=0.6,
    min_face_size=2.0,
):
    """Rotate an image and synchronously transform HBB boxes and facial landmarks."""
    if canvas_mode not in {"fixed", "expand_letterbox"}:
        raise ValueError(f"Unsupported canvas_mode={canvas_mode}")
    if border_mode not in {"constant", "reflect"}:
        raise ValueError(f"Unsupported border_mode={border_mode}")
    if not 0 <= drop_visible <= keep_visible <= 1:
        raise ValueError("Expected 0 <= drop_visible <= keep_visible <= 1")

    height, width = image.shape[:2]
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2] = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    corners = np.array(((0, 0), (width, 0), (width, height), (0, height)), dtype=np.float32)
    rotated_corners = _transform_points(corners, matrix)
    canvas_width, canvas_height = width, height
    if canvas_mode == "expand_letterbox":
        low = np.floor(rotated_corners.min(0))
        high = np.ceil(rotated_corners.max(0))
        canvas_width, canvas_height = max(int(high[0] - low[0]), 1), max(int(high[1] - low[1]), 1)
        translate = np.eye(3, dtype=np.float32)
        translate[:2, 2] = -low
        matrix = translate @ matrix

    if output_shape is not None:
        out_height, out_width = output_shape
        gain = min(out_width / canvas_width, out_height / canvas_height)
        letterbox = np.array(
            (
                (gain, 0, (out_width - canvas_width * gain) / 2),
                (0, gain, (out_height - canvas_height * gain) / 2),
                (0, 0, 1),
            ),
            dtype=np.float32,
        )
        matrix = letterbox @ matrix
    else:
        out_height, out_width = canvas_height, canvas_width

    value = (border_value,) * image.shape[2] if isinstance(border_value, int) and image.ndim == 3 else border_value
    rotated = cv2.warpAffine(
        image,
        matrix[:2],
        (out_width, out_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101 if border_mode == "reflect" else cv2.BORDER_CONSTANT,
        borderValue=value,
    )

    bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    quads = bboxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)
    transformed_quads = _transform_points(quads.reshape(-1, 2), matrix).reshape(-1, 4, 2)
    transformed_boxes = np.zeros_like(bboxes)
    visible = np.zeros(len(bboxes), dtype=np.float32)
    frame = np.array(((0, 0), (out_width, 0), (out_width, out_height), (0, out_height)), dtype=np.float32)
    for i, quad in enumerate(transformed_quads):
        quad = np.ascontiguousarray(quad, dtype=np.float32)
        area = abs(cv2.contourArea(quad))
        intersection, polygon = cv2.intersectConvexConvex(quad, frame)
        if area > 0 and intersection > 0 and polygon is not None:
            visible[i] = min(intersection / area, 1.0)
            points = polygon.reshape(-1, 2)
            transformed_boxes[i] = (*points.min(0), *points.max(0))

    widths = transformed_boxes[:, 2] - transformed_boxes[:, 0]
    heights = transformed_boxes[:, 3] - transformed_boxes[:, 1]
    status = np.where(visible < drop_visible, 0, np.where(visible < keep_visible, 1, 2)).astype(np.uint8)
    status[(widths < min_face_size) | (heights < min_face_size)] = 0

    transformed_keypoints = None
    if keypoints is not None:
        transformed_keypoints = np.asarray(keypoints, dtype=np.float32).copy()
        valid = transformed_keypoints[..., 2] > 0
        transformed_keypoints[..., :2][~valid] = 0
        transformed_keypoints[..., :2][valid] = _transform_points(transformed_keypoints[..., :2][valid], matrix)
        xy = transformed_keypoints[..., :2]
        outside = valid & ((xy[..., 0] < 0) | (xy[..., 1] < 0) | (xy[..., 0] > out_width) | (xy[..., 1] > out_height))
        transformed_keypoints[..., 2][outside] = 0

    metadata = {
        "angle_deg": float(angle),
        "canvas_mode": canvas_mode,
        "matrix_total": matrix[:2],
        "quads": transformed_quads,
        "visible_ratio": visible,
        "status": status,
        "output_shape": (out_height, out_width),
    }
    return rotated, transformed_boxes, transformed_keypoints, metadata


def _transform_points(points, matrix):
    """Apply a homogeneous affine matrix to an array of xy points."""
    points = np.asarray(points, dtype=np.float32)
    homogeneous = np.concatenate((points, np.ones((len(points), 1), dtype=np.float32)), axis=1)
    return (homogeneous @ matrix.T)[:, :2]


class SCRFDRandomRoll:
    """Apply configurable full-angle roll augmentation before SCRFD crop and resize transforms."""

    def __init__(self, degrees, config):
        self.degrees = float(degrees)
        self.full_probability = float(config.get("full_probability", 0.8))
        self.small_degrees = float(config.get("small_degrees", 15.0))
        self.expand_probability = float(config.get("expand_probability", 0.3))
        self.border_modes = config.get("border_modes", ("constant", "reflect", "random"))
        self.border_value = int(config.get("border_value", 114))
        self.drop_visible = float(config.get("drop_visible", 0.3))
        self.keep_visible = float(config.get("keep_visible", 0.6))
        self.min_face_size = float(config.get("min_face_size", 2.0))
        if isinstance(self.border_modes, str):
            self.border_modes = (self.border_modes,)
        if not 0 <= self.full_probability <= 1 or not 0 <= self.expand_probability <= 1:
            raise ValueError("SCRFD rotation probabilities must be between 0 and 1")
        if not set(self.border_modes) <= {"constant", "reflect", "random"}:
            raise ValueError(f"Unsupported SCRFD border modes: {self.border_modes}")

    def __call__(self, labels):
        """Rotate a training sample and mark partially visible faces as ignored class -1."""
        if not self.degrees:
            return labels
        angle_range = self.degrees if random.random() < self.full_probability else min(self.small_degrees, self.degrees)
        angle = random.uniform(-angle_range, angle_range)
        canvas_mode = "expand_letterbox" if random.random() < self.expand_probability else "fixed"
        border_mode = random.choice(self.border_modes)
        border_value = tuple(random.randint(0, 255) for _ in range(3)) if border_mode == "random" else self.border_value
        border_mode = "constant" if border_mode == "random" else border_mode

        image, instances = labels["img"], labels["instances"]
        instances.convert_bbox("xyxy")
        instances.denormalize(image.shape[1], image.shape[0])
        rotated, boxes, keypoints, metadata = rotate_face_sample(
            image,
            instances.bboxes,
            instances.keypoints,
            angle=angle,
            canvas_mode=canvas_mode,
            border_mode=border_mode,
            border_value=border_value,
            drop_visible=self.drop_visible,
            keep_visible=self.keep_visible,
            min_face_size=self.min_face_size,
        )
        selected = metadata["status"] > 0
        cls = labels["cls"][selected].copy()
        cls[metadata["status"][selected] == 1] = -1
        segments = np.zeros((0, 1000, 2), dtype=np.float32)
        labels["img"] = rotated
        labels["instances"] = Instances(
            boxes[selected],
            segments,
            None if keypoints is None else keypoints[selected],
            bbox_format="xyxy",
            normalized=False,
        )
        labels["cls"] = cls
        labels["resized_shape"] = rotated.shape[:2]
        return labels


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
            affine_index = next(i for i, transform in enumerate(spatial) if isinstance(transform, RandomPerspective))
            spatial[affine_index].degrees = 0.0
            spatial.insert(affine_index, SCRFDRandomRoll(hyp.degrees, self.data.get("scrfd_rotation", {})))
            spatial.insert(affine_index + 1, SCRFDSquareCrop())
            spatial.insert(affine_index + 2, LetterBox((self.imgsz, self.imgsz), scale_fill=True))
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
        pad=0.0,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
    )
