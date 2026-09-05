from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ultralytics.utils.metrics import bbox_iou, box_iou
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.torch_utils import TORCH_1_11


def _decode_boxes(centers, distances):
    return torch.cat((centers - distances[:, :2], centers + distances[:, 2:]), 1)


def _reduce_mean(value):
    if dist.is_available() and dist.is_initialized():
        value = value.clone()
        dist.all_reduce(value)
        value /= dist.get_world_size()
    return value


class SCRFDCriterion:
    """Official SCRFD ATSS/QFL/DIoU/Smooth-L1 objective without MMDetection dependencies."""

    def __init__(self, model):
        self.strides = (8, 16, 32)
        self.base_sizes = (16, 64, 256)
        self.topk = 9
        self.use_kps = model.model[-1].head.use_kps

    def __call__(self, preds, batch):
        if len(preds) == 2 and isinstance(preds[0], torch.Tensor):
            preds = preds[1]
        cls_levels, box_levels = preds[:2]
        kps_levels = preds[2] if self.use_kps else None
        image_size = batch["img"].shape[2:]
        anchors = [
            self._anchors(x.shape[2:], base, stride, x)
            for x, base, stride in zip(cls_levels, self.base_sizes, self.strides)
        ]
        counts = [len(x) for x in anchors]
        flat_anchors = torch.cat(anchors)
        all_labels, all_boxes, all_kps, all_kps_weights = [], [], [], []
        total_pos = 0
        for batch_index in range(len(batch["img"])):
            mask = batch["batch_idx"].view(-1).long() == batch_index
            scale = batch["bboxes"].new_tensor((image_size[1], image_size[0], image_size[1], image_size[0]))
            classes = batch["cls"][mask].view(-1)
            boxes = xywh2xyxy(batch["bboxes"][mask] * scale)
            gt_boxes, ignored_boxes = boxes[classes >= 0], boxes[classes < 0]
            if self.use_kps:
                gt_kps = batch["keypoints"][mask][classes >= 0].clone()
                if len(gt_kps):
                    gt_kps[..., 0] *= image_size[1]
                    gt_kps[..., 1] *= image_size[0]
            assigned = self._assign(flat_anchors, counts, gt_boxes)
            positive = assigned >= 0
            labels = torch.ones(len(flat_anchors), dtype=torch.long, device=flat_anchors.device)
            labels[positive] = 0
            if len(ignored_boxes):
                centers = (flat_anchors[:, :2] + flat_anchors[:, 2:]) / 2
                ignored = (
                    torch.cat(
                        (centers[:, None] - ignored_boxes[None, :, :2], ignored_boxes[None, :, 2:] - centers[:, None]),
                        -1,
                    ).amin(-1)
                    > 0
                ).any(1)
                labels[ignored & ~positive] = -1
            targets = torch.zeros_like(flat_anchors)
            kps_targets = flat_anchors.new_zeros((len(flat_anchors), 10)) if self.use_kps else None
            kps_weights = flat_anchors.new_zeros((len(flat_anchors), 10)) if self.use_kps else None
            if positive.any():
                targets[positive] = gt_boxes[assigned[positive]]
                if self.use_kps:
                    selected_kps = gt_kps[assigned[positive]]
                    kps_targets[positive] = selected_kps[..., :2].flatten(1)
                    kps_weights[positive] = selected_kps[..., 2:].expand(-1, -1, 2).flatten(1)
            total_pos += int(positive.sum())
            all_labels.append(labels)
            all_boxes.append(targets)
            if self.use_kps:
                all_kps.append(kps_targets)
                all_kps_weights.append(kps_weights)

        labels = torch.stack(all_labels)
        box_targets = torch.stack(all_boxes)
        if self.use_kps:
            kps_targets = torch.stack(all_kps)
            kps_weights = torch.stack(all_kps_weights)
        num_pos = _reduce_mean(batch["img"].new_tensor(total_pos)).clamp(min=1)
        cls_loss = batch["img"].new_tensor(0.0)
        box_loss = batch["img"].new_tensor(0.0)
        kps_loss = batch["img"].new_tensor(0.0)
        weight_sum = batch["img"].new_tensor(0.0)
        offset = 0
        for level, (cls, box, anchor, count, stride) in enumerate(
            zip(cls_levels, box_levels, anchors, counts, self.strides)
        ):
            b = cls.shape[0]
            cls = cls.permute(0, 2, 3, 1).reshape(b, count, 1)
            box = box.permute(0, 2, 3, 1).reshape(b, count, 4)
            if self.use_kps:
                kps = kps_levels[level].permute(0, 2, 3, 1).reshape(b, count, 10)
            level_labels = labels[:, offset : offset + count]
            positive = level_labels == 0
            quality = torch.zeros_like(level_labels, dtype=cls.dtype)
            if positive.any():
                centers = ((anchor[:, :2] + anchor[:, 2:]) / 2 / stride).expand(b, -1, -1)[positive]
                decoded = _decode_boxes(centers, box[positive])
                target_boxes = box_targets[:, offset : offset + count][positive] / stride
                weights = cls.detach().sigmoid().squeeze(-1)[positive]
                quality[positive] = bbox_iou(decoded.detach(), target_boxes, xywh=False).squeeze(-1)
                box_loss += (
                    (1 - bbox_iou(decoded, target_boxes, xywh=False, DIoU=True).squeeze(-1)) * weights
                ).sum() * 2
                if self.use_kps:
                    target_kps = kps_targets[:, offset : offset + count][positive].reshape(-1, 5, 2) / stride
                    target_offsets = (target_kps - centers.unsqueeze(1)).flatten(1)
                    kp_weight = kps_weights[:, offset : offset + count][positive] * weights[:, None]
                    kps_loss += (
                        F.smooth_l1_loss(kps[positive], target_offsets, beta=1 / 9, reduction="none") * kp_weight
                    ).sum() * 0.1
                weight_sum += weights.sum()
            pred_sigmoid = cls.sigmoid().squeeze(-1)
            qfl = (
                F.binary_cross_entropy_with_logits(cls.squeeze(-1), torch.zeros_like(pred_sigmoid), reduction="none")
                * pred_sigmoid.square()
            )
            if positive.any():
                q = quality[positive]
                qfl[positive] = (
                    F.binary_cross_entropy_with_logits(cls.squeeze(-1)[positive], q, reduction="none")
                    * (q - pred_sigmoid[positive]).abs().square()
                )
            cls_loss += qfl[level_labels >= 0].sum() / num_pos
            offset += count
        normalizer = _reduce_mean(weight_sum).clamp(min=1e-6)
        losses = {"cls_loss": cls_loss, "box_loss": box_loss / normalizer}
        if self.use_kps:
            losses["kps_loss"] = kps_loss / normalizer
        return sum(losses.values()), losses

    @staticmethod
    def _anchors(shape, base_size, stride, like):
        h, w = shape
        sy = torch.arange(h, device=like.device, dtype=like.dtype)
        sx = torch.arange(w, device=like.device, dtype=like.dtype)
        y, x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
        centers = torch.stack((x, y), -1).reshape(-1, 1, 2) * stride
        sizes = like.new_tensor((base_size, base_size * 2)).reshape(1, 2, 1).expand(len(centers), -1, 2)
        return torch.cat((centers - sizes / 2, centers + sizes / 2), -1).reshape(-1, 4)

    @torch.no_grad()
    def _assign(self, anchors, counts, gt_boxes):
        assigned = torch.full((len(anchors),), -1, dtype=torch.long, device=anchors.device)
        if not len(gt_boxes):
            return assigned
        overlaps = box_iou(anchors, gt_boxes)
        centers = (anchors[:, :2] + anchors[:, 2:]) / 2
        gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        distances = (centers[:, None] - gt_centers).square().sum(-1).sqrt()
        candidates, start = [], 0
        for count in counts:
            candidates.append(
                distances[start : start + count].topk(min(self.topk, count), dim=0, largest=False).indices + start
            )
            start += count
        candidates = torch.cat(candidates)
        candidate_overlaps = overlaps[candidates, torch.arange(len(gt_boxes), device=anchors.device)]
        is_positive = candidate_overlaps >= candidate_overlaps.mean(0) + candidate_overlaps.std(0)
        candidate_centers = centers[candidates]
        inside_distance = torch.cat(
            (candidate_centers - gt_boxes[None, :, :2], gt_boxes[None, :, 2:] - candidate_centers), -1
        ).amin(-1)
        gt_scale = ((gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])).clamp(min=1e-4).sqrt()
        inside = inside_distance / gt_scale > 0.001
        is_positive &= inside
        match = overlaps.new_full(overlaps.shape, -1e8)
        rows = candidates[is_positive]
        cols = torch.arange(len(gt_boxes), device=anchors.device).expand_as(candidates)[is_positive]
        match[rows, cols] = overlaps[rows, cols]
        values, indices = match.max(1)
        positive = values > -1e8
        assigned[positive] = indices[positive]
        return assigned
