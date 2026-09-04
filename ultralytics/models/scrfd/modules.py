from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.utils.torch_utils import TORCH_1_11


class ConvNormAct(nn.Sequential):
    """Convolution followed by optional normalization and ReLU."""

    def __init__(self, c1, c2, k=3, s=1, groups=1, norm=None, act=True):
        padding = k // 2
        layers = [nn.Conv2d(c1, c2, k, s, padding, groups=groups, bias=norm is None)]
        if norm == "BN":
            layers.append(nn.BatchNorm2d(c2))
        elif norm:
            layers.append(nn.GroupNorm(int(norm), c2))
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class DepthwiseSeparableConv(nn.Sequential):
    """MMCV-compatible depthwise separable convolution block."""

    def __init__(self, c1, c2, norm, stride=1):
        super().__init__(
            ConvNormAct(c1, c1, s=stride, groups=c1, norm=norm),
            ConvNormAct(c1, c2, k=1, norm=norm),
        )


class MobileNetV1(nn.Module):
    """SCRFD MobileNetV1 backbone."""

    def __init__(self, stage_blocks, stage_planes):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(3, stage_planes[0], s=2, norm="BN"),
            DepthwiseSeparableConv(stage_planes[0], stage_planes[1], "BN"),
        )
        self.stages = nn.ModuleList()
        for i, repeats in enumerate(stage_blocks):
            blocks = [DepthwiseSeparableConv(stage_planes[i + 1], stage_planes[i + 2], "BN", stride=2)]
            blocks.extend(
                DepthwiseSeparableConv(stage_planes[i + 2], stage_planes[i + 2], "BN") for _ in range(repeats - 1)
            )
            self.stages.append(nn.Sequential(*blocks))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


class BasicBlock(nn.Module):
    """Basic residual block used by the SCRFD ResNetV1e backbone."""

    def __init__(self, c1, c2, stride=1):
        super().__init__()
        self.conv1 = ConvNormAct(c1, c2, s=stride, norm="BN")
        self.conv2 = ConvNormAct(c2, c2, norm="BN", act=False)
        if stride != 1 or c1 != c2:
            downsample = [nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False)] if stride != 1 else []
            downsample.extend((nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2)))
            self.downsample = nn.Sequential(*downsample)
        else:
            self.downsample = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.downsample(x))


class Bottleneck(nn.Module):
    """Bottleneck residual block used by SCRFD-34G."""

    expansion = 4

    def __init__(self, c1, planes, stride=1):
        super().__init__()
        c2 = planes * self.expansion
        self.conv1 = ConvNormAct(c1, planes, k=1, norm="BN")
        self.conv2 = ConvNormAct(planes, planes, s=stride, norm="BN")
        self.conv3 = ConvNormAct(planes, c2, k=1, norm="BN", act=False)
        if stride != 1 or c1 != c2:
            downsample = [nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False)] if stride != 1 else []
            downsample.extend((nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2)))
            self.downsample = nn.Sequential(*downsample)
        else:
            self.downsample = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv3(self.conv2(self.conv1(x))) + self.downsample(x))


class ResNetV1e(nn.Module):
    """SCRFD ResNetV1e backbone with a deep stem and average-pool downsampling."""

    def __init__(self, stage_blocks, stage_planes, base_channels, block="BasicBlock"):
        super().__init__()
        stem_half = base_channels // 2
        self.stem = nn.Sequential(
            ConvNormAct(3, stem_half, s=2, norm="BN"),
            ConvNormAct(stem_half, stem_half, norm="BN"),
            ConvNormAct(stem_half, base_channels, norm="BN"),
        )
        self.maxpool = nn.MaxPool2d(2, 2)
        self.stages = nn.ModuleList()
        block_type = BasicBlock if block == "BasicBlock" else Bottleneck
        channels = base_channels
        for i, (repeats, planes) in enumerate(zip(stage_blocks, stage_planes)):
            stride = 1 if i == 0 else 2
            blocks = [block_type(channels, planes, stride)]
            channels = planes * block_type.expansion if block_type is Bottleneck else planes
            blocks.extend(block_type(channels, planes) for _ in range(repeats - 1))
            self.stages.append(nn.Sequential(*blocks))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.modules():
            if isinstance(module, BasicBlock):
                nn.init.zeros_(module.conv2[1].weight)
            elif isinstance(module, Bottleneck):
                nn.init.zeros_(module.conv3[1].weight)

    def forward(self, x):
        outputs = []
        x = self.maxpool(self.stem(x))
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


class PAFPN(nn.Module):
    """Path aggregation FPN used by the official SCRFD implementation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        in_channels = in_channels[1:]
        self.lateral = nn.ModuleList(ConvNormAct(c, out_channels, k=1, act=False) for c in in_channels)
        self.fpn = nn.ModuleList(ConvNormAct(out_channels, out_channels, act=False) for _ in in_channels)
        self.downsample = nn.ModuleList(
            ConvNormAct(out_channels, out_channels, s=2, act=False) for _ in range(len(in_channels) - 1)
        )
        self.pan = nn.ModuleList(
            ConvNormAct(out_channels, out_channels, act=False) for _ in range(len(in_channels) - 1)
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        laterals = [layer(x) for layer, x in zip(self.lateral, inputs[1:])]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="nearest"
            )
        outputs = [layer(x) for layer, x in zip(self.fpn, laterals)]
        for i in range(len(outputs) - 1):
            outputs[i + 1] = outputs[i + 1] + self.downsample[i](outputs[i])
        return [outputs[0], *(layer(x) for layer, x in zip(self.pan, outputs[1:]))]


class Scale(nn.Module):
    """Learnable scalar used by the official SCRFD regression head."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.scale


class SCRFDHead(nn.Module):
    """Anchor-based SCRFD classification, box and five-landmark head."""

    def __init__(
        self,
        channels,
        feat_channels,
        stacked_convs=2,
        norm="BN",
        dw_conv=False,
        strides_share=False,
        scale_mode=0,
        use_kps=True,
    ):
        super().__init__()
        self.register_buffer("stride", torch.tensor([8.0, 16.0, 32.0]), persistent=False)
        self.export = False
        self.strides_share = strides_share
        self.use_kps = use_kps
        count = 1 if strides_share else 3
        conv = DepthwiseSeparableConv if dw_conv else ConvNormAct
        self.towers = nn.ModuleList(
            nn.Sequential(
                *[conv(channels if i == 0 else feat_channels, feat_channels, norm=norm) for i in range(stacked_convs)]
            )
            for _ in range(count)
        )
        self.cls = nn.ModuleList(nn.Conv2d(feat_channels, 2, 3, padding=1) for _ in range(count))
        self.box = nn.ModuleList(nn.Conv2d(feat_channels, 8, 3, padding=1) for _ in range(count))
        self.kps = nn.ModuleList(nn.Conv2d(feat_channels, 20, 3, padding=1) for _ in range(count)) if use_kps else None
        self.scales = (
            nn.ModuleList(Scale() for _ in range(3)) if scale_mode and (strides_share or scale_mode == 2) else None
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for module in self.cls:
            nn.init.constant_(module.bias, -4.595)

    def forward(self, features):
        cls_scores, box_preds, kps_preds = [], [], []
        for i, x in enumerate(features):
            j = 0 if self.strides_share else i
            x = self.towers[j](x)
            cls_scores.append(self.cls[j](x))
            box = self.box[j](x)
            box_preds.append(self.scales[i](box) if self.scales is not None else box)
            if self.use_kps:
                kps_preds.append(self.kps[j](x))
        raw = (cls_scores, box_preds, kps_preds) if self.use_kps else (cls_scores, box_preds)
        if self.training:
            return raw
        decoded = self.decode(*raw)
        return decoded if self.export else (decoded, raw)

    def decode(self, cls_scores, box_preds, kps_preds=None):
        boxes, scores, keypoints = [], [], []
        for i, (cls, box, stride) in enumerate(zip(cls_scores, box_preds, self.stride)):
            b, _, h, w = cls.shape
            sy = torch.arange(h, device=cls.device, dtype=cls.dtype)
            sx = torch.arange(w, device=cls.device, dtype=cls.dtype)
            y, x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
            centers = torch.stack((x, y), -1).reshape(1, -1, 1, 2) * stride
            centers = centers.expand(b, -1, 2, -1).reshape(b, -1, 2)
            distances = box.permute(0, 2, 3, 1).reshape(b, -1, 4) * stride
            xyxy = torch.cat((centers - distances[..., :2], centers + distances[..., 2:]), -1)
            boxes.append(torch.cat(((xyxy[..., :2] + xyxy[..., 2:]) / 2, xyxy[..., 2:] - xyxy[..., :2]), -1))
            scores.append(cls.permute(0, 2, 3, 1).reshape(b, -1, 1).sigmoid())
            if self.use_kps:
                offsets = kps_preds[i].permute(0, 2, 3, 1).reshape(b, -1, 5, 2) * stride
                xy = centers.unsqueeze(2) + offsets
                keypoints.append(torch.cat((xy, torch.ones_like(xy[..., :1])), -1).flatten(2))
        output = (
            (torch.cat(boxes, 1), torch.cat(scores, 1), torch.cat(keypoints, 1))
            if self.use_kps
            else (torch.cat(boxes, 1), torch.cat(scores, 1))
        )
        return torch.cat(output, -1).transpose(1, 2)


class SCRFDNetwork(nn.Module):
    """SCRFD-500M-KPS network with official pixel normalization."""

    def __init__(self, cfg):
        super().__init__()
        self.kpt_shape = (5, 3)
        backbone, neck, head = cfg["backbone"], cfg["neck"], cfg["head"]
        mobile = backbone.get("type", "MobileNetV1") == "MobileNetV1"
        if mobile:
            self.backbone = MobileNetV1(backbone["stage_blocks"], backbone["stage_planes"])
        else:
            self.backbone = ResNetV1e(
                backbone["stage_blocks"],
                backbone["stage_planes"],
                backbone["base_channels"],
                backbone.get("block", "BasicBlock"),
            )
        self.neck = PAFPN(neck["in_channels"], neck["out_channels"])
        self.head = SCRFDHead(
            neck["out_channels"],
            head["feat_channels"],
            head["stacked_convs"],
            head["norm"],
            head.get("dw_conv", mobile),
            head["strides_share"],
            head["scale_mode"],
            head.get("use_kps", "kpt_shape" in cfg),
        )

    def forward(self, x):
        x = (x * 255.0 - 127.5) / 128.0
        return self.head(self.neck(self.backbone(x)))
