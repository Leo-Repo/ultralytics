from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ultralytics.engine.model import Model
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils import LOGGER

from .loss import SCRFDCriterion
from .modules import SCRFDNetwork


class SCRFDModel(BaseModel):
    """Ultralytics model wrapper for the official SCRFD architecture."""

    def __init__(self, cfg, ch=3, nc=1, data_kpt_shape=(5, 3), verbose=True):
        super().__init__()
        if ch != 3 or nc != 1:
            raise ValueError(f"SCRFD expects ch=3 and nc=1, received ch={ch}, nc={nc}")
        network = SCRFDNetwork(cfg)
        network.i, network.f, network.type = 0, -1, "SCRFDNetwork"
        network.np = sum(p.numel() for p in network.parameters())
        self.model = nn.Sequential(network)
        self.save = []
        self.yaml = cfg
        self.names = {0: "face"}
        self.nc = 1
        self.kpt_shape = list(data_kpt_shape)
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        self.inplace = True
        self.end2end = False
        if verbose:
            LOGGER.info(f"SCRFD summary: {network.np:,} parameters")

    def init_criterion(self):
        return SCRFDCriterion(self)


class SCRFD(Model):
    """SCRFD face detector using Ultralytics training, validation, prediction and export APIs.

    Examples:
        >>> from ultralytics import SCRFD
        >>> model = SCRFD("scrfd-500m-kps.yaml")
        >>> model.train(data="retinaface.yaml", imgsz=640, degrees=180)
    """

    def __init__(self, model: str | Path | Model = "scrfd-500m-kps.yaml", verbose: bool = False):
        super().__init__(model=model, task="pose", verbose=verbose)
        if hasattr(self.model, "model"):
            self.model.model[-1].kpt_shape = tuple(self.model.kpt_shape)
        self.overrides.update(
            imgsz=640,
            epochs=640,
            batch=16,
            optimizer="SGD",
            lr0=0.01,
            momentum=0.9,
            weight_decay=0.0005,
            mosaic=0.0,
            mixup=0.0,
            cutmix=0.0,
            copy_paste=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            conf=0.02,
            iou=0.45,
            max_det=10000,
            warmup_bias_lr=0.0,
            warmup_momentum=0.9,
            close_mosaic=0,
            patience=0,
            amp=False,
            hsv_h=0.05,
            hsv_s=0.5,
            hsv_v=0.125,
        )

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        from .predict import SCRFDPredictor
        from .train import SCRFDTrainer
        from .val import SCRFDValidator

        return {
            "pose": {
                "model": SCRFDModel,
                "trainer": SCRFDTrainer,
                "validator": SCRFDValidator,
                "predictor": SCRFDPredictor,
            }
        }
