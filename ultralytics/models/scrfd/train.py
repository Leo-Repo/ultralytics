from __future__ import annotations

from copy import copy
from pathlib import Path

from torch import optim

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils import RANK

from .data import build_scrfd_dataset
from .model import SCRFDModel
from .val import SCRFDPoseValidator, SCRFDValidator


class _SCRFDTrainer:
    """Shared SCRFD training behavior."""

    validator_cls = SCRFDValidator

    def get_model(self, cfg: str | Path | dict | None = None, weights=None, verbose=True):
        cfg = yaml_model_load(cfg) if isinstance(cfg, (str, Path)) else cfg
        model = self.set_model_names_for_load(
            SCRFDModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                data_kpt_shape=self.data.get("kpt_shape", (5, 3)),
                verbose=verbose and RANK == -1,
            )
        )
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        stride = max(int(self.model.stride.max()), 32)
        return build_scrfd_dataset(self.args, img_path, batch, self.data, mode, mode == "val", stride)

    def get_validator(self):
        return self.validator_cls(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def _setup_scheduler(self):
        self.lf = lambda epoch: 1.0 if epoch < self.epochs * 0.6875 else 0.1 if epoch < self.epochs * 0.85 else 0.01
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _get_warmup_iterations(self, num_batches):
        return min(1500, max((self.epochs - 1) * num_batches, 0))

    @staticmethod
    def build_optimizer(model, name="SGD", lr=0.01, momentum=0.9, decay=5e-4, iterations=1e5):
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)


class SCRFDTrainer(_SCRFDTrainer, DetectionTrainer):
    """Train landmark-free SCRFD variants with Ultralytics detection data and metrics."""


class SCRFDPoseTrainer(_SCRFDTrainer, PoseTrainer):
    """Train SCRFD-KPS variants with Ultralytics pose data and metrics."""

    validator_cls = SCRFDPoseValidator
