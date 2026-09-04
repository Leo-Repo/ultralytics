from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.pose import PoseValidator

from .data import build_scrfd_dataset


class _SCRFDValidator:
    """Validate SCRFD with its native image padding and Ultralytics pose metrics."""

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_scrfd_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)


class SCRFDValidator(_SCRFDValidator, DetectionValidator):
    """Validate landmark-free SCRFD variants."""


class SCRFDPoseValidator(_SCRFDValidator, PoseValidator):
    """Validate SCRFD-KPS boxes and five landmarks."""
