from ultralytics.models.yolo.pose import PoseValidator

from .data import build_scrfd_dataset


class SCRFDValidator(PoseValidator):
    """Validate SCRFD with its native image padding and Ultralytics pose metrics."""

    def build_dataset(self, img_path, mode="val", batch=None):
        return build_scrfd_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)
