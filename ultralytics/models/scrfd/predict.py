from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.pose import PosePredictor


class SCRFDPredictor(PosePredictor):
    """SCRFD predictor using the official zero-valued letterbox padding."""

    def pre_transform(self, images):
        same_shapes = len({image.shape for image in images}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes
            and self.args.rect
            and (self.model.format == "pt" or (getattr(self.model, "dynamic", False) and self.model.format != "imx")),
            stride=self.model.stride,
            padding_value=0,
            center=False,
        )
        return [letterbox(image=image) for image in images]
