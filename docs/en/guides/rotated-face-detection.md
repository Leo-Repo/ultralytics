---
comments: true
description: Train and deploy the sub-1M YOLO26 Face Lite oriented bounding box model.
keywords: YOLO26, OBB, rotated face detection, rectangular inference, ReLU
---

# Rotated face detection with YOLO26 Face Lite

`yolo26-face-obb.yaml` is a single-class OBB model for arbitrarily rotated faces. It keeps the YOLO26 OBB loss,
assigner, end-to-end head, and P3/P4/P5 feature pyramid, while replacing C2PSA with `C3k2` and using ReLU throughout
the feature network. Its input convention is RGB `(pixel - 128) / 128`.

Three size variants share the configuration:

| Model | Intended use | Approximate training parameters |
| --- | --- | ---: |
| `yolo26m-face-obb.yaml` | Accuracy-first | 0.86M |
| `yolo26s-face-obb.yaml` | Balanced | 0.67M |
| `yolo26n-face-obb.yaml` | Compact | 0.54M |

## Dataset

Use one class and the [Ultralytics OBB label format](../datasets/obb/index.md):

```text
0 x1 y1 x2 y2 x3 y3 x4 y4
```

Each coordinate is normalized by the source image width or height. Split video frames by person, home, or source
sequence before training to prevent leakage between train and validation sets.

## Train and validate

Training uses square `320×320` batches so the existing square augmentation pipeline remains available. Use
`degrees=180` for arbitrary in-plane rotation and inspect augmented labels before a full run.

```python
from ultralytics import YOLO

model = YOLO("yolo26m-face-obb.yaml")
model.train(data="face-obb.yaml", imgsz=320, rect=False, degrees=180, epochs=100)
```

The deployment input is `N×3×192×320` (height×width). Run a separate deployment-shape validation because square
training adds more vertical LetterBox padding and therefore does not guarantee identical accuracy at the rectangular
inference shape.

```python
model = YOLO("path/to/best.pt")
metrics = model.val(data="face-obb.yaml", imgsz=[192, 320], rect=True)
results = model.predict("image.jpg", imgsz=[192, 320])
```

Trainer, Validator, and Predictor all obtain the normalization mode from the model. A raw BGR OpenCV image is converted
to RGB before normalization. Tensor inputs passed directly to `predict()` must already be RGB floating-point tensors
using the same normalization convention.

## Export and audit ONNX

```python
onnx_path = model.export(format="onnx", imgsz=[192, 320], batch=1, dynamic=False)
```

The fixed ONNX input is `[1, 3, 192, 320]`. The exported metadata records `input_norm=minus128_div128`; consumers that
do not read Ultralytics metadata must apply this preprocessing themselves. Audit the exported graph with:

```bash
python tools/inspect_onnx_ops.py path/to/best.onnx
python tools/inspect_onnx_ops.py path/to/best.onnx --check --allow-sigmoid '^/model\.23/Sigmoid$'
```

The model has no attention Softmax. Sigmoid remains only in confidence decoding/post-processing and is intentionally
outside the feature-network restriction.
