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

Training uses the deployment shape `320×192` (width×height). Use `degrees=180` for arbitrary in-plane rotation and
inspect augmented labels for clipping before a full run.

```python
from ultralytics import YOLO

model = YOLO("yolo26m-face-obb.yaml")
model.train(data="face-obb.yaml", imgsz=[192, 320], rect=False, degrees=180, epochs=100)
```

At `192×320`, the P3/P4/P5 grids contain 1,260 locations (`24×40 + 12×20 + 6×10`), compared with 2,100 at
`320×320`. Rectangular training therefore uses approximately 60% of the convolutional work and activation memory while
keeping the parameter count unchanged. It also matches deployment preprocessing and reduces padding for landscape
sources. The tradeoff is increased clipping during large-angle rotation augmentation and smaller faces when squarer or
portrait source images are fitted into the landscape canvas. Include those source types in validation and inspect
recall by face size and rotation angle.

The deployment input is `N×3×192×320` (height×width). Validate at the same fixed shape:

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
onnx_path = model.export(format="onnx", imgsz=[192, 320], batch=1, dynamic=False, opset=11)
```

The model YAML also defaults ONNX-based export to opset 11. The fixed ONNX input is `[1, 3, 192, 320]`. The exported
metadata records `input_norm=minus128_div128`; consumers that do not read Ultralytics metadata must apply this
preprocessing themselves. Audit the exported graph with:

```bash
python tools/inspect_onnx_ops.py path/to/best.onnx
python tools/inspect_onnx_ops.py path/to/best.onnx --check --allow-sigmoid '^/model\.23/Sigmoid$'
```

The model has no attention Softmax. Sigmoid remains only in confidence decoding/post-processing and is intentionally
outside the feature-network restriction.

## SCRFD HBB and five-landmark path

SCRFD provides a second route for 360-degree face detection. It keeps horizontal bounding boxes and standard NMS while
using five facial landmarks as auxiliary supervision. Rotation changes only the input and labels; it does not add an
angle head or rotated NMS.

Enable online full-angle training with `degrees=180`. Set `degrees=0` to disable roll rotation. The image, all four HBB
corners, and valid landmarks share one affine matrix. The transformed corners produce a new enclosing HBB, landmarks
outside the canvas lose visibility, and partially visible faces are excluded from both positive and negative SCRFD
classification targets.

```python
from ultralytics import SCRFD

model = SCRFD("scrfd-500m-kps.yaml")
model.train(data="retinaface.yaml", imgsz=640, degrees=180, epochs=640)
```

The default online distribution uses 80% samples from the full configured range and 20% from ±15 degrees. It uses a
70/30 mixture of fixed-canvas and expanded-canvas rotation. Configure these dataset-owned settings in the data YAML:

```yaml
scrfd_rotation:
    full_probability: 0.8
    small_degrees: 15.0
    expand_probability: 0.3
    border_modes: [constant, reflect, random]
    border_value: 114
    drop_visible: 0.3
    keep_visible: 0.6
    min_face_size: 2.0
```

Training randomness follows the Ultralytics `seed` and `deterministic` settings. Rotation is applied before the native
SCRFD random square crop, resize, and official photometric distortion. Other SCRFD augmentation remains active when
`degrees=0`.

Five-column `labelv2.txt` rows whose final value is `1` retain their source ignore state. Rotation additionally marks a
face ignored when `drop_visible <= visible_ratio < keep_visible`; smaller visible ratios and boxes below
`min_face_size` are dropped. A landmark outside the canvas only loses its own visibility and does not by itself ignore
the whole face. Random square crop drops faces whose centers lie outside the selected crop instead of marking them
ignored.

### Deterministic rotation validation

Keep the native validation split unchanged, then generate a separate fixed-angle validation dataset. Generate the
default twelve 30-degree buckets at 640×640 with:

```bash
python tools/scrfd_rotation_val.py generate \
    --data retinaface.yaml \
    --output datasets/retinaface-rotation-val \
    --imgsz 640 640
```

The output includes synchronized Ultralytics HBB/KPS labels, one JSONL manifest and YAML per angle, an aggregate YAML,
and filtering statistics. Ignored targets are counted in the manifest but omitted from positive validation GT. Re-run
with `--existing verify` to byte-check deterministic images, labels, and manifests. Use
`--canvas-mode expand_letterbox` for the alternate canvas strategy.

Evaluate native and rotated validation separately. The angle evaluator reports each angle, the aggregate metrics,
worst-angle recall, and recall range:

```bash
python tools/scrfd_rotation_val.py evaluate \
    --model path/to/best.pt \
    --data datasets/retinaface-rotation-val \
    --output runs/scrfd-rotation-val \
    --imgsz 640
```
