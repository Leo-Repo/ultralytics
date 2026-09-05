# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Generate or evaluate deterministic fixed-angle SCRFD validation datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def generate(args):
    """Generate a fixed-angle validation dataset."""
    from ultralytics.data.converter import generate_rotated_face_val

    return generate_rotated_face_val(
        args.data,
        args.output,
        split=args.split,
        angles=args.angles,
        imgsz=tuple(args.imgsz),
        canvas_mode=args.canvas_mode,
        border_mode=args.border_mode,
        border_value=args.border_value,
        drop_visible=args.drop_visible,
        keep_visible=args.keep_visible,
        min_face_size=args.min_face_size,
        existing=args.existing,
        previews=args.previews,
    )


def evaluate(args):
    """Evaluate one SCRFD checkpoint on every generated angle and write CSV/JSON reports."""
    from ultralytics import SCRFD

    data_root, report_dir = Path(args.data), Path(args.output)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((data_root / "rotation_summary.json").read_text())
    counts = {float(item["angle_deg"]): item for item in summary["angles"]}
    model, rows = SCRFD(args.model), []
    for data_yaml in sorted(data_root.glob("angle_*.yaml")):
        angle = float(data_yaml.stem.removeprefix("angle_"))
        validation = model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=False,
            project=str(report_dir / "runs"),
            name=data_yaml.stem,
            exist_ok=True,
            verbose=False,
        )
        metrics = validation.results_dict
        small_recall = _small_face_recall(
            model,
            data_root / "images" / "val_rotation" / data_yaml.stem,
            data_root / "labels" / "val_rotation" / data_yaml.stem,
            args,
        )
        rows.append(
            {
                "angle_deg": angle,
                "images": counts[angle]["images"],
                "faces": counts[angle]["kept"],
                "precision": metrics["metrics/precision(B)"],
                "recall": metrics["metrics/recall(B)"],
                "ap50": metrics["metrics/mAP50(B)"],
                "ap75": float(validation.box.all_ap[:, 5].mean()),
                "map50_95": metrics["metrics/mAP50-95(B)"],
                "small_recall": small_recall,
            }
        )
    overall_metrics = model.val(
        data=str(data_root / "rotation-val.yaml"),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        project=str(report_dir / "runs"),
        name="overall",
        exist_ok=True,
        verbose=False,
    ).results_dict
    recalls = [row["recall"] for row in rows]
    report = {
        "model": str(Path(args.model).resolve()),
        "data": str(data_root.resolve()),
        "angles": rows,
        "overall": {
            "precision": overall_metrics["metrics/precision(B)"],
            "recall": overall_metrics["metrics/recall(B)"],
            "ap50": overall_metrics["metrics/mAP50(B)"],
            "map50_95": overall_metrics["metrics/mAP50-95(B)"],
            "worst_angle_recall": min(recalls),
            "recall_range": max(recalls) - min(recalls),
        },
    }
    with (report_dir / "per-angle.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["overall"], indent=2))
    return report


def _small_face_recall(model, image_root, label_root, args):
    """Return greedy IoU-0.5 recall for GT faces smaller than 32×32 pixels."""
    import torch

    from ultralytics.utils.metrics import box_iou

    matched_small = total_small = 0
    predictions = model.predict(
        source=str(image_root),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.45,
        max_det=10000,
        device=args.device,
        stream=True,
        verbose=False,
    )
    for result in predictions:
        label_file = label_root / Path(result.path).with_suffix(".txt").name
        rows = [line.split() for line in label_file.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        height, width = result.orig_shape
        xywh = torch.tensor([[float(value) for value in row[1:5]] for row in rows])
        xywh *= xywh.new_tensor((width, height, width, height))
        gt = torch.column_stack(
            (
                xywh[:, 0] - xywh[:, 2] / 2,
                xywh[:, 1] - xywh[:, 3] / 2,
                xywh[:, 0] + xywh[:, 2] / 2,
                xywh[:, 1] + xywh[:, 3] / 2,
            )
        )
        small = xywh[:, 2] * xywh[:, 3] < 32**2
        total_small += int(small.sum())
        predictions_xyxy = result.boxes.xyxy.cpu()
        if not len(predictions_xyxy):
            continue
        iou = box_iou(predictions_xyxy, gt)
        matched = torch.zeros(len(gt), dtype=torch.bool)
        while iou.numel() and iou.max() >= 0.5:
            flat = int(iou.argmax())
            prediction_index, gt_index = divmod(flat, len(gt))
            matched[gt_index] = True
            iou[prediction_index] = -1
            iou[:, gt_index] = -1
        matched_small += int((matched & small).sum())
    return matched_small / total_small if total_small else ""


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generator = commands.add_parser("generate")
    generator.add_argument("--data", required=True)
    generator.add_argument("--output", required=True)
    generator.add_argument("--split", default="val")
    generator.add_argument("--angles", type=int, nargs="+", default=list(range(0, 360, 30)))
    generator.add_argument("--imgsz", type=int, nargs=2, default=(640, 640), metavar=("HEIGHT", "WIDTH"))
    generator.add_argument("--canvas-mode", choices=("fixed", "expand_letterbox"), default="fixed")
    generator.add_argument("--border-mode", choices=("constant", "reflect"), default="constant")
    generator.add_argument("--border-value", type=int, default=114)
    generator.add_argument("--drop-visible", type=float, default=0.3)
    generator.add_argument("--keep-visible", type=float, default=0.6)
    generator.add_argument("--min-face-size", type=float, default=2.0)
    generator.add_argument("--existing", choices=("error", "overwrite", "skip", "verify"), default="error")
    generator.add_argument("--previews", type=int, default=0, help="annotated geometry previews per angle")

    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("--model", required=True)
    evaluator.add_argument("--data", required=True)
    evaluator.add_argument("--output", required=True)
    evaluator.add_argument("--imgsz", type=int, default=640)
    evaluator.add_argument("--batch", type=int, default=16)
    evaluator.add_argument("--device", default="")
    evaluator.add_argument("--workers", type=int, default=8)
    evaluator.add_argument("--conf", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    (generate if arguments.command == "generate" else evaluate)(arguments)
