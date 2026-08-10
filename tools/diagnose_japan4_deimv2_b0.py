"""Run a reproducible Val-only error diagnosis for the frozen Japan4 DEIMv2 B0.

The script deliberately never accepts a Test annotation path. It reproduces the
official COCO evaluation, saves capped Val predictions, and adds an exploratory
IoU=0.50 operating-point audit for FP/FN review. Thresholds optimized on Val are
diagnostic only and must not be reported as locked Test operating points.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

try:
    import resource
except ImportError:  # Windows can still import/review generated artifacts.
    resource = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig  # noqa: E402
from engine.misc import dist_utils  # noqa: E402
from engine.solver import TASKS  # noqa: E402


COCO_STAT_NAMES = (
    "AP50_95",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raise_nofile_limit(target: int = 65535) -> dict[str, int | bool]:
    if resource is None:
        return {"supported": False, "before": -1, "after": -1, "hard": -1}
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(max(soft, target), hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    return {"supported": True, "before": int(soft), "after": int(new_soft), "hard": int(hard)}


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def box_iou(box: list[float], boxes: list[list[float]]) -> np.ndarray:
    if not boxes:
        return np.empty((0,), dtype=np.float64)
    lhs = np.asarray(box, dtype=np.float64)
    rhs = np.asarray(boxes, dtype=np.float64)
    inter_x1 = np.maximum(lhs[0], rhs[:, 0])
    inter_y1 = np.maximum(lhs[1], rhs[:, 1])
    inter_x2 = np.minimum(lhs[2], rhs[:, 2])
    inter_y2 = np.minimum(lhs[3], rhs[:, 3])
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    lhs_area = max(0.0, lhs[2] - lhs[0]) * max(0.0, lhs[3] - lhs[1])
    rhs_area = np.maximum(0.0, rhs[:, 2] - rhs[:, 0]) * np.maximum(0.0, rhs[:, 3] - rhs[:, 1])
    union = lhs_area + rhs_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def area_bucket(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def load_ground_truth(annotation_path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in payload["images"]}
    ground_truth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        record = {
            "id": int(annotation["id"]),
            "image_id": int(annotation["image_id"]),
            "category_id": int(annotation["category_id"]),
            "bbox_xywh": [float(value) for value in annotation["bbox"]],
            "bbox_xyxy": xywh_to_xyxy(annotation["bbox"]),
            "area": float(annotation.get("area", annotation["bbox"][2] * annotation["bbox"][3])),
        }
        record["area_bucket"] = area_bucket(record["area"])
        ground_truth[record["image_id"]].append(record)
    categories = sorted(payload["categories"], key=lambda item: int(item["id"]))
    return images, ground_truth, categories


def collect_predictions(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device: str,
    seed: int,
    max_dets: int,
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    cfg = YAMLConfig(
        str(config_path),
        resume=str(checkpoint_path),
        device=device,
        seed=seed,
        output_dir=str(output_dir / "runtime"),
        test_only=True,
    )
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    model.eval()
    criterion = solver.criterion
    criterion.eval()
    evaluator = solver.evaluator
    evaluator.cleanup()

    predictions: list[dict[str, Any]] = []
    image_count = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for samples, targets in solver.val_dataloader:
            samples = samples.to(solver.device)
            targets = [{key: value.to(solver.device) for key, value in target.items()} for target in targets]
            outputs = model(samples)
            original_sizes = torch.stack([target["orig_size"] for target in targets], dim=0)
            results = solver.postprocessor(outputs, original_sizes)
            evaluator.update(
                {
                    int(target["image_id"].item()): result
                    for target, result in zip(targets, results)
                }
            )

            for target, result in zip(targets, results):
                image_count += 1
                image_id = int(target["image_id"].item())
                boxes = result["boxes"].detach().cpu().tolist()
                scores = result["scores"].detach().cpu().tolist()
                labels = result["labels"].detach().cpu().tolist()
                order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:max_dets]
                for index in order:
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(labels[index]),
                            "score": float(scores[index]),
                            "bbox_xyxy": [float(value) for value in boxes[index]],
                            "bbox_xywh": xyxy_to_xywh(boxes[index]),
                        }
                    )

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    coco_eval = evaluator.coco_eval["bbox"]
    coco_stats = coco_eval.stats.tolist()
    torch.save(coco_eval.eval, output_dir / "coco_eval.pth")
    runtime = {
        "images": image_count,
        "predictions_capped": len(predictions),
        "max_dets_per_image": max_dets,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }
    solver.cleanup()
    return predictions, coco_stats, runtime


def label_predictions(
    predictions: list[dict[str, Any]],
    ground_truth: dict[int, list[dict[str, Any]]],
    iou_threshold: float,
    localization_floor: float,
) -> tuple[list[dict[str, Any]], dict[int, set[int]]]:
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_image[prediction["image_id"]].append(prediction)

    labeled: list[dict[str, Any]] = []
    matched_by_image: dict[int, set[int]] = defaultdict(set)
    for image_id, image_predictions in predictions_by_image.items():
        image_ground_truth = ground_truth.get(image_id, [])
        matched = matched_by_image[image_id]
        for prediction in sorted(image_predictions, key=lambda item: item["score"], reverse=True):
            same_indices = [
                index
                for index, item in enumerate(image_ground_truth)
                if item["category_id"] == prediction["category_id"]
            ]
            same_boxes = [image_ground_truth[index]["bbox_xyxy"] for index in same_indices]
            same_ious = box_iou(prediction["bbox_xyxy"], same_boxes)
            best_same_iou = float(same_ious.max()) if len(same_ious) else 0.0
            unmatched_same = [
                (float(iou), same_indices[offset])
                for offset, iou in enumerate(same_ious)
                if same_indices[offset] not in matched
            ]
            best_unmatched_same = max(unmatched_same, default=(0.0, None), key=lambda item: item[0])

            all_boxes = [item["bbox_xyxy"] for item in image_ground_truth]
            all_ious = box_iou(prediction["bbox_xyxy"], all_boxes)
            best_any_iou = float(all_ious.max()) if len(all_ious) else 0.0
            best_any_index = int(all_ious.argmax()) if len(all_ious) else None
            best_any_class = (
                int(image_ground_truth[best_any_index]["category_id"])
                if best_any_index is not None
                else None
            )

            record = dict(prediction)
            record.update(
                {
                    "is_tp": False,
                    "matched_gt_index": None,
                    "best_same_iou": best_same_iou,
                    "best_any_iou": best_any_iou,
                    "best_any_class": best_any_class,
                }
            )
            if best_unmatched_same[1] is not None and best_unmatched_same[0] >= iou_threshold:
                matched.add(best_unmatched_same[1])
                record["is_tp"] = True
                record["matched_gt_index"] = best_unmatched_same[1]
                record["error_type"] = "tp"
            elif best_same_iou >= iou_threshold:
                record["error_type"] = "duplicate"
            elif best_any_iou >= iou_threshold and best_any_class != prediction["category_id"]:
                record["error_type"] = "class_confusion"
            elif best_any_iou >= localization_floor:
                record["error_type"] = "localization"
            else:
                record["error_type"] = "background_or_unlabeled"
            labeled.append(record)
    return labeled, matched_by_image


def operating_point(
    labeled: list[dict[str, Any]],
    ground_truth_count: int,
    threshold: float,
    category_id: int | None = None,
) -> dict[str, Any]:
    selected = [
        item
        for item in labeled
        if item["score"] >= threshold and (category_id is None or item["category_id"] == category_id)
    ]
    tp = sum(bool(item["is_tp"]) for item in selected)
    fp = len(selected) - tp
    fn = max(0, ground_truth_count - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / ground_truth_count if ground_truth_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = Counter(item["error_type"] for item in selected if not item["is_tp"])
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision_iou50": float(precision),
        "recall_iou50": float(recall),
        "f1_iou50": float(f1),
        "fp_error_types": dict(sorted(errors.items())),
    }


def best_f1_threshold(
    labeled: list[dict[str, Any]],
    ground_truth_count: int,
    category_id: int | None = None,
) -> dict[str, Any]:
    selected = [item for item in labeled if category_id is None or item["category_id"] == category_id]
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        grouped[float(item["score"])].append(item)
    tp = 0
    fp = 0
    best: dict[str, Any] | None = None
    for score in sorted(grouped, reverse=True):
        group = grouped[score]
        tp += sum(bool(item["is_tp"]) for item in group)
        fp += sum(not bool(item["is_tp"]) for item in group)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / ground_truth_count if ground_truth_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if best is None or f1 > best["f1_iou50"]:
            best = {
                "threshold": float(score),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(max(0, ground_truth_count - tp)),
                "precision_iou50": float(precision),
                "recall_iou50": float(recall),
                "f1_iou50": float(f1),
            }
    if best is None:
        return operating_point([], ground_truth_count, 1.0, category_id)
    return operating_point(labeled, ground_truth_count, best["threshold"], category_id)


def unmatched_ground_truth(
    labeled: list[dict[str, Any]],
    ground_truth: dict[int, list[dict[str, Any]]],
    images: dict[int, dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    predictions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    matched: dict[int, set[int]] = defaultdict(set)
    for item in labeled:
        predictions_by_image[item["image_id"]].append(item)
        if item["score"] >= threshold and item["is_tp"] and item["matched_gt_index"] is not None:
            matched[item["image_id"]].add(int(item["matched_gt_index"]))

    samples: list[dict[str, Any]] = []
    for image_id, image_ground_truth in ground_truth.items():
        image_predictions = predictions_by_image.get(image_id, [])
        for index, gt in enumerate(image_ground_truth):
            if index in matched[image_id]:
                continue
            same = [item for item in image_predictions if item["category_id"] == gt["category_id"]]
            any_predictions = image_predictions
            best_same = max(same, key=lambda item: box_iou(gt["bbox_xyxy"], [item["bbox_xyxy"]])[0], default=None)
            best_any = max(any_predictions, key=lambda item: box_iou(gt["bbox_xyxy"], [item["bbox_xyxy"]])[0], default=None)
            samples.append(
                {
                    "image_id": image_id,
                    "file_name": images[image_id]["file_name"],
                    "gt_id": gt["id"],
                    "gt_category_id": gt["category_id"],
                    "area": gt["area"],
                    "area_bucket": gt["area_bucket"],
                    "bbox_xywh": gt["bbox_xywh"],
                    "best_same_score": None if best_same is None else best_same["score"],
                    "best_same_iou": 0.0 if best_same is None else float(box_iou(gt["bbox_xyxy"], [best_same["bbox_xyxy"]])[0]),
                    "best_any_category_id": None if best_any is None else best_any["category_id"],
                    "best_any_score": None if best_any is None else best_any["score"],
                    "best_any_iou": 0.0 if best_any is None else float(box_iou(gt["bbox_xyxy"], [best_any["bbox_xyxy"]])[0]),
                }
            )
    return samples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def crop_tile(image_path: Path, box_xyxy: list[float], caption: str, color: str, size: int = 256) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    x1, y1, x2, y2 = box_xyxy
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    margin = max(48.0, max(width, height) * 1.5)
    crop_box = (
        max(0, math.floor(x1 - margin)),
        max(0, math.floor(y1 - margin)),
        min(image.width, math.ceil(x2 + margin)),
        min(image.height, math.ceil(y2 + margin)),
    )
    crop = image.crop(crop_box)
    draw = ImageDraw.Draw(crop)
    draw.rectangle(
        [x1 - crop_box[0], y1 - crop_box[1], x2 - crop_box[0], y2 - crop_box[1]],
        outline=color,
        width=4,
    )
    crop = ImageOps.fit(crop, (size, size - 28), method=Image.Resampling.BILINEAR)
    tile = Image.new("RGB", (size, size), "white")
    tile.paste(crop, (0, 28))
    ImageDraw.Draw(tile).text((4, 6), caption[:44], fill="black")
    return tile


def make_montage(
    rows: list[dict[str, Any]],
    image_root: Path,
    output_path: Path,
    mode: str,
    class_names: dict[int, str],
    limit: int = 16,
) -> None:
    tiles: list[Image.Image] = []
    for row in rows[:limit]:
        image_path = image_root / row["file_name"]
        if not image_path.is_file():
            continue
        if mode == "fp":
            box = row["bbox_xyxy"]
            caption = f'{row["image_id"]} {class_names[row["category_id"]]} {row["score"]:.3f} {row["error_type"]}'
            color = "red"
        else:
            box = xywh_to_xyxy(row["bbox_xywh"])
            caption = f'{row["image_id"]} {class_names[row["gt_category_id"]]} FN {row["area_bucket"]}'
            color = "orange"
        tiles.append(crop_tile(image_path, box, caption, color))
    if not tiles:
        return
    columns = 4
    rows_count = math.ceil(len(tiles) / columns)
    montage = Image.new("RGB", (columns * 256, rows_count * 256), "#dddddd")
    for index, tile in enumerate(tiles):
        montage.paste(tile, ((index % columns) * 256, (index // columns) * 256))
    montage.save(output_path, quality=92)


def extract_per_class_coco(eval_path: Path, categories: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    evaluation = torch.load(eval_path, map_location="cpu")
    precision = np.asarray(evaluation["precision"])
    recall = np.asarray(evaluation["recall"])
    result: dict[str, dict[str, float]] = {}
    for index, category in enumerate(categories):
        all_iou = precision[:, :, index, 0, 2]
        ap50 = precision[0, :, index, 0, 2]
        ap75 = precision[5, :, index, 0, 2]
        ar100 = recall[:, index, 0, 2]
        result[category["name"]] = {
            "AP50_95": float(all_iou[all_iou > -1].mean()),
            "AP50": float(ap50[ap50 > -1].mean()),
            "AP75": float(ap75[ap75 > -1].mean()),
            "AR100": float(ar100[ar100 > -1].mean()),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--fixed-threshold", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--localization-floor", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "test" in args.annotations.name.lower() or "test" in str(args.image_root).lower():
        raise ValueError("This diagnostic is Val-only; Test paths are forbidden.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    nofile = raise_nofile_limit()
    dist_utils.setup_distributed(0, "builtin", seed=args.seed)

    images, ground_truth, categories = load_ground_truth(args.annotations)
    class_names = {int(item["id"]): item["name"] for item in categories}
    predictions, coco_stats, runtime = collect_predictions(
        args.config,
        args.checkpoint,
        args.output_dir,
        args.device,
        args.seed,
        args.max_dets,
    )
    labeled, _ = label_predictions(
        predictions,
        ground_truth,
        args.iou_threshold,
        args.localization_floor,
    )

    gt_total = sum(len(items) for items in ground_truth.values())
    gt_by_class = Counter(item["category_id"] for items in ground_truth.values() for item in items)
    gt_by_size = Counter(item["area_bucket"] for items in ground_truth.values() for item in items)
    optimal = best_f1_threshold(labeled, gt_total)
    fixed = operating_point(labeled, gt_total, args.fixed_threshold)
    optimal_by_class = {
        class_names[category_id]: best_f1_threshold(labeled, gt_by_class[category_id], category_id)
        for category_id in sorted(class_names)
    }
    selected_threshold = optimal["threshold"]
    selected_predictions = [item for item in labeled if item["score"] >= selected_threshold]
    fp_rows = [
        {
            **item,
            "file_name": images[item["image_id"]]["file_name"],
            "predicted_class": class_names[item["category_id"]],
            "best_any_class_name": None if item["best_any_class"] is None else class_names[item["best_any_class"]],
        }
        for item in selected_predictions
        if not item["is_tp"]
    ]
    fp_rows.sort(key=lambda item: item["score"], reverse=True)
    fn_rows = unmatched_ground_truth(labeled, ground_truth, images, selected_threshold)
    fn_rows.sort(
        key=lambda item: (
            item["best_same_score"] is not None,
            item["best_same_score"] if item["best_same_score"] is not None else -1,
        ),
        reverse=True,
    )

    predictions_path = args.output_dir / "val_predictions_top100.json"
    predictions_path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    write_csv(args.output_dir / "fp_samples.csv", fp_rows)
    write_csv(args.output_dir / "fn_samples.csv", fn_rows)
    make_montage(fp_rows, args.image_root, args.output_dir / "top_fp_montage.jpg", "fp", class_names)
    make_montage(fn_rows, args.image_root, args.output_dir / "top_fn_montage.jpg", "fn", class_names)
    background_rows = [item for item in fp_rows if item["error_type"] == "background_or_unlabeled"]
    localization_rows = [item for item in fp_rows if item["error_type"] == "localization"]
    make_montage(
        background_rows,
        args.image_root,
        args.output_dir / "top_background_or_unlabeled_montage.jpg",
        "fp",
        class_names,
    )
    make_montage(
        localization_rows,
        args.image_root,
        args.output_dir / "top_localization_montage.jpg",
        "fp",
        class_names,
    )

    confusion = Counter(
        f'{class_names[item["category_id"]]}->{class_names[item["best_any_class"]]}'
        for item in selected_predictions
        if item["error_type"] == "class_confusion" and item["best_any_class"] is not None
    )
    fn_by_class = Counter(class_names[item["gt_category_id"]] for item in fn_rows)
    fn_by_size = Counter(item["area_bucket"] for item in fn_rows)
    coco_eval_path = args.output_dir / "coco_eval.pth"
    report = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Val-only; Test not read",
        "config": str(args.config),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256(args.checkpoint),
            "last_epoch": int(torch.load(args.checkpoint, map_location="cpu").get("last_epoch", -1)),
        },
        "annotations": {
            "path": str(args.annotations),
            "sha256": sha256(args.annotations),
            "images": len(images),
            "boxes": gt_total,
            "boxes_by_class": {class_names[key]: int(value) for key, value in sorted(gt_by_class.items())},
            "boxes_by_size": dict(sorted(gt_by_size.items())),
        },
        "runtime": {**runtime, "nofile": nofile},
        "official_coco": dict(zip(COCO_STAT_NAMES, (float(item) for item in coco_stats))),
        "official_coco_per_class": extract_per_class_coco(coco_eval_path, categories),
        "diagnostic_operating_points_iou50": {
            "fixed_0_25": fixed,
            "val_optimal_micro_f1": optimal,
            "val_optimal_by_class": optimal_by_class,
            "warning": "Val-optimized thresholds are exploratory and are not locked Test thresholds.",
        },
        "selected_threshold_error_audit": {
            "threshold": selected_threshold,
            "fp_by_error_type": optimal["fp_error_types"],
            "class_confusions": dict(sorted(confusion.items())),
            "fn_by_class": dict(sorted(fn_by_class.items())),
            "fn_by_size": dict(sorted(fn_by_size.items())),
        },
        "artifacts": {
            "predictions": str(predictions_path),
            "predictions_sha256": sha256(predictions_path),
            "coco_eval": str(coco_eval_path),
            "coco_eval_sha256": sha256(coco_eval_path),
            "fp_samples": str(args.output_dir / "fp_samples.csv"),
            "fn_samples": str(args.output_dir / "fn_samples.csv"),
            "top_fp_montage": str(args.output_dir / "top_fp_montage.jpg"),
            "top_fn_montage": str(args.output_dir / "top_fn_montage.jpg"),
            "top_background_or_unlabeled_montage": str(
                args.output_dir / "top_background_or_unlabeled_montage.jpg"
            ),
            "top_localization_montage": str(args.output_dir / "top_localization_montage.jpg"),
        },
    }
    report_path = args.output_dir / "diagnostics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
