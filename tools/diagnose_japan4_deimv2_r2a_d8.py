"""D8: zero-training counterfactual routing audit for the Japan4 R2-A refiner.

The audit reuses one frozen R2-A checkpoint and evaluates four deterministic
inference routes on Val: refiner-all, refiner-none, entropy-top25%, and a fixed
input-small rule.  It also preserves the pre-refiner Hungarian assignment and
records whether each matched query improves after refinement.  No threshold is
fit and no model parameter is changed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision

try:
    import resource
except ImportError:
    resource = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig  # noqa: E402
from engine.deim.dfine_utils import distance2bbox, weighting_function  # noqa: E402
from engine.solver import TASKS  # noqa: E402


STAT_NAMES = (
    "AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge",
    "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge",
)
ROUTES = ("all", "none", "entropy25", "input_small")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=29)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Engineering smoke test only; omit for formal D8.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


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


def reject_test_path(path: Path) -> None:
    if "test" in str(path).lower():
        raise ValueError(f"D8 is Val-only and rejects paths containing 'test': {path}")


def normalized_entropy(corners: torch.Tensor, reg_max: int) -> torch.Tensor:
    prob = corners.reshape(*corners.shape[:-1], 4, reg_max + 1).softmax(-1)
    entropy = -(prob * prob.clamp_min(1e-12).log()).sum(-1)
    return entropy.mean(-1) / math.log(reg_max + 1)


def top_fraction_mask(risk: torch.Tensor, fraction: float = 0.25) -> torch.Tensor:
    k = max(1, math.ceil(risk.shape[1] * fraction))
    indices = risk.topk(k, dim=1, sorted=False).indices
    return torch.zeros_like(risk, dtype=torch.bool).scatter_(1, indices, True)


def route_masks(pre_boxes: torch.Tensor, entropy: torch.Tensor) -> dict[str, torch.Tensor]:
    area = pre_boxes[..., 2] * pre_boxes[..., 3]
    # COCO small-object boundary mapped to the locked 640 x 640 input canvas.
    input_small = area < (32.0 / 640.0) ** 2
    return {
        "all": torch.ones_like(entropy, dtype=torch.bool),
        "none": torch.zeros_like(entropy, dtype=torch.bool),
        "entropy25": top_fraction_mask(entropy, 0.25),
        "input_small": input_small,
    }


def box_iou_aligned(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lhs_xyxy = torchvision.ops.box_convert(lhs, "cxcywh", "xyxy")
    rhs_xyxy = torchvision.ops.box_convert(rhs, "cxcywh", "xyxy")
    return torchvision.ops.box_iou(lhs_xyxy, rhs_xyxy).diag()


def gt_scale(target: dict[str, torch.Tensor], target_index: int) -> str:
    # Val keeps COCO's original-pixel area even though its boxes are resized
    # absolute xyxy coordinates for evaluation.
    area = float(target["area"][target_index])
    return "small" if area < 32**2 else ("medium" if area < 96**2 else "large")


def matcher_targets(targets: list[dict[str, torch.Tensor]], samples: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """Convert Val absolute xyxy boxes to the criterion's normalized cxcywh contract."""
    height, width = samples.shape[-2:]
    scale = samples.new_tensor([width, height, width, height])
    converted = []
    for target in targets:
        item = dict(target)
        item["boxes"] = torchvision.ops.box_convert(target["boxes"], "xyxy", "cxcywh") / scale
        converted.append(item)
    return converted


def bootstrap_image_ci(rows: list[dict[str, Any]], iterations: int, seed: int) -> list[float | None]:
    by_image: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_image[int(row["image_id"])].append(float(row["delta_iou"]))
    image_ids = sorted(by_image)
    if not image_ids:
        return [None, None]
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        sample = [rng.choice(image_ids) for _ in image_ids]
        deltas = [value for image_id in sample for value in by_image[image_id]]
        values.append(float(np.median(deltas)))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def subset_summary(rows: list[dict[str, Any]], key: str, iterations: int, seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for routed in (True, False):
        part = [row for row in rows if bool(row[key]) is routed]
        deltas = np.asarray([row["delta_iou"] for row in part], dtype=np.float64)
        name = "routed" if routed else "not_routed"
        result[name] = {
            "n": len(part),
            "positive_rate": float((deltas > 0).mean()) if len(part) else None,
            "mean_delta_iou": float(deltas.mean()) if len(part) else None,
            "median_delta_iou": float(np.median(deltas)) if len(part) else None,
            "median_image_bootstrap_95ci": bootstrap_image_ci(part, iterations, seed + int(routed)),
            "cross_to_iou75": int(sum(row["pre_iou"] < 0.75 <= row["post_iou"] for row in part)),
            "fall_below_iou75": int(sum(row["pre_iou"] >= 0.75 > row["post_iou"] for row in part)),
        }
    numerator = result["routed"]["positive_rate"]
    denominator = result["not_routed"]["positive_rate"]
    result["positive_rate_enrichment"] = (
        numerator / denominator if numerator is not None and denominator not in (None, 0.0) else None
    )
    return result


def group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(float(row["delta_iou"]))
    return {
        name: {
            "n": len(values),
            "positive_rate": float(np.mean(np.asarray(values) > 0)),
            "mean_delta_iou": float(np.mean(values)),
            "median_delta_iou": float(np.median(values)),
        }
        for name, values in sorted(groups.items())
    }


def metrics_dict(stats: list[float]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(STAT_NAMES, stats)}


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    reject_test_path(args.config)
    reject_test_path(args.checkpoint)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.config.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("Config and checkpoint must both exist")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    limits = raise_nofile_limit()
    cfg = YAMLConfig(
        str(args.config), resume=str(args.checkpoint), device=args.device,
        seed=args.seed, output_dir=str(args.output_dir / "runtime"), test_only=True,
    )
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    outer = model.decoder
    decoder = outer.decoder
    refiner = decoder.fdr_refiner
    if refiner is None or refiner.gate_mode != "all":
        raise RuntimeError("D8 requires the frozen R2-A checkpoint/config with gate_mode='all'")

    capture: dict[str, torch.Tensor] = {}

    def refiner_hook(_module, inputs, output):
        capture["pre_box"] = inputs[2].detach()
        capture["pre_corners"] = inputs[5].detach()
        capture["post_corners"] = output[0].detach()

    def lqe_hook(_module, inputs, _output):
        capture["raw_scores"] = inputs[0].detach()

    def decoder_hook(_module, _inputs, output):
        capture["ref_points"] = output[3][-1].detach()

    handles = [
        refiner.register_forward_hook(refiner_hook),
        decoder.lqe_layers[decoder.eval_idx].register_forward_hook(lqe_hook),
        decoder.register_forward_hook(decoder_hook),
    ]
    evaluators = {name: copy.deepcopy(solver.evaluator) for name in ROUTES}
    for evaluator in evaluators.values():
        evaluator.cleanup()

    project = weighting_function(decoder.reg_max, outer.up, outer.reg_scale)
    rows: list[dict[str, Any]] = []
    images = 0
    batches = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch_index, (samples, targets) in enumerate(solver.val_dataloader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            samples = samples.to(solver.device)
            targets = [{key: value.to(solver.device) for key, value in target.items()} for target in targets]
            capture.clear()
            actual = model(samples)
            required = {"pre_box", "pre_corners", "post_corners", "raw_scores", "ref_points"}
            if set(capture) != required:
                raise RuntimeError(f"Incomplete hook capture: {sorted(capture)}")

            pre_box = capture["pre_box"]
            pre_corners = capture["pre_corners"]
            post_corners = capture["post_corners"]
            raw_scores = capture["raw_scores"]
            entropy = normalized_entropy(pre_corners, decoder.reg_max)
            masks = route_masks(pre_box, entropy)
            route_outputs: dict[str, dict[str, torch.Tensor]] = {}
            for name, mask in masks.items():
                corners = pre_corners + mask.unsqueeze(-1) * (post_corners - pre_corners)
                boxes = distance2bbox(
                    capture["ref_points"], outer.integral(corners, project), outer.reg_scale)
                logits = decoder.lqe_layers[decoder.eval_idx](raw_scores, corners)
                route_outputs[name] = {"pred_logits": logits, "pred_boxes": boxes}

            max_box_diff = float((route_outputs["all"]["pred_boxes"] - actual["pred_boxes"]).abs().max())
            max_logit_diff = float((route_outputs["all"]["pred_logits"] - actual["pred_logits"]).abs().max())
            if max(max_box_diff, max_logit_diff) > 2e-5:
                raise RuntimeError(f"All-route reconstruction mismatch: box={max_box_diff}, logit={max_logit_diff}")

            original_sizes = torch.stack([target["orig_size"] for target in targets])
            for name in ROUTES:
                results = solver.postprocessor(route_outputs[name], original_sizes)
                evaluators[name].update({
                    int(target["image_id"].item()): result
                    for target, result in zip(targets, results)
                })

            normalized_targets = matcher_targets(targets, samples.tensors if hasattr(samples, "tensors") else samples)
            pre_match = solver.criterion.matcher(
                route_outputs["none"], normalized_targets, epoch=args.epoch)["indices"]
            for image_index, (query_indices, target_indices) in enumerate(pre_match):
                if not len(query_indices):
                    continue
                pre_iou = box_iou_aligned(
                    route_outputs["none"]["pred_boxes"][image_index, query_indices],
                    normalized_targets[image_index]["boxes"][target_indices],
                )
                post_iou = box_iou_aligned(
                    route_outputs["all"]["pred_boxes"][image_index, query_indices],
                    normalized_targets[image_index]["boxes"][target_indices],
                )
                for offset, (query_index, target_index) in enumerate(zip(query_indices, target_indices)):
                    box = pre_box[image_index, query_index]
                    aspect = float(torch.maximum(box[2] / box[3].clamp_min(1e-9), box[3] / box[2].clamp_min(1e-9)))
                    row = {
                        "image_id": int(targets[image_index]["image_id"].item()),
                        "query_id": int(query_index.item()),
                        "target_id": int(target_index.item()),
                        "class_id": int(targets[image_index]["labels"][target_index].item()),
                        "gt_scale": gt_scale(targets[image_index], int(target_index.item())),
                        "pre_iou": float(pre_iou[offset]),
                        "post_iou": float(post_iou[offset]),
                        "delta_iou": float(post_iou[offset] - pre_iou[offset]),
                        "entropy": float(entropy[image_index, query_index]),
                        "pred_area_norm": float(box[2] * box[3]),
                        "pred_short_side_norm": float(torch.minimum(box[2], box[3])),
                        "pred_aspect": aspect,
                        "route_entropy25": bool(masks["entropy25"][image_index, query_index]),
                        "route_input_small": bool(masks["input_small"][image_index, query_index]),
                        "is_short_side": bool(torch.minimum(box[2], box[3]) < 32.0 / 640.0),
                        "is_high_aspect": aspect >= 3.0,
                    }
                    rows.append(row)
            batches += 1
            images += len(targets)
            print(f"D8 batch {batches}/{len(solver.val_dataloader)} images={images} matched={len(rows)}", flush=True)

    for handle in handles:
        handle.remove()
    route_metrics: dict[str, dict[str, float]] = {}
    for name, evaluator in evaluators.items():
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        route_metrics[name] = metrics_dict(evaluator.coco_eval["bbox"].stats.tolist())

    entropy_summary = subset_summary(rows, "route_entropy25", args.bootstrap, args.seed)
    input_small_summary = subset_summary(rows, "route_input_small", args.bootstrap, args.seed + 100)
    all_metrics, none_metrics = route_metrics["all"], route_metrics["none"]
    entropy_metrics, small_metrics = route_metrics["entropy25"], route_metrics["input_small"]

    def retention(route: dict[str, float], metric: str) -> float | None:
        denominator = all_metrics[metric] - none_metrics[metric]
        return (route[metric] - none_metrics[metric]) / denominator if denominator > 1e-12 else None

    def recovery(route: dict[str, float], metric: str) -> float | None:
        harm = none_metrics[metric] - all_metrics[metric]
        return (route[metric] - all_metrics[metric]) / harm if harm > 1e-12 else None

    route_decisions = {}
    for name, metrics in (("entropy25", entropy_metrics), ("input_small", small_metrics)):
        subset = entropy_summary if name == "entropy25" else input_small_summary
        ci = subset["routed"]["median_image_bootstrap_95ci"]
        class_groups = group_summary([row for row in rows if row["route_" + name]], "class_id")
        nonnegative_classes = sum(group["median_delta_iou"] >= 0 for group in class_groups.values())
        route_decisions[name] = {
            "positive_rate_enrichment": subset["positive_rate_enrichment"],
            "routed_median_delta_iou": subset["routed"]["median_delta_iou"],
            "routed_median_95ci": ci,
            "APsmall_gain_retention_vs_all": retention(metrics, "APsmall"),
            "APmedium_harm_recovery_vs_all": recovery(metrics, "APmedium"),
            "AP50_delta_vs_none": metrics["AP50"] - none_metrics["AP50"],
            "nonnegative_class_medians": nonnegative_classes,
            "go": bool(
                subset["positive_rate_enrichment"] is not None
                and subset["positive_rate_enrichment"] >= 1.5
                and subset["routed"]["median_delta_iou"] is not None
                and subset["routed"]["median_delta_iou"] > 0
                and ci[0] is not None and ci[0] > 0
                and retention(metrics, "APsmall") is not None
                and retention(metrics, "APsmall") >= 0.75
                and recovery(metrics, "APmedium") is not None
                and recovery(metrics, "APmedium") >= 0.50
                and nonnegative_classes >= 3
                and metrics["AP50"] >= none_metrics["AP50"] - 0.002
            ),
        }

    fieldnames = list(rows[0]) if rows else []
    with (args.output_dir / "d8_query_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "diagnostic": "D8_R2A_refiner_benefit_routing",
        "formal": args.max_batches is None,
        "protocol": {
            "split": "Val-only",
            "checkpoint_epoch": args.epoch,
            "seed": args.seed,
            "routes": {
                "all": "original R2-A refiner on all queries",
                "none": "same R2-A weights with refiner residual disabled",
                "entropy25": "fixed per-image top-25% pre-FDR entropy",
                "input_small": "fixed predicted normalized area < (32/640)^2",
            },
            "matching": "Hungarian assignment fixed on the no-refiner counterfactual",
            "no_threshold_fitting": True,
        },
        "artifacts": {
            "config": str(args.config), "config_sha256": sha256(args.config),
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
        },
        "runtime": {
            "batches": batches, "images": images, "matched_queries": len(rows),
            "nofile": limits,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        },
        "coco_metrics": route_metrics,
        "matched_query": {
            "entropy25": entropy_summary,
            "input_small": input_small_summary,
            "by_gt_scale": group_summary(rows, "gt_scale"),
            "by_class": group_summary(rows, "class_id"),
            "short_side": subset_summary(rows, "is_short_side", args.bootstrap, args.seed + 200),
            "high_aspect": subset_summary(rows, "is_high_aspect", args.bootstrap, args.seed + 300),
        },
        "pre_registered_gate": {
            "requirements": [
                "routed positive-rate enrichment >= 1.5x",
                "routed median delta-IoU > 0 with image-bootstrap 95% CI > 0",
                "retain >= 75% of all-route APsmall gain over none",
                "recover >= 50% of all-route APmedium harm",
                ">= 3 classes have nonnegative routed median delta-IoU",
                "AP50 no worse than none by more than 0.002",
            ],
            "routes": route_decisions,
            "overall_go": any(item["go"] for item in route_decisions.values()),
        },
    }
    (args.output_dir / "d8_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary["pre_registered_gate"], indent=2), flush=True)
    solver.cleanup()


if __name__ == "__main__":
    main()
