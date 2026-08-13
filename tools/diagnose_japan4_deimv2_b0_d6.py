#!/usr/bin/env python3
"""D6 dual-view task-gradient audit for the frozen Japan4 DEIMv2-N B0.

D6-Actual uses the real post-LQE MAL objective. D6-Pure recomputes the same
matched classification objective from the raw, pre-LQE class logits. Both are
compared with the same localization objective at the final decoder layer.

The detector is never optimized. Backbone/encoder BatchNorm remains in eval
mode, Test paths are refused, and the script refuses to overwrite its output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine.core import YAMLConfig  # noqa: E402
from engine.deim.box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou  # noqa: E402
from engine.misc import dist_utils  # noqa: E402
from engine.solver import TASKS  # noqa: E402


EXPECTED_B0_SHA256 = "0f0a5623491066f18acde42d6b436f18e6eb0beddcc8cdd0df847da7e8f00420"
CLASS_NAMES = ["D00", "D10", "D20", "D40"]
MATCH_EPOCH = 123
LOC_PREFIXES = ("loss_bbox", "loss_giou", "loss_fgl", "loss_ddf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_test_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        if "test" in str(path).lower():
            raise ValueError(f"D6 is Train-only and refuses Test paths: {path}")


def raise_nofile_limit() -> tuple[int, int]:
    if os.name == "nt":
        return (-1, -1)
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(max(soft, 65535), hard)
    if target > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)


def deterministic_loader(
    config: Path,
    images: Path,
    annotations: Path,
    batch_size: int,
    workers: int,
    device: str,
    seed: int,
):
    overrides = {
        "dataset": {"img_folder": str(images), "ann_file": str(annotations)},
        "shuffle": False,
        "total_batch_size": batch_size,
        "num_workers": workers,
    }
    cfg = YAMLConfig(str(config), device=device, seed=seed, val_dataloader=overrides)
    return cfg.val_dataloader


def build_solver(config: Path, checkpoint: Path, output_dir: Path, device: str, seed: int):
    cfg = YAMLConfig(
        str(config),
        resume=str(checkpoint),
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return solver, model


def parameter_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    layer = model.decoder.decoder.layers[model.decoder.decoder.eval_idx]
    groups = {
        "self_attention": list(layer.self_attn.parameters()) + list(layer.norm1.parameters()),
        "cross_sampling_offsets": list(layer.cross_attn.sampling_offsets.parameters()),
        "cross_attention_weights": list(layer.cross_attn.attention_weights.parameters()),
        "cross_gateway_or_norm": (
            list(layer.gateway.parameters()) if layer.use_gateway else list(layer.norm2.parameters())
        ),
        "ffn": list(layer.swish_ffn.parameters()) + list(layer.norm3.parameters()),
    }
    seen: set[int] = set()
    for name, parameters in groups.items():
        duplicate = [parameter for parameter in parameters if id(parameter) in seen]
        if duplicate:
            raise RuntimeError(f"D6 parameter groups overlap at {name}")
        seen.update(id(parameter) for parameter in parameters)
        for parameter in parameters:
            parameter.requires_grad_(True)
    return groups


def flatten_gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    retain_graph: bool = True,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
        materialize_grads=False,
    )
    chunks = [
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(chunks) if chunks else loss.new_zeros(0)


def cosine_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | bool | None]:
    left = left.float()
    right = right.float()
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    active = left_norm > 0.0 and right_norm > 0.0
    cosine = float(F.cosine_similarity(left, right, dim=0).item()) if active else None
    return {
        "cosine": cosine,
        "left_norm": left_norm,
        "right_norm": right_norm,
        "active": active,
    }


def matched_indices(criterion: Any, outputs: dict[str, Any], targets: list[dict[str, torch.Tensor]], epoch: int):
    outputs_main = {key: value for key, value in outputs.items() if "aux" not in key}
    return criterion.matcher(outputs_main, targets, epoch=epoch)["indices"]


def num_boxes(targets: list[dict[str, torch.Tensor]]) -> float:
    return float(max(1, sum(len(target["labels"]) for target in targets)))


def raw_mal_loss(
    criterion: Any,
    raw_logits: torch.Tensor,
    boxes: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    diagnostic = {"pred_logits": raw_logits, "pred_boxes": boxes}
    return criterion.loss_labels_mal(diagnostic, targets, indices, num_boxes(targets))["loss_mal"] * criterion.weight_dict["loss_mal"]


def class_binary_mal(
    criterion: Any,
    logits: torch.Tensor,
    boxes: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    class_id: int,
) -> torch.Tensor | None:
    selected: list[tuple[int, int, int]] = []
    for batch_index, (source, target_ids) in enumerate(indices):
        labels = targets[batch_index]["labels"][target_ids.to(targets[batch_index]["labels"].device)]
        keep = (labels == class_id).to(source.device)
        for source_id, target_id in zip(source[keep], target_ids[keep]):
            selected.append((batch_index, int(source_id), int(target_id)))
    if not selected:
        return None

    channel_logits = logits[..., class_id]
    target_score = torch.zeros_like(channel_logits)
    positive_mask = torch.zeros_like(channel_logits, dtype=torch.bool)
    for batch_index, source_id, target_id in selected:
        iou = box_iou(
            box_cxcywh_to_xyxy(boxes[batch_index, source_id : source_id + 1]),
            box_cxcywh_to_xyxy(targets[batch_index]["boxes"][target_id : target_id + 1]),
        )[0][0, 0].detach()
        target_score[batch_index, source_id] = iou.pow(criterion.gamma)
        positive_mask[batch_index, source_id] = True

    probability = channel_logits.sigmoid().detach()
    if criterion.mal_alpha is not None:
        weight = criterion.mal_alpha * probability.pow(criterion.gamma) * (~positive_mask) + positive_mask
    else:
        weight = probability.pow(criterion.gamma) * (~positive_mask) + positive_mask
    return F.binary_cross_entropy_with_logits(channel_logits, target_score, weight=weight, reduction="sum") / len(selected)


def class_localization_loss(
    criterion: Any,
    outputs: dict[str, Any],
    targets: list[dict[str, torch.Tensor]],
    indices: list[tuple[torch.Tensor, torch.Tensor]],
    class_id: int,
) -> torch.Tensor | None:
    filtered: list[tuple[torch.Tensor, torch.Tensor]] = []
    count = 0
    for batch_index, (source, target_ids) in enumerate(indices):
        labels = targets[batch_index]["labels"][target_ids.to(targets[batch_index]["labels"].device)]
        keep = (labels == class_id).to(source.device)
        filtered.append((source[keep], target_ids[keep]))
        count += int(keep.sum().item())
    if count == 0:
        return None
    criterion._clear_cache()
    box_losses = criterion.loss_boxes(outputs, targets, filtered, float(count))
    local_losses = criterion.loss_local(outputs, targets, filtered, float(count))
    weighted = [
        value * criterion.weight_dict[name]
        for name, value in {**box_losses, **local_losses}.items()
        if name in criterion.weight_dict
    ]
    return sum(weighted)


def add_masks(
    boxes: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    yy = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    xx = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    core = torch.zeros((height, width), dtype=torch.bool, device=device)
    boundary = torch.zeros_like(core)
    for cx, cy, box_w, box_h in boxes:
        x1, x2 = cx - box_w / 2, cx + box_w / 2
        y1, y2 = cy - box_h / 2, cy + box_h / 2
        core_box = (
            (grid_x >= x1 + 0.2 * box_w)
            & (grid_x <= x2 - 0.2 * box_w)
            & (grid_y >= y1 + 0.2 * box_h)
            & (grid_y <= y2 - 0.2 * box_h)
        )
        expanded = (
            (grid_x >= x1 - 0.1 * box_w)
            & (grid_x <= x2 + 0.1 * box_w)
            & (grid_y >= y1 - 0.1 * box_h)
            & (grid_y <= y2 + 0.1 * box_h)
        )
        core |= core_box
        boundary |= expanded & ~core_box
    boundary &= ~core
    background = ~(core | boundary)
    return core, boundary, background


def feature_energy(
    gradient: torch.Tensor | None,
    targets: list[dict[str, torch.Tensor]],
) -> dict[str, float] | None:
    if gradient is None:
        return None
    energy = gradient.detach().float().square().sum(dim=1)
    totals = defaultdict(float)
    for batch_index, target in enumerate(targets):
        masks = add_masks(target["boxes"], energy.shape[-2], energy.shape[-1], energy.device)
        for name, mask in zip(("core", "boundary", "background"), masks):
            totals[name] += float(energy[batch_index][mask].sum().item())
    total = sum(totals.values())
    totals["total"] = total
    for name in ("core", "boundary", "background"):
        totals[f"{name}_fraction"] = totals[name] / total if total > 0 else 0.0
    return dict(totals)


def bootstrap_mean_ci(values: list[float], seed: int, samples: int = 4000) -> list[float] | None:
    if not values:
        return None
    data = np.asarray(values, dtype=np.float64)
    if len(data) == 1:
        return [float(data[0]), float(data[0])]
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(samples, len(data)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize_rows(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    keys = sorted({key for row in rows for key in row if key.endswith("_cosine")})
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
        summary[key.removesuffix("_cosine")] = {
            "n": len(values),
            "mean_cosine": float(np.mean(values)) if values else None,
            "median_cosine": float(np.median(values)) if values else None,
            "negative_fraction": float(np.mean(np.asarray(values) < 0)) if values else None,
            "bootstrap_mean_95ci": bootstrap_mean_ci(values, seed),
        }
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(
    model: torch.nn.Module,
    criterion: Any,
    loader: Any,
    device: torch.device,
    max_batches: int,
    match_epoch: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups = parameter_groups(model)
    parameters = [parameter for group in groups.values() for parameter in group]
    slices: dict[str, slice] = {}
    cursor = 0
    for name, group in groups.items():
        size = sum(parameter.numel() for parameter in group)
        slices[name] = slice(cursor, cursor + size)
        cursor += size

    lqe_raw: dict[str, torch.Tensor] = {}

    def lqe_hook(_module, inputs, _output):
        lqe_raw["logits"] = inputs[0]

    decoder = model.decoder.decoder
    handle = decoder.lqe_layers[decoder.eval_idx].register_forward_hook(lqe_hook)
    rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    model.eval()
    model.decoder.train()
    try:
        for batch_index, (samples, targets) in enumerate(loader):
            if batch_index >= max_batches:
                break
            samples = samples.to(device)
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            with torch.no_grad():
                features = model.encoder(model.backbone(samples))
            features = [feature.detach().requires_grad_(True) for feature in features]
            lqe_raw.clear()
            outputs = model.decoder(features, targets)
            if "logits" not in lqe_raw:
                raise RuntimeError("D6 failed to capture final raw logits")
            raw_logits = lqe_raw["logits"]
            if outputs.get("dn_meta") is not None:
                raw_logits = raw_logits.split(outputs["dn_meta"]["dn_num_split"], dim=1)[1]
            if raw_logits.shape != outputs["pred_logits"].shape:
                raise RuntimeError(f"Raw/post-LQE shape mismatch: {raw_logits.shape} vs {outputs['pred_logits'].shape}")

            losses = criterion(outputs, targets, epoch=match_epoch)
            actual_loss = losses["loss_mal"]
            localization_loss = sum(losses[name] for name in LOC_PREFIXES if name in losses)
            full_actual_loss = sum(value for name, value in losses.items() if name.startswith("loss_mal"))
            full_localization_loss = sum(
                value for name, value in losses.items() if name.startswith(LOC_PREFIXES)
            )
            indices = matched_indices(criterion, outputs, targets, match_epoch)
            pure_loss = raw_mal_loss(criterion, raw_logits, outputs["pred_boxes"], targets, indices)

            gradients = {
                "actual": flatten_gradients(actual_loss, parameters),
                "pure": flatten_gradients(pure_loss, parameters),
                "localization": flatten_gradients(localization_loss, parameters),
                "full_actual": flatten_gradients(full_actual_loss, parameters),
                "full_localization": flatten_gradients(full_localization_loss, parameters),
            }
            row: dict[str, Any] = {
                "batch": batch_index,
                "images": len(targets),
                "objects": int(sum(len(target["labels"]) for target in targets)),
                "actual_loss": float(actual_loss.detach().item()),
                "pure_loss": float(pure_loss.detach().item()),
                "localization_loss": float(localization_loss.detach().item()),
            }
            for name, group_slice in {"all_final": slice(0, cursor), **slices}.items():
                for view in ("actual", "pure", "full_actual"):
                    loc_key = "full_localization" if view == "full_actual" else "localization"
                    stats = cosine_stats(gradients[view][group_slice], gradients[loc_key][group_slice])
                    prefix = f"{view}_vs_loc_{name}"
                    row[f"{prefix}_cosine"] = stats["cosine"]
                    row[f"{prefix}_left_norm"] = stats["left_norm"]
                    row[f"{prefix}_right_norm"] = stats["right_norm"]

            for class_id, class_name in enumerate(CLASS_NAMES):
                class_actual = class_binary_mal(
                    criterion, outputs["pred_logits"], outputs["pred_boxes"], targets, indices, class_id
                )
                class_pure = class_binary_mal(
                    criterion, raw_logits, outputs["pred_boxes"], targets, indices, class_id
                )
                class_loc = class_localization_loss(criterion, outputs, targets, indices, class_id)
                if class_actual is None or class_pure is None or class_loc is None:
                    continue
                class_gradients = {
                    "actual": flatten_gradients(class_actual, parameters),
                    "pure": flatten_gradients(class_pure, parameters),
                    "loc": flatten_gradients(class_loc, parameters),
                }
                for name, group_slice in {"all_final": slice(0, cursor), **slices}.items():
                    for view in ("actual", "pure"):
                        stats = cosine_stats(class_gradients[view][group_slice], class_gradients["loc"][group_slice])
                        row[f"class_{class_name}_{view}_vs_loc_{name}_cosine"] = stats["cosine"]

            feature_targets = parameters + features
            feature_gradients: dict[str, list[torch.Tensor | None]] = {}
            for name, loss in (
                ("actual", actual_loss),
                ("pure", pure_loss),
                ("localization", localization_loss),
            ):
                all_gradients = torch.autograd.grad(
                    loss,
                    feature_targets,
                    retain_graph=True,
                    allow_unused=True,
                    materialize_grads=False,
                )
                feature_gradients[name] = list(all_gradients[len(parameters) :])
            for level, _feature in enumerate(features):
                spatial: dict[str, Any] = {"batch": batch_index, "level": level}
                for view in ("actual", "pure", "localization"):
                    energy = feature_energy(feature_gradients[view][level], targets)
                    if energy is not None:
                        spatial.update({f"{view}_{key}": value for key, value in energy.items()})
                spatial_rows.append(spatial)

            rows.append(row)
            print(
                f"D6 batch {batch_index + 1}/{max_batches}: "
                f"actual={row['actual_vs_loc_all_final_cosine']:.4f} "
                f"pure={row['pure_vs_loc_all_final_cosine']:.4f}",
                flush=True,
            )
            del outputs, losses, gradients
    finally:
        handle.remove()
        model.decoder.eval()

    metadata = {
        "parameter_groups": {
            name: {"parameters": len(group), "elements": sum(parameter.numel() for parameter in group)}
            for name, group in groups.items()
        },
        "note": (
            "DEIMv2-N has no learned value projection inside final MSDeformableAttention; "
            "P4/P5 value-feature gradients are therefore reported separately."
        ),
    }
    return rows, spatial_rows, metadata


def classify_case(summary: dict[str, Any]) -> dict[str, Any]:
    actual = summary.get("actual_vs_loc_all_final", {})
    pure = summary.get("pure_vs_loc_all_final", {})
    actual_ci = actual.get("bootstrap_mean_95ci")
    pure_ci = pure.get("bootstrap_mean_95ci")
    actual_negative = actual_ci is not None and actual_ci[1] < 0 and (actual.get("negative_fraction") or 0) >= 0.60
    pure_negative = pure_ci is not None and pure_ci[1] < 0 and (pure.get("negative_fraction") or 0) >= 0.60
    if actual_negative and pure_negative:
        case = "A"
        action = "GO_R4A_STRONG: true classification-localization conflict survives removal of LQE coupling."
    elif actual_negative and not pure_negative:
        case = "B"
        action = "STOP_R4A: conflict is dominated by LQE/quality coupling, not pure spatial task conflict."
    elif pure_negative and not actual_negative:
        case = "C"
        action = "R4A_CONDITIONAL: LQE mitigates conflict; any split must preserve the native LQE bridge."
    else:
        case = "D"
        action = "STOP_R4A: no stable final-layer task-gradient conflict under the preregistered gate."
    return {
        "case": case,
        "actual_gate": actual_negative,
        "pure_gate": pure_negative,
        "action": action,
        "gate_definition": "mean cosine bootstrap 95% upper bound < 0 and negative-batch fraction >= 0.60",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--match-epoch", type=int, default=MATCH_EPOCH)
    parser.add_argument("--allow-nonfinal-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reject_test_paths([args.config, args.checkpoint, args.train_images, args.train_annotations])
    if args.output_dir.exists():
        raise FileExistsError(f"D6 refuses existing output directory: {args.output_dir}")
    for path in (args.config, args.checkpoint, args.train_images, args.train_annotations):
        if not path.exists():
            raise FileNotFoundError(path)
    checkpoint_hash = sha256(args.checkpoint)
    if not args.allow_nonfinal_checkpoint and checkpoint_hash != EXPECTED_B0_SHA256:
        raise ValueError(f"Unexpected frozen B0 checkpoint SHA256: {checkpoint_hash}")
    args.output_dir.mkdir(parents=True)
    nofile = raise_nofile_limit()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    solver, model = build_solver(args.config, args.checkpoint, args.output_dir, args.device, args.seed)
    loader = deterministic_loader(
        args.config,
        args.train_images,
        args.train_annotations,
        args.batch_size,
        args.workers,
        args.device,
        args.seed,
    )
    try:
        rows, spatial_rows, metadata = audit(
            model, solver.criterion, loader, solver.device, args.max_batches, args.match_epoch
        )
    finally:
        solver.cleanup()
        dist_utils.cleanup()
    if not rows:
        raise RuntimeError("D6 produced no batches")
    write_csv(args.output_dir / "d6_gradient_rows.csv", rows)
    write_csv(args.output_dir / "d6_spatial_energy_rows.csv", spatial_rows)
    summary = summarize_rows(rows, args.seed)
    decision = classify_case(summary)
    report = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen B0; deterministic no-augmentation Train subset; no optimizer; Test not read.",
        "inputs": {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "train_annotations": str(args.train_annotations),
            "train_annotations_sha256": sha256(args.train_annotations),
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "match_epoch": args.match_epoch,
        },
        "runtime": {"device": args.device, "seed": args.seed, "nofile": nofile},
        "definitions": {
            "actual": "Final main post-LQE MAL gradient from the unchanged training graph.",
            "pure": "Final main pre-LQE raw-logit MAL gradient with actual final matching held fixed.",
            "localization": "Final main weighted bbox + GIoU + FGL; main DDF is absent by DEIMv2 design.",
            "full_actual": "Robustness view summing all MAL vs all bbox/GIoU/FGL/DDF loss keys.",
            "per_class": "Matched class-channel MAL vs matched class localization; diagnostic only.",
        },
        "metadata": metadata,
        "summary": summary,
        "spatial_energy_rows": len(spatial_rows),
        "decision": decision,
    }
    (args.output_dir / "d6_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)
    print("D6_STATUS=PASS", flush=True)


if __name__ == "__main__":
    main()
