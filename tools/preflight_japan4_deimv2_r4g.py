#!/usr/bin/env python3
"""Hard-gate preflight for the Japan4 DEIMv2-N R4-G experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS


LOC_PREFIXES = ("loss_bbox", "loss_giou", "loss_fgl", "loss_ddf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def max_tensor_diff(left: Any, right: Any) -> float:
    if torch.is_tensor(left):
        return float((left.detach().float() - right.detach().float()).abs().max().item())
    if isinstance(left, dict):
        keys = set(left) & set(right)
        return max((max_tensor_diff(left[key], right[key]) for key in keys), default=0.0)
    if isinstance(left, (list, tuple)):
        return max((max_tensor_diff(a, b) for a, b in zip(left, right)), default=0.0)
    return 0.0


def build_solver(config: Path, checkpoint: Path, device: str, batch_size: int, runtime_dir: Path):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    cfg = YAMLConfig(
        str(config),
        device=device,
        tuning=str(checkpoint),
        use_amp=False,
        output_dir=str(runtime_dir),
    )
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    for loader_name in ("train_dataloader", "val_dataloader"):
        cfg.yaml_cfg[loader_name]["total_batch_size"] = batch_size
        cfg.yaml_cfg[loader_name]["num_workers"] = 0
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    return cfg, solver


def grad_norm(loss: torch.Tensor, parameter: torch.nn.Parameter) -> float:
    gradient = torch.autograd.grad(
        loss, parameter, retain_graph=True, allow_unused=True, materialize_grads=False
    )[0]
    return 0.0 if gradient is None else float(gradient.detach().float().norm().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--r4g-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite {args.report}")
    for path in (args.base_config, args.r4g_config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the R4-G dynamic preflight")

    dist_utils.setup_distributed(seed=42)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime = args.report.parent / "r4g_preflight_runtime"
    base_cfg, base_solver = build_solver(
        args.base_config, args.checkpoint, args.device, args.batch_size, runtime / "base"
    )
    r4g_cfg, r4g_solver = build_solver(
        args.r4g_config, args.checkpoint, args.device, args.batch_size, runtime / "r4g"
    )
    base_model = dist_utils.de_parallel(base_solver.model)
    r4g_model = dist_utils.de_parallel(r4g_solver.model)

    base_params = sum(parameter.numel() for parameter in base_model.parameters())
    r4g_params = sum(parameter.numel() for parameter in r4g_model.parameters())
    decoder = r4g_model.decoder.decoder
    layer = decoder.layers[decoder.eval_idx]
    cls_scale = layer.gateway_cls_norm.scale
    loc_scale = layer.gateway.norm.scale

    state_base = base_model.state_dict()
    state_r4g = r4g_model.state_dict()
    shared_mismatch = [
        key for key, value in state_base.items()
        if key not in state_r4g or not torch.equal(value.cpu(), state_r4g[key].cpu())
    ]
    migrated_scale_diff = float((cls_scale.detach() - loc_scale.detach()).abs().max().item())

    loader = r4g_cfg.val_dataloader
    samples, targets = next(iter(loader))
    samples = samples.to(device)
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

    base_model.eval()
    r4g_model.eval()
    with torch.no_grad():
        base_eval = base_model(samples)
        r4g_eval = r4g_model(samples)
    logits_diff = max_tensor_diff(base_eval["pred_logits"], r4g_eval["pred_logits"])
    boxes_diff = max_tensor_diff(base_eval["pred_boxes"], r4g_eval["pred_boxes"])
    eval_tree_diff = max_tensor_diff(base_eval, r4g_eval)

    base_model.train()
    r4g_model.train()
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    base_train = base_model(samples, targets)
    base_losses = base_solver.criterion(base_train, targets, epoch=0)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    r4g_train = r4g_model(samples, targets)
    r4g_losses = r4g_solver.criterion(r4g_train, targets, epoch=0)
    loss_diffs = {
        name: abs(float(base_losses[name].detach()) - float(r4g_losses[name].detach()))
        for name in set(base_losses) & set(r4g_losses)
    }
    max_loss_diff = max(loss_diffs.values(), default=0.0)
    all_finite = all(torch.isfinite(value).all().item() for value in r4g_losses.values())

    actual_loss = r4g_losses["loss_mal"]
    localization_loss = sum(
        value for name, value in r4g_losses.items() if name in LOC_PREFIXES
    )
    gradient_norms = {
        "actual_to_cls_norm": grad_norm(actual_loss, cls_scale),
        "actual_to_loc_norm": grad_norm(actual_loss, loc_scale),
        "localization_to_cls_norm": grad_norm(localization_loss, cls_scale),
        "localization_to_loc_norm": grad_norm(localization_loss, loc_scale),
    }

    optimizer = r4g_cfg.optimizer
    weight_decay_by_parameter: dict[int, float] = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            weight_decay_by_parameter[id(parameter)] = float(group.get("weight_decay", 0.0))
    norm_weight_decay = {
        "cls": weight_decay_by_parameter[id(cls_scale)],
        "loc": weight_decay_by_parameter[id(loc_scale)],
    }

    checks = {
        "parameter_delta_is_128": r4g_params - base_params == 128,
        "all_shared_checkpoint_tensors_equal": not shared_mismatch,
        "migrated_norms_equal": migrated_scale_diff == 0.0,
        "eval_logits_equivalent": logits_diff <= args.tolerance,
        "eval_boxes_equivalent": boxes_diff <= args.tolerance,
        "eval_tree_equivalent": eval_tree_diff <= args.tolerance,
        "training_losses_equivalent": max_loss_diff <= args.tolerance,
        "losses_finite": all_finite,
        "classification_norm_receives_gradient": gradient_norms["actual_to_cls_norm"] > 0.0,
        "localization_norm_receives_gradient": gradient_norms["localization_to_loc_norm"] > 0.0,
        "localization_isolated_from_cls_norm": gradient_norms["localization_to_cls_norm"] == 0.0,
        "both_norms_no_weight_decay": norm_weight_decay == {"cls": 0.0, "loc": 0.0},
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "inputs": {
            "base_config": str(args.base_config),
            "r4g_config": str(args.r4g_config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "batch_size": args.batch_size,
        },
        "parameters": {"base": base_params, "r4g": r4g_params, "delta": r4g_params - base_params},
        "equivalence": {
            "migrated_scale_max_abs_diff": migrated_scale_diff,
            "eval_logits_max_abs_diff": logits_diff,
            "eval_boxes_max_abs_diff": boxes_diff,
            "eval_tree_max_abs_diff": eval_tree_diff,
            "max_loss_diff": max_loss_diff,
            "loss_diffs": loss_diffs,
            "shared_mismatch": shared_mismatch,
        },
        "gradient_norms": gradient_norms,
        "norm_weight_decay": norm_weight_decay,
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"R4G_PREFLIGHT_STATUS={status}")
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
