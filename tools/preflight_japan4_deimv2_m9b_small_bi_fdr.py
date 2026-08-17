#!/usr/bin/env python3
"""Hard-gate preflight for Japan4 DEIMv2-N M9-B small-only BI-FDR."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS
from engine.deim.box_ops import box_cxcywh_to_xyxy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def max_tensor_diff(left: Any, right: Any) -> float:
    if torch.is_tensor(left):
        if left.shape != right.shape:
            return float("inf")
        return (
            float((left.detach().float() - right.detach().float()).abs().max().item())
            if left.numel()
            else 0.0
        )
    if isinstance(left, dict):
        if set(left) != set(right):
            return float("inf")
        return max((max_tensor_diff(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max((max_tensor_diff(a, b) for a, b in zip(left, right)), default=0.0)
    return 0.0


def all_finite(values) -> bool:
    return all(torch.isfinite(value).all().item() for value in values)


def build_solver(config: Path, checkpoint: Path, device: str, batch_size: int, runtime: Path):
    seed_all(42)
    cfg = YAMLConfig(
        str(config),
        device=device,
        tuning=str(checkpoint),
        use_amp=False,
        output_dir=str(runtime),
    )
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    for loader_name in ("train_dataloader", "val_dataloader"):
        cfg.yaml_cfg[loader_name]["total_batch_size"] = batch_size
        cfg.yaml_cfg[loader_name]["num_workers"] = 0
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    return cfg, solver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--m9b-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    args = parser.parse_args()

    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite {args.report}")
    for path in (args.base_config, args.m9b_config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the M9-B dynamic preflight")

    raw_m9b = yaml.safe_load(args.m9b_config.read_text(encoding="utf-8"))
    raw_config_valid = (
        set(raw_m9b) == {"__include__", "epoches", "output_dir", "DEIMCriterion"}
        and raw_m9b["__include__"] == ["deimv2_hgnetv2_n_japan4.yml"]
        and raw_m9b["epoches"] == 32
        and raw_m9b["DEIMCriterion"]
        == {
            "use_boundary_interval_fdr": True,
            "boundary_interval_epsilon": 0.0015625,
            "boundary_interval_small_area_threshold": 0.0025,
        }
    )

    dist_utils.setup_distributed(seed=42)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime = args.report.parent / "m9b_small_bi_fdr_preflight_runtime"
    base_cfg, base_solver = build_solver(
        args.base_config, args.checkpoint, args.device, args.batch_size, runtime / "base"
    )
    m9b_cfg, m9b_solver = build_solver(
        args.m9b_config, args.checkpoint, args.device, args.batch_size, runtime / "m9b"
    )
    base_model = dist_utils.de_parallel(base_solver.model)
    m9b_model = dist_utils.de_parallel(m9b_solver.model)

    resize_ops = [
        op for op in m9b_cfg.yaml_cfg["train_dataloader"]["dataset"]["transforms"]["ops"]
        if op.get("type") == "Resize"
    ]
    fixed_resize_640 = (
        len(resize_ops) == 1
        and list(resize_ops[0]["size"]) == [640, 640]
        and m9b_cfg.yaml_cfg["train_dataloader"]["collate_fn"].get(
            "base_size_repeat"
        ) is None
    )

    base_state = base_model.state_dict()
    m9b_state = m9b_model.state_dict()
    state_mismatches = [
        key
        for key in base_state
        if key not in m9b_state
        or not torch.equal(base_state[key].cpu(), m9b_state[key].cpu())
    ]
    base_params = sum(parameter.numel() for parameter in base_model.parameters())
    m9b_params = sum(parameter.numel() for parameter in m9b_model.parameters())

    samples, targets = next(iter(m9b_cfg.val_dataloader))
    samples = samples.to(device)
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
    base_model.eval()
    m9b_model.eval()
    with torch.no_grad():
        base_eval = base_model(samples)
        m9b_eval = m9b_model(samples)
    eval_diff = max_tensor_diff(base_eval, m9b_eval)

    base_model.train()
    m9b_model.train()
    seed_all(123)
    base_train = base_model(samples, targets)
    seed_all(123)
    m9b_train = m9b_model(samples, targets)
    train_output_diff = max_tensor_diff(base_train, m9b_train)
    base_losses = base_solver.criterion(base_train, targets, epoch=0)
    m9b_losses = m9b_solver.criterion(m9b_train, targets, epoch=0)
    loss_keys_match = set(base_losses) == set(m9b_losses)
    fgl_keys = sorted(key for key in base_losses if key.startswith("loss_fgl"))
    non_fgl_keys = sorted(set(base_losses) - set(fgl_keys))
    non_fgl_diff = max(
        (max_tensor_diff(base_losses[key], m9b_losses[key]) for key in non_fgl_keys),
        default=0.0,
    )

    boxes_cxcywh = torch.tensor(
        [[0.50, 0.50, 0.04, 0.04], [0.465, 0.535, 0.31, 0.49]],
        device=device,
    )
    active_mask = m9b_solver.criterion.boundary_interval_active_mask(boxes_cxcywh)
    ref_points = torch.tensor(
        [[0.50, 0.50, 0.08, 0.08], [0.45, 0.55, 0.20, 0.35]],
        device=device,
    )
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    reg_scale = m9b_train["reg_scale"]
    up = m9b_train["up"]
    native, global_dense = m9b_solver.criterion.boundary_interval_fdr_targets(
        ref_points, boxes_xyxy, reg_scale, up
    )
    _, gated_dense = m9b_solver.criterion.boundary_interval_fdr_targets(
        ref_points, boxes_xyxy, reg_scale, up, active_box_mask=active_mask
    )
    native_dense = m9b_solver.criterion._two_bin_target_distribution(*native)
    small_rows = slice(0, 4)
    medium_rows = slice(4, 8)
    small_matches_global = max_tensor_diff(
        gated_dense[small_rows], global_dense[small_rows]
    )
    medium_matches_native = max_tensor_diff(
        gated_dense[medium_rows], native_dense[medium_rows]
    )
    medium_global_effect = max_tensor_diff(
        global_dense[medium_rows], native_dense[medium_rows]
    )
    mass_error = float((gated_dense.sum(dim=-1) - 1.0).abs().max().item())

    generator = torch.Generator(device=device).manual_seed(20260817)
    base_logits = torch.randn(
        gated_dense.shape, generator=generator, device=device, requires_grad=True
    )
    gated_logits = base_logits.detach().clone().requires_grad_(True)
    weights = torch.linspace(0.4, 0.9, gated_dense.shape[0], device=device)
    native_loss = base_solver.criterion.unimodal_distribution_focal_loss(
        base_logits, *native, weights, avg_factor=2
    )
    gated_loss = m9b_solver.criterion.dense_distribution_loss(
        gated_logits, gated_dense, weights, avg_factor=2
    )
    native_grad = torch.autograd.grad(native_loss, base_logits)[0]
    gated_grad = torch.autograd.grad(gated_loss, gated_logits)[0]
    small_grad_diff = max_tensor_diff(
        native_grad[small_rows], gated_grad[small_rows]
    )
    medium_grad_diff = max_tensor_diff(
        native_grad[medium_rows], gated_grad[medium_rows]
    )

    fp32_losses_finite = all_finite(m9b_losses.values())
    m9b_model.zero_grad(set_to_none=True)
    sum(m9b_losses.values()).backward()
    fp32_gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in m9b_model.parameters()
    )
    del base_losses, m9b_losses
    m9b_model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    seed_all(456)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        amp_outputs = m9b_model(samples, targets)
    with torch.autocast(device_type="cuda", enabled=False):
        amp_losses = m9b_solver.criterion(amp_outputs, targets, epoch=0)
        amp_total = sum(amp_losses.values())
    amp_total.backward()
    amp_losses_finite = all_finite(amp_losses.values())
    amp_gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in m9b_model.parameters()
    )

    checks = {
        "raw_yaml_has_only_registered_overrides": raw_config_valid,
        "training_pipeline_is_fixed_640": fixed_resize_640,
        "threshold_is_exactly_32_over_640_squared": abs(
            m9b_solver.criterion.boundary_interval_small_area_threshold
            - (32.0 / 640.0) ** 2
        ) <= 1e-12,
        "model_state_keys_identical": set(base_state) == set(m9b_state),
        "checkpoint_loaded_state_identical": not state_mismatches,
        "parameter_count_identical": base_params == m9b_params,
        "eval_output_exactly_b0": eval_diff == 0.0,
        "train_output_exactly_b0": train_output_diff <= args.tolerance,
        "loss_keys_identical": loss_keys_match,
        "only_fgl_loss_values_change": non_fgl_diff <= args.tolerance,
        "synthetic_mask_selects_only_small": active_mask.tolist() == [True, False],
        "small_target_exactly_matches_m9a": small_matches_global <= args.tolerance,
        "medium_target_exactly_matches_b0": medium_matches_native <= args.tolerance,
        "global_m9a_would_change_medium": medium_global_effect > args.tolerance,
        "gated_target_mass_is_one": mass_error <= args.tolerance,
        "small_fgl_gradient_changes": small_grad_diff > args.tolerance,
        "medium_fgl_gradient_exactly_b0": medium_grad_diff <= args.tolerance,
        "synthetic_gradients_finite": all_finite((native_grad, gated_grad)),
        "fp32_losses_finite": fp32_losses_finite,
        "fp32_gradients_finite": fp32_gradients_finite,
        "amp_losses_finite": amp_losses_finite,
        "amp_gradients_finite": amp_gradients_finite,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "inputs": {
            "base_config": str(args.base_config),
            "m9b_config": str(args.m9b_config),
            "checkpoint_sha256": sha256(args.checkpoint),
            "epsilon_pixels_at_640": (
                m9b_solver.criterion.boundary_interval_epsilon * 640.0
            ),
            "training_scale_small_area_threshold": (
                m9b_solver.criterion.boundary_interval_small_area_threshold
            ),
        },
        "implementation": {
            "routing": "augmented normalized w*h < (32/640)^2",
            "short_side_routing": False,
            "model_changes": 0,
            "matcher_changes": 0,
            "other_loss_changes": 0,
            "parameter_delta": m9b_params - base_params,
        },
        "equivalence": {
            "eval_max_abs_diff": eval_diff,
            "train_max_abs_diff": train_output_diff,
            "non_fgl_loss_max_abs_diff": non_fgl_diff,
            "state_mismatches": state_mismatches,
        },
        "routing_probe": {
            "active_mask": active_mask.tolist(),
            "small_matches_global_max_error": small_matches_global,
            "medium_matches_native_max_error": medium_matches_native,
            "global_medium_target_difference": medium_global_effect,
            "target_mass_max_error": mass_error,
            "small_gradient_difference": small_grad_diff,
            "medium_gradient_difference": medium_grad_diff,
        },
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"M9B_SMALL_BI_FDR_PREFLIGHT_STATUS={report['status']}")
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
