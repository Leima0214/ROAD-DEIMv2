#!/usr/bin/env python3
"""Hard-gate preflight for Japan4 DEIMv2-N M9-A BI-FDR."""

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
    parser.add_argument("--m9a-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    args = parser.parse_args()

    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite {args.report}")
    for path in (args.base_config, args.m9a_config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the M9-A dynamic preflight")

    raw_m9a = yaml.safe_load(args.m9a_config.read_text(encoding="utf-8"))
    raw_config_valid = (
        set(raw_m9a) == {"__include__", "epoches", "output_dir", "DEIMCriterion"}
        and raw_m9a["__include__"] == ["deimv2_hgnetv2_n_japan4.yml"]
        and raw_m9a["epoches"] == 32
        and raw_m9a["DEIMCriterion"]
        == {
            "use_boundary_interval_fdr": True,
            "boundary_interval_epsilon": 0.0015625,
        }
    )

    dist_utils.setup_distributed(seed=42)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime = args.report.parent / "m9a_bi_fdr_preflight_runtime"
    base_cfg, base_solver = build_solver(
        args.base_config, args.checkpoint, args.device, args.batch_size, runtime / "base"
    )
    m9a_cfg, m9a_solver = build_solver(
        args.m9a_config, args.checkpoint, args.device, args.batch_size, runtime / "m9a"
    )
    base_model = dist_utils.de_parallel(base_solver.model)
    m9a_model = dist_utils.de_parallel(m9a_solver.model)

    base_state = base_model.state_dict()
    m9a_state = m9a_model.state_dict()
    state_mismatches = [
        key
        for key in base_state
        if key not in m9a_state
        or not torch.equal(base_state[key].cpu(), m9a_state[key].cpu())
    ]
    base_params = sum(parameter.numel() for parameter in base_model.parameters())
    m9a_params = sum(parameter.numel() for parameter in m9a_model.parameters())

    samples, targets = next(iter(m9a_cfg.val_dataloader))
    samples = samples.to(device)
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

    base_model.eval()
    m9a_model.eval()
    with torch.no_grad():
        base_eval = base_model(samples)
        m9a_eval = m9a_model(samples)
    eval_diff = max_tensor_diff(base_eval, m9a_eval)

    base_model.train()
    m9a_model.train()
    seed_all(123)
    base_train = base_model(samples, targets)
    seed_all(123)
    m9a_train = m9a_model(samples, targets)
    train_output_diff = max_tensor_diff(base_train, m9a_train)
    base_losses = base_solver.criterion(base_train, targets, epoch=0)
    m9a_losses = m9a_solver.criterion(m9a_train, targets, epoch=0)
    loss_keys_match = set(base_losses) == set(m9a_losses)
    fgl_keys = sorted(key for key in base_losses if key.startswith("loss_fgl"))
    non_fgl_keys = sorted(set(base_losses) - set(fgl_keys))
    non_fgl_diff = max(
        (max_tensor_diff(base_losses[key], m9a_losses[key]) for key in non_fgl_keys),
        default=0.0,
    )
    actual_fgl_diffs = {
        key: max_tensor_diff(base_losses[key], m9a_losses[key]) for key in fgl_keys
    }

    ref_points = torch.tensor(
        [[0.50, 0.50, 0.30, 0.20], [0.45, 0.55, 0.20, 0.35]],
        device=device,
    )
    boxes_xyxy = torch.tensor(
        [[0.28, 0.37, 0.73, 0.64], [0.31, 0.29, 0.62, 0.78]],
        device=device,
    )
    reg_scale = m9a_train["reg_scale"]
    up = m9a_train["up"]
    native, interval_dense = m9a_solver.criterion.boundary_interval_fdr_targets(
        ref_points, boxes_xyxy, reg_scale, up
    )
    _, zero_dense = m9a_solver.criterion.boundary_interval_fdr_targets(
        ref_points, boxes_xyxy, reg_scale, up, epsilon=0.0
    )
    native_dense = m9a_solver.criterion._two_bin_target_distribution(*native)
    mass_error = float((interval_dense.sum(dim=-1) - 1.0).abs().max().item())
    zero_width_error = max_tensor_diff(native_dense, zero_dense)
    interval_target_diff = max_tensor_diff(native_dense, interval_dense)
    support_sizes = (interval_dense > 0).sum(dim=-1)

    generator = torch.Generator(device=device).manual_seed(20260817)
    base_logits = torch.randn(
        interval_dense.shape, generator=generator, device=device, requires_grad=True
    )
    zero_logits = base_logits.detach().clone().requires_grad_(True)
    interval_logits = base_logits.detach().clone().requires_grad_(True)
    weights = torch.linspace(0.4, 0.9, interval_dense.shape[0], device=device)
    native_loss = base_solver.criterion.unimodal_distribution_focal_loss(
        base_logits, *native, weights, avg_factor=2
    )
    zero_dense_loss = m9a_solver.criterion.dense_distribution_loss(
        zero_logits, zero_dense, weights, avg_factor=2
    )
    interval_loss = m9a_solver.criterion.dense_distribution_loss(
        interval_logits, interval_dense, weights, avg_factor=2
    )
    native_grad = torch.autograd.grad(native_loss, base_logits)[0]
    zero_grad = torch.autograd.grad(zero_dense_loss, zero_logits)[0]
    interval_grad = torch.autograd.grad(interval_loss, interval_logits)[0]
    zero_loss_error = float((native_loss - zero_dense_loss).abs().item())
    zero_grad_error = max_tensor_diff(native_grad, zero_grad)
    interval_loss_diff = float((native_loss - interval_loss).abs().item())
    interval_grad_diff = max_tensor_diff(native_grad, interval_grad)

    fp32_losses_finite = all_finite(m9a_losses.values())
    m9a_model.zero_grad(set_to_none=True)
    sum(m9a_losses.values()).backward()
    fp32_gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in m9a_model.parameters()
    )

    del base_losses, m9a_losses
    m9a_model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    seed_all(456)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        amp_outputs = m9a_model(samples, targets)
    with torch.autocast(device_type="cuda", enabled=False):
        amp_losses = m9a_solver.criterion(amp_outputs, targets, epoch=0)
        amp_total = sum(amp_losses.values())
    amp_total.backward()
    amp_losses_finite = all_finite(amp_losses.values())
    amp_gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in m9a_model.parameters()
    )

    checks = {
        "raw_yaml_has_only_registered_overrides": raw_config_valid,
        "base_bi_fdr_off_and_m9a_on": (
            not base_solver.criterion.use_boundary_interval_fdr
            and m9a_solver.criterion.use_boundary_interval_fdr
        ),
        "model_state_keys_identical": set(base_state) == set(m9a_state),
        "checkpoint_loaded_state_identical": not state_mismatches,
        "parameter_count_identical": base_params == m9a_params,
        "eval_output_exactly_b0": eval_diff == 0.0,
        "train_output_exactly_b0": train_output_diff <= args.tolerance,
        "loss_keys_identical": loss_keys_match,
        "only_fgl_loss_values_change": non_fgl_diff <= args.tolerance,
        "interval_target_is_nonnegative": bool((interval_dense >= 0).all().item()),
        "interval_target_mass_is_one": mass_error <= args.tolerance,
        "zero_width_recovers_native_target": zero_width_error <= args.tolerance,
        "zero_width_recovers_native_loss": zero_loss_error <= args.tolerance,
        "zero_width_recovers_native_gradient": zero_grad_error <= args.tolerance,
        "one_pixel_interval_changes_target": interval_target_diff > args.tolerance,
        "one_pixel_interval_expands_support": bool((support_sizes > 2).any().item()),
        "one_pixel_interval_changes_loss": interval_loss_diff > args.tolerance,
        "one_pixel_interval_changes_gradient": interval_grad_diff > args.tolerance,
        "synthetic_gradients_finite": all_finite((native_grad, interval_grad)),
        "fp32_losses_finite": fp32_losses_finite,
        "fp32_gradients_finite": fp32_gradients_finite,
        "amp_losses_finite": amp_losses_finite,
        "amp_gradients_finite": amp_gradients_finite,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "inputs": {
            "base_config": str(args.base_config),
            "m9a_config": str(args.m9a_config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "epsilon_normalized": m9a_solver.criterion.boundary_interval_epsilon,
            "epsilon_at_640_pixels": (
                m9a_solver.criterion.boundary_interval_epsilon * 640.0
            ),
        },
        "implementation": {
            "center_mass": 0.5,
            "endpoint_mass_each": 0.25,
            "model_changes": 0,
            "matcher_changes": 0,
            "mal_l1_giou_ddf_changes": 0,
            "parameter_delta": m9a_params - base_params,
        },
        "equivalence": {
            "eval_max_abs_diff": eval_diff,
            "train_max_abs_diff": train_output_diff,
            "non_fgl_loss_max_abs_diff": non_fgl_diff,
            "actual_fgl_differences": actual_fgl_diffs,
            "state_mismatches": state_mismatches,
        },
        "target_probe": {
            "mass_max_abs_error": mass_error,
            "zero_width_target_max_abs_error": zero_width_error,
            "interval_target_max_abs_difference": interval_target_diff,
            "support_sizes": support_sizes.tolist(),
            "zero_width_loss_abs_error": zero_loss_error,
            "zero_width_gradient_max_abs_error": zero_grad_error,
            "interval_loss_abs_difference": interval_loss_diff,
            "interval_gradient_max_abs_difference": interval_grad_diff,
        },
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"M9A_BI_FDR_PREFLIGHT_STATUS={report['status']}")
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
