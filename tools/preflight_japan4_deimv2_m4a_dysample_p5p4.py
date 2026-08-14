#!/usr/bin/env python3
"""Hard-gate preflight for Japan4 DEIMv2-N M4-A DySample P5-to-P4."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS


EXPECTED_EXTRA_KEYS = {
    "encoder.p5_to_p4_upsample.init_pos",
    "encoder.p5_to_p4_upsample.offset.weight",
    "encoder.p5_to_p4_upsample.offset.bias",
}


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
        return float((left.detach().float() - right.detach().float()).abs().max().item())
    if isinstance(left, dict):
        if set(left) != set(right):
            return float("inf")
        return max((max_tensor_diff(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max((max_tensor_diff(a, b) for a, b in zip(left, right)), default=0.0)
    return 0.0


def build_solver(config: Path, checkpoint: Path, device: str, batch_size: int, runtime_dir: Path):
    seed_all(42)
    cfg = YAMLConfig(
        str(config), device=device, tuning=str(checkpoint), use_amp=False,
        output_dir=str(runtime_dir)
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
    parser.add_argument("--m4a-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    args = parser.parse_args()

    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite {args.report}")
    for path in (args.base_config, args.m4a_config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the M4-A dynamic preflight")

    dist_utils.setup_distributed(seed=42)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime = args.report.parent / "m4a_dysample_p5p4_preflight_runtime"
    base_cfg, base_solver = build_solver(
        args.base_config, args.checkpoint, args.device, args.batch_size, runtime / "base")
    m4a_cfg, m4a_solver = build_solver(
        args.m4a_config, args.checkpoint, args.device, args.batch_size, runtime / "m4a")
    base_model = dist_utils.de_parallel(base_solver.model)
    m4a_model = dist_utils.de_parallel(m4a_solver.model)
    base_state, m4a_state = base_model.state_dict(), m4a_model.state_dict()
    extra_keys = set(m4a_state) - set(base_state)
    missing_keys = set(base_state) - set(m4a_state)
    common_mismatches = [
        key for key in base_state
        if key in m4a_state and not torch.equal(base_state[key].cpu(), m4a_state[key].cpu())
    ]
    base_params = sum(parameter.numel() for parameter in base_model.parameters())
    m4a_params = sum(parameter.numel() for parameter in m4a_model.parameters())

    dysample = m4a_model.encoder.p5_to_p4_upsample
    offset_weight = dysample.offset.weight
    offset_bias = dysample.offset.bias
    init_pos = dysample.init_pos
    probe = torch.randn(1, 128, 5, 7, device=device)
    with torch.no_grad():
        probe_output = dysample(probe)
        nearest_output = F.interpolate(probe, scale_factor=2.0, mode="nearest")
    probe_diff = max_tensor_diff(probe_output, nearest_output)

    samples, targets = next(iter(m4a_cfg.val_dataloader))
    samples = samples.to(device)
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

    route_shapes = []

    def record_route(_module, inputs, output):
        route_shapes.append({
            "input": list(inputs[0].shape),
            "output": list(output.shape),
        })

    route_handle = dysample.register_forward_hook(record_route)
    base_model.eval()
    m4a_model.eval()
    with torch.no_grad():
        base_eval = base_model(samples)
        m4a_eval = m4a_model(samples)
    route_handle.remove()
    eval_diff = max_tensor_diff(base_eval, m4a_eval)

    # Capture deployment before any training-mode forward mutates BN statistics.
    base_deploy = copy.deepcopy(base_model).deploy()
    m4a_deploy = copy.deepcopy(m4a_model).deploy()
    base_deploy_params = sum(parameter.numel() for parameter in base_deploy.parameters())
    m4a_deploy_params = sum(parameter.numel() for parameter in m4a_deploy.parameters())
    with torch.no_grad():
        base_deploy_outputs = base_deploy(samples)
        m4a_deploy_outputs = m4a_deploy(samples)
    deploy_diff = max_tensor_diff(base_deploy_outputs, m4a_deploy_outputs)
    deploy_route_present = m4a_deploy.encoder.p5_to_p4_upsample is not None

    base_model.train()
    m4a_model.train()
    seed_all(123)
    base_train = base_model(samples, targets)
    base_losses = base_solver.criterion(base_train, targets, epoch=0)
    seed_all(123)
    m4a_train = m4a_model(samples, targets)
    m4a_losses = m4a_solver.criterion(m4a_train, targets, epoch=0)
    loss_key_match = set(base_losses) == set(m4a_losses)
    fp32_losses_finite = all(torch.isfinite(value).all().item() for value in m4a_losses.values())
    total_loss = sum(m4a_losses.values())
    gradient_norms = {
        "dysample_offset_weight": grad_norm(total_loss, offset_weight),
        "normal_decoder_l0_self_attention": grad_norm(
            total_loss,
            m4a_model.decoder.decoder.layers[0].self_attn.in_proj_weight),
    }

    del base_train, base_losses, m4a_train, m4a_losses, total_loss
    torch.cuda.empty_cache()
    m4a_model.zero_grad(set_to_none=True)
    seed_all(456)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        amp_outputs = m4a_model(samples, targets)
    with torch.autocast(device_type="cuda", enabled=False):
        amp_losses = m4a_solver.criterion(amp_outputs, targets, epoch=0)
        amp_total = sum(amp_losses.values())
    amp_total.backward()
    amp_losses_finite = all(torch.isfinite(value).all().item() for value in amp_losses.values())
    amp_gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in m4a_model.parameters()
    )
    amp_offset_grad = offset_weight.grad
    amp_offset_gradient_nonzero = (
        amp_offset_grad is not None and float(amp_offset_grad.float().norm().item()) > 0.0)

    route_shape_valid = (
        len(route_shapes) == 1
        and route_shapes[0]["input"][1] == 128
        and route_shapes[0]["output"][1] == 128
        and route_shapes[0]["output"][2] == 2 * route_shapes[0]["input"][2]
        and route_shapes[0]["output"][3] == 2 * route_shapes[0]["input"][3]
    )
    checks = {
        "actual_encoder_is_HybridEncoder": type(m4a_model.encoder).__name__ == "HybridEncoder",
        "base_uses_nearest_and_m4a_uses_dysample": (
            not base_model.encoder.use_dysample_p5p4
            and m4a_model.encoder.use_dysample_p5p4),
        "two_level_p5_p4_contract": list(m4a_model.encoder.feat_strides) == [16, 32],
        "official_lp_groups4_without_scope": (
            dysample.scale == 2 and dysample.groups == 4 and not hasattr(dysample, "scope")),
        "only_expected_state_keys_added": extra_keys == EXPECTED_EXTRA_KEYS and not missing_keys,
        "all_common_checkpoint_state_identical": not common_mismatches,
        "parameter_delta_is_4128": m4a_params - base_params == 4128,
        "offset_shape_is_32x128x1x1": list(offset_weight.shape) == [32, 128, 1, 1],
        "offset_bias_is_zero_initialized": torch.count_nonzero(offset_bias).item() == 0,
        "offset_weight_is_small_nonzero": (
            torch.count_nonzero(offset_weight).item() > 0
            and float(offset_weight.detach().abs().max().item()) < 0.01),
        "init_position_shape_is_expected": list(init_pos.shape) == [1, 32, 1, 1],
        "standalone_output_shape_is_2x": list(probe_output.shape) == [1, 128, 10, 14],
        "replacement_changes_nearest_function": probe_diff > args.tolerance,
        "actual_model_route_called_once_with_2x_shape": route_shape_valid,
        "eval_output_structure_identical": set(base_eval) == set(m4a_eval),
        "replacement_changes_initial_eval_output": eval_diff > args.tolerance,
        "base_loss_keys_identical": loss_key_match,
        "fp32_losses_finite": fp32_losses_finite,
        "dysample_offset_gradient_nonzero": gradient_norms["dysample_offset_weight"] > 0.0,
        "normal_decoder_gradient_nonzero": gradient_norms["normal_decoder_l0_self_attention"] > 0.0,
        "amp_losses_finite": amp_losses_finite,
        "amp_gradients_finite": amp_gradients_finite,
        "amp_offset_gradient_nonzero": amp_offset_gradient_nonzero,
        "deploy_route_retained": deploy_route_present,
        "deploy_output_structure_identical": set(base_deploy_outputs) == set(m4a_deploy_outputs),
        "replacement_changes_deploy_output": deploy_diff > args.tolerance,
        "deployed_parameter_delta_is_4128": m4a_deploy_params - base_deploy_params == 4128,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "inputs": {
            "base_config": str(args.base_config),
            "m4a_config": str(args.m4a_config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "batch_size": args.batch_size,
        },
        "implementation": {
            "route": "HybridEncoder P5-to-P4 nearest -> LP DySample",
            "scale": dysample.scale,
            "groups": dysample.groups,
            "dyscope": False,
            "criterion_changes": 0,
            "extra_state_keys": sorted(extra_keys),
            "parameter_delta": m4a_params - base_params,
            "deployed_parameter_delta": m4a_deploy_params - base_deploy_params,
            "route_shapes": route_shapes,
        },
        "initialization": {
            "offset_weight_std": float(offset_weight.detach().float().std().item()),
            "offset_weight_max_abs": float(offset_weight.detach().abs().max().item()),
            "offset_bias_max_abs": float(offset_bias.detach().abs().max().item()),
            "standalone_vs_nearest_max_abs_diff": probe_diff,
        },
        "forward_differences": {
            "eval_max_abs_diff": eval_diff,
            "deploy_max_abs_diff": deploy_diff,
            "common_state_mismatches": common_mismatches,
        },
        "gradient_norms": gradient_norms,
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"M4A_DYSAMPLE_P5P4_PREFLIGHT_STATUS={status}")
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
