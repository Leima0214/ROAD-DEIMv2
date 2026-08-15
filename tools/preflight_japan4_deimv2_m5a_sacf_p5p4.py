#!/usr/bin/env python3
"""Hard-gate preflight for Japan4 DEIMv2-N M5-A SACF P5-to-P4."""

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
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS


EXPECTED_EXTRA_KEYS = {
    "encoder.p5_to_p4_competitive_fusion.gate.weight",
    "encoder.p5_to_p4_competitive_fusion.gate.bias",
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
        return float(
            (left.detach().float() - right.detach().float()).abs().max().item()
        )
    if isinstance(left, dict):
        if set(left) != set(right):
            return float("inf")
        return max(
            (max_tensor_diff(left[key], right[key]) for key in left), default=0.0
        )
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max(
            (max_tensor_diff(a, b) for a, b in zip(left, right)), default=0.0
        )
    return 0.0


def build_solver(
    config: Path,
    checkpoint: Path,
    device: str,
    batch_size: int,
    runtime_dir: Path,
):
    seed_all(42)
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


def parameter_group_count(optimizer, parameter: torch.nn.Parameter) -> int:
    return sum(
        candidate is parameter
        for group in optimizer.param_groups
        for candidate in group["params"]
    )


def grad_norm(gradient: torch.Tensor | None) -> float:
    return 0.0 if gradient is None else float(gradient.detach().float().norm().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--m5a-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    args = parser.parse_args()

    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite {args.report}")
    for path in (args.base_config, args.m5a_config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the M5-A dynamic preflight")

    raw_m5a = yaml.safe_load(args.m5a_config.read_text(encoding="utf-8"))
    raw_config_valid = (
        set(raw_m5a) == {"__include__", "epoches", "output_dir", "HybridEncoder"}
        and raw_m5a["__include__"] == ["deimv2_hgnetv2_n_japan4.yml"]
        and raw_m5a["epoches"] == 32
        and raw_m5a["HybridEncoder"] == {"use_sacf_p5p4": True}
    )

    dist_utils.setup_distributed(seed=42)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime = args.report.parent / "m5a_sacf_p5p4_preflight_runtime"
    base_cfg, base_solver = build_solver(
        args.base_config, args.checkpoint, args.device, args.batch_size, runtime / "base"
    )
    m5a_cfg, m5a_solver = build_solver(
        args.m5a_config, args.checkpoint, args.device, args.batch_size, runtime / "m5a"
    )
    base_model = dist_utils.de_parallel(base_solver.model)
    m5a_model = dist_utils.de_parallel(m5a_solver.model)
    base_state, m5a_state = base_model.state_dict(), m5a_model.state_dict()
    extra_keys = set(m5a_state) - set(base_state)
    missing_keys = set(base_state) - set(m5a_state)
    common_mismatches = [
        key
        for key in base_state
        if key in m5a_state
        and not torch.equal(base_state[key].cpu(), m5a_state[key].cpu())
    ]
    base_params = sum(parameter.numel() for parameter in base_model.parameters())
    m5a_params = sum(parameter.numel() for parameter in m5a_model.parameters())

    fusion = m5a_model.encoder.p5_to_p4_competitive_fusion
    gate_weight = fusion.gate.weight
    gate_bias = fusion.gate.bias
    probe_high = torch.randn(1, 128, 10, 14, device=device)
    probe_low = torch.randn(1, 128, 10, 14, device=device)
    with torch.no_grad():
        probe_logits = fusion.gate(torch.concat([probe_high, probe_low], dim=1))
        probe_weights = probe_logits.softmax(dim=1)
        probe_output = fusion(probe_high, probe_low)
        probe_reference = probe_high + probe_low
    probe_diff = max_tensor_diff(probe_output, probe_reference)
    gate_half_diff = float((probe_weights - 0.5).abs().max().item())

    samples, targets = next(iter(m5a_cfg.val_dataloader))
    samples = samples.to(device)
    targets = [
        {key: value.to(device) for key, value in target.items()}
        for target in targets
    ]

    route_calls = []

    def record_route(_module, inputs, output):
        with torch.no_grad():
            weights = _module.gate(torch.concat(list(inputs), dim=1)).softmax(dim=1)
        route_calls.append(
            {
                "high": list(inputs[0].shape),
                "low": list(inputs[1].shape),
                "output": list(output.shape),
                "weight_min": float(weights.min().item()),
                "weight_max": float(weights.max().item()),
            }
        )

    route_handle = fusion.register_forward_hook(record_route)
    base_model.eval()
    m5a_model.eval()
    with torch.no_grad():
        base_eval = base_model(samples)
        m5a_eval = m5a_model(samples)
    route_handle.remove()
    eval_diff = max_tensor_diff(base_eval, m5a_eval)

    # Capture deployment before any training-mode forward mutates BN statistics.
    base_deploy = copy.deepcopy(base_model).deploy()
    m5a_deploy = copy.deepcopy(m5a_model).deploy()
    base_deploy_params = sum(parameter.numel() for parameter in base_deploy.parameters())
    m5a_deploy_params = sum(parameter.numel() for parameter in m5a_deploy.parameters())
    with torch.no_grad():
        base_deploy_outputs = base_deploy(samples)
        m5a_deploy_outputs = m5a_deploy(samples)
    deploy_diff = max_tensor_diff(base_deploy_outputs, m5a_deploy_outputs)
    deploy_route = m5a_deploy.encoder.p5_to_p4_competitive_fusion

    base_model.train()
    m5a_model.train()
    seed_all(123)
    base_train = base_model(samples, targets)
    base_losses = base_solver.criterion(base_train, targets, epoch=0)
    seed_all(123)
    m5a_train = m5a_model(samples, targets)
    m5a_losses = m5a_solver.criterion(m5a_train, targets, epoch=0)
    train_output_diff = max_tensor_diff(base_train, m5a_train)
    loss_key_match = set(base_losses) == set(m5a_losses)
    loss_value_diff = max_tensor_diff(base_losses, m5a_losses)
    fp32_losses_finite = all(
        torch.isfinite(value).all().item() for value in m5a_losses.values()
    )

    base_total = sum(base_losses.values())
    m5a_total = sum(m5a_losses.values())
    base_decoder_parameter = base_model.decoder.decoder.layers[0].self_attn.in_proj_weight
    m5a_decoder_parameter = m5a_model.decoder.decoder.layers[0].self_attn.in_proj_weight
    base_encoder_parameter = base_model.encoder.input_proj[0].conv.weight
    m5a_encoder_parameter = m5a_model.encoder.input_proj[0].conv.weight
    base_gradients = torch.autograd.grad(
        base_total, [base_decoder_parameter, base_encoder_parameter]
    )
    m5a_gradients = torch.autograd.grad(
        m5a_total,
        [gate_weight, gate_bias, m5a_decoder_parameter, m5a_encoder_parameter],
    )
    gradient_norms = {
        "sacf_gate_weight": grad_norm(m5a_gradients[0]),
        "sacf_gate_bias": grad_norm(m5a_gradients[1]),
        "normal_decoder_l0_self_attention": grad_norm(m5a_gradients[2]),
        "normal_encoder_input_projection": grad_norm(m5a_gradients[3]),
    }
    common_gradient_diffs = {
        "decoder_l0_self_attention": max_tensor_diff(
            base_gradients[0], m5a_gradients[2]
        ),
        "encoder_input_projection": max_tensor_diff(
            base_gradients[1], m5a_gradients[3]
        ),
    }

    del base_train, base_losses, m5a_train, m5a_losses, base_total, m5a_total
    torch.cuda.empty_cache()
    m5a_model.zero_grad(set_to_none=True)
    seed_all(456)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        amp_outputs = m5a_model(samples, targets)
    with torch.autocast(device_type="cuda", enabled=False):
        amp_losses = m5a_solver.criterion(amp_outputs, targets, epoch=0)
        amp_total = sum(amp_losses.values())
    amp_total.backward()
    amp_losses_finite = all(
        torch.isfinite(value).all().item() for value in amp_losses.values()
    )
    amp_gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in m5a_model.parameters()
    )
    amp_gate_gradient_nonzero = (
        gate_weight.grad is not None
        and float(gate_weight.grad.detach().float().norm().item()) > 0.0
    )

    route_shape_valid = (
        len(route_calls) == 1
        and route_calls[0]["high"] == route_calls[0]["low"]
        and route_calls[0]["high"][1] == 128
        and route_calls[0]["output"][1] == 128
        and route_calls[0]["output"][2:] == route_calls[0]["high"][2:]
    )
    checks = {
        "raw_yaml_has_only_registered_overrides": raw_config_valid,
        "actual_encoder_is_HybridEncoder": type(m5a_model.encoder).__name__
        == "HybridEncoder",
        "base_off_and_m5a_sacf_on": (
            not base_model.encoder.use_sacf_p5p4
            and m5a_model.encoder.use_sacf_p5p4
        ),
        "two_level_p5_p4_sum_contract": (
            list(m5a_model.encoder.feat_strides) == [16, 32]
            and m5a_model.encoder.fuse_op == "sum"
        ),
        "only_expected_state_keys_added": extra_keys == EXPECTED_EXTRA_KEYS
        and not missing_keys,
        "all_common_checkpoint_state_identical": not common_mismatches,
        "parameter_delta_is_514": m5a_params - base_params == 514,
        "gate_shape_is_2x256x1x1": list(gate_weight.shape) == [2, 256, 1, 1],
        "gate_weight_is_zero_initialized": torch.count_nonzero(gate_weight).item()
        == 0,
        "gate_bias_is_zero_initialized": torch.count_nonzero(gate_bias).item() == 0,
        "standalone_gate_is_exactly_half": gate_half_diff == 0.0,
        "standalone_fusion_is_exact_b0_sum": probe_diff == 0.0,
        "actual_model_route_called_once_with_expected_shapes": route_shape_valid,
        "initial_eval_output_structure_identical": set(base_eval) == set(m5a_eval),
        "initial_eval_output_matches_b0": eval_diff <= args.tolerance,
        "initial_train_output_matches_b0": train_output_diff <= args.tolerance,
        "initial_loss_keys_match_b0": loss_key_match,
        "initial_loss_values_match_b0": loss_value_diff <= args.tolerance,
        "fp32_losses_finite": fp32_losses_finite,
        "sacf_gate_weight_gradient_nonzero": gradient_norms["sacf_gate_weight"] > 0.0,
        "sacf_gate_bias_gradient_nonzero": gradient_norms["sacf_gate_bias"] > 0.0,
        "normal_decoder_gradient_nonzero": gradient_norms[
            "normal_decoder_l0_self_attention"
        ]
        > 0.0,
        "normal_encoder_gradient_nonzero": gradient_norms[
            "normal_encoder_input_projection"
        ]
        > 0.0,
        "initial_common_decoder_gradient_matches_b0": common_gradient_diffs[
            "decoder_l0_self_attention"
        ]
        <= args.tolerance,
        "initial_common_encoder_gradient_matches_b0": common_gradient_diffs[
            "encoder_input_projection"
        ]
        <= args.tolerance,
        "gate_weight_is_in_optimizer_once": parameter_group_count(
            m5a_solver.optimizer, gate_weight
        )
        == 1,
        "gate_bias_is_in_optimizer_once": parameter_group_count(
            m5a_solver.optimizer, gate_bias
        )
        == 1,
        "amp_losses_finite": amp_losses_finite,
        "amp_gradients_finite": amp_gradients_finite,
        "amp_gate_gradient_nonzero": amp_gate_gradient_nonzero,
        "deploy_route_retained": deploy_route is not None,
        "deploy_output_structure_identical": set(base_deploy_outputs)
        == set(m5a_deploy_outputs),
        "initial_deploy_output_matches_b0": deploy_diff <= args.tolerance,
        "deployed_parameter_delta_is_514": m5a_deploy_params - base_deploy_params
        == 514,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "inputs": {
            "base_config": str(args.base_config),
            "m5a_config": str(args.m5a_config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "batch_size": args.batch_size,
        },
        "implementation": {
            "route": "P5/P4 sum -> identity-initialized spatial competition",
            "nearest_upsampling_retained": True,
            "scale_softmax": True,
            "initial_scale_multiplier": 1.0,
            "criterion_changes": 0,
            "extra_state_keys": sorted(extra_keys),
            "parameter_delta": m5a_params - base_params,
            "deployed_parameter_delta": m5a_deploy_params - base_deploy_params,
            "route_calls": route_calls,
        },
        "initialization": {
            "gate_weight_max_abs": float(gate_weight.detach().abs().max().item()),
            "gate_bias_max_abs": float(gate_bias.detach().abs().max().item()),
            "gate_half_max_abs_diff": gate_half_diff,
            "standalone_vs_sum_max_abs_diff": probe_diff,
        },
        "forward_differences": {
            "eval_max_abs_diff": eval_diff,
            "train_max_abs_diff": train_output_diff,
            "loss_max_abs_diff": loss_value_diff,
            "deploy_max_abs_diff": deploy_diff,
            "common_state_mismatches": common_mismatches,
        },
        "gradient_norms": gradient_norms,
        "common_gradient_diffs": common_gradient_diffs,
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"M5A_SACF_P5P4_PREFLIGHT_STATUS={status}")
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
