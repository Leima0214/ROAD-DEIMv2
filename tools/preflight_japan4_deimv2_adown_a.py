#!/usr/bin/env python3
"""Hard-gate preflight for Japan4 DEIMv2-N ADown-A."""

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
from engine.backbone.hgnetv2 import AdaptiveDownsample, ConvBNAct


EXPECTED_EXTRA_KEYS = {
    "backbone.stages.2.downsample.offset.weight",
    "backbone.stages.2.downsample.offset.bias",
}
EXPECTED_PARAMETER_DELTA = 256 * 18 * 3 * 3 + 18


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
        return float(
            (left.detach().float() - right.detach().float()).abs().max().item()
        ) if left.numel() else 0.0
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
    solver.optimizer = cfg.optimizer
    return cfg, solver


def optimizer_parameter_records(solver: Any, names: set[str]) -> list[dict[str, Any]]:
    model = dist_utils.de_parallel(solver.model)
    id_to_name = {id(parameter): name for name, parameter in model.named_parameters() if name in names}
    records: list[dict[str, Any]] = []
    for group_index, group in enumerate(solver.optimizer.param_groups):
        for parameter in group["params"]:
            name = id_to_name.get(id(parameter))
            if name is not None:
                records.append(
                    {
                        "name": name,
                        "group": group_index,
                        "lr": float(group["lr"]),
                        "weight_decay": float(group["weight_decay"]),
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--adown-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-tolerance", type=float, default=1e-4)
    parser.add_argument("--loss-tolerance", type=float, default=5e-4)
    args = parser.parse_args()

    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite {args.report}")
    for path in (args.base_config, args.adown_config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ADown-A dynamic preflight")

    raw_adown = yaml.safe_load(args.adown_config.read_text(encoding="utf-8"))
    raw_config_valid = (
        set(raw_adown) == {"__include__", "epoches", "output_dir", "HGNetv2"}
        and raw_adown["__include__"] == ["deimv2_hgnetv2_n_japan4.yml"]
        and raw_adown["epoches"] == 32
        and raw_adown["HGNetv2"] == {"adaptive_downsample_stage": 2}
    )

    dist_utils.setup_distributed(seed=42)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime = args.report.parent / "adown_a_preflight_runtime"
    base_cfg = base_solver = adown_cfg = adown_solver = None
    try:
        base_cfg, base_solver = build_solver(
            args.base_config, args.checkpoint, args.device, args.batch_size, runtime / "base"
        )
        adown_cfg, adown_solver = build_solver(
            args.adown_config, args.checkpoint, args.device, args.batch_size, runtime / "adown"
        )
        base_model = dist_utils.de_parallel(base_solver.model)
        adown_model = dist_utils.de_parallel(adown_solver.model)

        base_state = base_model.state_dict()
        adown_state = adown_model.state_dict()
        extra_keys = set(adown_state) - set(base_state)
        missing_keys = set(base_state) - set(adown_state)
        shared_mismatches = [
            key for key in base_state
            if key in adown_state and not torch.equal(base_state[key].cpu(), adown_state[key].cpu())
        ]
        base_params = sum(parameter.numel() for parameter in base_model.parameters())
        adown_params = sum(parameter.numel() for parameter in adown_model.parameters())

        base_down = base_model.backbone.stages[2].downsample
        adown_down = adown_model.backbone.stages[2].downsample
        other_stage_types_match = all(
            type(base_model.backbone.stages[index].downsample)
            is type(adown_model.backbone.stages[index].downsample)
            for index in (0, 1, 3)
        )
        offset_weight_max = float(adown_down.offset.weight.detach().abs().max().item())
        offset_bias_max = float(adown_down.offset.bias.detach().abs().max().item())

        offset_names = {
            "backbone.stages.2.downsample.offset.weight",
            "backbone.stages.2.downsample.offset.bias",
        }
        optimizer_records = optimizer_parameter_records(adown_solver, offset_names)
        optimizer_counts = {
            name: sum(record["name"] == name for record in optimizer_records) for name in offset_names
        }
        optimizer_lr_correct = all(abs(record["lr"] - 0.0004) <= 1e-12 for record in optimizer_records)

        samples, targets = next(iter(adown_cfg.train_dataloader))
        samples = samples.to(device)
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        base_capture: dict[str, torch.Tensor] = {}
        adown_capture: dict[str, torch.Tensor] = {}
        base_handle = base_down.register_forward_hook(
            lambda _module, _inputs, output: base_capture.__setitem__("output", output.detach())
        )
        adown_handle = adown_down.register_forward_hook(
            lambda _module, _inputs, output: adown_capture.__setitem__("output", output.detach())
        )
        try:
            base_model.eval()
            adown_model.eval()
            with torch.no_grad():
                base_eval = base_model(samples)
                adown_eval = adown_model(samples)
            downsample_eval_diff = max_tensor_diff(base_capture["output"], adown_capture["output"])
            eval_diff = max_tensor_diff(base_eval, adown_eval)
        finally:
            base_handle.remove()
            adown_handle.remove()

        base_model.train()
        adown_model.train()
        seed_all(123)
        base_train = base_model(samples, targets)
        seed_all(123)
        adown_train = adown_model(samples, targets)
        train_output_diff = max_tensor_diff(base_train, adown_train)
        base_losses = base_solver.criterion(base_train, targets, epoch=0)
        adown_losses = adown_solver.criterion(adown_train, targets, epoch=0)
        loss_keys_match = set(base_losses) == set(adown_losses)
        loss_diffs = {
            key: max_tensor_diff(base_losses[key], adown_losses[key])
            for key in sorted(set(base_losses) & set(adown_losses))
        }
        max_loss_diff = max(loss_diffs.values(), default=0.0)
        fp32_losses_finite = all_finite(adown_losses.values())

        adown_model.zero_grad(set_to_none=True)
        fp32_total = sum(adown_losses.values())
        fp32_total.backward()
        offset_weight_grad = adown_down.offset.weight.grad
        offset_bias_grad = adown_down.offset.bias.grad
        conv_grad = adown_down.conv.weight.grad
        offset_grad_norm = float(offset_weight_grad.detach().float().norm().item())
        offset_bias_grad_norm = float(offset_bias_grad.detach().float().norm().item())
        conv_grad_norm = float(conv_grad.detach().float().norm().item())
        fp32_gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in adown_model.parameters()
        )

        del base_train, adown_train, base_losses, adown_losses
        adown_model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        seed_all(456)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            amp_outputs = adown_model(samples, targets)
        with torch.autocast(device_type="cuda", enabled=False):
            amp_losses = adown_solver.criterion(amp_outputs, targets, epoch=0)
            amp_total = sum(amp_losses.values())
        amp_total.backward()
        amp_losses_finite = all_finite(amp_losses.values())
        amp_gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in adown_model.parameters()
        )
        amp_offset_grad_norm = float(adown_down.offset.weight.grad.detach().float().norm().item())

        checks = {
            "raw_yaml_has_only_registered_overrides": raw_config_valid,
            "base_stage3_is_original_convbnact": type(base_down) is ConvBNAct,
            "adown_stage3_is_adaptive_downsample": isinstance(adown_down, AdaptiveDownsample),
            "all_other_downsample_stage_types_unchanged": other_stage_types_match,
            "only_expected_state_keys_added": extra_keys == EXPECTED_EXTRA_KEYS and not missing_keys,
            "all_shared_checkpoint_states_identical": not shared_mismatches,
            "parameter_delta_exact": adown_params - base_params == EXPECTED_PARAMETER_DELTA,
            "offset_predictor_zero_initialized": offset_weight_max == 0.0 and offset_bias_max == 0.0,
            "offset_parameters_each_in_optimizer_once": all(
                optimizer_counts[name] == 1 for name in offset_names
            ),
            "offset_parameters_use_backbone_lr": optimizer_lr_correct,
            "downsample_eval_numerically_b0": downsample_eval_diff <= args.output_tolerance,
            "eval_output_numerically_b0": eval_diff <= args.output_tolerance,
            "train_output_numerically_b0": train_output_diff <= args.output_tolerance,
            "loss_keys_identical": loss_keys_match,
            "initial_loss_values_numerically_b0": max_loss_diff <= args.loss_tolerance,
            "fp32_losses_finite": fp32_losses_finite,
            "fp32_offset_weight_gradient_nonzero": offset_grad_norm > 0.0,
            "fp32_offset_bias_gradient_nonzero": offset_bias_grad_norm > 0.0,
            "fp32_original_depthwise_gradient_nonzero": conv_grad_norm > 0.0,
            "fp32_gradients_finite": fp32_gradients_finite,
            "amp_losses_finite": amp_losses_finite,
            "amp_gradients_finite": amp_gradients_finite,
            "amp_offset_gradient_nonzero": amp_offset_grad_norm > 0.0,
        }
        report = {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "inputs": {
                "base_config": str(args.base_config),
                "base_config_sha256": sha256(args.base_config),
                "adown_config": str(args.adown_config),
                "adown_config_sha256": sha256(args.adown_config),
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": sha256(args.checkpoint),
                "device": args.device,
                "batch_size": args.batch_size,
            },
            "architecture": {
                "base_parameters": base_params,
                "adown_parameters": adown_params,
                "parameter_delta": adown_params - base_params,
                "expected_parameter_delta": EXPECTED_PARAMETER_DELTA,
                "extra_state_keys": sorted(extra_keys),
                "missing_state_keys": sorted(missing_keys),
                "shared_state_mismatch_count": len(shared_mismatches),
                "shared_state_mismatches": shared_mismatches,
                "optimizer_records": optimizer_records,
                "offset_weight_abs_max": offset_weight_max,
                "offset_bias_abs_max": offset_bias_max,
            },
            "equivalence": {
                "downsample_eval_max_abs_diff": downsample_eval_diff,
                "eval_output_max_abs_diff": eval_diff,
                "train_output_max_abs_diff": train_output_diff,
                "max_initial_loss_abs_diff": max_loss_diff,
                "per_loss_abs_diff": loss_diffs,
                "output_tolerance": args.output_tolerance,
                "loss_tolerance": args.loss_tolerance,
            },
            "gradients": {
                "fp32_offset_weight_norm": offset_grad_norm,
                "fp32_offset_bias_norm": offset_bias_grad_norm,
                "fp32_original_depthwise_norm": conv_grad_norm,
                "amp_offset_weight_norm": amp_offset_grad_norm,
            },
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        if report["status"] != "PASS":
            raise SystemExit(2)
    finally:
        for solver in (base_solver, adown_solver):
            if solver is not None:
                solver.cleanup()
        dist_utils.cleanup()


if __name__ == "__main__":
    main()
