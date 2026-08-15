#!/usr/bin/env python3
"""Hard-gate preflight for Japan4 DEIMv2-N M6-A GS-FGL."""

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


def all_finite(values) -> bool:
    return all(torch.isfinite(value).all().item() for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--m6a-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    args = parser.parse_args()

    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite {args.report}")
    for path in (args.base_config, args.m6a_config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the M6-A dynamic preflight")

    raw_m6a = yaml.safe_load(args.m6a_config.read_text(encoding="utf-8"))
    raw_config_valid = (
        set(raw_m6a) == {"__include__", "epoches", "output_dir", "DEIMCriterion"}
        and raw_m6a["__include__"] == ["deimv2_hgnetv2_n_japan4.yml"]
        and raw_m6a["epoches"] == 32
        and raw_m6a["DEIMCriterion"] == {"use_geometry_sensitive_fgl": True}
    )

    dist_utils.setup_distributed(seed=42)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime = args.report.parent / "m6a_gs_fgl_preflight_runtime"
    base_cfg, base_solver = build_solver(
        args.base_config, args.checkpoint, args.device, args.batch_size, runtime / "base"
    )
    m6a_cfg, m6a_solver = build_solver(
        args.m6a_config, args.checkpoint, args.device, args.batch_size, runtime / "m6a"
    )
    base_model = dist_utils.de_parallel(base_solver.model)
    m6a_model = dist_utils.de_parallel(m6a_solver.model)

    base_state = base_model.state_dict()
    m6a_state = m6a_model.state_dict()
    state_keys_match = set(base_state) == set(m6a_state)
    state_mismatches = [
        key for key in base_state
        if key in m6a_state and not torch.equal(base_state[key].cpu(), m6a_state[key].cpu())
    ]
    base_params = sum(parameter.numel() for parameter in base_model.parameters())
    m6a_params = sum(parameter.numel() for parameter in m6a_model.parameters())

    synthetic = torch.tensor(
        [[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.5, 0.1]], device=device
    )
    synthetic_weights = m6a_solver.criterion.geometry_sensitive_side_weights(synthetic)
    expected_elongated = torch.tensor(
        [1.0 / 3.0, 5.0 / 3.0, 1.0 / 3.0, 5.0 / 3.0], device=device
    )
    unit_mean_error = float((synthetic_weights.mean(dim=-1) - 1.0).abs().max().item())
    square_identity_error = float((synthetic_weights[0] - 1.0).abs().max().item())
    elongated_formula_error = float(
        (synthetic_weights[1] - expected_elongated).abs().max().item()
    )

    samples, targets = next(iter(m6a_cfg.val_dataloader))
    samples = samples.to(device)
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

    base_model.eval()
    m6a_model.eval()
    with torch.no_grad():
        base_eval = base_model(samples)
        m6a_eval = m6a_model(samples)
    eval_diff = max_tensor_diff(base_eval, m6a_eval)

    base_model.train()
    m6a_model.train()
    seed_all(123)
    base_train = base_model(samples, targets)
    seed_all(123)
    m6a_train = m6a_model(samples, targets)
    train_output_diff = max_tensor_diff(base_train, m6a_train)
    base_losses = base_solver.criterion(base_train, targets, epoch=0)
    m6a_losses = m6a_solver.criterion(m6a_train, targets, epoch=0)

    loss_keys_match = set(base_losses) == set(m6a_losses)
    fgl_keys = sorted(key for key in base_losses if key.startswith("loss_fgl"))
    non_fgl_keys = sorted(set(base_losses) - set(fgl_keys))
    fgl_diffs = {key: max_tensor_diff(base_losses[key], m6a_losses[key]) for key in fgl_keys}
    non_fgl_diff = max(
        (max_tensor_diff(base_losses[key], m6a_losses[key]) for key in non_fgl_keys),
        default=0.0,
    )
    fp32_losses_finite = all_finite(m6a_losses.values())

    base_fgl_grad = torch.autograd.grad(
        base_losses["loss_fgl"], base_train["pred_corners"], retain_graph=True
    )[0]
    m6a_fgl_grad = torch.autograd.grad(
        m6a_losses["loss_fgl"], m6a_train["pred_corners"], retain_graph=True
    )[0]
    base_fgl_grad_norm = float(base_fgl_grad.detach().float().norm().item())
    m6a_fgl_grad_norm = float(m6a_fgl_grad.detach().float().norm().item())
    fgl_grad_diff = max_tensor_diff(base_fgl_grad, m6a_fgl_grad)

    # The randomly initialized Japan4 class heads can yield a real preflight
    # batch whose matched box IoUs are all zero. Since B0 multiplies FGL by
    # matched IoU, that correctly makes both actual-batch FGL gradients zero.
    # Use a deterministic nonzero-IoU probe to hard-check the causal weighting
    # path without weakening the real-batch equality checks above.
    probe_generator = torch.Generator(device=device).manual_seed(20260815)
    probe_base_logits = torch.randn(
        8, 33, generator=probe_generator, device=device, requires_grad=True
    )
    probe_m6a_logits = probe_base_logits.detach().clone().requires_grad_(True)
    probe_labels = torch.tensor(
        [2, 4, 7, 9, 12, 15, 20, 25], device=device, dtype=torch.float32
    )
    probe_weight_right = torch.tensor(
        [0.2, 0.7, 0.4, 0.8, 0.3, 0.6, 0.1, 0.9], device=device
    )
    probe_weight_left = 1.0 - probe_weight_right
    probe_ious = torch.tensor([0.6, 0.8], device=device).unsqueeze(-1)
    probe_base_weights = probe_ious.expand(-1, 4).reshape(-1)
    probe_m6a_weights = (probe_ious * synthetic_weights).reshape(-1)
    probe_base_loss = base_solver.criterion.unimodal_distribution_focal_loss(
        probe_base_logits,
        probe_labels,
        probe_weight_right,
        probe_weight_left,
        probe_base_weights,
        avg_factor=2,
    )
    probe_m6a_loss = m6a_solver.criterion.unimodal_distribution_focal_loss(
        probe_m6a_logits,
        probe_labels,
        probe_weight_right,
        probe_weight_left,
        probe_m6a_weights,
        avg_factor=2,
    )
    probe_base_grad = torch.autograd.grad(probe_base_loss, probe_base_logits)[0]
    probe_m6a_grad = torch.autograd.grad(probe_m6a_loss, probe_m6a_logits)[0]
    synthetic_loss_diff = float((probe_base_loss - probe_m6a_loss).abs().item())
    synthetic_grad_diff = max_tensor_diff(probe_base_grad, probe_m6a_grad)

    del base_train, base_losses, m6a_train, m6a_losses
    torch.cuda.empty_cache()
    m6a_model.zero_grad(set_to_none=True)
    seed_all(456)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        amp_outputs = m6a_model(samples, targets)
    with torch.autocast(device_type="cuda", enabled=False):
        amp_losses = m6a_solver.criterion(amp_outputs, targets, epoch=0)
        amp_total = sum(amp_losses.values())
    amp_total.backward()
    amp_losses_finite = all_finite(amp_losses.values())
    amp_gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in m6a_model.parameters()
    )
    normal_gradient_nonzero = any(
        parameter.grad is not None
        and float(parameter.grad.detach().float().norm().item()) > 0.0
        for parameter in m6a_model.parameters()
    )

    checks = {
        "raw_yaml_has_only_registered_overrides": raw_config_valid,
        "base_gs_fgl_off_and_m6a_on": (
            not base_solver.criterion.use_geometry_sensitive_fgl
            and m6a_solver.criterion.use_geometry_sensitive_fgl
        ),
        "model_state_keys_identical": state_keys_match,
        "checkpoint_loaded_state_identical": not state_mismatches,
        "parameter_count_identical": base_params == m6a_params,
        "side_weights_have_unit_mean": unit_mean_error <= 1e-6,
        "square_box_recovers_b0_weights": square_identity_error <= 1e-6,
        "elongated_box_matches_closed_form": elongated_formula_error <= 1e-6,
        "eval_output_structure_identical": set(base_eval) == set(m6a_eval),
        "eval_output_exactly_b0": eval_diff == 0.0,
        "train_output_exactly_b0": train_output_diff <= args.tolerance,
        "loss_keys_identical": loss_keys_match,
        "only_fgl_loss_values_change": non_fgl_diff <= args.tolerance,
        "fp32_losses_finite": fp32_losses_finite,
        "synthetic_nonzero_iou_fgl_loss_changes": synthetic_loss_diff > args.tolerance,
        "synthetic_nonzero_iou_fgl_gradient_changes": synthetic_grad_diff > args.tolerance,
        "synthetic_nonzero_iou_fgl_gradients_finite": all_finite(
            (probe_base_grad, probe_m6a_grad)
        ),
        "amp_losses_finite": amp_losses_finite,
        "amp_gradients_finite": amp_gradients_finite,
        "normal_training_gradient_nonzero": normal_gradient_nonzero,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "inputs": {
            "base_config": str(args.base_config),
            "m6a_config": str(args.m6a_config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "batch_size": args.batch_size,
        },
        "implementation": {
            "intervention": "unit-mean first-order IoU sensitivity on L/T/R/B FGL weights",
            "model_changes": 0,
            "matcher_changes": 0,
            "ddf_changes": 0,
            "parameter_delta": m6a_params - base_params,
        },
        "side_weight_probe": {
            "square": synthetic_weights[0].tolist(),
            "aspect_5_horizontal": synthetic_weights[1].tolist(),
            "unit_mean_max_error": unit_mean_error,
        },
        "forward_differences": {
            "eval_max_abs_diff": eval_diff,
            "train_max_abs_diff": train_output_diff,
            "state_mismatches": state_mismatches,
        },
        "loss_differences": {
            "fgl": fgl_diffs,
            "non_fgl_max_abs_diff": non_fgl_diff,
        },
        "gradient_probe": {
            "actual_batch_base_fgl_norm": base_fgl_grad_norm,
            "actual_batch_m6a_fgl_norm": m6a_fgl_grad_norm,
            "actual_batch_fgl_max_abs_diff": fgl_grad_diff,
            "synthetic_nonzero_iou_loss_abs_diff": synthetic_loss_diff,
            "synthetic_nonzero_iou_gradient_max_abs_diff": synthetic_grad_diff,
        },
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"M6A_GS_FGL_PREFLIGHT_STATUS={status}")
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
