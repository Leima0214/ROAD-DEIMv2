#!/usr/bin/env python3
"""Hard-gate preflight for the Japan4 DEIMv2-N M2 EASE-L2 experiment."""

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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS


EXPECTED_EXTRA_KEYS = {
    "decoder.decoder.ease_relation_mlp.0.weight",
    "decoder.decoder.ease_relation_mlp.0.bias",
    "decoder.decoder.ease_relation_mlp.2.weight",
    "decoder.decoder.ease_relation_mlp.2.bias",
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
    parser.add_argument("--m2-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=2e-6)
    args = parser.parse_args()

    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite {args.report}")
    for path in (args.base_config, args.m2_config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the M2 dynamic preflight")

    dist_utils.setup_distributed(seed=42)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime = args.report.parent / "m2_ease_l2_preflight_runtime"
    base_cfg, base_solver = build_solver(
        args.base_config, args.checkpoint, args.device, args.batch_size, runtime / "base")
    m2_cfg, m2_solver = build_solver(
        args.m2_config, args.checkpoint, args.device, args.batch_size, runtime / "m2")
    base_model = dist_utils.de_parallel(base_solver.model)
    m2_model = dist_utils.de_parallel(m2_solver.model)
    base_state, m2_state = base_model.state_dict(), m2_model.state_dict()
    extra_keys = set(m2_state) - set(base_state)
    missing_keys = set(base_state) - set(m2_state)
    common_mismatches = [
        key for key in base_state
        if key in m2_state and not torch.equal(base_state[key].cpu(), m2_state[key].cpu())
    ]
    base_params = sum(parameter.numel() for parameter in base_model.parameters())
    m2_params = sum(parameter.numel() for parameter in m2_model.parameters())

    samples, targets = next(iter(m2_cfg.val_dataloader))
    samples = samples.to(device)
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

    base_model.eval()
    m2_model.eval()
    with torch.no_grad():
        base_eval = base_model(samples)
        m2_eval = m2_model(samples)
    eval_diff = max_tensor_diff(base_eval, m2_eval)

    base_model.train()
    m2_model.train()
    seed_all(123)
    base_train = base_model(samples, targets)
    base_losses = base_solver.criterion(base_train, targets, epoch=0)
    seed_all(123)
    m2_train = m2_model(samples, targets)
    m2_losses = m2_solver.criterion(m2_train, targets, epoch=0)
    train_diff = max_tensor_diff(base_train, m2_train)
    loss_key_match = set(base_losses) == set(m2_losses)
    loss_diffs = {
        key: abs(float(base_losses[key].detach()) - float(m2_losses[key].detach()))
        for key in set(base_losses) & set(m2_losses)
    }
    loss_diff = max(loss_diffs.values(), default=float("inf"))

    decoder = m2_model.decoder.decoder
    total_loss = sum(m2_losses.values())
    gradient_norms = {
        "relation_output_weight": grad_norm(total_loss, decoder.ease_relation_mlp[2].weight),
        "normal_l2_self_attention": grad_norm(
            total_loss, decoder.layers[2].self_attn.in_proj_weight),
    }
    fp32_losses_finite = all(torch.isfinite(value).all().item() for value in m2_losses.values())

    del base_train, base_losses, m2_train, m2_losses, total_loss
    torch.cuda.empty_cache()
    m2_model.zero_grad(set_to_none=True)
    seed_all(456)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        amp_outputs = m2_model(samples, targets)
    with torch.autocast(device_type="cuda", enabled=False):
        amp_losses = m2_solver.criterion(amp_outputs, targets, epoch=0)
        amp_total = sum(amp_losses.values())
    amp_total.backward()
    amp_losses_finite = all(torch.isfinite(value).all().item() for value in amp_losses.values())
    amp_gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in m2_model.parameters()
    )
    amp_relation_grad = decoder.ease_relation_mlp[2].weight.grad
    amp_relation_gradient_nonzero = (
        amp_relation_grad is not None and float(amp_relation_grad.float().norm().item()) > 0.0)

    base_deploy = copy.deepcopy(base_model).deploy()
    m2_deploy = copy.deepcopy(m2_model).deploy()
    base_deploy_params = sum(parameter.numel() for parameter in base_deploy.parameters())
    m2_deploy_params = sum(parameter.numel() for parameter in m2_deploy.parameters())
    with torch.no_grad():
        base_deploy_outputs = base_deploy(samples)
        m2_deploy_outputs = m2_deploy(samples)
    deploy_diff = max_tensor_diff(base_deploy_outputs, m2_deploy_outputs)
    native_l1_score_retained = isinstance(m2_deploy.decoder.dec_score_head[1], torch.nn.Linear)
    native_l1_lqe_retained = not isinstance(
        m2_deploy.decoder.decoder.lqe_layers[1], torch.nn.Identity)
    base_l1_deploy_heads_removed = (
        isinstance(base_deploy.decoder.dec_score_head[1], torch.nn.Identity)
        and isinstance(base_deploy.decoder.decoder.lqe_layers[1], torch.nn.Identity)
    )

    checks = {
        "actual_decoder_is_DEIMTransformer": type(m2_model.decoder).__name__ == "DEIMTransformer",
        "decoder_has_three_layers": len(decoder.layers) == 3,
        "ease_enabled_only_in_m2": (
            not base_model.decoder.decoder.use_ease_l2 and decoder.use_ease_l2),
        "only_expected_state_keys_added": extra_keys == EXPECTED_EXTRA_KEYS and not missing_keys,
        "all_common_checkpoint_state_identical": not common_mismatches,
        "parameter_delta_is_168": m2_params - base_params == 168,
        "relation_output_is_zero_initialized": (
            torch.count_nonzero(decoder.ease_relation_mlp[2].weight).item() == 0
            and torch.count_nonzero(decoder.ease_relation_mlp[2].bias).item() == 0),
        "eval_output_structure_identical": set(base_eval) == set(m2_eval),
        "initial_eval_output_equivalent": eval_diff <= args.tolerance,
        "initial_train_output_equivalent": train_diff <= args.tolerance,
        "base_loss_keys_identical": loss_key_match,
        "base_loss_values_equivalent": loss_diff <= args.tolerance,
        "fp32_losses_finite": fp32_losses_finite,
        "relation_module_gradient_nonzero": gradient_norms["relation_output_weight"] > 0.0,
        "normal_l2_gradient_nonzero": gradient_norms["normal_l2_self_attention"] > 0.0,
        "amp_losses_finite": amp_losses_finite,
        "amp_gradients_finite": amp_gradients_finite,
        "amp_relation_gradient_nonzero": amp_relation_gradient_nonzero,
        "native_l1_score_retained_for_m2_deploy": native_l1_score_retained,
        "native_l1_lqe_retained_for_m2_deploy": native_l1_lqe_retained,
        "base_l1_deploy_heads_still_removed": base_l1_deploy_heads_removed,
        "initial_deploy_output_equivalent": deploy_diff <= args.tolerance,
        "deployed_parameter_delta_is_2093": m2_deploy_params - base_deploy_params == 2093,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "inputs": {
            "base_config": str(args.base_config), "m2_config": str(args.m2_config),
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
            "batch_size": args.batch_size,
        },
        "implementation": {
            "route": "native supervised L1 detached score/LQE + IoU relation -> L2 MSelf-Attention",
            "matching_queries_only_during_dn_training": True,
            "criterion_changes": 0,
            "extra_state_keys": sorted(extra_keys),
            "parameter_delta": m2_params - base_params,
            "deployed_parameter_delta": m2_deploy_params - base_deploy_params,
        },
        "equivalence": {
            "eval_max_abs_diff": eval_diff,
            "train_max_abs_diff": train_diff,
            "loss_max_abs_diff": loss_diff,
            "deploy_max_abs_diff": deploy_diff,
            "common_state_mismatches": common_mismatches,
        },
        "gradient_norms": gradient_norms,
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"M2_EASE_L2_PREFLIGHT_STATUS={status}")
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
