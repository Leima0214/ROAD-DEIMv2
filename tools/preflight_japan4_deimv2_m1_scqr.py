#!/usr/bin/env python3
"""Hard-gate preflight for the Japan4 DEIMv2-N M1 SCQR experiment."""

from __future__ import annotations

import argparse
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


SCQR_LOSS_KEYS = {
    "loss_mal_scqr",
    "loss_bbox_scqr",
    "loss_giou_scqr",
    "loss_fgl_scqr",
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
    parser.add_argument("--m1-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite {args.report}")
    for path in (args.base_config, args.m1_config, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the M1 dynamic preflight")

    dist_utils.setup_distributed(seed=42)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime = args.report.parent / "m1_scqr_preflight_runtime"
    base_cfg, base_solver = build_solver(
        args.base_config, args.checkpoint, args.device, args.batch_size, runtime / "base"
    )
    m1_cfg, m1_solver = build_solver(
        args.m1_config, args.checkpoint, args.device, args.batch_size, runtime / "m1"
    )
    base_model = dist_utils.de_parallel(base_solver.model)
    m1_model = dist_utils.de_parallel(m1_solver.model)

    base_params = sum(parameter.numel() for parameter in base_model.parameters())
    m1_params = sum(parameter.numel() for parameter in m1_model.parameters())
    state_base = base_model.state_dict()
    state_m1 = m1_model.state_dict()
    state_mismatches = [
        key for key, value in state_base.items()
        if key not in state_m1 or not torch.equal(value.cpu(), state_m1[key].cpu())
    ]
    state_mismatches.extend(key for key in state_m1 if key not in state_base)

    loader = m1_cfg.val_dataloader
    samples, targets = next(iter(loader))
    samples = samples.to(device)
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

    base_model.eval()
    m1_model.eval()
    with torch.no_grad():
        base_eval = base_model(samples)
        m1_eval = m1_model(samples)
    eval_diff = max_tensor_diff(base_eval, m1_eval)

    base_model.train()
    m1_model.train()
    seed_all(123)
    base_train = base_model(samples, targets)
    base_losses = base_solver.criterion(base_train, targets, epoch=0)
    seed_all(123)
    m1_train = m1_model(samples, targets)
    m1_losses = m1_solver.criterion(m1_train, targets, epoch=0)

    train_has_scqr = "scqr_outputs" in m1_train
    m1_main_train = {key: value for key, value in m1_train.items() if key != "scqr_outputs"}
    train_output_diff = max_tensor_diff(base_train, m1_main_train)
    m1_base_losses = {key: value for key, value in m1_losses.items() if not key.endswith("_scqr")}
    loss_key_match = set(base_losses) == set(m1_base_losses)
    loss_diffs = {
        key: abs(float(base_losses[key].detach()) - float(m1_base_losses[key].detach()))
        for key in set(base_losses) & set(m1_base_losses)
    }
    max_base_loss_diff = max(loss_diffs.values(), default=float("inf"))
    scqr_loss_keys = {key for key in m1_losses if key.endswith("_scqr")}
    scqr_total = sum(m1_losses[key] for key in sorted(scqr_loss_keys))

    decoder = m1_model.decoder.decoder
    gradient_norms = {
        "scqr_to_l0": grad_norm(scqr_total, decoder.layers[0].self_attn.in_proj_weight),
        "scqr_to_l1": grad_norm(scqr_total, decoder.layers[1].self_attn.in_proj_weight),
        "scqr_to_l2": grad_norm(scqr_total, decoder.layers[2].self_attn.in_proj_weight),
    }
    fp32_scqr_finite = all(torch.isfinite(m1_losses[key]).all().item() for key in scqr_loss_keys)

    del base_train, base_losses, m1_train, m1_losses, scqr_total
    torch.cuda.empty_cache()
    m1_model.zero_grad(set_to_none=True)
    seed_all(456)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        amp_outputs = m1_model(samples, targets)
    with torch.autocast(device_type="cuda", enabled=False):
        amp_losses = m1_solver.criterion(amp_outputs, targets, epoch=0)
        amp_total = sum(amp_losses.values())
    amp_total.backward()
    amp_losses_finite = all(torch.isfinite(value).all().item() for value in amp_losses.values())
    amp_gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in m1_model.parameters()
    )

    checks = {
        "actual_decoder_is_DEIMTransformer": type(m1_model.decoder).__name__ == "DEIMTransformer",
        "decoder_has_three_layers": len(decoder.layers) == 3,
        "scqr_enabled_only_in_m1": (
            not base_model.decoder.decoder.use_scqr and m1_model.decoder.decoder.use_scqr
        ),
        "lambda_scqr_is_one": m1_solver.criterion.scqr_loss_weight == 1.0,
        "parameter_count_unchanged": base_params == m1_params,
        "checkpoint_state_identical": not state_mismatches,
        "eval_has_no_scqr_branch": "scqr_outputs" not in m1_eval,
        "eval_output_identical": eval_diff <= args.tolerance,
        "train_has_independent_scqr_output": train_has_scqr,
        "base_train_output_identical": train_output_diff <= args.tolerance,
        "base_loss_keys_identical": loss_key_match,
        "base_loss_values_identical": max_base_loss_diff <= args.tolerance,
        "scqr_loss_keys_exact": scqr_loss_keys == SCQR_LOSS_KEYS,
        "fp32_scqr_losses_finite": fp32_scqr_finite,
        "scqr_gradient_reaches_l0": gradient_norms["scqr_to_l0"] > 0.0,
        "scqr_gradient_skips_l1": gradient_norms["scqr_to_l1"] == 0.0,
        "scqr_gradient_reaches_l2": gradient_norms["scqr_to_l2"] > 0.0,
        "amp_losses_finite": amp_losses_finite,
        "amp_gradients_finite": amp_gradients_finite,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "status": status,
        "inputs": {
            "base_config": str(args.base_config),
            "m1_config": str(args.m1_config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "batch_size": args.batch_size,
        },
        "implementation": {
            "instantiated_decoder": type(m1_model.decoder).__name__,
            "decoder_layers": len(decoder.layers),
            "scqr_route": "state_L0 -> shared_L2",
            "scqr_assignment": "independent Hungarian; excluded from GO union",
            "scqr_loss_weight": m1_solver.criterion.scqr_loss_weight,
            "scqr_loss_keys": sorted(scqr_loss_keys),
        },
        "equivalence": {
            "base_parameters": base_params,
            "m1_parameters": m1_params,
            "state_mismatches": state_mismatches,
            "eval_max_abs_diff": eval_diff,
            "base_train_output_max_abs_diff": train_output_diff,
            "base_loss_max_abs_diff": max_base_loss_diff,
            "base_loss_diffs": loss_diffs,
        },
        "gradient_norms": gradient_norms,
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"M1_SCQR_PREFLIGHT_STATUS={status}")
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
