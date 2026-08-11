#!/usr/bin/env python3
"""Fail-fast preflight for the Japan4 DEIMv2-N R1 P3-to-P4 screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config


EXPECTED_PRETRAINED_SHA256 = (
    "e76c71a53534d767bb09bb4eaabba8c11aefb48df1dc4f9e120a5b58677ae2c6"
)
EXPECTED_B0_MATCHED_STATE_ITEMS = 628


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved_config(path: Path) -> dict[str, Any]:
    # Passing a fresh dictionary avoids yaml_utils.load_config's mutable default.
    config = load_config(str(path), {})
    config.pop("__include__", None)
    return config


def assert_single_variable_contract(
    b0_path: Path, r1_path: Path
) -> dict[str, Any]:
    b0 = resolved_config(b0_path)
    r1 = resolved_config(r1_path)

    expected = {
        "epoches": 32,
        "output_dir": "./outputs/deimv2_hgnetv2_n_japan4_r1_p3p4_32e",
        "return_idx": [1, 2, 3],
        "detail_in_channels": 256,
        "detail_gate_init": 0.0,
        "detail_act": "silu",
    }
    observed = {
        "epoches": r1["epoches"],
        "output_dir": r1["output_dir"],
        "return_idx": r1["HGNetv2"]["return_idx"],
        "detail_in_channels": r1["HybridEncoder"].get("detail_in_channels"),
        "detail_gate_init": r1["HybridEncoder"].get("detail_gate_init"),
        "detail_act": r1["HybridEncoder"].get("detail_act"),
    }
    if observed != expected:
        raise AssertionError(f"Unexpected R1 overrides: {observed}")

    normalized = copy.deepcopy(r1)
    normalized["epoches"] = b0["epoches"]
    normalized["output_dir"] = b0["output_dir"]
    normalized["HGNetv2"]["return_idx"] = b0["HGNetv2"]["return_idx"]
    for key in ("detail_in_channels", "detail_gate_init", "detail_act"):
        normalized["HybridEncoder"].pop(key)
    if normalized != b0:
        raise AssertionError(
            "R1 changes settings outside epoch/output/P3 detail adapter contract"
        )

    if b0["epoches"] != 160:
        raise AssertionError(f"Frozen B0 must remain 160E, got {b0['epoches']}")
    if r1["flat_epoch"] < r1["epoches"]:
        raise AssertionError("The 32E screen is no longer inside B0's flat-LR prefix")
    if r1["lr_gamma"] != 1.0:
        raise AssertionError("Frozen B0 lr_gamma drifted from 1.0")

    return {
        "b0_epoches": b0["epoches"],
        "r1_epoches": r1["epoches"],
        "flat_epoch": r1["flat_epoch"],
        "lr_gamma": r1["lr_gamma"],
        "comparison_epoch": 31,
        "only_intended_overrides": True,
    }


def build_model(config_path: Path, device: torch.device, seed: int):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    config = YAMLConfig(str(config_path), device=str(device))
    # Mirrors train.py -t: the complete COCO checkpoint is the only parent.
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = config.model.to(device)
    return config, model


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]
    if not isinstance(payload, dict) or not all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        raise TypeError(f"Unsupported checkpoint payload in {path}")
    return payload


def transfer_pretrained(
    b0_model: torch.nn.Module,
    r1_model: torch.nn.Module,
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required COCO parent checkpoint: {path}")
    digest = sha256_file(path)
    if digest != EXPECTED_PRETRAINED_SHA256:
        raise AssertionError(
            f"COCO parent SHA256 mismatch: {digest} != {EXPECTED_PRETRAINED_SHA256}"
        )

    pretrained = checkpoint_state(path)
    b0_state = b0_model.state_dict()
    r1_state = r1_model.state_dict()
    matched = {
        name: tensor
        for name, tensor in pretrained.items()
        if name in b0_state and tensor.shape == b0_state[name].shape
    }
    if len(matched) != EXPECTED_B0_MATCHED_STATE_ITEMS:
        raise AssertionError(
            f"Expected {EXPECTED_B0_MATCHED_STATE_ITEMS} matched B0 tensors, got {len(matched)}"
        )
    r1_matched = {
        name
        for name, tensor in pretrained.items()
        if name in r1_state and tensor.shape == r1_state[name].shape
    }
    if r1_matched != set(matched):
        raise AssertionError("R1 changed the transferable B0 checkpoint tensor set")

    b0_model.load_state_dict(matched, strict=False)
    r1_model.load_state_dict(matched, strict=False)
    parameter_names = {name for name, _ in b0_model.named_parameters()}
    total_numel = sum(parameter.numel() for parameter in b0_model.parameters())
    matched_numel = sum(
        b0_state[name].numel() for name in matched if name in parameter_names
    )
    return {
        "path": str(path),
        "sha256": digest,
        "state_items": len(pretrained),
        "matched_state_items": len(matched),
        "matched_parameter_numel": matched_numel,
        "b0_parameter_numel": total_numel,
        "b0_parameter_coverage": matched_numel / total_numel,
        "r1_transfer_set_matches_b0": True,
    }


def tensor_leaves(value: Any, prefix: str = "output") -> Iterator[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_leaves(item, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from tensor_leaves(item, f"{prefix}[{index}]")


def identity_check(
    b0_model: torch.nn.Module,
    r1_model: torch.nn.Module,
    device: torch.device,
    input_size: int,
) -> dict[str, Any]:
    b0_state = b0_model.state_dict()
    r1_state_before_sync = r1_model.state_dict()
    shared_initialization_mismatch = [
        name
        for name, tensor in b0_state.items()
        if name not in r1_state_before_sync
        or tensor.shape != r1_state_before_sync[name].shape
        or not torch.equal(tensor, r1_state_before_sync[name])
    ]
    if shared_initialization_mismatch:
        raise AssertionError(
            "R1 perturbed shared B0 seed initialization: "
            f"{shared_initialization_mismatch[:10]}"
        )

    incompatible = r1_model.load_state_dict(b0_state, strict=False)
    expected_missing = {
        name for name in r1_model.state_dict() if name.startswith("encoder.detail_")
    }
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise AssertionError(
            "B0-to-R1 state compatibility failed: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    if float(r1_model.encoder.detail_gate.detach().cpu()) != 0.0:
        raise AssertionError("R1 detail gate must initialize to exactly zero")

    for model in (b0_model, r1_model):
        model.encoder.eval_spatial_size = None
        model.decoder.eval_spatial_size = None
        model.eval()

    generator = torch.Generator(device=device).manual_seed(20260811)
    image = torch.randn(
        1, 3, input_size, input_size, generator=generator, device=device
    )
    with torch.inference_mode():
        b0_output = b0_model(image)
        r1_output = r1_model(image)

    b0_tensors = dict(tensor_leaves(b0_output))
    r1_tensors = dict(tensor_leaves(r1_output))
    if b0_tensors.keys() != r1_tensors.keys():
        raise AssertionError("B0 and R1 output structures differ")

    max_abs = 0.0
    for name, b0_tensor in b0_tensors.items():
        r1_tensor = r1_tensors[name]
        if b0_tensor.shape != r1_tensor.shape:
            raise AssertionError(
                f"Output shape mismatch for {name}: {b0_tensor.shape} != {r1_tensor.shape}"
            )
        difference = float((b0_tensor - r1_tensor).abs().max().detach().cpu())
        max_abs = max(max_abs, difference)
    if max_abs > 1e-6:
        raise AssertionError(f"Zero-gate identity error is too large: {max_abs}")

    return {
        "shared_state_items": len(b0_state),
        "shared_seed42_state_exact_before_sync": True,
        "new_r1_state_items": sorted(expected_missing),
        "output_tensor_count": len(b0_tensors),
        "max_abs_output_difference": max_abs,
        "tolerance": 1e-6,
    }


def gradient_check(
    r1_model: torch.nn.Module,
    device: torch.device,
    input_size: int,
) -> dict[str, Any]:
    r1_model.backbone.eval()
    r1_model.encoder.train()
    generator = torch.Generator(device=device).manual_seed(20260812)
    images = torch.randn(
        2, 3, input_size, input_size, generator=generator, device=device
    )
    with torch.no_grad():
        features = [feature.detach() for feature in r1_model.backbone(images)]
    expected_shapes = [
        (2, 256, input_size // 8, input_size // 8),
        (2, 512, input_size // 16, input_size // 16),
        (2, 1024, input_size // 32, input_size // 32),
    ]
    observed_shapes = [tuple(feature.shape) for feature in features]
    if observed_shapes != expected_shapes:
        raise AssertionError(
            f"Unexpected HGNetv2-B0 P3/P4/P5 shapes: {observed_shapes}"
        )

    adapter_parameters = list(r1_model.encoder.detail_downsample.named_parameters())
    r1_model.encoder.zero_grad(set_to_none=True)
    outputs = r1_model.encoder(features)
    loss = sum(output.float().square().mean() for output in outputs)
    loss.backward()

    gate_grad = r1_model.encoder.detail_gate.grad
    if gate_grad is None or not torch.isfinite(gate_grad) or gate_grad.abs().item() == 0:
        raise AssertionError(f"Gate did not receive a finite nonzero gradient: {gate_grad}")
    adapter_zero_grad_names = [
        name
        for name, parameter in adapter_parameters
        if parameter.grad is not None and torch.count_nonzero(parameter.grad).item() == 0
    ]
    if len(adapter_zero_grad_names) != len(adapter_parameters):
        raise AssertionError(
            "At gate=0 every adapter gradient must be exactly zero on the first backward"
        )

    with torch.no_grad():
        r1_model.encoder.detail_gate.fill_(1e-3)
    r1_model.encoder.zero_grad(set_to_none=True)
    outputs_unlocked = r1_model.encoder(features)
    unlocked_loss = sum(output.float().square().mean() for output in outputs_unlocked)
    unlocked_loss.backward()
    nonzero_adapter_gradients = [
        name
        for name, parameter in adapter_parameters
        if parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad).item() > 0
    ]
    if not nonzero_adapter_gradients:
        raise AssertionError("Adapter gradients did not unlock after gate departed zero")

    return {
        "backbone_feature_shapes": observed_shapes,
        "encoder_output_shapes": [tuple(output.shape) for output in outputs],
        "gate_gradient_at_zero": float(gate_grad.detach().cpu()),
        "adapter_gradient_tensors_zero_at_gate_zero": len(adapter_zero_grad_names),
        "adapter_gradient_tensors_nonzero_at_gate_1e3": len(nonzero_adapter_gradients),
        "first_backward_semantics": "gate-only",
    }


def optimizer_check(config: YAMLConfig, model: torch.nn.Module) -> dict[str, Any]:
    optimizer = config.optimizer
    gate = model.encoder.detail_gate
    gate_group = None
    for index, group in enumerate(optimizer.param_groups):
        if any(parameter is gate for parameter in group["params"]):
            gate_group = {
                "index": index,
                "lr": group["lr"],
                "weight_decay": group["weight_decay"],
            }
            break
    if gate_group is None:
        raise AssertionError("detail_gate is absent from all optimizer parameter groups")
    return {"detail_gate_group": gate_group, "parameter_groups": len(optimizer.param_groups)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--b0-config",
        type=Path,
        default=Path("configs/deimv2/deimv2_hgnetv2_n_japan4.yml"),
    )
    parser.add_argument(
        "--r1-config",
        type=Path,
        default=Path("configs/deimv2/deimv2_hgnetv2_n_japan4_r1_p3p4_32e.yml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("weights/deimv2_hgnetv2_n_coco_hf.pth"),
    )
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/japan4_deimv2_n_r1_p3p4_32e_preflight.json"),
    )
    args = parser.parse_args()

    if args.input_size < 320 or args.input_size % 32:
        raise ValueError("--input-size must be a multiple of 32 and at least 320")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    protocol = assert_single_variable_contract(args.b0_config, args.r1_config)
    b0_config, b0_model = build_model(args.b0_config, device, seed=42)
    r1_config, r1_model = build_model(args.r1_config, device, seed=42)
    del b0_config

    pretrained = {"skipped": True}
    if not args.skip_checkpoint:
        pretrained = transfer_pretrained(b0_model, r1_model, args.checkpoint)

    identity = identity_check(b0_model, r1_model, device, args.input_size)
    del b0_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gradients = gradient_check(r1_model, device, args.input_size)
    optimizer = optimizer_check(r1_config, r1_model)

    report = {
        "status": "PASS",
        "scope": "R1 32E engineering preflight; not detector-performance evidence",
        "device": str(device),
        "torch": torch.__version__,
        "protocol": protocol,
        "pretrained": pretrained,
        "identity": identity,
        "gradients": gradients,
        "optimizer": optimizer,
        "model": {
            "r1_parameter_numel": sum(parameter.numel() for parameter in r1_model.parameters()),
            "detail_adapter_parameter_numel": sum(
                parameter.numel()
                for parameter in r1_model.encoder.detail_downsample.parameters()
            ) + r1_model.encoder.detail_gate.numel(),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
