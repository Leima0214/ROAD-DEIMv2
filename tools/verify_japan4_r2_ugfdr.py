#!/usr/bin/env python3
"""Static and synthetic preflight for the Japan4 UG-FDR 32E screens."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.core import yaml_utils  # noqa: E402
from engine.deim.deim_decoder import DEIMTransformer  # noqa: E402


B0_CONFIG = ROOT / "configs/deimv2/deimv2_hgnetv2_n_japan4.yml"
R2A_CONFIG = ROOT / "configs/deimv2/deimv2_hgnetv2_n_japan4_r2a_fdr_refiner_all_32e.yml"
R2B_CONFIG = ROOT / "configs/deimv2/deimv2_hgnetv2_n_japan4_r2b_fdr_refiner_entropy25_32e.yml"
L3_CONFIG = ROOT / "configs/deimv2/deimv2_hgnetv2_n_japan4_r2ctrl_full_l3_32e.yml"


def load_config(path: Path) -> dict:
    # yaml_utils.load_config has a mutable compatibility argument; always pass
    # a fresh mapping so separate experiment configs cannot contaminate one
    # another during this audit.
    return yaml_utils.load_config(str(path), cfg={})


def flatten(value, prefix="") -> dict:
    if not isinstance(value, dict):
        return {prefix: value}
    output = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        output.update(flatten(item, path))
    return output


def changed_paths(base: dict, candidate: dict) -> set[str]:
    base_flat = flatten(base)
    candidate_flat = flatten(candidate)
    keys = set(base_flat) | set(candidate_flat)
    return {
        key for key in keys
        if base_flat.get(key, object()) != candidate_flat.get(key, object())
    }


def assert_config_scope(base: dict, candidate: dict, allowed: set[str]) -> list[str]:
    changes = changed_paths(base, candidate)
    unexpected = sorted(
        key for key in changes
        if key != "__include__" and key not in allowed
    )
    if unexpected:
        raise AssertionError(f"Unexpected config differences: {unexpected}")
    if candidate["epoches"] != 32:
        raise AssertionError("Every R2 screen must stop at 32 epochs")
    return sorted(key for key in changes if key != "__include__")


def decoder_kwargs(config: dict) -> dict:
    kwargs = copy.deepcopy(config["DEIMTransformer"])
    # Small dynamic tensors keep the preflight CPU-friendly; all architectural
    # widths, FDR bins, heads, levels and points remain the actual N settings.
    kwargs.update({
        "num_classes": 4,
        "num_queries": 20,
        "num_denoising": 0,
        "eval_spatial_size": None,
    })
    return kwargs


def build_decoder(config: dict, seed: int = 42) -> DEIMTransformer:
    torch.manual_seed(seed)
    return DEIMTransformer(**decoder_kwargs(config))


def synthetic_features(batch_size: int = 1) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(20260811)
    return [
        torch.randn(batch_size, 128, 8, 8, generator=generator),
        torch.randn(batch_size, 128, 4, 4, generator=generator),
    ]


def max_output_difference(left: dict, right: dict) -> dict[str, float]:
    return {
        "boxes": float((left["pred_boxes"] - right["pred_boxes"]).abs().max()),
        "logits": float((left["pred_logits"] - right["pred_logits"]).abs().max()),
    }


def verify_identity_and_gate(base: dict, r2a: dict, r2b: dict) -> dict:
    b0_model = build_decoder(base).eval()
    r2a_model = build_decoder(r2a).eval()
    r2b_model = build_decoder(r2b).eval()

    b0_state = b0_model.state_dict()
    r2a_state = r2a_model.state_dict()
    shared = sorted(set(b0_state) & set(r2a_state))
    unequal = [key for key in shared if not torch.equal(b0_state[key], r2a_state[key])]
    if unequal:
        raise AssertionError(f"R2 perturbed B0 shared initialization: {unequal[:5]}")

    extra = sorted(set(r2a_state) - set(b0_state))
    if not extra or any(not key.startswith("decoder.fdr_refiner.") for key in extra):
        raise AssertionError(f"Unexpected R2-only state keys: {extra}")

    features = synthetic_features()
    with torch.no_grad():
        b0_output = b0_model(features)
        r2a_output = r2a_model(features)
        r2b_output = r2b_model(features)
    r2a_diff = max_output_difference(b0_output, r2a_output)
    r2b_diff = max_output_difference(b0_output, r2b_output)
    if max(*r2a_diff.values(), *r2b_diff.values()) > 1e-7:
        raise AssertionError(f"Zero-init identity failed: R2A={r2a_diff}, R2B={r2b_diff}")

    corners = torch.randn(2, 20, 4 * 33, generator=torch.Generator().manual_seed(7))
    all_gate = r2a_model.decoder.fdr_refiner.build_gate(corners)
    entropy_gate = r2b_model.decoder.fdr_refiner.build_gate(corners)
    expected_routed = math.ceil(corners.shape[1] * 0.25)
    if not torch.equal(all_gate.sum(dim=1), torch.full((2,), 20., dtype=all_gate.dtype)):
        raise AssertionError("R2-A did not route every query")
    if not torch.equal(
            entropy_gate.sum(dim=1),
            torch.full((2,), float(expected_routed), dtype=entropy_gate.dtype)):
        raise AssertionError("R2-B did not route exactly the fixed top 25%")

    dn_corners = torch.randn(2, 30, 4 * 33, generator=torch.Generator().manual_seed(8))
    dn_gate = r2b_model.decoder.fdr_refiner.build_gate(
        dn_corners,
        dn_meta={"dn_num_split": [10, 20]})
    dn_routed = dn_gate[:, :10].sum(dim=1)
    matching_routed = dn_gate[:, 10:].sum(dim=1)
    if not torch.equal(dn_routed, torch.full((2,), 3., dtype=dn_gate.dtype)):
        raise AssertionError("R2-B did not rank denoising queries within their own segment")
    if not torch.equal(matching_routed, torch.full((2,), 5., dtype=dn_gate.dtype)):
        raise AssertionError("R2-B matching-query route changed when DN queries were present")

    return {
        "shared_tensor_count": len(shared),
        "r2_only_tensor_count": len(extra),
        "r2a_initial_max_diff": r2a_diff,
        "r2b_initial_max_diff": r2b_diff,
        "r2a_routed_per_20": [int(value) for value in all_gate.sum(dim=1)],
        "r2b_routed_per_20": [int(value) for value in entropy_gate.sum(dim=1)],
        "r2b_dn_routed_per_10": [int(value) for value in dn_routed],
        "r2b_matching_routed_per_20_with_dn": [int(value) for value in matching_routed],
    }


def verify_two_stage_gradients(r2a: dict) -> dict:
    model = build_decoder(r2a).train()
    refiner = model.decoder.fdr_refiner
    optimizer = torch.optim.SGD(refiner.parameters(), lr=1e-3)

    first_output = model(synthetic_features())
    first_output["pred_boxes"].sum().backward()
    final_head = refiner.residual_head.layers[-1]
    first_head_grad = float(final_head.weight.grad.abs().sum())
    first_cross_grad = sum(
        0. if parameter.grad is None else float(parameter.grad.abs().sum())
        for parameter in refiner.cross_attn.parameters()
    )
    if not math.isfinite(first_head_grad) or first_head_grad <= 0.:
        raise AssertionError("Zero-initialized residual head did not receive a finite first-step gradient")
    if first_cross_grad != 0.:
        raise AssertionError("Cross-attention should remain locked on the exact zero-residual first step")

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second_output = model(synthetic_features())
    second_output["pred_boxes"].sum().backward()
    second_cross_grad = sum(
        0. if parameter.grad is None else float(parameter.grad.abs().sum())
        for parameter in refiner.cross_attn.parameters()
    )
    if not math.isfinite(second_cross_grad) or second_cross_grad <= 0.:
        raise AssertionError("Cross-attention did not unlock after the residual head left zero")

    return {
        "first_head_grad_l1": first_head_grad,
        "first_cross_attn_grad_l1": first_cross_grad,
        "second_cross_attn_grad_l1": second_cross_grad,
    }


def verify_denoising_forward(r2b: dict) -> dict:
    kwargs = decoder_kwargs(r2b)
    kwargs["num_denoising"] = 10
    torch.manual_seed(42)
    model = DEIMTransformer(**kwargs).train()
    targets = [{
        "labels": torch.tensor([0, 1], dtype=torch.long),
        "boxes": torch.tensor([
            [0.30, 0.35, 0.12, 0.08],
            [0.70, 0.65, 0.18, 0.10],
        ], dtype=torch.float32),
    }]
    output = model(synthetic_features(), targets=targets)
    if output["pred_boxes"].shape != (1, 20, 4):
        raise AssertionError(f"Unexpected matching-query output shape: {output['pred_boxes'].shape}")
    if "dn_outputs" not in output or "dn_meta" not in output:
        raise AssertionError("Denoising outputs disappeared after entropy routing")
    return {
        "matching_box_shape": list(output["pred_boxes"].shape),
        "aux_output_count": len(output["aux_outputs"]),
        "dn_output_count": len(output["dn_outputs"]),
        "dn_num_split": output["dn_meta"]["dn_num_split"],
    }


def main() -> None:
    base = load_config(B0_CONFIG)
    r2a = load_config(R2A_CONFIG)
    r2b = load_config(R2B_CONFIG)
    l3 = load_config(L3_CONFIG)

    common_refiner_changes = {
        "epoches",
        "output_dir",
        "DEIMTransformer.fdr_refiner",
        "DEIMTransformer.fdr_refiner_gate",
        "DEIMTransformer.fdr_refiner_fraction",
        "DEIMTransformer.fdr_refiner_num_points",
    }
    config_changes = {
        "r2a": assert_config_scope(base, r2a, common_refiner_changes),
        "r2b": assert_config_scope(base, r2b, common_refiner_changes),
        "l3_control": assert_config_scope(base, l3, {
            "epoches",
            "output_dir",
            "DEIMTransformer.num_layers",
        }),
    }

    # The depth control must build as a genuine four-layer decoder, but it is
    # intentionally excluded from the identity claim made for R2-A/B.
    l3_model = build_decoder(l3)
    if len(l3_model.decoder.layers) != 4:
        raise AssertionError("Full-L3 control did not build four decoder layers")

    b0_model = build_decoder(base)
    r2a_model = build_decoder(r2a)
    b0_params = sum(parameter.numel() for parameter in b0_model.parameters())
    r2a_params = sum(parameter.numel() for parameter in r2a_model.parameters())

    report = {
        "status": "PASS",
        "scope": "synthetic engineering preflight; not detector-performance evidence",
        "config_changes": config_changes,
        "decoder_parameter_delta": r2a_params - b0_params,
        "decoder_parameter_delta_fraction": (r2a_params - b0_params) / b0_params,
        "identity_and_gate": verify_identity_and_gate(base, r2a, r2b),
        "gradient_semantics": verify_two_stage_gradients(r2a),
        "denoising_forward": verify_denoising_forward(r2b),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
