#!/usr/bin/env python3
"""Static and synthetic preflight for the Japan4 R3-A MASF screen."""

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
from engine.deim.hybrid_encoder import HybridEncoder  # noqa: E402


B0_CONFIG = ROOT / "configs/deimv2/deimv2_hgnetv2_n_japan4.yml"
R3A_CONFIG = ROOT / "configs/deimv2/deimv2_hgnetv2_n_japan4_r3a_masf_p4_32e.yml"


def load_config(path: Path) -> dict:
    return yaml_utils.load_config(str(path), cfg={})


def flatten(value, prefix="") -> dict:
    if not isinstance(value, dict):
        return {prefix: value}
    output = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        output.update(flatten(item, path))
    return output


def verify_scope(base: dict, candidate: dict) -> list[str]:
    base_flat, candidate_flat = flatten(base), flatten(candidate)
    keys = set(base_flat) | set(candidate_flat)
    changed = sorted(
        key for key in keys
        if key != "__include__"
        and base_flat.get(key, object()) != candidate_flat.get(key, object())
    )
    allowed = {
        "epoches", "output_dir",
        "HybridEncoder.masf_p4", "HybridEncoder.masf_kernel_size",
    }
    unexpected = sorted(set(changed) - allowed)
    if unexpected:
        raise AssertionError(f"Unexpected config differences: {unexpected}")
    if candidate["epoches"] != 32:
        raise AssertionError("R3-A must stop at 32 epochs")
    return changed


def encoder_kwargs(config: dict) -> dict:
    kwargs = copy.deepcopy(config["HybridEncoder"])
    kwargs["eval_spatial_size"] = None
    return kwargs


def synthetic_features() -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(20260812)
    return [
        torch.randn(2, 512, 8, 8, generator=generator),
        torch.randn(2, 1024, 4, 4, generator=generator),
    ]


def main() -> None:
    base, candidate = load_config(B0_CONFIG), load_config(R3A_CONFIG)
    changes = verify_scope(base, candidate)

    torch.manual_seed(42)
    b0 = HybridEncoder(**encoder_kwargs(base)).eval()
    torch.manual_seed(42)
    r3a = HybridEncoder(**encoder_kwargs(candidate)).eval()

    b0_state, r3a_state = b0.state_dict(), r3a.state_dict()
    shared = sorted(set(b0_state) & set(r3a_state))
    unequal = [key for key in shared if not torch.equal(b0_state[key], r3a_state[key])]
    if unequal:
        raise AssertionError(f"R3-A perturbed B0 shared initialization: {unequal[:5]}")
    extra = sorted(set(r3a_state) - set(b0_state))
    if not extra or any(not key.startswith("p4_masf.") for key in extra):
        raise AssertionError(f"Unexpected R3-A-only tensors: {extra}")

    features = synthetic_features()
    with torch.no_grad():
        b0_output, r3a_output = b0(features), r3a(features)
    max_diff = max(
        float((left - right).abs().max())
        for left, right in zip(b0_output, r3a_output)
    )
    if max_diff > 1e-7:
        raise AssertionError(f"Zero-init identity failed: max diff={max_diff}")

    train_model = HybridEncoder(**encoder_kwargs(candidate)).train()
    optimizer = torch.optim.SGD(train_model.p4_masf.parameters(), lr=1e-3)
    first = train_model(synthetic_features())[0].square().mean()
    first.backward()
    bn_scale_grad = float(train_model.p4_masf.project[-1].weight.grad.abs().sum())
    strip_grad_first = sum(
        0.0 if p.grad is None else float(p.grad.abs().sum())
        for name, p in train_model.p4_masf.named_parameters()
        if name.startswith(("horizontal.", "vertical.", "gate."))
    )
    if not math.isfinite(bn_scale_grad) or bn_scale_grad <= 0:
        raise AssertionError("Zero-init output scale has no first-step gradient")
    if strip_grad_first != 0:
        raise AssertionError("Strip/gate paths must stay locked on the identity step")

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    train_model(synthetic_features())[0].square().mean().backward()
    strip_grad_second = sum(
        0.0 if p.grad is None else float(p.grad.abs().sum())
        for name, p in train_model.p4_masf.named_parameters()
        if name.startswith(("horizontal.", "vertical.", "gate."))
    )
    if not math.isfinite(strip_grad_second) or strip_grad_second <= 0:
        raise AssertionError("Strip/gate paths did not unlock after the identity step")

    b0_params = sum(p.numel() for p in b0.parameters())
    r3a_params = sum(p.numel() for p in r3a.parameters())
    report = {
        "status": "PASS",
        "scope": "synthetic engineering preflight; not detector-performance evidence",
        "config_changes": changes,
        "shared_tensor_count": len(shared),
        "r3a_only_tensor_count": len(extra),
        "initial_max_output_diff": max_diff,
        "first_bn_scale_grad_l1": bn_scale_grad,
        "first_strip_gate_grad_l1": strip_grad_first,
        "second_strip_gate_grad_l1": strip_grad_second,
        "b0_encoder_params": b0_params,
        "r3a_encoder_params": r3a_params,
        "added_encoder_params": r3a_params - b0_params,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
