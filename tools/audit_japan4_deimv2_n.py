#!/usr/bin/env python3
"""Static GPU audit for the official DEIMv2-N model on Japan4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.solver import TASKS


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def finite_gradients(model: torch.nn.Module) -> dict[str, object]:
    gradients = [(name, parameter.grad) for name, parameter in model.named_parameters() if parameter.grad is not None]
    nonfinite = [name for name, grad in gradients if not torch.isfinite(grad).all()]
    nonzero = [name for name, grad in gradients if torch.count_nonzero(grad).item() > 0]
    squared_norm = sum(float(grad.detach().float().pow(2).sum()) for _, grad in gradients)
    return {
        "gradient_tensors": len(gradients),
        "nonzero_gradient_tensors": len(nonzero),
        "nonfinite_gradient_tensors": nonfinite,
        "global_l2_norm": math.sqrt(squared_norm),
    }


def loss_values(loss_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().cpu()) for name, value in loss_dict.items()}


def prepare_checkpoint(safetensors_path: Path, checkpoint_path: Path) -> dict[str, torch.Tensor]:
    if not safetensors_path.is_file():
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            "Intellindust/DEIMv2_HGNetv2_N_COCO",
            "model.safetensors",
            local_dir=str(safetensors_path.parent),
        )
        if Path(downloaded).resolve() != safetensors_path.resolve():
            raise AssertionError(f"Unexpected Hugging Face download path: {downloaded}")
    state = load_file(str(safetensors_path), device="cpu")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state}, checkpoint_path)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/deimv2/deimv2_hgnetv2_n_japan4.yml"
    )
    parser.add_argument(
        "--safetensors", type=Path,
        default=Path("weights/hf_deimv2_n/model.safetensors"),
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("weights/deimv2_hgnetv2_n_coco_hf.pth"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("reports/japan4_deimv2_n_static_audit.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this audit")
    dist_utils.setup_distributed(seed=42)
    torch.cuda.set_device(torch.device(args.device))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    pretrained_state = prepare_checkpoint(args.safetensors, args.checkpoint)
    cfg = YAMLConfig(
        args.config,
        device=args.device,
        tuning=str(args.checkpoint),
        use_amp=True,
        output_dir="./reports/static_audit_runtime",
    )
    # This mirrors train.py -t: do not download/load a second backbone state.
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    for loader_name in ("train_dataloader", "val_dataloader"):
        cfg.yaml_cfg[loader_name]["total_batch_size"] = args.batch_size
        cfg.yaml_cfg[loader_name]["num_workers"] = 0

    model_before = cfg.model
    current_state = model_before.state_dict()
    matched_keys = {
        key for key, value in current_state.items()
        if key in pretrained_state and value.shape == pretrained_state[key].shape
    }
    mismatched_keys = {
        key for key, value in current_state.items()
        if key in pretrained_state and value.shape != pretrained_state[key].shape
    }
    missing_keys = set(current_state) - set(pretrained_state)
    random_before = {key: tensor_sha256(current_state[key]) for key in mismatched_keys | missing_keys}
    parameter_names = {name for name, _ in model_before.named_parameters()}
    total_parameter_numel = sum(parameter.numel() for parameter in model_before.parameters())
    matched_parameter_numel = sum(current_state[key].numel() for key in matched_keys if key in parameter_names)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver._setup()
    model = solver.model
    raw_model = dist_utils.de_parallel(model)
    loaded_state = raw_model.state_dict()
    transfer_mismatch = [
        key for key in matched_keys if not torch.equal(loaded_state[key].cpu(), pretrained_state[key])
    ]
    random_overwritten = [
        key for key, digest in random_before.items() if tensor_sha256(loaded_state[key]) != digest
    ]
    if transfer_mismatch:
        raise AssertionError(f"Matched pretrained tensors not retained: {transfer_mismatch[:5]}")
    if random_overwritten:
        raise AssertionError(f"Mismatched classification tensors were overwritten: {random_overwritten[:5]}")

    train_loader = cfg.train_dataloader
    val_loader = cfg.val_dataloader
    samples, targets = next(iter(train_loader))
    label_values = sorted({int(label) for target in targets for label in target["labels"].tolist()})
    if not label_values or min(label_values) < 0 or max(label_values) >= 4:
        raise AssertionError(f"Japan4 labels are not in 0..3: {label_values}")
    samples = samples.to(args.device)
    targets = [{key: value.to(args.device) for key, value in target.items()} for target in targets]
    metas = {"epoch": 0, "step": 0, "global_step": 0, "epoch_step": len(train_loader)}

    model.train()
    solver.criterion.train()
    model.zero_grad(set_to_none=True)
    outputs_fp32 = model(samples, targets=targets)
    losses_fp32 = solver.criterion(outputs_fp32, targets, **metas)
    total_fp32 = sum(losses_fp32.values())
    if not torch.isfinite(total_fp32):
        raise FloatingPointError(f"Non-finite FP32 loss: {loss_values(losses_fp32)}")
    total_fp32.backward()
    gradients_fp32 = finite_gradients(model)
    if gradients_fp32["nonfinite_gradient_tensors"]:
        raise FloatingPointError("Non-finite FP32 gradients")

    optimizer = cfg.optimizer
    scaler = torch.amp.GradScaler("cuda")
    amp_attempts = []
    for amp_attempt in range(1, 9):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs_amp = model(samples, targets=targets)
        with torch.autocast(device_type="cuda", enabled=False):
            losses_amp = solver.criterion(outputs_amp, targets, **metas)
            total_amp = sum(losses_amp.values())
        if not torch.isfinite(total_amp):
            raise FloatingPointError(f"Non-finite AMP loss: {loss_values(losses_amp)}")
        scale = float(scaler.get_scale())
        scaler.scale(total_amp).backward()
        scaler.unscale_(optimizer)
        gradients_amp = finite_gradients(model)
        amp_attempts.append(
            {
                "attempt": amp_attempt,
                "scale": scale,
                "finite_after_unscale": not gradients_amp["nonfinite_gradient_tensors"],
            }
        )
        if not gradients_amp["nonfinite_gradient_tensors"]:
            break
        scaler.update()
    else:
        raise FloatingPointError("AMP gradients remained non-finite after dynamic scale backoff")

    model.eval()
    val_samples, val_targets = next(iter(val_loader))
    val_samples = val_samples.to(args.device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        val_outputs = model(val_samples)
        original_sizes = torch.stack([target["orig_size"] for target in val_targets]).to(args.device)
        predictions = solver.postprocessor(val_outputs, original_sizes)

    report = {
        "status": "PASS",
        "config": args.config,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "pretrained": {
            "source": "Intellindust/DEIMv2_HGNetv2_N_COCO/model.safetensors",
            "safetensors_sha256": hashlib.sha256(args.safetensors.read_bytes()).hexdigest(),
            "state_items": len(pretrained_state),
            "matched_state_items": len(matched_keys),
            "mismatched_state_items": sorted(mismatched_keys),
            "missing_state_items": sorted(missing_keys),
            "matched_parameter_numel": matched_parameter_numel,
            "total_parameter_numel": total_parameter_numel,
            "parameter_coverage": matched_parameter_numel / total_parameter_numel,
            "matched_tensors_exact_after_solver_setup": not transfer_mismatch,
            "random_class_tensors_unchanged": not random_overwritten,
        },
        "data": {
            "train_images": len(train_loader.dataset),
            "val_images": len(val_loader.dataset),
            "batch_shape": list(samples.shape),
            "batch_target_counts": [int(target["labels"].numel()) for target in targets],
            "observed_labels": label_values,
        },
        "model": {
            "parameters": total_parameter_numel,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "output_shapes_fp32": {
                key: list(value.shape) for key, value in outputs_fp32.items() if torch.is_tensor(value)
            },
        },
        "fp32": {
            "losses": loss_values(losses_fp32),
            "total_loss": float(total_fp32.detach().cpu()),
            "gradients": gradients_fp32,
        },
        "amp": {
            "losses": loss_values(losses_amp),
            "total_loss": float(total_amp.detach().cpu()),
            "gradients": gradients_amp,
            "scale_attempts": amp_attempts,
            "final_scale": amp_attempts[-1]["scale"],
            "pred_logits_dtype": str(outputs_amp["pred_logits"].dtype),
            "pred_boxes_finite": bool(torch.isfinite(outputs_amp["pred_boxes"]).all()),
        },
        "validation_forward": {
            "batch_shape": list(val_samples.shape),
            "prediction_batches": len(predictions),
            "scores_finite": all(bool(torch.isfinite(item["scores"]).all()) for item in predictions),
            "boxes_finite": all(bool(torch.isfinite(item["boxes"]).all()) for item in predictions),
        },
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
