# Japan4 DEIMv2-N M4-A: DySample P5-to-P4 32E

## Experiment boundary

- Parent: frozen Japan4 DEIMv2-N B0 commit `4e4fc4e3c253adb9198aa9d6ed51a20116937cfc`.
- Single intervention: replace the HybridEncoder's only P5-to-P4 nearest-neighbor interpolation with LP-style DySample.
- Fixed design: scale `2`, groups `4`, no dynamic scope branch, offset multiplier `0.25`.
- The implementation uses standard PyTorch `grid_sample`; it adds no external package or custom CUDA operator.
- Expected parameter delta: `4,128`; all B0-common tensors must remain bit-identical after checkpoint loading.
- P3, fusion blocks, PAN downsampling, backbone, decoder, FDR, LQE, matcher, criterion, GO union, data, augmentation, optimizer and schedule remain B0.
- Unlike a zero-initialized residual, DySample's regular sampling grid is bilinear and therefore its initial function is intentionally not identical to B0 nearest interpolation. This is part of the registered replacement.
- Budget: 32 epochs, seed 42. Do not tune groups/style/scope, resume, stack another module, or run a long schedule unless the screen passes.

## Promotion gate

- E31 AP50:95 >= `0.237331` (matched B0 E31 `0.234331` + `0.003`).
- AP75 >= `0.190171` and APsmall >= `0.087867`.
- If the aggregate gate passes, run the matched E29 per-class audit before promotion.

## Hard preflight

The launcher must report `M4A_DYSAMPLE_P5P4_PREFLIGHT_STATUS=PASS` before training. It checks:

- actual two-level `[16, 32]` HybridEncoder and exactly one P5-to-P4 DySample call;
- official LP contract (`scale=2`, `groups=4`, no scope) and exact 2x spatial shape;
- all B0-common checkpoint tensors are bit-identical;
- only the offset weight/bias and fixed initial-position buffer are added;
- parameter and deployed-parameter deltas are exactly `4,128`;
- replacement output differs from nearest interpolation, without changing output structure;
- loss keys are unchanged; offset and normal decoder gradients are nonzero;
- FP32/AMP losses and all AMP gradients are finite;
- the DySample route remains active after deployment conversion.

## Remote artifacts

- Worktree: `/root/ROAD-DEIMv2-M4A`
- Output: `/root/ROAD-DEIMv2-M4A/outputs/deimv2_n_japan4_m4a_dysample_p5p4_32e_seed42_20260814`
- Log: `/root/ROAD-DEIMv2-M4A/logs/deimv2_n_japan4_m4a_dysample_p5p4_32e_seed42_20260814.log`
- Preflight report: `/root/ROAD-DEIMv2-M4A/reports/deimv2_n_japan4_m4a_dysample_p5p4_32e_seed42_20260814_preflight.json`

## Fresh launch

```bash
cd /root/ROAD-DEIMv2-M4A
mkdir -p logs
nohup bash tools/run_japan4_deimv2_m4a_dysample_p5p4_32e.sh \
  > logs/deimv2_n_japan4_m4a_dysample_p5p4_32e_seed42_20260814_launcher.log 2>&1 &
echo $! > logs/deimv2_n_japan4_m4a_dysample_p5p4_32e_seed42_20260814_launcher.pid
```

Monitor:

```bash
tail -n 100 -f /root/ROAD-DEIMv2-M4A/logs/deimv2_n_japan4_m4a_dysample_p5p4_32e_seed42_20260814_launcher.log
```

## Decision rule

- Run exactly one matched seed-42 32E screen.
- Promote only if all three registered gates pass.
- If M4-A fails, freeze the learnable P5-to-P4 upsampling route rather than sweeping DySample variants or replacing it with CARAFE as a rescue.
