# Japan4 DEIMv2-N M5-A: SACF P5-to-P4 32E

## Decision and research question

M4-A ended normally but failed its promotion gate: E31 AP was `0.233755`
versus matched B0 `0.234331`. It improved AP50, AP75, and medium/large recall,
while reducing aggregate AP and APmedium. M5-A does not rescue or retune
DySample. It asks a different, falsifiable question:

> Can explicit per-location competition between the existing P5 semantic
> feature and P4 spatial feature retain useful context without forcing the
> same cross-scale allocation at every spatial location?

## Experiment boundary

- Parent: frozen Japan4 DEIMv2-N B0 commit
  `4e4fc4e3c253adb9198aa9d6ed51a20116937cfc`.
- Single model intervention: replace only the existing P5/P4 top-down concat
  policy with Spatially Adaptive Competitive Fusion (SACF).
- P5 still uses the original nearest-neighbor 2x interpolation.
- Given upsampled P5 `U` and P4 `L`, SACF is:

  ```text
  [g5, g4] = softmax(Conv1x1(concat(U, L)), scale_dimension)
  output = concat(2 * g5 * U, 2 * g4 * L)
  ```

- The `256 -> 2` gate weight and bias are both initialized to exactly zero.
  Therefore `g5 = g4 = 0.5`, and the initial function is exactly the B0
  `concat(U, L)` function.
- Expected parameter delta: `514`; all B0-common checkpoint tensors must be
  bit-identical after checkpoint loading.
- DySample, CARAFE, P3, fusion-block replacement, PAN, backbone, decoder, FDR,
  LQE, matcher, criterion, GO union, data, augmentation, optimizer, and
  schedule remain B0.
- Budget: 32 epochs, seed 42. Do not tune the gate temperature, gate depth,
  initialization, loss weight, or insert a second gate.

## Registered outcomes

Primary model-promotion gate, evaluated at E31 under the matched protocol:

- AP50:95 >= `0.237331` (matched B0 E31 `0.234331` + `0.003`);
- AP75 >= `0.190171`;
- APsmall >= `0.087867`.

All three conditions are required. If they pass, run the matched E29 per-class
audit and one additional 32E confirmation seed before considering 160E.

Mechanism-support outcomes are registered separately and cannot rescue a
failed primary gate:

- APmedium should not be below matched B0 E31 `0.197459`;
- learned gate weights must be input-dependent rather than spatially constant;
- neither scale may collapse to a near-global zero weight.

If AP improves but the mechanism outcomes fail, report an empirical gain
without claiming that spatial scale allocation caused it.

## Hard preflight

The launcher must report `M5A_SACF_P5P4_PREFLIGHT_STATUS=PASS` before training.
The preflight checks:

- the M5-A YAML contains only the registered include, 32E/output, and SACF
  override;
- the instantiated model is the actual two-level `[16, 32]`, concat-mode
  `HybridEncoder`;
- only the `2 x 256 x 1 x 1` gate weight and two-element bias are added;
- parameter and deployed-parameter deltas are exactly `514`;
- all B0-common checkpoint tensors are bit-identical;
- standalone fusion, full eval output, training output, losses, and deployed
  output initially match B0 within the fixed tolerance;
- representative shared encoder and decoder gradients initially match B0;
- gate weight/bias and normal-path gradients are nonzero;
- both gate parameters occur exactly once in the optimizer;
- FP32/AMP losses and all AMP gradients are finite;
- the SACF route is called exactly once and survives deployment conversion.

## Remote artifact contract

- Worktree: `/root/ROAD-DEIMv2-M5A`
- Output:
  `/root/ROAD-DEIMv2-M5A/outputs/deimv2_n_japan4_m5a_sacf_p5p4_32e_seed42_20260814`
- Training log:
  `/root/ROAD-DEIMv2-M5A/logs/deimv2_n_japan4_m5a_sacf_p5p4_32e_seed42_20260814.log`
- Launcher log:
  `/root/ROAD-DEIMv2-M5A/logs/deimv2_n_japan4_m5a_sacf_p5p4_32e_seed42_20260814_launcher.log`
- Preflight report:
  `/root/ROAD-DEIMv2-M5A/reports/deimv2_n_japan4_m5a_sacf_p5p4_32e_seed42_20260814_preflight.json`

## Fresh launch

```bash
cd /root/ROAD-DEIMv2-M5A
mkdir -p logs
nohup bash tools/run_japan4_deimv2_m5a_sacf_p5p4_32e.sh \
  > logs/deimv2_n_japan4_m5a_sacf_p5p4_32e_seed42_20260814_launcher.log 2>&1 &
echo $! > logs/deimv2_n_japan4_m5a_sacf_p5p4_32e_seed42_20260814_launcher.pid
```

Monitor:

```bash
tail -n 100 -f /root/ROAD-DEIMv2-M5A/logs/deimv2_n_japan4_m5a_sacf_p5p4_32e_seed42_20260814_launcher.log
```

## Stop rule

- Run exactly one matched seed-42 32E screen.
- A failed primary gate freezes SACF and other learned weighting variants at
  this P5/P4 fusion site; do not rescue it with temperature sweeps, channel
  gates, deeper MLPs, or a second PAN gate.
- Do not run 160E unless the registered 32E gate and confirmation rule pass.
