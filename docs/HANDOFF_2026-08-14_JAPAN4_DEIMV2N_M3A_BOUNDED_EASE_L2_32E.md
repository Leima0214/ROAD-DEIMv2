# Japan4 DEIMv2-N M3-A: Bounded EASE-L2 32E

## Experiment boundary

- Parent: frozen Japan4 DEIMv2-N B0 commit `4e4fc4e3c253adb9198aa9d6ed51a20116937cfc`.
- Direct parent: frozen M2 implementation commit `644d41804bf5e7c3b6b1466b6595e9cf30eb1bfb`.
- M2 diagnosis: its mean absolute log-bias grew from `0.0155` at E4 to `0.8768` at E31, while matched AP changed from an early lead to `-0.005353` at E31.
- Single intervention versus M2: replace its wide asymmetric `log(clamp(2 * sigmoid(raw), 1e-4, 2))` mapping with the symmetric `0.25 * tanh(raw)` mapping.
- This bounds every learned attention multiplier to `[exp(-0.25), exp(0.25)]`, approximately `[0.779, 1.284]`.
- The native supervised L1 score/LQE output remains detached; its class-agnostic confidence and box IoU still form the same pairwise relation for L2 self-attention.
- The relation MLP remains `1 -> 16 -> 8 heads`; its output layer remains zero-initialized, so initial log-bias is exactly zero and the initial model equals B0.
- Only the 300 matching queries are modulated during training. Denoising-query attention retains the native B0 mask and values.
- Matcher, criterion, GO union, all losses, query count, encoder, deformable cross-attention, FFN, data, augmentation, optimizer and schedule remain B0.
- The module remains active during inference. It is an EASE-inspired bounded MSelf-Attention adaptation, not a reproduction of full EASE-DETR.
- Deployment retains the native L1 score head and LQE required by the relation route; the extra deployed parameter count versus B0 is `2,093` (`1,925` retained native parameters plus the `168`-parameter relation MLP).
- Budget: 32 epochs, seed 42. Do not resume, tune, stack another module, or run 160E unless the screen passes.

## Promotion gate

- E31 AP50:95 >= `0.237331` (matched B0 E31 `0.234331` + `0.003`).
- AP75 >= `0.190171` and APsmall >= `0.087867`.
- If the aggregate gate passes, run the matched E29 per-class audit before promotion.

## Remote artifacts

- Worktree: `/root/ROAD-DEIMv2-M3A`
- Output: `/root/ROAD-DEIMv2-M3A/outputs/deimv2_n_japan4_m3a_bounded_ease_l2_32e_seed42_20260814`
- Log: `/root/ROAD-DEIMv2-M3A/logs/deimv2_n_japan4_m3a_bounded_ease_l2_32e_seed42_20260814.log`

## Hard preflight

The dynamic preflight must report `PASS` before launch. It checks:

- the instantiated decoder is the three-layer `DEIMTransformer`;
- the cap is exactly `0.25` and saturates at both `+0.25` and `-0.25`;
- all B0-common checkpoint tensors are bit-identical and the only new tensors are the 168 relation-MLP parameters;
- initial eval outputs, train outputs, base loss keys/values and deploy outputs equal B0;
- matcher, criterion and GO union are untouched;
- relation-module and normal L2 gradients are nonzero and finite in FP32/AMP;
- DN attention retains its native mask, while only matching-query attention receives the bounded bias;
- deployment retains the native L1 score/LQE route and has the expected `+2,093` parameter delta.

## Fresh launch

Run only after the GPU has no active experiment process and the dynamic preflight can fit:

```bash
cd /root/ROAD-DEIMv2-M3A
mkdir -p logs
nohup bash tools/run_japan4_deimv2_m3a_bounded_ease_l2_32e.sh \
  > logs/deimv2_n_japan4_m3a_bounded_ease_l2_32e_seed42_20260814_launcher.log 2>&1 &
echo $! > logs/deimv2_n_japan4_m3a_bounded_ease_l2_32e_seed42_20260814_launcher.pid
```

The launcher performs the hard preflight first and starts training only after a `PASS`.

Monitor the complete preflight/training stream with:

```bash
tail -n 100 -f /root/ROAD-DEIMv2-M3A/logs/deimv2_n_japan4_m3a_bounded_ease_l2_32e_seed42_20260814_launcher.log
```

After completion, verify the recorded process exit code:

```bash
cat /root/ROAD-DEIMv2-M3A/logs/deimv2_n_japan4_m3a_bounded_ease_l2_32e_seed42_20260814.exitcode
```

## Decision rule

- Run exactly one seed-42 32E screen. Do not tune the cap, resume, stack gates or launch 160E.
- Promote only if all three registered gates pass.
- If M3-A fails, freeze the final-layer query-competition family rather than adding entropy, scale or class gates as rescue variants.
