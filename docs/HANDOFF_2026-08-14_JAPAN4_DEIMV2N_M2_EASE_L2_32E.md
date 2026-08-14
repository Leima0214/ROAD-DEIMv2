# Japan4 DEIMv2-N M2: EASE-L2 32E

## Experiment boundary

- Parent: frozen Japan4 DEIMv2-N B0 commit `4e4fc4e3c253adb9198aa9d6ed51a20116937cfc`.
- Single intervention: the detached L1 state is evaluated by the native final scoring/LQE interface; its class-agnostic confidence and box IoU form an EASE-style pairwise relation that modulates L2 self-attention.
- The relation MLP is `1 -> 16 -> 8 heads`; its output layer is zero-initialized, so the initial log-decay is exactly zero.
- Only the 300 matching queries are modulated during training. Denoising-query attention retains the native B0 mask and values.
- Matcher, criterion, GO union, all losses, query count, encoder, deformable cross-attention, FFN, data, augmentation, optimizer and schedule remain B0.
- The module is active during inference. It is an EASE-inspired MSelf-Attention adaptation, not a reproduction of full EASE-DETR.
- Budget: 32 epochs, seed 42. Do not resume, tune, stack another module, or run 160E unless the screen passes.

## Promotion gate

- E31 AP50:95 >= `0.237331` (matched B0 E31 `0.234331` + `0.003`).
- AP75 >= `0.190171` and APsmall >= `0.087867`.
- If the aggregate gate passes, run the matched E29 per-class audit before promotion.

## Remote artifacts

- Worktree: `/root/ROAD-DEIMv2-M2`
- Output: `/root/ROAD-DEIMv2-M2/outputs/deimv2_n_japan4_m2_ease_l2_32e_seed42_20260814`
- Log: `/root/ROAD-DEIMv2-M2/logs/deimv2_n_japan4_m2_ease_l2_32e_seed42_20260814.log`
