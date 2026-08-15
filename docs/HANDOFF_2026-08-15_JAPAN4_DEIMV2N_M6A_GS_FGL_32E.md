# Japan4 DEIMv2-N M6-A: Geometry-Sensitive FGL 32E

## Frozen question

Does replacing the shared whole-box FGL quality weight with unit-mean
first-order side-IoU sensitivity improve the strict localization of elongated
road defects without changing inference or the aggregate FGL loss scale?

## Single intervention

- B0 uses the same detached whole-box IoU for L/T/R/B FGL terms.
- M6-A multiplies that IoU by `[1/w, 1/h, 1/w, 1/h]`, normalized to mean one
  per matched target.
- A square target exactly recovers B0 weighting. A horizontal 5:1 target uses
  `[1/3, 5/3, 1/3, 5/3]`.
- Matcher, GO union, DDF, model graph, optimizer, augmentation, data and seed
  remain unchanged. Parameter/FLOP/inference deltas are zero.

## Protocol and gate

- Fresh start from the same official COCO checkpoint.
- Japan4 Train/Val only; Test remains sealed.
- Seed 42, AMP, batch 32, resolution 640, E0-E31 prefix of the official 160E
  schedule.
- Compare E31 only with frozen B0 E31: AP 0.234331, AP50 0.502853,
  AP75 0.190171, APsmall 0.087867, APmedium 0.197459, APlarge 0.269977.
- GO requires delta AP >= +0.003 and delta AP75 >= +0.005, D10/short-axis
  localization improving, and AP50 decline no worse than 0.002.
- Do not tune side weights, add entropy, resume, access Test or launch 160E.

## Remote artifacts

- Worktree: `/root/ROAD-DEIMv2-M6A`
- Output: `/root/ROAD-DEIMv2-M6A/outputs/deimv2_n_japan4_m6a_gs_fgl_32e_seed42_20260815`
- Launcher log: `/root/ROAD-DEIMv2-M6A/logs/deimv2_n_japan4_m6a_gs_fgl_32e_seed42_20260815_launcher.log`
- Training log: `/root/ROAD-DEIMv2-M6A/logs/deimv2_n_japan4_m6a_gs_fgl_32e_seed42_20260815.log`
- Preflight report: `/root/ROAD-DEIMv2-M6A/reports/deimv2_n_japan4_m6a_gs_fgl_32e_seed42_20260815_preflight.json`
