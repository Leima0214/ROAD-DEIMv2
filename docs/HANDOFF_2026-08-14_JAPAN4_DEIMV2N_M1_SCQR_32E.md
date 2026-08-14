# Japan4 DEIMv2-N M1: SCQR 32E

## Experiment boundary

- Parent: frozen Japan4 DEIMv2-N B0 at commit `4e4fc4e3c253adb9198aa9d6ed51a20116937cfc`.
- Single intervention: during training, pass the complete post-L0 matching-query state through the shared final L2 decoder layer and supervise the resulting prediction independently.
- `scqr_outputs` receives its own Hungarian match and the existing MAL, L1, GIoU, and FGL losses with `scqr_loss_weight: 1.0`.
- `scqr_outputs` never enters the original main/aux/encoder GO union. The original B0 loss graph and all inference outputs remain unchanged.
- Screen budget: 32 epochs, seed 42. This is a promotion screen, not a formal 160E result.

## Hard preflight gate

`tools/preflight_japan4_deimv2_m1_scqr.py` must report `PASS` before launch. It verifies:

- the instantiated decoder is the three-layer `DEIMTransformer`;
- B0 and M1 parameters/checkpoint state are identical;
- evaluation has no SCQR output and is tensor-identical to B0;
- training-time B0 outputs and every original loss are tensor-identical;
- the only new losses are `loss_mal_scqr`, `loss_bbox_scqr`, `loss_giou_scqr`, and `loss_fgl_scqr`;
- SCQR gradients reach L0 and the shared L2, skip L1, and remain finite under AMP.

## Remote run

- Worktree: `/root/ROAD-DEIMv2-M1`
- Output: `/root/ROAD-DEIMv2-M1/outputs/deimv2_n_japan4_m1_scqr_32e_seed42_20260814`
- Log: `/root/ROAD-DEIMv2-M1/logs/deimv2_n_japan4_m1_scqr_32e_seed42_20260814.log`

Do not resume, tune the SCQR weight, alter data/augmentation/schedule, or add another module during M1.
