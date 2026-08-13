# Japan4 DEIMv2-N B0 D6：双视角任务梯度诊断

## 目的与边界

D6 只验证最终 Decoder 层是否存在足以支持分类/定位解耦的任务梯度冲突。它不训练 detector、不更新权重、不读取 Test，也不是精度结果。

- `D6-Actual`：真实最终层 post-LQE `loss_mal` 梯度。
- `D6-Pure`：固定 Actual 的最终匹配，用 LQE 前 raw class logits 重算同一 MAL 梯度。
- `Localization`：同一最终输出的加权 `bbox + GIoU + FGL`。DEIMv2 主输出不产生 DDF；脚本另报包含全部辅助/DN loss key 的 full-objective 稳健性视角。
- 统计位置：最终层 self-attention、deformable sampling offsets、attention weights、gateway/norm 和 FFN。
- DEIMv2-N 的最终 MSDeformableAttention 没有独立 value projection 参数，因此脚本不虚构该分组，而是报告 P4/P5 value-feature 的空间梯度能量。

主判决使用 deterministic no-augmentation Train 子集，默认 `64 × batch4 = 256` 张图。FP32 诊断保留原目标函数和计算图拓扑，避免 AMP 下小梯度舍入影响余弦。

## A/B/C/D 判决

对 `all_final` 的逐批余弦预注册工程门：bootstrap mean cosine 95% CI 上界 `< 0`，且负余弦批次比例 `>= 60%`。

| 情况 | Actual | Pure | 决策 |
|---|---|---|---|
| A | 冲突 | 冲突 | 强支持 R4-A；LQE 去除后冲突仍存在 |
| B | 冲突 | 不冲突 | 停止 R4-A；冲突主要来自 LQE/质量耦合 |
| C | 不冲突 | 冲突 | 条件支持；LQE 在缓和冲突，任何解耦必须保留 LQE 桥 |
| D | 不冲突 | 不冲突 | 停止 R4-A |

最终还必须查看 offsets/attention weights/FFN 分块和 D00/D10/D20/D40 方向。上述阈值是本项目工程门，不是论文通用定律。

## 正式命令

```bash
cd /root/ROAD-DEIMv2
ulimit -n 65535
python -u tools/diagnose_japan4_deimv2_b0_d6.py \
  --config configs/deimv2/deimv2_hgnetv2_n_japan4.yml \
  --checkpoint outputs/formal_deimv2_n_japan4_b0_160e_seed42_20260810/best_stg1.pth \
  --train-images /JAPAN4-DETR/JAPAN4-DETR/images/train \
  --train-annotations /JAPAN4-DETR/JAPAN4-DETR/annotations/deimv2/instances_train.json \
  --output-dir outputs/deimv2_n_japan4_b0_d6_20260813 \
  --device cuda:0 --seed 42 --batch-size 4 --workers 0 --max-batches 64
```

产物：

- `d6_diagnostics.json`
- `d6_gradient_rows.csv`
- `d6_spatial_energy_rows.csv`
- 外层重定向保存的 `remote_run.log`

脚本拒绝覆盖既有输出目录。正式运行前先用新的 smoke 目录执行 `--max-batches 1`，smoke 只验证计算链，不进入判决。
