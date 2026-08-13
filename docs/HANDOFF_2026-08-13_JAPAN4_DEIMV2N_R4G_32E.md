# Japan4 DEIMv2-N R4-G 32E 实验边界

## 唯一变量

R4-G 只修改最终 Decoder 层的 post-gateway RMSNorm。原共享 scale 保留为定位流，新增一个从同一预训练 scale 严格复制的分类流 scale（128 参数）。

- self-attention、norm1、deformable cross-attention、gateway linear 只前向一次；
- gateway RMSNorm 后分叉；
- FFN 与 norm3 参数共享，但对两个流分别前向；
- score head 读取分类流，FDR/bbox head 读取定位流；
- LQE、matcher、loss、optimizer、数据、增强、EMA 和 32E matched 协议不变；
- 从与 B0 相同的 `weights/deimv2_hgnetv2_n_coco_hf.pth` fresh start，禁止 resume B0 最终权重。

配置：`configs/deimv2/deimv2_hgnetv2_n_japan4_r4g_gateway_norm_32e.yml`

## 正式训练前硬门

```bash
cd /root/ROAD-DEIMv2
ulimit -n 65535
/opt/conda/bin/python -u tools/preflight_japan4_deimv2_r4g.py \
  --base-config configs/deimv2/deimv2_hgnetv2_n_japan4.yml \
  --r4g-config configs/deimv2/deimv2_hgnetv2_n_japan4_r4g_gateway_norm_32e.yml \
  --checkpoint weights/deimv2_hgnetv2_n_coco_hf.pth \
  --report reports/japan4_deimv2_n_r4g_preflight_20260813.json \
  --device cuda:0 --batch-size 2
```

必须同时通过：新增参数恰为128、旧参数完全同源、双 Norm 初值一致、logits/boxes/loss FP32 等价、损失有限、定位损失不进入分类 Norm、新旧 Norm 均 weight decay 0。

## 正式命令

```bash
tmux new-session -d -s deimv2_n_japan4_r4g_32e_20260813 "bash -lc 'cd /root/ROAD-DEIMv2; ulimit -n 65535; set -o pipefail; CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python -u train.py -c configs/deimv2/deimv2_hgnetv2_n_japan4_r4g_gateway_norm_32e.yml -t weights/deimv2_hgnetv2_n_coco_hf.pth --use-amp --seed 42 -d cuda:0 --output-dir outputs/deimv2_n_japan4_r4g_gateway_norm_32e_seed42_20260813 2>&1 | tee logs/deimv2_n_japan4_r4g_gateway_norm_32e_seed42_20260813.log; rc=\${PIPESTATUS[0]}; echo \$rc > logs/deimv2_n_japan4_r4g_gateway_norm_32e_seed42_20260813.exitcode; exec bash'"
```

日志：

```bash
tail -f /root/ROAD-DEIMv2/logs/deimv2_n_japan4_r4g_gateway_norm_32e_seed42_20260813.log
```

## 32E 判决

matched B0 E31：AP `0.234330567`，AP75 `0.190171299`。

- 强通过：AP 至少 `+0.005`，AP75 至少 `+0.003`，APsmall 与 D10 不退；
- 暂定通过：AP 至少 `+0.003` 且关键指标不退，之后必须补配对 B0/R4-G seed43；
- 失败：AP 增益低于 `+0.002`，或 AP75 下降至少 `0.002`；
- 机制改善但 AP 增益不足：记录后冻结，不晋级160E。

这些是本项目工程门，不是论文通用阈值。32E 只与冻结 B0 的 E31 比较，不与独立最佳 epoch 错位比较。
