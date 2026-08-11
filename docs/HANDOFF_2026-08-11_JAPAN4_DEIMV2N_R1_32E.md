# Japan4 DEIMv2-N R1：32E 短测试交接

## 结论与边界

本分支只准备并运行一个 **32E 筛选实验**，不运行 160E。R1 是单变量改动：把 HGNetv2-B0 的 stride-8 P3 特征经轻量深度可分离 stride-2 投影后，用零初始化标量门控残差注入现有 P4 encoder 输入。encoder/decoder 仍只接收 P4/P5 两层。

- 分支：`codex/japan4-r1-p3-p4-detail`
- 配置：`configs/deimv2/deimv2_hgnetv2_n_japan4_r1_p3p4_32e.yml`
- 输出：`outputs/deimv2_hgnetv2_n_japan4_r1_p3p4_32e`
- 对照点：冻结 B0 的 E31，而不是最终 E123/E159
- 性质：futility screen；可以淘汰明显弱方案，不能证明最终性能或统计显著性
- 本轮禁止：160E、Test、resume、叠加第二模块、改变数据/增强/loss/matcher/LR/optimizer/seed

旧冻结文档中的 30E 筛选由本交接更新为 32E；B0 的代码、权重、结果和冻结分支均不改写。

## 为什么接受这条 R1

B0 已观察到 AP50 与 AP75 间隔大、定位型 FP 占首位、small FN 较高且 D10 最弱，因此“恢复 stride-8 细节是否改善严格定位”是有靶点的假设。但这些证据没有证明“缺 P3 就是原因”，所以 R1 只是一项可证伪测试。

FPN 指出高分辨率浅层特征具有更准确的空间定位；DEIMv2 的较大模型也用多尺度适配补充细粒度信息。这些只支持可行性，不构成本数据集上的效果证据。零初始化残差门控借鉴 ReZero 的身份起点思想，使 R1 在初始前向上等价于 B0。

- FPN: <https://arxiv.org/abs/1612.03144>
- ReZero: <https://arxiv.org/abs/2003.04887>
- DEIMv2: <https://arxiv.org/abs/2509.20787>

没有采纳完整 P3 检测层、额外 attention、DySample、Wavelet、分类/质量头或 loss/matcher 改动，因为它们会扩大算力和混杂因素，无法回答这次唯一问题。

## 实现冻结

R1 相对 B0 只有以下差异：

1. `HGNetv2.return_idx: [2, 3] -> [1, 2, 3]`，额外暴露 P3；
2. P3 经 `DWConv 3x3, stride=2, 256->256` 和 `Conv 1x1, 256->128`；
3. 原始标量 `detail_gate` 初始化为 `0.0`，融合为 `P4 + gate * detail(P3)`；
4. `epoches: 32` 和独立输出目录；
5. 日志逐 epoch 记录训练模型的 `detail_gate` 和用于 Val 的 `ema_detail_gate`。

新增可训练参数为 35,841（约为 B0 参数量的 1.01%）。适配器使用隔离的确定性随机流，不能改变 B0 共享张量在 seed 42 下的初始化。门控为零时首个 backward 的数学预期是：gate 梯度可非零，adapter 梯度严格为零；gate 离开零后 adapter 梯度才解锁。

32E 继承 B0 的 `flat_epoch: 78`、`lr_gamma: 1.0`、warmup、增强阶段和全部优化设置。因此 E0-E31 位于 B0 相同的 flat-LR 前缀；没有把官方长程阶段压缩进 32E。

## 租到 GPU 后的前检

```bash
cd /root/ROAD-DEIMv2
git fetch origin
git switch codex/japan4-r1-p3-p4-detail
git pull --ff-only origin codex/japan4-r1-p3-p4-detail
ulimit -n 65535
sha256sum weights/deimv2_hgnetv2_n_coco_hf.pth
python tools/verify_japan4_r1_p3p4.py --device cuda:0
```

权重 SHA256 必须为：

```text
e76c71a53534d767bb09bb4eaabba8c11aefb48df1dc4f9e120a5b58677ae2c6
```

预检必须输出 `"status": "PASS"`。它会检查：配置只有允许差异；官方权重的 B0/R1 可迁移集合一致；seed 42 下所有共享张量逐项相同；gate=0 的 B0/R1 输出等价；P3/P4/P5 和 encoder 形状正确；两阶段梯度语义正确；gate 已进入 optimizer。该检查只做合成前向/反向，不是训练或性能证据。

## 32E 启动命令

必须 fresh start 于同一份官方 COCO 预训练权重，不能使用 B0 的 `best_stg1.pth`，也不能使用 `-r`：

```bash
cd /root/ROAD-DEIMv2
mkdir -p logs
ulimit -n 65535
nohup python -u train.py \
  -c configs/deimv2/deimv2_hgnetv2_n_japan4_r1_p3p4_32e.yml \
  -t weights/deimv2_hgnetv2_n_coco_hf.pth \
  -d cuda:0 \
  --seed 42 \
  --use-amp \
  > logs/deimv2_n_japan4_r1_p3p4_32e_seed42.log 2>&1 &
echo $!
```

查看日志：

```bash
tail -f /root/ROAD-DEIMv2/logs/deimv2_n_japan4_r1_p3p4_32e_seed42.log
```

## E31 同点判定

冻结 B0 E31：

| 指标 | B0 E31 |
|---|---:|
| AP50:95 | 0.234331 |
| AP50 | 0.502853 |
| AP75 | 0.190171 |
| AP-small | 0.087867 |
| AR100 | 0.521263 |

R1 E31 的判定只用于决定该假设是否还值得投入：

- 明确淘汰：AP50:95 ≤ 0.229331（比 B0 低至少 0.005），或 AP75 与 AP-small 同时清楚回退，或出现 NaN/OOM/梯度异常/gate 非有限；
- 值得保留：AP50:95 ≥ 0.237331（至少 +0.003），且 AP75、AP-small 均不低于 B0 E31；
- 灰区：介于两者之间或出现 AP 与严格定位/小目标指标互换。先做误差诊断，不自动追加训练。

不以单个“最好 epoch”替代 E31 同点主比较；可以附报 best-within-32E，但必须标成探索性结果。即便通过，本轮仍在 32E 结束，不自动启动或续跑 160E。

## 停止条件与交付物

立即停止并保留现场：preflight 非 PASS、权重哈希不符、数据成员/标签变化、NaN/Inf、CUDA OOM、持续数据加载错误、输出目录已含不明旧实验。

完成后至少回收：

- 完整控制台日志和 `output_dir/log.txt`；
- `last.pth`、`best_stg1.pth`（若生成）及 SHA256；
- E31 与 best-within-32E 的 COCO 指标；
- 每 epoch 的 `detail_gate` 与 `ema_detail_gate` 轨迹；
- 参数量、实测显存与训练耗时；
- 若结果在灰区或通过，再做 D00/D10、AP75/AP-small、定位 FP 与背景/疑似漏标 FP 诊断。
