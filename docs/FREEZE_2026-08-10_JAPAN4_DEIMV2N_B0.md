# Japan4 DEIMv2-N B0 正式冻结报告（2026-08-10）

## 冻结结论

**状态：`FROZEN B0`。** Japan4 DEIMv2-N 已完整运行 160E，并完成独立 Val-only 重评、逐类/尺度/错误类型诊断、人工抽查、速度测量、跨框架数据一致性核对和本地备份。未发现会使所选最佳权重失效的致命问题；从本报告起，B0 的代码、数据、训练协议、最佳权重和评估结果均不再改写。

- 正式模型只认 `best_stg1.pth`，内部 `last_epoch=123`（日志从 E0 计数，即第 124 个 epoch）。
- 正式选择指标只认 Val COCO AP50:95；Test 未读取，继续锁定。
- `last.pth`、E148-E159 和 Val 最优阈值仅用于解释，不得替代正式 B0。
- 后续实验必须从 B0 fresh start，只允许一次一个因果变量；不得在 B0 上补模块、改超参或重命名结果。

## 1. 冻结身份与协议

| 项目 | 冻结值 |
|---|---|
| 仓库 / 分支 | `Leima0214/ROAD-DEIMv2` / `codex/japan4-deimv2-n` |
| 正式运行仓库 HEAD | `dfa7b03c6e41ea46c83fcf1389574506af1d4a0d` |
| 训练行为冻结 commit | `c74592bb072a1ee56e8e5d0ff8da4de2d742adfe` |
| 模型 | 官方 `DEIMv2-N` / HGNetv2-B0，stride 16/32 两层 encoder/decoder 输入 |
| 数据 | Japan4：D00、D10、D20、D40 |
| train / val / test | 6320 / 790 / 790；正式训练与选择只读 train/val |
| train / val 标注框 | 13,175 / 1,647 |
| 输入 / batch / workers | 640 / 32 / 8 |
| epoch / seed / device | 160 / 42 / CUDA:0 |
| 优化 | 官方 AdamW、增强、matcher、loss、EMA、AMP 均保持 |
| 唯一历史修正 | `flat_epoch: 7800 -> 78`；`lr_gamma=1.0`，E78 后没有数值 LR 衰减 |
| 初始化 | 官方 DEIMv2-N COCO 权重，628/637 state items，按参数量覆盖 99.9238% |
| 参数 / GFLOPs | 3,551,477 total；3,551,475 trainable；6.8204 GFLOPs |
| 正式输出目录 | `/root/ROAD-DEIMv2/outputs/formal_deimv2_n_japan4_b0_160e_seed42_20260810` |
| 正式控制台日志 | `/root/ROAD-DEIMv2/logs/formal_deimv2_n_japan4_b0_160e_seed42_20260810.log` |

正式运行完成 E0-E159，退出码为 0，总训练时间 4:44:19。训练日志未发现 traceback、CUDA OOM、NaN/Inf 或数据加载错误。Linux 运行前必须把 `ulimit -n` 从默认 1024 提高到 65535，否则多进程验证可能触发 `received 0 items of ancdata`；该要求属于运行环境前置条件，不是模型改动。

### 数据边界

Japan4 四类协议从源标注中移除了 5,482 个非目标 D43/D44/D50 框，但对应像素仍留在图像中。因此这些区域会作为未标注背景参与训练和评估。该边界已冻结，不能通过本轮后处理或偷偷补标改变；它也是解释“背景误检”的重要混杂因素。

## 2. 正式结果

### 2.1 全局 COCO Val

| 指标 | B0 |
|---|---:|
| AP50:95 | **0.275671** |
| AP50 | 0.588856 |
| AP75 | 0.215733 |
| AP-small / medium / large | 0.125651 / 0.233079 / 0.325649 |
| AR1 / AR10 / AR100 | 0.252799 / 0.480295 / 0.556736 |
| AR-small / medium / large | 0.297842 / 0.529638 / 0.632068 |

### 2.2 逐类 COCO Val

| 类别 | AP50:95 | AP50 | AP75 | AR100 | 诊断 |
|---|---:|---:|---:|---:|---|
| D00 | 0.248975 | 0.540622 | 0.179185 | 0.554815 | 次弱；细长裂缝的定位和可见度不足 |
| D10 | **0.191857** | 0.481201 | **0.112823** | **0.481658** | 最弱类，后续首要目标 |
| D20 | **0.365611** | **0.691634** | **0.347117** | **0.667258** | 当前最强类，不应作为首要改进对象 |
| D40 | 0.296241 | 0.641965 | 0.223807 | 0.523214 | AP 尚可，但背景/疑似漏标触发较多 |

### 2.3 实测效率

RTX 4090、PyTorch autocast FP16、batch=1、640、deploy model、100 次 warmup + 500 次计时：

- 平均 11.593 ms，median 11.489 ms，p95 11.661 ms，约 86.26 FPS。
- 推理峰值 allocated 71.96 MiB、reserved 98 MiB。
- 正式训练日志中的 PyTorch 最大 allocated 8,321 MiB；训练时 `nvidia-smi` 观察约 9,994 MiB。

这只是该环境下的 PyTorch 延迟，不等于 TensorRT 延迟，也不能与缺少同硬件实测的 YOLO 数字混用。

## 3. B0 薄弱点诊断

### 3.1 主要弱点：框定位，而不是分类混淆

独立诊断在 Val 上以 IoU=0.5、Val 最优 micro-F1 阈值 0.376580 统计：TP=1001、FP=713、FN=646，Precision=0.5840、Recall=0.6078、F1=0.5957。

| FP 类型 | 数量 | FP 占比 | 解释边界 |
|---|---:|---:|---|
| localization | **339** | **47.5%** | 与同类 GT 有重叠但 IoU<0.5；是最大明确错误源 |
| background_or_unlabeled | 287 | 40.3% | 自动分类；同时混有真实背景、非目标损伤和疑似漏标 |
| class_confusion | 48 | 6.7% | 占比低，不支持先改分类头 |
| duplicate | 39 | 5.5% | 不是当前主矛盾 |

AP50 0.588856 与 AP75 0.215733 的大间隔、定位 FP 占首位，以及 FN 图中的大量 IoU 约 0.33-0.50 近失配共同说明：模型经常“看到了损伤”，但框偏移、过宽或过窄，严格定位质量不足。分类混淆仅 48 个，主要为 D00→D20（24）和 D20→D00（16），不足以支撑优先改分类/质量排序分支。

### 3.2 次要弱点：小目标和细裂缝细节

在相同诊断阈值下，FN 按尺度为：

| 尺度 | FN / GT | 漏检率 |
|---|---:|---:|
| small | 71 / 135 | **52.6%** |
| medium | 371 / 812 | 45.7% |
| large | 204 / 700 | 29.1% |

结合 AP-small 0.125651、D10 最低 AP/AP75/AR100，以及 FN 可视化中的细、浅、低对比裂缝，后续模块应明确针对 stride-8 细节到现有 stride-16 路径的保真，而不是泛化为“增加容量”。

### 3.3 数据混杂：疑似漏标会放大细节增强的误检风险

对自动判为 `background_or_unlabeled` 的最高置信度 16 个样本进行人工抽查：13 个标为 `suspected_unlabeled_damage`，另有 2 个不确定样本、1 个路面纹理/设施伪影。该抽查受到“最高置信度样本”选择偏差影响，**不能外推为全量漏标率，也不能把自动错误类型当成标注错误的因果证明**；但它足以确认一个设计风险：更强的浅层/高频细节可能同时恢复细裂缝并放大道路接缝、修补纹理和未标注损伤。

因此，后续设计必须带零门控、身份初始化和明确的 FP 约束，不能直接增加一个强 P3 检测层。

### 3.4 收敛与 checkpoint 语义

- E0-E77 最佳为 E77 AP 0.250898；E78-E147 最佳为 E123 AP 0.275671。
- E148 进入最后阶段时，trainer 重新载入 `best_stg1.pth`；E148-E159 不是从 E147 连续训练的普通 12E 延长。
- 最后阶段最佳 E152 AP 0.275225，没有超过 E123；因此没有 `best_stg2.pth`，正式模型仍为 `best_stg1.pth`。
- `last.pth` 只在 `epoch < stop_epoch(148)` 时保存，所以其内部 E147 是预期的实现语义，不代表训练中止。
- JSON 日志中的 `n_parameters=2` 是局部变量被复用于“不可训练参数数目”的记录错误；静态审计的总参数 3,551,477 才是正式值。该问题不影响权重、前向或指标，但后续报告不得引用日志中的 2。

这些属于需要披露的重要实现限制，但不推翻 E123 最佳 checkpoint。为保持 B0 可追溯性，本轮不回改 trainer 再伪造一次“更整洁”的基线。

## 4. 与 YOLO26n 冻结参考的可比结论

已按文件名、类别名和 bbox（3 位小数）核对 DEIM 与 YOLO26 Japan4-cleanV3 Val：图像成员差异为 0，归一化标注差异为 0。原始 JSON 哈希不同来自 ID 和浮点序列化，不能据此误判数据不一致。

| 模型 | Epoch | AP50:95 | AP50 | AP75 | AP-small | AR100 | Params | GFLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n B0-V3 | 100 | 0.24142 | 0.52184 | 0.19609 | 0.09924 | 0.50709 | 2,505,360 | 5.6267 |
| DEIMv2-N B0 | 160 | **0.275671** | **0.588856** | **0.215733** | **0.125651** | **0.556736** | 3,551,477 | 6.8204 |
| DEIM - YOLO | — | +0.034251 | +0.067016 | +0.019643 | +0.026411 | +0.049646 | +41.75% | +21.22% |

该表支持把 DEIMv2-N 晋升为 Paper 1 当前诊断母体，但不支持“DEIM 架构因果显著优于 YOLO”或“Pareto 更优”：两者 epoch、架构专属 optimizer/schedule 不同，只有单 seed，YOLO checkpoint 缺失，无法进行同图预测配对 bootstrap，也缺少同硬件延迟。

## 5. 冻结决策与证据等级

### 接受

1. 接受 `best_stg1.pth`（E123）为 Japan4 DEIMv2-N 正式 B0。
2. 接受 AP50:95=0.275671 为后续 full-run 唯一基准；30E 筛选另用 B0 E29 的匹配快照。
3. 接受“定位质量 + 小/细 D00/D10”为已观察到的首要薄弱点。
4. 接受“疑似漏标背景污染”为设计约束，不把它写成已经证明的数据错误率。

### 不接受

1. 不接受 `last.pth`、E159 或 trial run 作为 B0。
2. 不接受 Val 最优阈值作为锁定 Test 阈值或正式部署阈值。
3. 不接受用单次结果宣称统计显著、架构因果优越或硬件 Pareto 优越。
4. 不接受重跑 B0 来掩盖 trainer 的最后阶段/日志语义；只在派生实验中修复记录逻辑，并与 B0 分开命名。

## 6. 唯一后续因果路线：R1 零门控 P3→P4 细节残差

后续只允许这一条路线进入代码预检：从 HGNetv2 暴露 stride-8 特征，经轻量投影和下采样后，以**零初始化可学习门控**残差注入现有 stride-16 encoder 输入。现有 stride-16/32 两层 encoder、decoder 张量形状、query 数、loss、matcher、数据、增强、optimizer 和 scheduler 全部保持不变。

选择它的因果理由是：它直接对应已证实的小/细裂缝与定位弱点；零门控使初始化严格等价于 B0，并限制浅层纹理/疑似漏标带来的 FP 放大。它不是第三检测层，也不是多模块堆叠。

### R1 静态与动态预检门槛

在任何正式训练前必须同时通过：

1. B0 可迁移张量按 name+shape 100% 覆盖；只允许新增适配器/门控张量随机或零初始化。
2. gate=0 时固定输入的 logits/boxes 与 B0 在数值容差内一致。
3. 单 batch forward/loss/backward/AMP 有限，适配器和 gate 梯度非零且有限。
4. 记录参数、GFLOPs、训练显存和相同 RTX 4090 延迟；禁止用 FLOPs 代替实测延迟。

### R1 30E 筛选门槛

匹配 B0 E29：AP50:95 0.230410、AP50 0.501152、AP75 0.183265、AP-small 0.087562、AR100 0.522314。

R1 只有同时满足以下条件才可晋升 160E：

- AP50:95 ≥ 0.233410（至少 +0.003）；
- AP75、AP-small 不低于 B0 E29；
- 无 NaN/OOM/训练不稳定，RTX 4090 同协议平均延迟增幅 ≤10%；
- 错误审计不得出现背景/疑似漏标 FP 的明显失控。

这是工程晋级门槛，不是统计显著性声明。

### R1 160E 接受门槛

只有 30E 通过后才运行匹配 160E。最终至少要求：AP50:95 ≥ 0.278671（B0 +0.003），AP75 与 AP-small 均不回退，D00/D10 平均 AP 至少 +0.005；仍只用 Val 选择，Test 保持锁定。

### 明确禁止

- 不重标数据，不改 train/val/test 成员；
- 不增加完整第三 decoder feature level，不扫 query/decoder 深度；
- 不改 loss、matcher、LR、scheduler、增强或训练周期；
- 不引入蒸馏，不搬回 YOLO 历史模块，不同时叠加第二个模块；
- R1 预检完成前不启动训练。

## 7. 冻结证据与恢复位置

| 证据 | 位置 |
|---|---|
| 诊断 JSON | [`reports/japan4_deimv2_n_b0_diagnostics.json`](../reports/japan4_deimv2_n_b0_diagnostics.json) |
| FP / FN 明细 | [`reports/japan4_deimv2_n_b0_fp_samples.csv`](../reports/japan4_deimv2_n_b0_fp_samples.csv) / [`reports/japan4_deimv2_n_b0_fn_samples.csv`](../reports/japan4_deimv2_n_b0_fn_samples.csv) |
| 人工 FP 抽查 | [`reports/japan4_deimv2_n_b0_manual_fp_review.csv`](../reports/japan4_deimv2_n_b0_manual_fp_review.csv) |
| 背景/疑似漏标图 | [`reports/japan4_deimv2_n_b0_top_background_or_unlabeled_montage.jpg`](../reports/japan4_deimv2_n_b0_top_background_or_unlabeled_montage.jpg) |
| 定位错误图 | [`reports/japan4_deimv2_n_b0_top_localization_montage.jpg`](../reports/japan4_deimv2_n_b0_top_localization_montage.jpg) |
| FP / FN 总览图 | [`reports/japan4_deimv2_n_b0_top_fp_montage.jpg`](../reports/japan4_deimv2_n_b0_top_fp_montage.jpg) / [`reports/japan4_deimv2_n_b0_top_fn_montage.jpg`](../reports/japan4_deimv2_n_b0_top_fn_montage.jpg) |
| 训练 JSONL / 控制台日志 | [`reports/japan4_deimv2_n_b0_training_log.jsonl`](../reports/japan4_deimv2_n_b0_training_log.jsonl) / [`reports/japan4_deimv2_n_b0_console.log`](../reports/japan4_deimv2_n_b0_console.log) |
| 可复现诊断脚本 | [`tools/diagnose_japan4_deimv2_b0.py`](../tools/diagnose_japan4_deimv2_b0.py) |
| SHA256 清单 | [`reports/japan4_deimv2_n_b0_freeze_manifest.sha256`](../reports/japan4_deimv2_n_b0_freeze_manifest.sha256) |

远端保留全部正式 checkpoints 和诊断输出；本地忽略目录 `outputs/frozen_b0_20260810/` 保存 `best_stg1.pth`、`last.pth`、日志、Val predictions、COCOeval 与诊断 JSON。正式权重以 SHA256 为身份，不以文件名或“最新时间”判断。
