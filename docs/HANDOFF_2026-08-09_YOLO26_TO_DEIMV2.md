# Paper 1 研究交接：从 YOLO26 转向实时 DETR / DEIMv2（2026-08-09）

> **状态更新（2026-08-10）：** 本文是历史路线交接。Japan4 DEIMv2-N 正式 B0 已完成并冻结；当前结果、薄弱点诊断和唯一后续路线以 [`FREEZE_2026-08-10_JAPAN4_DEIMV2N_B0.md`](FREEZE_2026-08-10_JAPAN4_DEIMV2N_B0.md) 为准。

## 1. 交接摘要

Paper 1 当前已停止继续给 YOLO26 叠加局部模块，研究主线转向实时 DETR 架构。第一条正式基线选用官方 `DEIMv2-N`，在冻结的 Japan4 四类、80/10/10 数据协议上训练 160E。

这不是因为“YOLO26无法改进”，也不是宣称所有已尝试模块都没有价值。更准确的结论是：

1. YOLO26n 在 Japan4-cleanV3 上已经形成较强的预训练特征、PAN融合和端到端预测平衡；
2. 多种浅层增强、方向先验、区域辅助、额外跨尺度融合和随机替换预训练层，能产生局部正信号，却没有稳定转化为整体AP；
3. 失败模式反复表现为 Recall/AR或候选覆盖增加，但分类置信度、尺度分配、AP50/AP75或最终排序受损；
4. YOLO26路线剩余较自然的研究空间逐渐集中到O2O/O2M、匹配、quality/ranking和监督传递，而这不是当前希望继续深入的研究主题；
5. 因而更合理的策略是更换架构母体，先建立一个可信的实时DETR基线，再根据它在Japan4上的真实弱点设计道路针对性改进。

明日唯一任务：从官方COCO预训练权重 fresh start 运行 `DEIMv2-N Japan4 160E B0`。今天的4轮链路试跑不得续训，也不得作为正式结果。

## 2. YOLO26阶段的冻结结论

### 2.1 可比协议

历史存在 Japan7、Japan4-Positive-v2 和 Japan4-cleanV3 等不同协议。它们的绝对AP不能混合比较。转向DETR前最可信的YOLO对照来自 Japan4-cleanV3：

| 模型 | 周期 | AP50:95 | AP50 | AP75 | 备注 |
|---|---:|---:|---:|---:|---|
| YOLO26n B0 | 30E | 0.23570 | 0.49134 | 0.18518 | cleanV3 matched baseline |
| YOLO26n B0 | 100E | 0.24142 | 0.52184 | 0.19609 | cleanV3 frozen reference |

这些数值是研究背景参考。DEIMv2最终比较时仍须确认图像成员、标签、分辨率和COCO评估语义完全匹配后，才可以作架构优劣结论。

### 2.2 代表性负证据

以下仅列能够支持“停止继续堆YOLO模块”的代表性 cleanV3 结果，不把静态审计或未完成设计冒充训练失败：

| 路线 | 结果 | 相对对应B0 | 冻结原因 |
|---|---:|---:|---|
| G1 Gaussian Region Guidance 30E | 0.2242 | -0.0115 | 辅助区域监督与检测主任务竞争，Recall明显下降 |
| S1 Strip Regression 100E | 0.24091 | -0.00051 | 总AP未提高，门控塌缩为近似均匀分配，条带机制贡献接近零 |
| MSHC M1 30E | 0.19891 | -0.03679 | 随机模块替换预训练关键层，特征破坏严重 |
| RoadMSHC-R1 30E | 0.21970 | -0.01600 | 保留主路径后恢复明显，但四类仍全部低于B0 |
| RoadAdaptiveScaleFusion-N1 30E | 0.21215 | -0.02355 | 候选覆盖/Recall信号未转化为判别与排序收益 |

此外，早期DySample、WTC等在其他Japan4协议或30E阶段出现过小幅正信号，但没有稳定形成100E整体优势。它们不能与cleanV3结果直接混合，也不足以证明继续堆叠模块值得作为主线。

### 2.3 YOLO26阶段真正留下的知识

- 预训练兼容性比模块名称重要；随机替换成熟P3/P4/Neck层风险很高。
- bbox长宽比不等价于裂缝像素方向；强水平/垂直先验可能监督错误。
- 高频、浅层细节和更多候选同时也会放大道路纹理、阴影和背景误检。
- Recall或AR上涨不等于AP上涨；分类置信度、定位质量和候选排序是独立问题。
- FLOPs下降不保证RTX 4090或TensorRT真实延迟下降。
- 30E信号可能在100E消失；所有晋级结论必须基于匹配协议和冻结门槛。

因此，YOLO26工作应作为完整负证据和任务诊断保留，而不是删除或否定。当前仅停止把它作为Paper 1的继续开发母体。

## 3. 新研究路线的定义

### 3.1 不是回到原始DETR

原始DETR训练周期长、实时性和小模型工程并不适合作为2026年的主要母体。本项目转向的是“实时端到端DETR”路线，候选思想包括：

- RT-DETRv2：结构清晰、实时、易于改造encoder与多尺度融合；
- DEIMv2：提供Atto/Femto/Pico/N等轻量系列、官方权重和完整训练工程；
- 后续可选RF-DETR/LW-DETR等效率母体，但必须先经过官方实现和协议审计。

当前实际落地的第一母体是官方 `DEIMv2-N`，不是原始DETR，也不是继续修改D-FINE旧实验。

### 3.2 为什么先选DEIMv2-N

- 官方仓库、配置和COCO预训练权重齐全；
- N模型约3.55M参数、约6.82 GFLOPs，规模与轻量YOLO对照接近；
- 自定义COCO数据接入成熟；
- 当前环境已完成真实batch、loss、backward和AMP验证；
- 先用N建立可信B0，比立即设计Road-DETR模块更能避免重复“先爱上模块、再找证据”的问题。

DEIMv2-N只是母体竞技场的第一条正式基线。只有它完整160E结果出来后，才判断下一步是道路针对性encoder/多尺度改造、效率压缩，还是补跑另一个实时DETR母体。

## 4. 项目与版本状态

### 4.1 仓库

- 官方上游：`https://github.com/Intellindust-AI-Lab/DEIMv2.git`
- 个人仓库：`https://github.com/Leima0214/ROAD-DEIMv2.git`
- 本地路径：`F:\deeplearning\ROAD-DEIMv2`
- 远程GPU路径：`/root/ROAD-DEIMv2`
- 开发分支：`codex/japan4-deimv2-n`
- 上游基点：`0fff8d4`
- 训练配置冻结commit：`c74592bb072a1ee56e8e5d0ff8da4de2d742adfe`
- 第一份操作交接commit：`98f396e94d7065a3629190c1a123cc467c41d6d0`

分支中的Japan4相关文件：

```text
configs/dataset/japan4_detection.yml
configs/deimv2/deimv2_hgnetv2_n_japan4.yml
tools/prepare_japan4_deimv2_annotations.py
tools/audit_japan4_deimv2_n.py
reports/japan4_deimv2_annotation_conversion.json
reports/japan4_deimv2_n_static_audit_b32.json
docs/HANDOFF_2026-08-09_JAPAN4_DEIMV2N_B0.md
```

### 4.2 冻结训练配置

```text
model       = DEIMv2 HGNetv2-N
dataset     = Japan4
epochs      = 160
imgsz       = 640
batch       = 32
workers     = 8
seed        = 42
device      = cuda:0
AMP         = True
pretrained  = official DEIMv2-N COCO
selection   = Val only
Test        = forbidden before final locked evaluation
```

## 5. Japan4-DETR数据准备

### 5.1 来源与输出

- 官方RDD2022 Japan源：`F:\deeplearning\image_data\Japan`
- 本地DETR数据：`F:\deeplearning\JAPAN4-DETR`
- 远程挂载：`/JAPAN4-DETR/JAPAN4-DETR`
- 格式：COCO detection
- 类别：D00、D10、D20、D40
- 划分：train/val/test = 80/10/10
- 划分继承：`F:\deeplearning\Japan4-cleanV3\audit\split_assignment.csv`
- 近重复组跨split数量：0
- 源图像hash核验：全部通过

| Split | Images | Boxes |
|---|---:|---:|
| Train | 6320 | 13175 |
| Val | 790 | 1647 |
| Test | 790 | 1647 |

Train逐类框数：D00 3239、D10 3183、D20 4958、D40 1795。

### 5.2 两套category ID视图

原始COCO转换文件使用标准类别ID `1..4`：

```text
annotations/instances_{train,val,test}.json
```

DEIMv2当前自定义数据criterion直接使用category ID，因此额外生成非破坏性的 `0..3` 视图：

```text
annotations/deimv2/instances_{train,val,test}.json
```

正式DEIMv2训练只使用 `annotations/deimv2` 下的train/val文件。

### 5.3 协议限制

为保持既有Japan4协议，含目标类和非目标类的混合图像会保留，但D43/D44/D50等非目标框被移除。由此有5482个非目标框对应的区域在Japan4任务中成为未标注背景。这是冻结协议的已知限制，不能在模型间比较时临时修改。

## 6. DEIMv2-N预训练与静态审计

### 6.1 权重

- 官方来源：`Intellindust/DEIMv2_HGNetv2_N_COCO/model.safetensors`
- safetensors SHA256：`0de14388a703d86c95200589b32a4497b0b00dceb1d2d381c6a14fbd958856cb`
- 转换后训练权重：`weights/deimv2_hgnetv2_n_coco_hf.pth`
- 转换后 SHA256：`e76c71a53534d767bb09bb4eaabba8c11aefb48df1dc4f9e120a5b58677ae2c6`

预训练匹配：

```text
checkpoint state items       637
matched state items          628
matched parameter coverage   99.9238%
```

未匹配的9项均为80类COCO到4类Japan4产生的分类/denoising分类张量差异；`decoder.reg_scale`和`decoder.up`为当前模型新增状态。没有为了提高加载率进行纯shape强塞。

### 6.2 已通过项目

- train images=6320、val images=790；
- 真实batch shape=`[32,3,640,640]`；
- 观测标签=`0,1,2,3`；
- FP32 forward/loss/backward全部通过；
- 405/405个梯度张量非零且有限；
- AMP forward/loss/backward通过；
- AMP动态缩放回退到8192后梯度全部有限；
- Val forward、scores和boxes全部有限；
- 参数量=3,551,477；
- forward FLOPs约6.8204G；
- batch32静态审计峰值显存约11,014MiB；
- 实际训练峰值约12,000MiB。

审计报告：`reports/japan4_deimv2_n_static_audit_b32.json`。

## 7. Scheduler审计

训练前发现官方N配置中的：

```text
flat_epoch: 7800
```

这显然超出160E训练范围。按照冻结指令，本分支只将其改为：

```text
flat_epoch: 78
```

没有顺手修改任何其他模型、数据或训练项。真实scheduler dry-run：

```text
epochs       = 160
iters/epoch  = 197
warmup_iter  = 2000
flat_epoch   = 78
flat_iter    = 15366
no_aug_epoch = 12
total_iters  = 31520
base LR      = [0.0004, 0.0004, 0.0008, 0.0008]
```

必须保留的知情说明：当前官方配置同时设置 `lr_gamma=1.0`，所以warmup后目标最小LR等于base LR，E78后不会出现数值上的余弦下降。为了保持“只改7800→78”的单变量约束，本轮没有修改它。明日B0继续使用冻结配置，不能临时改scheduler。

## 8. 今日训练链路试跑

### 8.1 首次环境故障

第一次启动在第一个optimizer step之前出现：

```text
RuntimeError: received 0 items of ancdata
```

诊断结果：`/dev/shm`充足，服务器默认 `ulimit -n=1024`，workers=8时文件描述符不足。零步失败运行已远程归档：

```text
outputs/failed_prestart_deimv2_n_japan4_b0_160e_seed42_ancdata
logs/failed_prestart_deimv2_n_japan4_b0_160e_seed42_ancdata.log
```

修复只发生在启动shell：

```bash
ulimit -n 65535
```

没有把workers从8改小，也没有更改训练配置。

### 8.2 成功试跑

提高文件描述符上限后，训练正常完成E0-E3，并进入E4首个batch：

```text
E3 Val AP50:95 = 0.0728197
E3 Val AR100   = 0.428
GPU利用率正常
训练显存约12GB
best_stg1.pth与last.pth正常生成
```

早期指标只证明数据、优化器、评估和checkpoint链路工作，不作为正式科学结果。

用户决定明日再跑正式实验后，试跑已人工终止。最终核验：

```text
train.py进程        无
DataLoader workers  无
tmux session         无
GPU utilization     0%
GPU memory           1 MiB
server shutdown      未执行
```

远程试跑工件：

```text
/root/ROAD-DEIMv2/outputs/formal_deimv2_n_japan4_b0_160e_seed42_20260809
/root/ROAD-DEIMv2/logs/formal_deimv2_n_japan4_b0_160e_seed42_20260809.log
```

试跑工件SHA256：

```text
best_stg1.pth  f6b4975887c3d4fc648401bc4009d000e6f17062b462b21403698ba52159ee16
last.pth       7bb591f753b0c7439f7e4689ef9301d70737bcca194709942bdb37d7751c3a67
training log   77eb964db71be2af671096ba8c80b566890cd43091abe522f50e531bdb0eb4d6
```

## 9. 明日实验计划

### 9.1 唯一正式实验

```text
Experiment: DEIMv2-N Japan4 160E B0
Purpose: 建立实时DETR新路线的首条可信母体基线
Start: official COCO pretrained checkpoint
Resume: False
Selection: Val only
Test: forbidden
```

今天的4轮试跑不得resume。正式运行必须使用新的输出目录 `formal_deimv2_n_japan4_b0_160e_seed42_20260810`。

### 9.2 启动前检查

明日GPU端口可能变化，用实际端口替换 `<PORT>`：

```bash
ssh -p <PORT> root@xj-member.bitahub.com
cd /root/ROAD-DEIMv2
git checkout codex/japan4-deimv2-n
git pull --ff-only origin codex/japan4-deimv2-n
git lfs pull

git rev-parse HEAD
sha256sum weights/deimv2_hgnetv2_n_coco_hf.pth
test -f /JAPAN4-DETR/JAPAN4-DETR/annotations/deimv2/instances_train.json
test -f /JAPAN4-DETR/JAPAN4-DETR/annotations/deimv2/instances_val.json
pgrep -af 'python.*train.py' || true
nvidia-smi
```

必须确认：

- 分支正确且包含 `c74592b`；
- 预训练权重hash为 `e76c71a...e2c6`；
- train/val标注存在；
- 没有其他训练进程；
- GPU空闲。

如果是全新服务器且权重不存在，先安装依赖并重新运行静态审计脚本。该脚本会从官方Hugging Face下载safetensors、生成训练用pth并重新验证batch32：

```bash
/opt/conda/bin/python -m pip install -r requirements.txt
/opt/conda/bin/python -u tools/audit_japan4_deimv2_n.py \
  --device cuda:0 \
  --batch-size 32 \
  --report reports/japan4_deimv2_n_static_audit_b32.json
```

### 9.3 正式启动命令

```bash
cd /root/ROAD-DEIMv2
mkdir -p logs

tmux new-session -d -s deimv2_n_japan4_b0_160e_20260810 "bash -lc 'ulimit -n 65535; set -o pipefail; CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python -u train.py -c configs/deimv2/deimv2_hgnetv2_n_japan4.yml -t weights/deimv2_hgnetv2_n_coco_hf.pth --use-amp --seed 42 -d cuda:0 --output-dir outputs/formal_deimv2_n_japan4_b0_160e_seed42_20260810 2>&1 | tee logs/formal_deimv2_n_japan4_b0_160e_seed42_20260810.log; rc=\${PIPESTATUS[0]}; echo \$rc > logs/formal_deimv2_n_japan4_b0_160e_seed42_20260810.exitcode; exec bash'"
```

查看日志：

```bash
tail -f /root/ROAD-DEIMv2/logs/formal_deimv2_n_japan4_b0_160e_seed42_20260810.log
```

或：

```bash
tmux attach -t deimv2_n_japan4_b0_160e_20260810
```

从tmux脱离但保持训练：`Ctrl+B`，松开后按 `D`。

### 9.4 启动后最低核验

```bash
pgrep -af 'python.*train.py'
nvidia-smi
tail -n 30 logs/formal_deimv2_n_japan4_b0_160e_seed42_20260810.log
```

确认只有一条训练，日志越过 `Start training` 并连续完成多个batch，没有 `ancdata`、NaN、Inf或CUDA OOM。前期AP偏低不是停止条件。

### 9.5 完成条件与交付指标

训练只有满足下列条件才算完成：

- 退出码为0；
- 完整160E；
- `best_stg1.pth`和`last.pth`存在且可读取；
- 训练/Val曲线完整；
- 未读取Test；
- 记录best epoch以及E140-E160趋势；
- 记录权重和日志SHA256。

至少汇报：

```text
Precision / Recall
AP50
AP50:95
AP75
AP-small / medium / large
AR100
D00 / D10 / D20 / D40逐类AP和AP75
best epoch
Params / GFLOPs
PyTorch FP16 batch=1 latency
peak VRAM
```

## 10. B0之后的决策边界

DEIMv2-N B0是诊断母体，不预设一定优于YOLO26。结果出来后先回答：

1. 与YOLO26相比，DEIMv2-N整体AP、AP75、AP-small和AR100分别如何；
2. D00/D10细裂缝与D20/D40结构性损伤中，哪一类最受益或最受损；
3. 短板来自浅层细节、多尺度encoder、query预算、收敛速度还是数据协议；
4. 3.55M参数和真实延迟是否形成有意义的精度—效率折中。

下一阶段只允许从结果驱动选择一条路线：

- 若DEIMv2-N具有较强AP/定位表现：围绕其真实弱点设计一个Road-DETR单变量改动；
- 若精度接近但效率有优势：优先研究query/decoder预算或轻量encoder的Pareto压缩；
- 若明显弱于YOLO26但存在明确类别/尺度优势：判断该优势是否足以形成针对性路线；
- 若所有关键指标均无优势：停止把DEIMv2-N当唯一母体，再公平审计RT-DETRv2-R18或DEIMv2-Pico，而不是立即给N堆模块救火。

可能的后续研究方向只作为候选，不得在B0前实施：

```text
Detail-preserving downsampling for D00/D10
Road dynamic scale fusion in the hybrid encoder
query count / decoder depth efficiency sweep
road-region weighted VFM distillation
```

## 11. 当前禁止事项

- 不从今日试跑checkpoint续训；
- 不在B0中修改模型、loss、matcher、augmentation、LR或scheduler；
- 不读取Test；
- 不并发运行第二模型；
- 不在B0结果出来前开发道路模块；
- 不把YOLO26不同数据协议的AP与DEIMv2直接混比；
- 不因早期AP低而提前停止160E；
- 不为了“救DETR”临时加attention、P2、DySample或其他旧YOLO模块。

## 12. 文件索引

- 主研究交接：`docs/HANDOFF_2026-08-09_YOLO26_TO_DEIMV2.md`
- 明日操作交接：`docs/HANDOFF_2026-08-09_JAPAN4_DEIMV2N_B0.md`
- 数据转换审计：`reports/japan4_deimv2_annotation_conversion.json`
- 模型静态审计：`reports/japan4_deimv2_n_static_audit_b32.json`
- 数据转换工具：`tools/prepare_japan4_deimv2_annotations.py`
- 模型审计工具：`tools/audit_japan4_deimv2_n.py`

当前状态：今日实验已结束，GPU空闲，明日从官方预训练 fresh start 运行唯一的DEIMv2-N Japan4 160E B0。
