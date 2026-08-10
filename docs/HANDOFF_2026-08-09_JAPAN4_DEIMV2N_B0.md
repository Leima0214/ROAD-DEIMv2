# Japan4 DEIMv2-N B0 实验交接（2026-08-09）

> **状态更新（2026-08-10）：** 本文中的启动说明已完成使命，不得再次启动或 resume B0。正式冻结结果、薄弱点诊断、artifact 哈希与下一条唯一因果路线见 [`FREEZE_2026-08-10_JAPAN4_DEIMV2N_B0.md`](FREEZE_2026-08-10_JAPAN4_DEIMV2N_B0.md)。

## 当前结论

DEIMv2-N 已完成 Japan4-cleanV3 数据、预训练权重、真实 batch、forward、loss、backward、AMP 和正式训练链路验证，可以在下一次 GPU 会话中直接从预训练权重启动 160E B0。

2026-08-09 的运行仅作为链路试跑：完整完成 E0-E3，并进入 E4 第一个 batch 后由人工停止。该运行不是正式实验结果，明日禁止 resume，必须 fresh start。

## 冻结版本与协议

- Git 仓库：`Leima0214/ROAD-DEIMv2`
- 分支：`codex/japan4-deimv2-n`
- 冻结 commit：`c74592bb072a1ee56e8e5d0ff8da4de2d742adfe`
- 模型配置：`configs/deimv2/deimv2_hgnetv2_n_japan4.yml`
- 数据配置：`configs/dataset/japan4_detection.yml`
- 数据根目录：`/JAPAN4-DETR/JAPAN4-DETR`
- 数据划分：train 6320、val 790、test 790；正式训练与模型选择只读 train/val，禁止 Test
- 类别：D00、D10、D20、D40，训练标号 `0..3`
- 训练周期：160
- batch size：32
- 图像尺寸：640
- workers：8
- device：CUDA 0
- seed：42
- AMP：开启
- 初始化：DEIMv2-N COCO 预训练权重；4类分类相关张量随机初始化
- 模型参数量：3,551,477，其中可训练参数 3,551,475
- 静态审计峰值显存：11,014 MiB；实际试跑约12,000 MiB

预训练审计：628/637 checkpoint state items 合法匹配，按参数量覆盖率为 99.9238%。不兼容项均为 COCO 80类到Japan4四类的分类相关张量，属于预期情况。

## Scheduler 定论

只实施了一个配置修改：

```text
flat_epoch: 7800 -> 78
```

真实配置 dry-run：

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

需要知情记录：官方配置当前 `lr_gamma=1.0`，因此 warmup 后最小 LR 与 base LR 相同，E78 后不会产生数值上的余弦下降。为遵守“只改 7800→78”的冻结要求，本轮没有修改 `lr_gamma`。正式 B0 必须继续使用这一冻结配置，不得临时改 scheduler。

## 已验证状态

- Japan4数据加载：PASS
- DEIMv2-N预训练加载：PASS
- FP32 forward/loss/backward：PASS
- AMP forward/loss/backward：PASS
- AMP动态缩放在 scale 8192 时梯度全部有限：PASS
- batch=32真实训练：PASS
- Val COCO evaluator：PASS
- checkpoint保存：PASS
- 无Test读取：确认

静态审计报告：

- `reports/japan4_deimv2_n_static_audit_b32.json`
- `reports/japan4_deimv2_annotation_conversion.json`

## 2026-08-09试跑记录

正式训练链路第一次启动在首个 optimizer step 前因远程 `ulimit -n=1024` 触发 DataLoader `RuntimeError: received 0 items of ancdata`。这不是模型或数据故障。零步失败运行已在远程归档为：

```text
outputs/failed_prestart_deimv2_n_japan4_b0_160e_seed42_ancdata
logs/failed_prestart_deimv2_n_japan4_b0_160e_seed42_ancdata.log
```

将启动 shell 的文件描述符上限设为 `65535` 后，保持 `workers=8`、batch=32和全部训练协议不变，训练正常运行。

链路试跑路径：

```text
/root/ROAD-DEIMv2/outputs/formal_deimv2_n_japan4_b0_160e_seed42_20260809
/root/ROAD-DEIMv2/logs/formal_deimv2_n_japan4_b0_160e_seed42_20260809.log
```

试跑完整完成 E0-E3，E3 Val AP50:95 为 `0.0728197`，随后在 E4 第一个 batch 人工停止。该数值仅证明训练正常收敛，不作为论文或B0最终指标。

保留工件及 SHA256：

```text
best_stg1.pth  f6b4975887c3d4fc648401bc4009d000e6f17062b462b21403698ba52159ee16
last.pth       7bb591f753b0c7439f7e4689ef9301d70737bcca194709942bdb37d7751c3a67
training log   77eb964db71be2af671096ba8c80b566890cd43091abe522f50e531bdb0eb4d6
```

结束核验：训练主进程及DataLoader workers均已停止，tmux会话已关闭，GPU为0%利用率、1MiB显存占用。服务器没有执行关机。

## 明日唯一下一实验

从冻结的 COCO 预训练权重 fresh start 运行 `DEIMv2-N Japan4 160E B0`。不得从本日试跑的 `last.pth` 或 `best_stg1.pth` resume。

### 1. 连接并做启动前检查

端口若租赁平台发生变化，用新端口替换 `<PORT>`：

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

预期：HEAD为本文档记录的commit或包含本文档的后续commit；转换后预训练权重 SHA256 为：

```text
e76c71a53534d767bb09bb4eaabba8c11aefb48df1dc4f9e120a5b58677ae2c6
```

### 2. 启动 fresh 160E B0

使用全新输出目录，避免覆盖本日试跑：

```bash
cd /root/ROAD-DEIMv2
mkdir -p logs

tmux new-session -d -s deimv2_n_japan4_b0_160e_20260810 "bash -lc 'ulimit -n 65535; set -o pipefail; CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python -u train.py -c configs/deimv2/deimv2_hgnetv2_n_japan4.yml -t weights/deimv2_hgnetv2_n_coco_hf.pth --use-amp --seed 42 -d cuda:0 --output-dir outputs/formal_deimv2_n_japan4_b0_160e_seed42_20260810 2>&1 | tee logs/formal_deimv2_n_japan4_b0_160e_seed42_20260810.log; rc=\${PIPESTATUS[0]}; echo \$rc > logs/formal_deimv2_n_japan4_b0_160e_seed42_20260810.exitcode; exec bash'"
```

### 3. 查看日志

```bash
tail -f /root/ROAD-DEIMv2/logs/formal_deimv2_n_japan4_b0_160e_seed42_20260810.log
```

或进入tmux：

```bash
tmux attach -t deimv2_n_japan4_b0_160e_20260810
```

从tmux脱离但保持训练：`Ctrl+B`，松开后按 `D`。

### 4. 启动后最低核验

```bash
pgrep -af 'python.*train.py'
nvidia-smi
tail -n 30 logs/formal_deimv2_n_japan4_b0_160e_seed42_20260810.log
```

必须确认：只有一条训练、GPU显存和利用率正常、日志已越过 `Start training` 并至少完成多个batch、没有 `ancdata`、NaN、Inf或CUDA OOM。

## 禁止事项与停止条件

禁止：

- resume本日4轮试跑；
- 修改模型、数据划分、增强、loss、学习率或scheduler；
- 读取Test；
- 同时启动第二条训练；
- 覆盖或删除本日试跑工件。

仅在出现明确工程异常时停止，例如持续NaN/Inf、CUDA OOM、DataLoader再次崩溃或loss长期完全不更新。前期AP低不构成停止理由。

训练完成后应核验退出码为0、完整160轮日志、`best_stg1.pth`、`last.pth`和Val曲线，再进行统一Val分析。
