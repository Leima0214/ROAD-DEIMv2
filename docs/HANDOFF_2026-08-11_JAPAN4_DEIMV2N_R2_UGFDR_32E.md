# Japan4 DEIMv2-N R2：UG-FDR 32E 验证交接

## 实验边界

本分支从冻结 B0 提交 `4e4fc4e` 创建，不包含已失败的 R1 P3 模块。三个配置均为 32E 筛选，不是正式 160E 结果：

| 配置 | 唯一方法差异 | 目的 |
|---|---|---|
| `deimv2_hgnetv2_n_japan4_r2a_fdr_refiner_all_32e.yml` | 所有 query 使用轻量 FDR 残差精修 | 检验小模块是否具有可修复性 |
| `deimv2_hgnetv2_n_japan4_r2b_fdr_refiner_entropy25_32e.yml` | 同一模块，仅最高平均熵 25% query 应用残差 | 检验不确定性门控能否保护容易样本 |
| `deimv2_hgnetv2_n_japan4_r2ctrl_full_l3_32e.yml` | 标准完整第四 Decoder 层 | 可选深度/容量对照，不是主方法 |

R2-A/R2-B 保持 B0 的三层 Decoder 和分类 logits 路径，新增模块仅进行一次 P4/P5 object-centric deformable memory read，并输出四边 FDR logits 残差：

- 无 P3；
- 无 self-attention；
- 无新分类头；
- 无新 loss family、matcher 或数据设置；
- 残差末层严格零初始化，E0 前向与 B0 等价；
- R2-B 使用停止梯度的四边平均归一化熵，每张图固定 top-25%；
- 当前门控是 accuracy-first 的功能门控，仍以 dense 方式计算 refiner，不宣称获得稀疏推理加速。

轻量 refiner 给 DEIMTransformer 增加 `46,052` 个参数；相对完整约 3.55M B0，约为 `1.3%`。最终必须重新实测整模型参数、FLOPs 与 RTX 4090 延迟。

## 首次拉取与前检

```bash
cd /root/ROAD-DEIMv2
git fetch origin
git switch codex/japan4-ug-fdr-r2-validation
git pull --ff-only origin codex/japan4-ug-fdr-r2-validation
ulimit -n 65535
sha256sum weights/deimv2_hgnetv2_n_coco_hf.pth
python tools/verify_japan4_r2_ugfdr.py
```

官方 COCO 初始化权重 SHA256 必须为：

```text
e76c71a53534d767bb09bb4eaabba8c11aefb48df1dc4f9e120a5b58677ae2c6
```

前检必须输出 `"status": "PASS"`。它验证配置差异、B0/R2 共享初始化、零初始化输出等价、20 个合成 query 中 R2-B 精确路由 5 个，以及两阶段梯度解锁。它不是检测精度证据。

## 推荐运行顺序

先只运行 R2-A。不要并行运行三个实验，也不要在看到中途波动后改配置。

```bash
cd /root/ROAD-DEIMv2
mkdir -p logs
ulimit -n 65535
nohup python -u train.py \
  -c configs/deimv2/deimv2_hgnetv2_n_japan4_r2a_fdr_refiner_all_32e.yml \
  -t weights/deimv2_hgnetv2_n_coco_hf.pth \
  -d cuda:0 \
  --seed 42 \
  --use-amp \
  > logs/deimv2_n_japan4_r2a_fdr_refiner_all_32e_seed42.log 2>&1 &
echo $!
```

查看日志：

```bash
tail -f /root/ROAD-DEIMv2/logs/deimv2_n_japan4_r2a_fdr_refiner_all_32e_seed42.log
```

R2-A 完成并满足下列条件后，才允许把配置名替换为 R2-B 运行熵门控版本：

- 相对 B0 E31，AP50:95 至少不明显下降；
- Q4/AP75 有明确改善，或出现“Q4 改善但容易样本受损”这一可由门控修复的模式；
- 没有 D10、AP-small、NaN、OOM 或梯度异常。

完整 L3 只在需要确认“标准额外 Decoder 深度是否存在上限收益”时单独运行。它的新第四层没有官方 N checkpoint 对应权重，因此其结果同时包含新增容量与初始化影响，不得当作 UG-FDR 的直接消融。

## E31 主判定

冻结 B0 E31：AP50:95 `0.234331`，AP75 `0.190171`，AP-small `0.087867`。

- 强通过：AP50:95 `>= 0.239331`，且 AP75 至少 `+0.005`；
- 暂定通过：AP50:95 `>= 0.237331`，且 AP75、AP-small、D10 无实质回退；补独立种子，不自动跑 160E；
- 停止：AP 增益 `< 0.002` 且 Q4/AP75 不改善，或 AP75/D10 明显下降；
- 仅 Q4 改善、总 AP 持平：只允许 R2-B 这一次门控验证，不能直接晋级正式实验。

统一以 E31 对 E31 为主比较；best-within-32E 只能附报。禁止 resume B0/R1 权重、访问 Test、压缩 160E 调度、阈值扫描、同时修改 loss/LR/matcher/增强或叠加第二模块。
