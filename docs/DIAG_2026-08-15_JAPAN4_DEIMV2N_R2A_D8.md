# Japan4 DEIMv2-N D8：R2-A Refiner 收益路由反事实审计

日期：2026-08-15  
结论：**STOP。R2-A refiner 不具备可利用的路由收益，停止 R2-B/门控救援。**

## 诊断对象与因果边界

- 冻结权重：R2-A E29 `checkpoint0029.pth`。
- 数据：Japan4 Val，790 张；不使用 Test。
- 不训练、不调参；同一前向重建四条确定性路线：
  - `all`：R2-A 原始全 query refiner；
  - `none`：仅关闭 refiner residual；
  - `entropy25`：每张图 pre-FDR 熵最高的 25% query；
  - `input_small`：预测归一化面积 `< (32/640)^2`。
- 逐-query 分析把 Hungarian assignment 固定在 `none`，避免重匹配制造假收益。
- `all` 重建与原 R2-A 输出最大误差 `< 2e-5`。

第一版正式运行发现 Val loader 的标注是 640 像素绝对 `xyxy`，不符合 matcher 的归一化 `cxcywh` 输入契约。其 COCO 结果有效，但逐-query 结果作废。以下全部结论来自修正后重跑的 `full_v2`。

## 四路 COCO 结果

| 路线 | AP | AP50 | AP75 | APsmall | APmedium | APlarge |
|---|---:|---:|---:|---:|---:|---:|
| all | 0.227219 | 0.487423 | 0.183476 | 0.099876 | 0.183176 | 0.266732 |
| none | 0.227141 | 0.487497 | 0.183476 | 0.099884 | 0.183114 | 0.266495 |
| entropy25 | 0.227141 | 0.487499 | 0.183476 | 0.099883 | 0.183113 | 0.266496 |
| input_small | 0.227157 | 0.487458 | 0.183475 | 0.099878 | 0.183112 | 0.266495 |

关键反事实：`all - none` 只有 `+0.000078 AP`，AP75 几乎为零变化，APsmall 反而 `-0.000009`。因此 R2-A 相对 B0 的历史 APsmall 差异不能归因于 refiner 在推理时直接修正小目标；更可能来自训练期间共享参数漂移或普通随机波动。

## 固定匹配的 1,647 个 query

- GT-small：中位 `delta IoU = +0.000073`；GT-medium：`+0.000036`；GT-large：`-0.000031`。量级均不足以改变任何一个 IoU=0.75 crossing。
- `entropy25`：仅覆盖 79 个匹配 query；正改善率富集 `0.846x`，中位 `delta IoU=-0.000068`，95% CI 跨 0。
- `input_small`：覆盖 91 个匹配 query；正改善率富集 `1.129x`，中位 `delta IoU=+0.000080`，95% CI 跨 0。
- 两条路线均未达到预注册的 `1.5x` 富集，也未产生 IoU=0.75 上穿。

## 判决

D8 的两条固定门控均为 `GO=false`：

1. 熵不是 refiner 收益的有效选择变量；
2. 固定小框规则有微弱方向性，但效应约为万分之一 IoU、置信区间跨零；
3. refiner residual 本身没有产生 R2-A 曾被观察到的 APsmall 增益；
4. 因而继续训练 R2-B、学习门控或扫描阈值都会变成对噪声的救援。

冻结结论：**关闭 R2 系列，不启动新的 routing 训练。后续模型改进不能再建立在“R2-A refiner 已证明有效”这一前提上。**

## 产物

- 正式摘要：`/root/ROAD-DEIMv2-D8/reports/d8_r2a_20260815_full_v2/d8_summary.json`
- query 明细：`/root/ROAD-DEIMv2-D8/reports/d8_r2a_20260815_full_v2/d8_query_rows.csv`
- 完整日志：`/root/ROAD-DEIMv2-D8/logs/d8_r2a_20260815_full_v2.log`
- 代码分支：`codex/japan4-d8-r2-benefit-routing`
