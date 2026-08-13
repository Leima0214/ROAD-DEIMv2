# Japan4 DEIMv2-N B0 D6 双视角梯度诊断

## 冻结边界

- 父权重：`best_stg1.pth`，SHA256 `0f0a5623491066f18acde42d6b436f18e6eb0beddcc8cdd0df847da7e8f00420`。
- 数据：Train no-augmentation 子集，`64 × batch4 = 256` 张图；不读取 Test。
- `D6-Actual`：真实 post-LQE 最终主 `loss_mal`。
- `D6-Pure`：固定同一最终匹配，用 LQE 前 raw class logits 重算 MAL。
- `Localization`：最终主输出的加权 bbox + GIoU + FGL。
- 只反向统计梯度；没有 optimizer、参数更新或模型训练。

## 主结论

D6 判为 **Case A**：Actual 与 Pure 都和定位目标稳定冲突，说明冲突不是 LQE 人为耦合出来的。但原子分组同时否定了“复制整块 cross-attention + FFN”的初始设想；冲突主要集中在最终层的归一化参数，而不是 attention/FFN 核心权重。

| 视角 / 参数块 | mean cosine | bootstrap mean 95% CI | 负批次比例 | 判断 |
|---|---:|---:|---:|---|
| Actual, all-final | -0.0691 | [-0.0778, -0.0605] | 98.4% | 冲突 |
| Pure, all-final | -0.0780 | [-0.0873, -0.0689] | 100.0% | 冲突 |
| Actual, self-attn core | +0.0659 | [+0.0555, +0.0767] | 3.1% | 协同 |
| Actual, norm1 | -0.3680 | [-0.4063, -0.3302] | 96.9% | 强冲突 |
| Actual, sampling offsets | +0.0243 | [-0.0353, +0.0819] | 46.9% | 不确定 |
| Actual, attention weights | +0.1355 | [+0.1111, +0.1590] | 14.1% | 协同 |
| Actual, gateway linear | -0.0207 | [-0.0321, -0.0092] | 67.2% | 弱冲突 |
| Actual, gateway RMSNorm | -0.4322 | [-0.4526, -0.4114] | 100.0% | 最强冲突 |
| Actual, FFN core | +0.0304 | [+0.0269, +0.0336] | 0.0% | 协同 |
| Actual, norm3 | -0.0271 | [-0.0530, -0.0005] | 60.9% | 边缘冲突 |

Pure 视角复现同一结构：`norm1=-0.3722`、`gateway RMSNorm=-0.4412`，同时 attention weights、self-attention core 和 FFN core 仍为正。因此 LQE 不是主要冲突源，核心注意力与 FFN 也不应被整体拆开。

全目标稳健性视角（包含辅助/DN loss key）仍显示 `all-final=-0.0931`、`gateway RMSNorm=-0.4631`、`norm1=-0.2881`；但 gateway linear 转为 `+0.0185`，进一步说明首要靶点是 normalization，而不是门控线性层。

## 类别与空间证据

四类的 all-final Actual/Pure 均为负：D00 `-0.0405/-0.0478`、D10 `-0.0236/-0.0249`、D20 `-0.0430/-0.0535`、D40 `-0.0307/-0.0323`。四类 gateway RMSNorm 也全部稳定为负；gateway linear 在类别分解中大多为正或不确定。因此总体的弱线性门控冲突不能作为复制整套 gateway 的充分证据。

按网格数校正后的梯度能量密度显示：

- P4：定位边界密度是背景的 `0.884×`，高于 Actual 的 `0.563×` 与 Pure 的 `0.501×`。
- P5：定位边界密度是背景的 `1.083×`，Actual/Pure 分别为 `1.267×/1.333×`。

这说明分类与定位的空间需求存在层级差异，但没有给出“复制 deformable offsets/weights”的一致证据；D6 的参数梯度反而显示这两块总体中性或协同。

## 对下一实验的约束

1. **停止原版 R4-A（整块 final cross-attention + FFN 双分支）**：它会复制 D6 已证明协同的核心参数，变量和成本都过大。
2. **允许一个更小的 R4-A：最终层任务特异归一化流**。最终层共享 self-attention、deformable attention、gateway linear 和 FFN 权重；从 `norm1` 后建立 classification/localization 两条流，各自使用 `norm1` 与 gateway RMSNorm。定位流进入 FDR/bbox head，分类流进入 score head，原 LQE 继续用分类分数与定位 corners 组合。
3. 首版只复制 `norm1 + gateway RMSNorm`，新增参数约 `256`；`norm3` 暂时共享，因为主视角只有边缘冲突、全目标视角为正。除该机制外，训练协议保持与 32E 筛选协议一致。
4. 这是机制准入证据，不保证 AP 上涨。R4-A 仍需用同协议短测试与 B0 matched epoch 比较；若没有稳定改善 AP/AP75，则停止该路线，不追加更多解耦模块救援。

## 产物

- `reports/d6_b0_20260813_atomic_full/d6_diagnostics.json`
- `reports/d6_b0_20260813_atomic_full/d6_gradient_rows.csv`
- `reports/d6_b0_20260813_atomic_full/d6_spatial_energy_rows.csv`
- `reports/d6_b0_20260813_atomic_full/remote_run.log`

对应 SHA256：

- JSON：`4493725ca991d1b55830a2e943c460afa68154e1cfabb7330b3cfd0695d4b3bb`
- gradient CSV：`2bb64b685b2959b1acfb8d47a8075a0f11cf27d06b339a414bb8befb6cb85670`
- spatial CSV：`b8669af47fb02082ea89d1789f2d591e31e871a0f7134d6ea1db37727884b244`
- remote log：`aa64d826559b9bc0ca8ec0eb194cd558ffc03a4880a223d439b0ba2ba0efee6e`
