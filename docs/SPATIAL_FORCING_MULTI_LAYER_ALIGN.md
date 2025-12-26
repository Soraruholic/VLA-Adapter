# Spatial Forcing Multi-Layer Alignment

## 概述

多层对齐（Multi-Layer Alignment）是 Spatial Forcing 框架的扩展功能，支持在 VGGT 和 VLA 模型之间进行**多层独立对齐**（Independent Layer-wise Alignment）。与原有的单层对齐不同，多层对齐允许同时对齐多对对应的层，并为每对层分配独立的损失权重。

## 方案设计

### 层间独立对齐 (Independent Layer-wise Alignment)

每对对应的 VGGT 和 VLM 层独立计算对齐损失，最终的总损失为各层损失的加权和：

```
Total_Loss = Σ (coeff_i × AlignLoss(VGGT_layer_i, VLA_layer_i))
```

**优势**：
- 可以为不同层分配不同的权重，灵活控制各层的贡献
- 支持配置任意数量的层对齐
- 支持负索引（如 -1 表示最后一层）

## 配置参数

在 `FinetuneConfig` 中新增以下参数：

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `use_multi_layer_align` | bool | False | 是否启用多层对齐 |
| `multi_layer_vggt` | str | "-3,-2,-1" | VGGT 层索引（逗号分隔） |
| `multi_layer_vla` | str | "-3,-2,-1" | VLA 层索引（逗号分隔） |
| `multi_layer_coeffs` | str | "0.2,0.3,0.5" | 各层损失权重（逗号分隔） |
| `share_multi_layer_projector` | bool | True | 是否在所有层对之间共享投影器 |

### 层索引说明

- 支持正索引和负索引
- 负索引表示从末尾计数，如 `-1` 表示最后一层
- VGGT 有 24 层（索引 0-23 或 -24 到 -1）
- VLA 层数取决于具体模型

### 权重系数说明

- 各层的损失权重可以任意设置
- 建议权重之和在 0.5-1.0 范围内
- 通常给予较深层（靠近输出的层）更高的权重

## 使用示例

### 基础配置

```bash
python vla-scripts/finetune.py \
    --use_spatial_forcing=True \
    --use_multi_layer_align=True \
    --multi_layer_vggt="-3,-2,-1" \
    --multi_layer_vla="-3,-2,-1" \
    --multi_layer_coeffs="0.2,0.3,0.5" \
    --share_multi_layer_projector=True
```

### 高级配置：使用独立投影器

```bash
python vla-scripts/finetune.py \
    --use_spatial_forcing=True \
    --use_multi_layer_align=True \
    --multi_layer_vggt="-6,-4,-2,-1" \
    --multi_layer_vla="-6,-4,-2,-1" \
    --multi_layer_coeffs="0.1,0.2,0.3,0.4" \
    --share_multi_layer_projector=False
```

## 实现细节

### MultiLayerAlignProjector 类

位于 `prismatic/models/projectors.py`，主要功能：

1. **投影维度对齐**：将 VLA 的隐藏状态投影到 VGGT 特征维度（2048）
2. **支持共享/独立投影器**：
   - `share_projector=True`：所有层对共享一个投影器
   - `share_projector=False`：每个层对使用独立的投影器
3. **支持视图级别的共享**：可配置是否在不同视图之间共享投影器

### 损失计算流程

```
1. 解析配置参数，获取层索引列表和权重列表
2. 对于每个层对 (vggt_layer_i, vla_layer_i)：
   a. 提取 VLA 隐藏状态的视觉 token
   b. 提取 VGGT 特征（包含 frame 和 global 特征的拼接）
   c. 对 VGGT 特征进行空间重采样以匹配 VLA 的分辨率
   d. 通过投影器对齐维度
   e. 计算余弦相似度损失
3. 计算加权总损失
```

### 架构图

```
VLA Model                          VGGT Model
    │                                  │
    ▼                                  ▼
┌─────────┐                     ┌─────────┐
│ Layer -3│ ◄──────────────────►│ Layer -3│  × coeff_0
├─────────┤                     ├─────────┤
│ Layer -2│ ◄──────────────────►│ Layer -2│  × coeff_1
├─────────┤                     ├─────────┤
│ Layer -1│ ◄──────────────────►│ Layer -1│  × coeff_2
└─────────┘                     └─────────┘
    │                                  │
    └──────────► Total Loss ◄──────────┘
```

## 模式对比

| 特性 | Legacy 模式 | Dual Align 模式 | Multi-Layer 模式 |
|------|------------|-----------------|------------------|
| 对齐层数 | 单层 | 单层（分 Frame/Global） | 多层 |
| 特征类型 | 拼接特征 (2048) | Frame (1024) + Global (1024) | 拼接特征 (2048) |
| 独立损失 | 否 | Frame/Global 分开 | 每层独立 |
| 权重配置 | 单一系数 | frame_coeff + global_coeff | 每层独立系数 |

## 训练监控

启用多层对齐后，以下 metrics 会被记录到 W&B：

- `align_loss`：总对齐损失（加权和）
- `layer_0_align_loss`：第 0 层对齐损失
- `layer_1_align_loss`：第 1 层对齐损失
- `layer_N_align_loss`：第 N 层对齐损失

## 注意事项

1. **互斥模式**：`use_multi_layer_align`、`use_dual_align` 和 Legacy 模式是互斥的
2. **参数长度一致**：`multi_layer_vggt`、`multi_layer_vla` 和 `multi_layer_coeffs` 的元素数量必须相同
3. **显存考虑**：多层对齐会增加显存占用，尤其是使用独立投影器时
4. **checkpoint 保存**：`MultiLayerAlignProjector` 的权重会自动保存到 checkpoint

## 推荐配置

### 常规训练（推荐）

```bash
--use_multi_layer_align=True
--multi_layer_vggt="-3,-2,-1"
--multi_layer_vla="-3,-2,-1"
--multi_layer_coeffs="0.2,0.3,0.5"
--share_multi_layer_projector=True
```

### 深度特征对齐

```bash
--use_multi_layer_align=True
--multi_layer_vggt="-6,-5,-4,-3,-2,-1"
--multi_layer_vla="-6,-5,-4,-3,-2,-1"
--multi_layer_coeffs="0.05,0.1,0.15,0.2,0.25,0.25"
--share_multi_layer_projector=True
```

### 稀疏层对齐

```bash
--use_multi_layer_align=True
--multi_layer_vggt="-12,-6,-1"
--multi_layer_vla="-12,-6,-1"
--multi_layer_coeffs="0.2,0.3,0.5"
--share_multi_layer_projector=False
```
