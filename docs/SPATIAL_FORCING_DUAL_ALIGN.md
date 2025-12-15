# Spatial Forcing 双路对齐实现文档

## 概述

本文档描述了 Spatial Forcing 功能的双路对齐（Dual Alignment）模式的实现细节。该模式将 VGGT 的 2048 维特征分离为 Frame-level（1024维）和 Global-level（1024维）两部分，分别与 VLA 的 vision tokens 进行对齐。

---

## 背景

### VGGT 输出结构

VGGT 的 Aggregator 采用交替注意力机制（Alternating Attention）：

1. **Frame Attention**：每帧独立处理，形状 `(B*S, P, 1024)`，捕捉帧内空间特征
2. **Global Attention**：跨帧处理，形状 `(B, S*P, 1024)`，建立视角间关系

最终输出是两者的 concat：`[B, S, P, 2048]`

```
VGGT Output (2048D) = Frame Features (1024D) || Global Features (1024D)
```

### 设计哲学

对于每个视角：
- 应与对应的 **Frame-level 特征**对齐（帧内空间信息）
- 同时应与 **Global-level 特征**对齐（跨视角融合信息）

---

## 实现细节

### 1. 新增配置参数

**文件**: `vla-scripts/finetune.py`

```python
# ========== [SPATIAL FORCING] Dual Alignment Configuration ==========
use_dual_align: bool = False                     # 是否启用双路对齐模式
share_frame_projector: bool = True               # 不同视角是否共享 Frame projector
frame_align_layer: int = -1                      # Frame 对齐使用的 VGGT 层索引
global_align_layer: int = -1                     # Global 对齐使用的 VGGT 层索引
frame_loss_coeff: float = 0.5                    # Frame 对齐损失系数
global_loss_coeff: float = 0.5                   # Global 对齐损失系数
```

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `use_dual_align` | bool | False | 启用双路对齐模式（新）；False 使用原有 concat 对齐（legacy） |
| `share_frame_projector` | bool | True | True：所有视角共享同一个 Frame projector；False：每个视角独立 projector |
| `frame_align_layer` | int | -1 | VGGT 的哪一层用于 Frame 特征提取（-1 表示最后一层） |
| `global_align_layer` | int | -1 | VGGT 的哪一层用于 Global 特征提取（-1 表示最后一层） |
| `frame_loss_coeff` | float | 0.5 | Frame 对齐损失的权重系数 |
| `global_loss_coeff` | float | 0.5 | Global 对齐损失的权重系数 |

---

### 2. 新增 Projector 类

**文件**: `prismatic/models/projectors.py`

#### FrameAlignProjector

```python
class FrameAlignProjector(nn.Module):
    """
    [DUAL ALIGN] Frame-level alignment projector.
    Projects LLM embeddings to vggt_dim (1024) to align with VGGT frame-level features.
    
    支持两种模式：
    - share_projector=True: 所有视角共享同一个 projector
    - share_projector=False: 每个视角有独立的 projector（参数量 × N）
    """
```

**参数**:
- `llm_dim`: LLM 隐藏层维度
- `vggt_dim`: VGGT 特征维度（1024）
- `align_loss_type`: 对齐损失类型（"cosine"）
- `use_vlm_norm`: 是否对 VLM embeddings 应用 LayerNorm
- `num_views`: 视角数量（仅在 share_projector=False 时使用）
- `share_projector`: 是否在视角间共享 projector

#### GlobalAlignProjector

```python
class GlobalAlignProjector(nn.Module):
    """
    [DUAL ALIGN] Global-level alignment projector.
    Projects LLM embeddings to vggt_dim (1024) to align with VGGT global-level features.
    
    Global features 编码了跨视角关系，因此所有视角共享同一个 projector。
    """
```

---

### 3. 修改初始化逻辑

**文件**: `vla-scripts/finetune.py`

**位置**: `# ========== [SPATIAL FORCING] Initialize VGGT and AlignProjector ==========`

```python
if cfg.use_dual_align:
    # 双路对齐模式
    frame_align_projector = FrameAlignProjector(
        llm_dim=llm_hidden_size,
        vggt_dim=vggt_hidden_size,
        align_loss_type=cfg.align_loss_type,
        use_vlm_norm=cfg.use_vlm_norm,
        num_views=cfg.num_images_in_input,
        share_projector=cfg.share_frame_projector,
    ).to(device_id)
    
    global_align_projector = GlobalAlignProjector(
        llm_dim=llm_hidden_size,
        vggt_dim=vggt_hidden_size,
        align_loss_type=cfg.align_loss_type,
        use_vlm_norm=cfg.use_vlm_norm,
    ).to(device_id)
else:
    # Legacy 模式（原有逻辑）
    align_projector = AlignProjector(...)
```

---

### 4. 修改对齐损失计算

**文件**: `vla-scripts/finetune.py`

**位置**: `# ========== [SPATIAL FORCING] Second Layer Selection: Alignment Loss ==========`

#### 特征提取

```python
if sf_dual_enabled:
    # 提取 Frame 特征（前 1024 维）
    frame_layer_features = vggt_output["features"][cfg.frame_align_layer]
    frame_hidden = frame_layer_features[:, :, patch_start_idx:, :1024]  # [B, N, P, 1024]
    
    # 提取 Global 特征（后 1024 维）
    global_layer_features = vggt_output["features"][cfg.global_align_layer]
    global_hidden = global_layer_features[:, :, patch_start_idx:, 1024:]  # [B, N, P, 1024]
```

#### 损失计算

```python
# 计算 Frame 对齐损失
frame_align_loss = frame_align_projector(vision_hidden, frame_hidden_pooled)

# 计算 Global 对齐损失
global_align_loss = global_align_projector(vision_hidden, global_hidden_pooled)

# 组合对齐损失
align_loss = cfg.frame_loss_coeff * frame_align_loss + cfg.global_loss_coeff * global_align_loss
```

---

### 5. 优化器参数更新

**文件**: `vla-scripts/finetune.py`

```python
if cfg.use_spatial_forcing:
    if cfg.use_dual_align:
        # 双路对齐模式：添加两个 projector 的参数
        trainable_params += frame_align_projector.parameters()
        trainable_params += global_align_projector.parameters()
    else:
        # Legacy 模式
        trainable_params += align_projector.parameters()
```

---

### 6. Metrics 日志

双路对齐模式会额外记录：
- `frame_align_loss`: Frame 对齐损失
- `global_align_loss`: Global 对齐损失
- `align_loss`: 总对齐损失（加权和）

---

## 使用方法

### Legacy 模式（原有）

```bash
./train_sf.sh
```

配置示例：
```bash
--use_spatial_forcing True
--use_dual_align False  # 或不指定，默认 False
--align_loss_coeff 0.5
--vggt_layers_align -1
```

### 双路对齐模式（新）

```bash
./train_sf_dual.sh
```

配置示例：
```bash
--use_spatial_forcing True
--use_dual_align True
--share_frame_projector True
--frame_align_layer -1
--global_align_layer -1
--frame_loss_coeff 0.5
--global_loss_coeff 0.5
```

---

## 架构图

```
                    VLA Vision Tokens [B, N*P, D_llm]
                           │
            ┌──────────────┴──────────────┐
            ▼                             ▼
    FrameAlignProjector              GlobalAlignProjector
    (per-view or shared)              (shared across views)
    D_llm → 1024                       D_llm → 1024
            │                             │
            ▼                             ▼
    Frame Features                   Global Features
    [B, N, P, 1024]                  [B, N, P, 1024]
    (from VGGT frame)                (from VGGT global)
            │                             │
            ▼                             ▼
      Frame Align Loss              Global Align Loss
            │                             │
            └──────────────┬──────────────┘
                           ▼
            Total Align Loss = α*frame + β*global
```

---

## 配置建议

### 默认配置

```python
use_dual_align = True
share_frame_projector = True
frame_align_layer = -1   # 最后层
global_align_layer = -1  # 最后层
frame_loss_coeff = 0.5
global_loss_coeff = 0.5
```

### 实验性配置

1. **不同层对齐**：浅层更关注局部，深层更关注全局
   ```python
   frame_align_layer = 12   # 中间层
   global_align_layer = -1  # 最后层
   ```

2. **独立视角 projector**：增加表达能力，但参数量增加
   ```python
   share_frame_projector = False
   ```

3. **调整损失权重**：根据任务特点调整
   ```python
   frame_loss_coeff = 0.7   # 更重视帧内对齐
   global_loss_coeff = 0.3  # 减少全局约束
   ```

---

## 向后兼容性

- 原有的 `use_dual_align=False` 模式完全保留
- `AlignProjector` 类保持不变
- 原有训练脚本 `train_sf.sh` 无需修改

---

## 文件变更清单

| 文件 | 变更类型 | 说明 |
|-----|---------|------|
| `prismatic/models/projectors.py` | 修改 | 添加 `FrameAlignProjector` 和 `GlobalAlignProjector` 类 |
| `vla-scripts/finetune.py` | 修改 | 添加配置参数、初始化逻辑、损失计算、优化器参数、checkpoint保存 |
| `train_sf_dual.sh` | 新增 | 双路对齐模式的训练脚本示例 |
| `docs/SPATIAL_FORCING_DUAL_ALIGN.md` | 新增 | 本文档 |

---

## Checkpoint 保存与加载

### 保存逻辑

双路对齐模式下，checkpoint 会分别保存 Frame 和 Global projector：

```python
# 双路对齐模式
frame_align_projector--{step}_checkpoint.pt
global_align_projector--{step}_checkpoint.pt

# Legacy 模式
align_projector--{step}_checkpoint.pt
```

### 涉及的函数修改

1. **`save_training_checkpoint`**: 新增 `frame_align_projector` 和 `global_align_projector` 参数
2. **`run_forward_pass`**: 新增 `frame_align_projector` 和 `global_align_projector` 参数
3. **`run_validation`**: 新增 `frame_align_projector` 和 `global_align_projector` 参数

### 加载 Checkpoint（推理时）

```python
# 加载 Frame projector
frame_align_projector = FrameAlignProjector(
    llm_dim=llm_hidden_size,
    vggt_dim=1024,
    num_views=num_images_in_input,
    share_projector=True,
)
frame_align_projector.load_state_dict(
    torch.load("frame_align_projector--latest_checkpoint.pt")
)

# 加载 Global projector
global_align_projector = GlobalAlignProjector(
    llm_dim=llm_hidden_size,
    vggt_dim=1024,
)
global_align_projector.load_state_dict(
    torch.load("global_align_projector--latest_checkpoint.pt")
)
```

---

## 变更详情

### prismatic/models/projectors.py

**新增类**：
1. `FrameAlignProjector` (line 120-247)
   - 支持共享/独立视角 projector
   - `align_dimension()`: 投影 LLM embeddings
   - `compute_align_loss_cosine()`: 计算余弦相似度损失
   - `forward()`: 完整的对齐损失计算

2. `GlobalAlignProjector` (line 250-346)
   - 横跨所有视角的共享 projector
   - 结构与 FrameAlignProjector 类似

**保留类**：
- `AlignProjector` (原有，用于 legacy 模式)

### vla-scripts/finetune.py

**配置参数** (line 142-148):
```python
use_dual_align: bool = False
share_frame_projector: bool = True
frame_align_layer: int = -1
global_align_layer: int = -1
frame_loss_coeff: float = 0.5
global_loss_coeff: float = 0.5
```

**初始化逻辑** (line 984-1025):
- 根据 `use_dual_align` 分支初始化不同的 projector

**损失计算** (line 444-562):
- 支持 `sf_legacy_enabled` 和 `sf_dual_enabled` 两种模式
- 双路模式下分别计算 frame 和 global 损失

**优化器** (line 1171-1183):
- 根据模式添加对应 projector 的参数

---

## 调试提示

1. 确认 VGGT 特征维度正确分离：
   ```python
   print(f"Frame features shape: {frame_hidden.shape}")  # [B, N, P, 1024]
   print(f"Global features shape: {global_hidden.shape}")  # [B, N, P, 1024]
   ```

2. 检查损失值是否合理：
   - 初始 frame_align_loss 和 global_align_loss 应在 0.5-1.0 范围
   - 训练过程中应逐渐下降

3. 确认 projector 参数被正确优化：
   ```python
   for name, param in frame_align_projector.named_parameters():
       print(f"{name}: requires_grad={param.requires_grad}")
   ```
