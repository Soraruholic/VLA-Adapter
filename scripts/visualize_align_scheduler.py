#!/usr/bin/env python3
"""
Visualization of different scheduler functions for alignment loss coefficient.

Requirements:
- Monotonically increasing from ~0 to peak_value
- Reaches peak at peak_step (default 10k)
- Various curve shapes for different training dynamics
"""

import numpy as np
import matplotlib.pyplot as plt


def step_scheduler(step, warmup_steps=5000, peak_value=0.5, **kwargs):
    """
    阶跃式调度器：在warmup_steps之前为0，之后直接跳变为peak_value
    
    适用场景：希望模型先学习基础动作预测，再突然引入对齐约束
    """
    return peak_value if step >= warmup_steps else 0.0


def linear_scheduler(step, warmup_steps=0, peak_step=10000, peak_value=0.5, **kwargs):
    """
    线性调度器：从warmup_steps开始线性增长到peak_step
    
    适用场景：希望对齐约束平滑引入，避免训练不稳定
    """
    if step < warmup_steps:
        return 0.0
    elif step >= peak_step:
        return peak_value
    else:
        progress = (step - warmup_steps) / (peak_step - warmup_steps)
        return peak_value * progress


def cosine_scheduler(step, warmup_steps=0, peak_step=10000, peak_value=0.5, **kwargs):
    """
    余弦调度器：使用余弦曲线平滑过渡，前期增长慢，中期加速，后期趋缓
    
    适用场景：希望早期给模型更多自由学习空间，中期加速引入对齐
    """
    if step < warmup_steps:
        return 0.0
    elif step >= peak_step:
        return peak_value
    else:
        progress = (step - warmup_steps) / (peak_step - warmup_steps)
        # 使用 (1 - cos(π * progress)) / 2 实现S型曲线
        return peak_value * (1 - np.cos(np.pi * progress)) / 2


def exponential_scheduler(step, warmup_steps=0, peak_step=10000, peak_value=0.5, gamma=5.0, **kwargs):
    """
    指数调度器：指数增长，前期非常缓慢，后期快速上升
    
    gamma控制曲线陡峭程度，gamma越大后期越陡峭
    适用场景：希望前期几乎不受对齐约束影响，后期快速增强
    """
    if step < warmup_steps:
        return 0.0
    elif step >= peak_step:
        return peak_value
    else:
        progress = (step - warmup_steps) / (peak_step - warmup_steps)
        # 使用 (exp(gamma * progress) - 1) / (exp(gamma) - 1) 归一化到 [0, 1]
        return peak_value * (np.exp(gamma * progress) - 1) / (np.exp(gamma) - 1)


def sigmoid_scheduler(step, warmup_steps=0, peak_step=10000, peak_value=0.5, steepness=10.0, **kwargs):
    """
    Sigmoid调度器：S型曲线，前期和后期平缓，中期快速变化
    
    steepness控制S曲线的陡峭程度
    适用场景：希望在中间阶段快速过渡到目标值
    """
    if step < warmup_steps:
        return 0.0
    elif step >= peak_step:
        return peak_value
    else:
        progress = (step - warmup_steps) / (peak_step - warmup_steps)
        # 将progress映射到sigmoid的中心区域
        x = steepness * (progress - 0.5)
        sigmoid_val = 1 / (1 + np.exp(-x))
        # 归一化到[0, 1]范围
        sigmoid_min = 1 / (1 + np.exp(steepness * 0.5))
        sigmoid_max = 1 / (1 + np.exp(-steepness * 0.5))
        normalized = (sigmoid_val - sigmoid_min) / (sigmoid_max - sigmoid_min)
        return peak_value * normalized


def polynomial_scheduler(step, warmup_steps=0, peak_step=10000, peak_value=0.5, power=2.0, **kwargs):
    """
    多项式调度器：使用幂函数 progress^power
    
    power=1: 线性
    power=2: 二次（前期慢，后期快）
    power=0.5: 平方根（前期快，后期慢）
    适用场景：需要精细控制增长曲线形状
    """
    if step < warmup_steps:
        return 0.0
    elif step >= peak_step:
        return peak_value
    else:
        progress = (step - warmup_steps) / (peak_step - warmup_steps)
        return peak_value * (progress ** power)


def two_stage_linear_scheduler(step, warmup_steps=5000, peak_step=10000, peak_value=0.5, **kwargs):
    """
    两阶段线性调度器：先warmup阶段为0，然后线性增长到peak
    
    结合了阶跃和线性的优点
    适用场景：希望先让模型稳定，然后平滑引入对齐约束
    """
    if step < warmup_steps:
        return 0.0
    elif step >= peak_step:
        return peak_value
    else:
        progress = (step - warmup_steps) / (peak_step - warmup_steps)
        return peak_value * progress


def visualize_schedulers():
    """可视化所有scheduler函数"""
    max_steps = 15000
    steps = np.arange(0, max_steps + 1, 100)
    
    # 配置参数
    config = {
        'warmup_steps': 5000,
        'peak_step': 10000,
        'peak_value': 0.5,
    }
    
    schedulers = {
        '1. Step (阶跃式)': (step_scheduler, {'warmup_steps': 5000, 'peak_value': 0.5}),
        '2. Two-Stage Linear (两阶段线性)': (two_stage_linear_scheduler, {'warmup_steps': 5000, 'peak_step': 10000, 'peak_value': 0.5}),
        '3. Linear (纯线性)': (linear_scheduler, {'warmup_steps': 0, 'peak_step': 10000, 'peak_value': 0.5}),
        '4. Cosine (余弦)': (cosine_scheduler, {'warmup_steps': 0, 'peak_step': 10000, 'peak_value': 0.5}),
        '5. Polynomial p=2 (二次)': (polynomial_scheduler, {'warmup_steps': 0, 'peak_step': 10000, 'peak_value': 0.5, 'power': 2.0}),
        '6. Polynomial p=0.5 (平方根)': (polynomial_scheduler, {'warmup_steps': 0, 'peak_step': 10000, 'peak_value': 0.5, 'power': 0.5}),
        '7. Exponential (指数)': (exponential_scheduler, {'warmup_steps': 0, 'peak_step': 10000, 'peak_value': 0.5, 'gamma': 5.0}),
        '8. Sigmoid (S型)': (sigmoid_scheduler, {'warmup_steps': 0, 'peak_step': 10000, 'peak_value': 0.5, 'steepness': 10.0}),
    }
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(schedulers)))
    
    for idx, (name, (scheduler_fn, params)) in enumerate(schedulers.items()):
        ax = axes[idx]
        values = [scheduler_fn(s, **params) for s in steps]
        ax.plot(steps, values, color=colors[idx], linewidth=2)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Step')
        ax.set_ylabel('Coeff')
        ax.set_xlim(0, max_steps)
        ax.set_ylim(-0.05, 0.6)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='peak=0.5')
        ax.axvline(x=5000, color='red', linestyle=':', alpha=0.5, label='warmup=5k')
        ax.axvline(x=10000, color='green', linestyle=':', alpha=0.5, label='peak_step=10k')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('/home/icrlab02/vla_ws/VLA-Adapter/scripts/align_scheduler_comparison.png', dpi=150)
    print(f"图片已保存到: /home/icrlab02/vla_ws/VLA-Adapter/scripts/align_scheduler_comparison.png")
    plt.show()
    
    # 打印所有方案对比表
    print("\n" + "="*80)
    print("Align Loss Coefficient Scheduler 方案对比")
    print("="*80)
    
    print("""
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 方案                    │ 特点                           │ 适用场景              │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 1. Step (阶跃式)        │ 简单粗暴，直接跳变              │ 快速实验，baseline    │
│ 2. Two-Stage Linear     │ 先warmup再线性增长              │ 稳定训练，推荐使用    │
│ 3. Linear (纯线性)      │ 从0开始线性增长                 │ 简单平滑过渡          │
│ 4. Cosine (余弦)        │ S型曲线，中期变化快             │ 常用的warmup策略      │
│ 5. Polynomial p=2       │ 前期慢后期快（凸函数）          │ 延迟引入对齐约束      │
│ 6. Polynomial p=0.5     │ 前期快后期慢（凹函数）          │ 早期就引入对齐约束    │
│ 7. Exponential (指数)   │ 前期极慢，后期爆发增长          │ 极端延迟对齐          │
│ 8. Sigmoid (S型)        │ 中间快速过渡，两端平缓          │ 明确的过渡阶段        │
└─────────────────────────────────────────────────────────────────────────────────┘
    """)
    
    # 打印各scheduler在关键step的值
    print("\n关键step处的系数值:")
    print("-" * 80)
    key_steps = [0, 2500, 5000, 7500, 10000, 12500, 15000]
    header = f"{'Scheduler':<30}" + "".join([f"{s:>8}" for s in key_steps])
    print(header)
    print("-" * 80)
    
    for name, (scheduler_fn, params) in schedulers.items():
        values = [scheduler_fn(s, **params) for s in key_steps]
        row = f"{name:<30}" + "".join([f"{v:>8.3f}" for v in values])
        print(row)


if __name__ == "__main__":
    visualize_schedulers()
