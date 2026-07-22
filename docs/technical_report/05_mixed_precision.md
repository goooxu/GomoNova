# 第五章：混合精度训练

## 本章你将学到

- FP32、FP16、BF16 的区别
- 为什么选择 BF16
- BatchNorm 为什么必须保持 FP32
- `torch.amp.autocast` 的工作原理

## 5.1 数字精度基础

计算机用二进制存储小数。不同格式用不同位数：

| 格式 | 总位数 | 符号 | 指数 | 尾数 | 范围 | 精度 |
|------|--------|------|------|------|------|------|
| FP32 | 32 | 1 | 8 | 23 | ±3.4×10³⁸ | ~7位有效数字 |
| FP16 | 16 | 1 | 5 | 10 | ±65504 | ~3位有效数字 |
| BF16 | 16 | 1 | 8 | 7 | ±3.4×10³⁸ | ~2位有效数字 |

**关键区别：**
- FP16：精度高但范围小（容易溢出/下溢）
- BF16：范围大但精度低（和 FP32 相同的指数范围）

## 5.2 为什么选 BF16

| | FP16 | BF16 |
|--|------|------|
| 需要 GradScaler？ | 是（防下溢） | **不需要** |
| 训练稳定性 | 需要小心调参 | 开箱即用 |
| 速度 | 快 | 快（相同） |
| 显存 | 省一半 | 省一半 |

BF16 的指数范围和 FP32 一样大，所以梯度不会突然变成 0（下溢）或无穷大（溢出）。不需要 GradScaler 这个"保险机制"，代码更简洁。

## 5.3 BatchNorm 的特殊性

PyTorch 的 `batch_norm` CUDA 内核**强制要求**权重和 running stats 为 FP32：

```
RuntimeError: Expected weight to have type Float but got BFloat16
```

原因：BN 需要计算均值和方差，这些统计量对精度非常敏感。BF16 只有 7 位尾数，累积误差会导致训练不稳定。

**解决方案：** 模型整体转 BF16，然后把 BN 层转回 FP32：

```python
# pipeline.py 中的初始化
network = GomoNovaNet(...).bfloat16().to(device)  # 全部转 BF16

for m in network.modules():
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.float()  # BN 转回 FP32
```

## 5.4 autocast 的工作原理

`torch.amp.autocast` 自动管理精度转换：

```python
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    logits, value = network(x)  # 内部自动处理精度
    loss = compute_loss(logits, value, ...)
```

在 autocast 上下文中：
- **Conv2d、Linear** → 自动用 BF16 计算（快）
- **BatchNorm** → 自动用 FP32 计算（精确）
- **Softmax、Log** → 自动用 FP32（数值稳定）
- **损失计算** → 自动用 FP32（避免精度丢失）

你不需要手动转换每个操作的精度，autocast 根据操作类型自动选择。

## 5.5 实际代码中的精度分布

```
模型权重:
  Conv2d.weight      → BF16
  Linear.weight      → BF16
  BatchNorm.weight   → FP32  ← 特殊处理
  BatchNorm.running_mean → FP32
  pos_enc (位置编码)  → BF16

训练时:
  输入 x             → FP32 → autocast 自动转 BF16
  前向计算           → BF16（Conv/Linear）+ FP32（BN）
  损失计算           → FP32
  梯度               → BF16
  参数更新           → BF16（AdamW 直接更新 BF16 权重）

推理时:
  输入 x             → FP32 → autocast 转 BF16
  输出 policy        → BF16 → .float() 转 FP32 → numpy
```

## 5.6 Checkpoint 大小对比

| 精度 | 模型权重大小 | 加上优化器状态 |
|------|-------------|---------------|
| FP32 | ~50MB | ~145MB |
| BF16 | ~25MB | ~73MB |

BF16 的 checkpoint 只有 FP32 的一半大小，备份和传输更快。

## 5.7 动手实验

```python
import torch
from gomonova.nn.network import GomoNovaNet

# 观察不同层的精度
net = GomoNovaNet(channels=96, num_blocks=8,
                  policy_channels=48, value_channels=24)
net = net.bfloat16()  # 全部转 BF16
for m in net.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        m.float()  # BN 转回 FP32

# 检查各层精度
for name, param in net.named_parameters():
    if 'bn' in name or 'batch' in name.lower():
        assert param.dtype == torch.float32, f"{name} should be FP32"
    else:
        assert param.dtype == torch.bfloat16, f"{name} should be BF16"

print("精度检查通过！")

# 验证前向传播正常
x = torch.randn(1, 8, 15, 15).cuda()
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    policy, value = net.cuda()(x)
print(f"Policy dtype: {policy.dtype}")  # bfloat16
print(f"Value dtype: {value.dtype}")    # bfloat16
```
