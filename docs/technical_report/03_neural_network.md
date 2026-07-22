# 第三章：神经网络架构

## 本章你将学到

- 如何把棋盘编码为神经网络输入
- MSAR-Net 的设计动机和结构
- 多尺度卷积为什么适合五子棋
- SE 注意力的作用
- 策略头和价值头的设计

## 3.1 输入编码

> 源文件：`gomonova/nn/encoder.py`

神经网络不能直接理解"棋盘"，需要把棋盘转换为数字张量。GomoNova 用 **8 个 15×15 的二值平面**（类似 8 张黑白图片叠在一起）：

| 通道 | 内容 | 示例 |
|------|------|------|
| 0 | 当前方的棋子 | 黑棋下时=所有黑子位置为1 |
| 1 | 对方的棋子 | 黑棋下时=所有白子位置为1 |
| 2 | 当前方最后一手 | 只有一个位置为1 |
| 3 | 当前方倒数第二手 | 只有一个位置为1 |
| 4 | 对方最后一手 | 只有一个位置为1 |
| 5 | 对方倒数第二手 | 只有一个位置为1 |
| 6 | 所有已占位置 | 有子的地方为1 |
| 7 | 当前方是黑棋？ | 黑棋下时全1，白棋下时全0 |

```python
def board_to_planes(board):
    planes = np.zeros((8, 15, 15), dtype=np.float32)
    me, opp = board.current, 3 - board.current

    planes[0] = (board.cells == me)       # 我的子
    planes[1] = (board.cells == opp)      # 对手的子
    planes[6] = (board.cells != 0)        # 所有子
    planes[7] = 1.0 if me == BLACK else 0.0  # 谁在下

    # 最近两手的位置标记
    for i, pos in enumerate(board.last_moves_for(me, 2)):
        r, c = divmod(pos, 15)
        planes[2 + i, r, c] = 1.0
    for i, pos in enumerate(board.last_moves_for(opp, 2)):
        r, c = divmod(pos, 15)
        planes[4 + i, r, c] = 1.0

    return planes
```

**为什么用"颜色相对"编码？** 通道 0 永远是"当前方"，通道 1 永远是"对方"。这样同一个网络可以执黑也可以执白，不需要两套权重。

**为什么标记最近两手？** 让网络知道"刚才发生了什么"，有助于判断威胁和意图。

## 3.2 卷积神经网络基础回顾

在看 MSAR-Net 之前，快速回顾三个核心组件：

### Conv2d（卷积层）

```python
nn.Conv2d(in_channels, out_channels, kernel_size, padding=...)
```

用一个 kernel_size × kernel_size 的小窗口在输入上滑动，每个位置产生一个输出值。kernel_size 越大，"看到"的范围越大（感受野越大）。

### BatchNorm2d（批归一化）

```python
nn.BatchNorm2d(channels)
```

把每个通道的值归一化到均值0、方差1。作用：稳定训练、加速收敛。

### Mish（激活函数）

```python
F.mish(x)  # = x * tanh(softplus(x))
```

比 ReLU 更平滑，允许小的负值通过，训练更稳定。

## 3.3 MSAR-Block：多尺度注意力残差块

> 源文件：`gomonova/nn/blocks.py`

### 设计动机

五子棋中的威胁跨越不同的空间尺度：

| 尺度 | 棋形 | 需要的感受野 |
|------|------|-------------|
| 局部 | 活三、冲四 | 3×3（3格连线） |
| 中程 | 跳三、断连 | 5×5（间隔一格的三） |
| 远程 | 做杀（两个威胁的组合） | 7×7（两个威胁的空间关系） |

传统 AlphaZero 架构只用 3×3 卷积，需要堆很多层才能"看到"远程关系。MSAR-Net 在**每一层**同时处理三个尺度。

### 结构

```
输入 x (C × 15 × 15)
│
├─ Branch-3×3: BN→Mish→Conv(3×3)→BN→Mish→Conv(3×3)  → 局部战术
├─ Branch-5×5: BN→Mish→Conv(5×5)→BN→Mish→Conv(5×5)  → 中程棋形
├─ Branch-7×7: BN→Mish→Conv(7×7)→BN→Mish→Conv(7×7)  → 远程结构
│
└─ 融合:
     Concat(3个分支) → Conv(1×1, 3C→C) → BN
     → SE注意力（通道加权）
     → + x（残差连接）
```

```python
class MSARBlock(nn.Module):
    def __init__(self, channels, se_reduction=4):
        self.branch3 = _ConvBranch(channels, 3)  # 3×3 路径
        self.branch5 = _ConvBranch(channels, 5)  # 5×5 路径
        self.branch7 = _ConvBranch(channels, 7)  # 7×7 路径
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),  # 3C → C
            nn.BatchNorm2d(channels),
        )
        self.se = SEBlock(channels, se_reduction)

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        out = torch.cat([b3, b5, b7], dim=1)  # 拼接三个尺度
        out = self.fuse(out)                    # 融合回 C 通道
        out = self.se(out)                      # 通道注意力
        return F.mish(out + x)                  # 残差 + 激活
```

**残差连接（`out + x`）：** 让梯度可以直接流过，避免深层网络的梯度消失。

## 3.4 SE 注意力

> Squeeze-and-Excitation：让网络自己学习"哪些通道重要"。

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        mid = channels // reduction
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),    # 压缩
            nn.ReLU(),
            nn.Linear(mid, channels),    # 恢复
            nn.Sigmoid(),                # 输出 0~1 的权重
        )

    def forward(self, x):
        # Squeeze: 全局平均池化，把 15×15 压缩成 1 个数
        w = x.mean(dim=(2, 3))       # [B, C]
        # Excitation: 学习每个通道的重要性
        w = self.fc(w)               # [B, C]，值在 0~1
        # Scale: 用权重缩放原始特征
        return x * w[:, :, None, None]
```

> 类比：SE 就像一个"调音台"，自动调节每个通道（每种特征）的音量。面对活三时，3×3 通道的"音量"会被调大；面对远程做杀时，7×7 通道会被调大。

## 3.5 完整网络

> 源文件：`gomonova/nn/network.py`

```python
class GomoNovaNet(nn.Module):
    def __init__(self, channels=96, num_blocks=8, ...):
        # 1. Stem: 把 8 通道输入扩展到 C 通道
        self.stem = Conv2d(8, C, k=3) + BN + Mish
        self.pos_enc = nn.Parameter(...)  # 可学习位置编码

        # 2. Tower: 8 个 MSAR 块
        self.tower = [MSARBlock(C) for _ in range(8)]

        # 3. Policy Head: 输出 225 个位置的得分
        # 4. Value Head: 输出局面评估 [-1, 1]

    def forward(self, x):
        h = self.stem(x) + self.pos_enc   # 特征提取 + 位置信息
        h = self.tower(h)                  # 8 层多尺度处理

        policy_logits = self.policy_head(h)  # [B, 225]
        value = self.value_head(h)           # [B, 1]
        return policy_logits, value
```

### 策略头：局部 + 全局门控

```python
# 局部分支：每个位置独立打分
local_logits = Conv1x1(h) → flatten  # 225 个分数

# 全局分支：理解整体局势后给每个位置加偏置
gap = GlobalAvgPool(h)               # 压缩成 1 个向量
global_bias = Linear(gap) → 225      # 全局战略偏置

# 门控：学习"多大程度上听全局的"
gate = sigmoid(Linear(gap))          # 0~1 的标量

policy = local_logits + gate * global_bias
```

**为什么需要全局门控？** 因为没有搜索（MCTS），网络必须自己做"战略规划"。局部分支看局部棋形，全局分支决定"往哪个方向发展"。门控让网络自己平衡两者。

### 价值头

```python
value = Conv1x1(h) → GlobalAvgPool → Linear(C→128) → Mish → Linear(128→1) → tanh
```

输出 [-1, 1]：+1 = 当前方必胜，-1 = 当前方必败，0 = 均势。

## 3.6 参数量

当前配置（C=96, N=8）：

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Stem | 7,104 | 0.06% |
| 位置编码 | 21,600 | 0.17% |
| Tower (8×MSAR) | 12,508,608 | 99.4% |
| Policy Head | 40,195 | 0.32% |
| Value Head | 5,825 | 0.05% |
| **总计** | **12,583,332** | 100% |

每个 MSAR 块中，7×7 分支最大（903K 参数），3×3 最小（166K）。

## 3.7 动手实验

```python
import torch
from gomonova.nn.network import GomoNovaNet

# 创建网络
net = GomoNovaNet(channels=96, num_blocks=8, policy_channels=48, value_channels=24)
print(f"参数量: {net.num_params():,}")

# 模拟一次前向传播
x = torch.randn(1, 8, 15, 15)  # 假装输入
policy, value = net(x)
print(f"策略输出: {policy.shape}")   # [1, 225]
print(f"价值输出: {value.shape}")    # [1, 1]
print(f"价值范围: [{value.min():.3f}, {value.max():.3f}]")  # tanh → [-1, 1]
```
