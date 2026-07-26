# 第三章：神经网络架构

## 本章你将学到

- 如何把棋盘编码为 16 通道的神经网络输入
- MSAR-Net 的设计动机和结构
- 多尺度卷积为什么适合五子棋
- SE 通道注意力的作用
- 策略头的"逐位置门控"设计
- 价值头如何评估局面优劣

> 涉及源文件：`gomonova/nn/encoder.py`、`gomonova/nn/blocks.py`、`gomonova/nn/network.py`

GomoNova 的网络叫 **MSAR-Net**（Multi-Scale Attentive Residual Network，多尺度注意力残差网络）。它是原创设计，但借鉴了卷积网络的通用最佳实践。本章会先回顾 PyTorch 基础组件，再逐层拆解这个网络。

## 3.1 输入编码：棋盘 → 16 通道张量

> 源文件：`gomonova/nn/encoder.py`

神经网络不能直接理解"棋盘"，需要把它转换为数字张量。GomoNova 用 **16 个 15×15 的二值平面**（想象成 16 张黑白图片叠在一起）：

| 通道 | 内容 |
|------|------|
| 0 | 当前方的所有棋子 |
| 1 | 对方的所有棋子 |
| 2–7 | 当前方最近 6 手的位置（每手一个平面） |
| 8–13 | 对方最近 6 手的位置 |
| 14 | 所有已占位置（占位图） |
| 15 | 回合偏置（黑棋下时全 1，白棋下时全 0） |

```python
NUM_HISTORY = 6
INPUT_CHANNELS = 2 + 2 * NUM_HISTORY + 2   # = 16

def board_to_planes(board):
    planes = np.zeros((16, 15, 15), dtype=np.float32)
    current, opponent = board.current, 3 - board.current

    planes[0] = (board.cells == current)      # 我的子
    planes[1] = (board.cells == opponent)     # 对手的子

    # 最近 6 手的位置标记
    for i, pos in enumerate(board.last_moves_for(current, 6)):
        r, c = divmod(pos, 15); planes[2 + i, r, c] = 1.0
    for i, pos in enumerate(board.last_moves_for(opponent, 6)):
        r, c = divmod(pos, 15); planes[8 + i, r, c] = 1.0

    planes[14] = (board.cells != 0)           # 占位图
    if current == BLACK: planes[15] = 1.0     # 回合偏置
    return planes
```

**三个设计要点：**

1. **颜色相对编码。** 通道 0 永远是"当前方"，通道 1 永远是"对方"。这样同一套网络权重既能执黑也能执白，无需两套模型。

2. **6 手历史。** 标记最近 6 手的位置，让网络知道"刚才发生了什么"。这对判断威胁和意图至关重要——比如"对手上一手下在这里，是不是在做活三？"早期版本只用 2 手历史，增加到 6 手后棋力明显提升。

3. **回合偏置。** 通道 15 告诉网络"现在谁在下"。虽然颜色相对编码已经隐含了这个信息，但显式的回合偏置能帮助网络区分"该我进攻"还是"该我防守"。

## 3.2 卷积网络基础回顾

在看 MSAR-Net 之前，快速回顾三个核心组件（PyTorch 初学者重点）：

### Conv2d（卷积层）

```python
nn.Conv2d(in_channels, out_channels, kernel_size, padding=...)
```

用一个 `kernel_size × kernel_size` 的小窗口在输入上滑动，每个位置产生一个输出值。`kernel_size` 越大，单次"看到"的范围（感受野）越大。`padding=kernel_size//2` 保证输出尺寸不变。

### BatchNorm2d（批归一化）

```python
nn.BatchNorm2d(channels)
```

把每个通道的激活值归一化到均值 0、方差 1（再用可学习参数缩放平移）。作用：稳定训练、允许更大学习率、加速收敛。

### Mish（激活函数）

```python
F.mish(x)   # = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
```

比 ReLU 更平滑，允许小的负值通过，梯度更稳定。GomoNova 全程用 Mish 而非 ReLU。

## 3.3 MSAR-Block：多尺度注意力残差块

> 源文件：`gomonova/nn/blocks.py`

### 设计动机

五子棋中的威胁跨越不同的空间尺度：

| 尺度 | 棋形 | 需要的感受野 |
|------|------|-------------|
| 局部 | 活三、冲四 | 3×3（紧邻的三/四） |
| 中程 | 跳三、断连 | 5×5（间隔一格的三） |
| 远程 | 做杀（双威胁组合） | 7×7（两个威胁的空间关系） |

经典的 AlphaZero 架构只用 3×3 卷积，需要堆叠很多层才能"看到"远程关系。MSAR-Net 的思路是：**在每一层同时处理三个尺度**，让局部、中程、远程信息在每一层都充分交互。

### 结构

```
输入 x (C × 15 × 15)
│
├─ Branch-3×3: BN→Mish→Conv(3×3)→BN→Mish→Conv(3×3)  → 局部战术
├─ Branch-5×5: BN→Mish→Conv(5×5)→BN→Mish→Conv(5×5)  → 中程棋形
├─ Branch-7×7: BN→Mish→Conv(7×7)→BN→Mish→Conv(7×7)  → 远程结构
│
└─ 融合:
     Concat(3 个分支, 3C 通道) → Conv(1×1, 3C→C) → BN
     → SE 注意力（通道加权）
     → + x（残差连接）→ Mish
```

```python
class MSARBlock(nn.Module):
    def __init__(self, channels, se_reduction=4):
        self.branch3 = _ConvBranch(channels, 3)
        self.branch5 = _ConvBranch(channels, 5)
        self.branch7 = _ConvBranch(channels, 7)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),  # 3C → C
            nn.BatchNorm2d(channels),
        )
        self.se = SEBlock(channels, se_reduction)

    def forward(self, x):
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        out = torch.cat([b3, b5, b7], dim=1)   # 拼接三个尺度 → 3C 通道
        out = self.fuse(out)                    # 1×1 卷积融合回 C 通道
        out = self.se(out)                      # 通道注意力
        return F.mish(out + x)                  # 残差 + 激活
```

每个 `_ConvBranch` 是"BN→Mish→Conv→BN→Mish→Conv"的预激活结构（pre-activation），两层卷积。**残差连接（`out + x`）**让梯度可以直接流过，避免深层网络的梯度消失——这是能堆 10 层的关键。

## 3.4 SE 注意力

> Squeeze-and-Excitation：让网络自己学习"哪些通道重要"。

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),     # 压缩
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),     # 恢复
            nn.Sigmoid(),                 # 输出 0~1 的权重
        )

    def forward(self, x):
        w = x.mean(dim=(2, 3))            # Squeeze: 全局平均池化 → [B, C]
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)  # Excitation: 学习通道权重
        return x * w                       # Scale: 按权重缩放每个通道
```

> **类比：** SE 就像一个自动"调音台"。每个通道代表一种特征（某种棋形检测器），SE 学会根据当前局面调节每个通道的"音量"。面对活三时，3×3 局部通道的音量被调大；面对远程做杀时，7×7 远程通道被调大。

## 3.5 完整网络

> 源文件：`gomonova/nn/network.py`

```python
class GomoNovaNet(nn.Module):
    def __init__(self, channels=192, num_blocks=10,
                 policy_channels=96, value_channels=48, se_reduction=4):
        # 1. Stem: 16 通道输入 → C 通道
        self.stem = Conv2d(16, C, k=3) + BN + Mish
        self.pos_enc = nn.Parameter(...)      # 可学习位置编码

        # 2. Tower: 10 个 MSAR 块
        self.tower = [MSARBlock(C) for _ in range(10)]

        # 3. Policy Head: 输出 225 个位置的得分
        # 4. Value Head: 输出局面评估 [-1, 1]

    def forward(self, x):
        h = self.stem(x) + self.pos_enc       # 特征提取 + 位置信息
        h = self.tower(h)                      # 10 层多尺度处理
        policy_logits = ...(h)                 # [B, 225]
        value = ...(h)                         # [B, 1]
        return policy_logits, value
```

**可学习位置编码 `pos_enc`：** 一个形状为 `(C, 15, 15)` 的可学习参数，加在 Stem 输出上。卷积本身是"平移等变"的（不知道绝对位置），位置编码让网络能区分"天元"和"边角"——这对五子棋很重要（中心价值高于边角）。

## 3.6 策略头：局部 + 逐位置全局门控

策略头是 GomoNova 最有特色的部分。它输出 225 个位置的"落子得分"（logits），由两部分融合：

```python
# 局部分支：每个位置独立打分
local_logits = self.policy_local(p).flatten(1)    # Conv1x1 → [B, 225]

# 全局分支：理解整体局势后，给每个位置一个战略偏置
gap = p.mean(dim=(2, 3))                          # 全局平均池化 → [B, pc]
global_bias = self.policy_global(gap)             # MLP → [B, 225]

# 逐位置门控：每个位置独立决定"多大程度上听全局的"
gate = torch.sigmoid(self.policy_gate(p).flatten(1))   # Conv1x1 → [B, 225]

policy_logits = local_logits + gate * global_bias
```

**为什么需要门控？** 因为对战时不做搜索（第 04 章详述），网络必须自己完成"战略规划"。局部分支看局部棋形（这里能不能成四），全局分支决定战略方向（往哪个区域发展）。门控让网络**逐位置**地平衡两者：

- 在战术激烈的局部（已有棋子纠缠），门控关小，主要听局部的
- 在空旷需要选大场的区域，门控开大，多听全局战略

> **一个重要的演进：** 早期版本用的是**标量门控**（整盘棋共用一个 gate 值），表达力有限——它只能统一地"全局多一点"或"局部多一点"。改成**逐位置门控**（每个位置一个 gate）后，网络能在同一盘棋里对不同区域采取不同策略，棋力有明显提升。

## 3.7 价值头

```python
value = Conv1x1(h) → GlobalAvgPool → Linear(C→256) → Mish → Linear(256→1) → tanh
```

输出 `[-1, 1]` 的标量：**从当前方视角**的局面评估。

- `+1` = 当前方必胜
- `-1` = 当前方必败
- `0` = 均势

价值头让网络不仅能"选招"，还能"判断形势"。这个判断有两个用途：① 训练时作为监督信号（最终胜负是 ground truth）；② MCTS 搜索时评估 leaf 节点（第 04 章）。

## 3.8 参数量

当前配置（C=192, N=10, policy_channels=96, value_channels=48）：

| 组件 | 说明 |
|------|------|
| Stem | 16→192 卷积 + 位置编码 |
| Tower | 10 × MSAR 块（参数主体） |
| Policy Head | 局部 + 全局门控 |
| Value Head | 全局评估 |
| **总计** | **62,709,060（约 62.7M）** |

每个 MSAR 块中，7×7 分支参数最多（卷积核大），3×3 最少。Tower 占了绝大部分参数。

> 为什么是 62.7M 而不是更大或更小？这是在"棋力"和"训练速度"之间的权衡。更大的模型棋力更强但训练更慢；更小的模型训练快但棋力不足。62.7M 在 4× GPU 上约 1 天能训完 3000 轮，且棋力达到预期。详见第 05、06 章。

## 3.9 动手实验

```python
import torch
from gomonova.nn.network import GomoNovaNet
from gomonova.nn.encoder import INPUT_CHANNELS

net = GomoNovaNet()   # 默认配置：192 通道 / 10 块
print(f"输入通道数: {INPUT_CHANNELS}")        # 16
print(f"参数量: {net.num_params():,}")        # 62,709,060

# 模拟一次前向传播
x = torch.randn(2, INPUT_CHANNELS, 15, 15)    # batch=2 的假输入
policy, value = net(x)
print(f"策略输出: {policy.shape}")            # [2, 225]
print(f"价值输出: {value.shape}")             # [2, 1]
print(f"价值范围: [{value.min():.3f}, {value.max():.3f}]")  # tanh → [-1, 1]

# 把策略 logits 转成概率
probs = torch.softmax(policy[0], dim=0)
print(f"概率之和: {probs.sum():.4f}")         # 1.0
print(f"最可能的位置: {probs.argmax().item()}")
```

**下一步：** 第 04 章是全书核心——我们会从零讲清自博弈和 MCTS 的原理，理解"为什么训练时搜索、对战时不搜索"这个关键设计。
