# 第四章：训练管线

## 本章你将学到

- 自博弈如何生成训练数据
- 经验回放缓冲的作用
- 为什么 REINFORCE 失败了，结果加权 CE 如何解决问题
- 学习率调度和评估晋升机制

## 4.1 自博弈：自己和自己下棋

> 源文件：`gomonova/training/selfplay.py`

训练数据从哪来？没有人类棋谱，就让网络自己和自己下。

### 批量并行对弈

关键优化：不是下完一局再下一局，而是 **512 局同时下**（lockstep）：

```python
def play_games_fast(network, device, num_games=512, ...):
    boards = [Board() for _ in range(num_games)]  # 512 个棋盘

    while 还有活跃棋局:
        # 把所有活跃棋盘叠成一个 batch
        planes = np.stack([board_to_planes(b) for b in active_boards])
        x = torch.from_numpy(planes).to(device)  # [N, 8, 15, 15]

        # 一次 GPU 前向推理，得到所有局面的策略
        logits, _ = network(x)
        policies = softmax(logits)  # [N, 225]

        # 每局根据策略选一步棋
        for i, board in enumerate(active_boards):
            move = sample(policies[i], temperature)
            board.play(move)
```

**为什么快？** 512 个局面一次 GPU 调用就处理完。GPU 擅长并行，batch=512 和 batch=1 耗时几乎相同（~12ms）。

### 温度采样

```python
temperature = 1.0 if move_num < 30 else 0.1

if temperature < 0.01:
    move = argmax(policy)          # 贪心：选概率最高的
else:
    tempered = policy ** (1/T)     # 温度越高，分布越平坦
    tempered /= tempered.sum()
    move = random.choice(225, p=tempered)  # 按概率随机选
```

- 前 30 手 T=1.0：多探索，尝试不同走法
- 30 手后 T=0.1：少探索，下最优走法

> 类比：开局像"随便试试"，中后盘像"认真下"。

## 4.2 经验回放

> 源文件：`gomonova/training/replay.py`

> 类比：经验回放 = 棋手的复盘笔记本。每下完一局，把所有局面记下来。训练时随机翻看笔记复习。

```python
class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.planes = []   # 棋盘状态
        self.moves = []    # 实际下的位置
        self.outcomes = [] # 最终结果 (+1赢/-1输/0和)

    def add(self, planes, move, outcome):
        if len < capacity:
            追加
        else:
            覆盖最旧的  # 环形缓冲

    def sample(self, batch_size):
        indices = random.choice(len, batch_size)
        return planes[indices], moves[indices], outcomes[indices]
```

**为什么需要它？**
1. 打破数据相关性：连续局面高度相关，随机打乱后训练更稳定
2. 重复利用：一局棋的数据可以被多次学习
3. 容量限制：只保留最近 100 万个局面，避免过时的数据

## 4.3 损失函数

> 源文件：`gomonova/nn/losses.py`

### 失败的尝试：REINFORCE

最初用策略梯度（REINFORCE）：

```
loss = -log π(实际走的步) × (对局结果 - 基线)
```

**问题：** 方差太大。赢了就疯狂强化，输了就疯狂压制。损失从 -3 爆炸到 -289，训练完全不稳定。

### 当前方案：结果加权交叉熵

核心思想：**只从胜利中学习**。

```python
def policy_weighted_ce(logits, moves, outcomes, entropy_weight=0.005):
    log_probs = log_softmax(logits)
    selected = log_probs[moves]  # 实际走法的 log 概率

    # 关键：根据结果加权
    weights = where(outcomes > 0.5, 1.0,    # 赢了：全力学习
            where(outcomes < -0.5, 0.1,     # 输了：轻微惩罚
                                       0.3)) # 和棋：适度学习

    ce_loss = -(selected * weights).mean()

    # 熵正则：防止策略崩溃（只会下一种棋）
    entropy = -(probs * log_probs).sum(dim=1).mean()
    loss = ce_loss - entropy_weight * entropy

    return loss, entropy
```

**为什么稳定？** 交叉熵是有界的（不像 REINFORCE 可以无限大），加权只是调节学习强度。

### 价值损失

```python
value_loss = MSE(网络预测的价值, 实际对局结果)
```

让网络学会判断"这个局面我大概能赢还是输"。

### 总损失

```python
total = policy_loss + value_loss + 1e-4 * L2_regularization
```

## 4.4 训练器

> 源文件：`gomonova/training/trainer.py`

```python
class Trainer:
    def train_step(self, replay, batch_size):
        # 1. 从回放池抽样
        planes, moves, outcomes = replay.sample(batch_size)

        # 2. 前向传播（BF16 混合精度）
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, value = network(planes)
            loss = total_loss(logits, value, moves, outcomes, network)

        # 3. 反向传播 + 更新
        loss.backward()
        clip_grad_norm_(network.parameters(), max_norm=1.0)  # 防梯度爆炸
        optimizer.step()
```

### 学习率调度

```
轮次:  0 ─── 20 ──────────────────────── 3000
LR:    0 → 3e-4（线性升温）→ 1e-5（余弦衰减）
```

- 前 20 轮：学习率从 0 线性增到 3e-4（避免初期大梯度破坏随机权重）
- 之后：余弦曲线缓慢降到 1e-5（后期精细调整）

## 4.5 评估与晋升

> 源文件：`gomonova/training/evaluator.py`

每轮训练后，新模型 vs 最佳模型对弈 50 局：

```python
def play_match(net_a, net_b, num_games=50):
    # 50 局同时下（批量并行）
    # 偶数局：A 执黑先手
    # 奇数局：B 执黑先手
    # 返回 A 的胜率
```

**晋升条件：** 胜率 ≥ 55% → 新模型替换最佳模型，保存 checkpoint。

**为什么是 55% 而不是 50%？** 需要一定优势才替换，避免噪声导致的虚假晋升。

## 4.6 主循环

> 源文件：`gomonova/training/pipeline.py`

```python
for iteration in range(3000):
    # ① 自博弈：生成 512 局新数据
    samples = generate_games(network, num_games=512)
    replay.add_batch(samples)

    # ② 训练：128 步梯度更新
    metrics = trainer.train_epoch(replay, batch_size=1024, steps=128)

    # ③ 评估：新模型 vs 最佳模型
    result = play_match(network, best_network, num_games=50)

    # ④ 晋升
    if result["winrate"] >= 0.55:
        best_network = copy(network)
        save_checkpoint("best.pt", network)
```

## 4.7 数据增强：D4 对称

每个训练样本被随机旋转/翻转后额外生成 3 个副本：

```python
for planes, move, outcome in game_samples:
    buffer.add(planes, move, outcome)  # 原始
    for _ in range(3):
        t = random_transform()  # 8 种对称中随机选一种
        buffer.add(transform(planes, t), transform(move, t), outcome)
```

效果：训练数据量 ×4，且网络学会"旋转后本质相同"。

## 4.8 动手实验

```python
# 观察训练过程中损失的变化
# 运行训练并观察输出
# python scripts/train.py --config configs/train_smoke.yaml

# 或手动检查 replay buffer 的工作方式
from gomonova.training.replay import ReplayBuffer
import numpy as np

buf = ReplayBuffer(capacity=100)
for i in range(200):  # 加入 200 个样本（超过容量）
    buf.add(np.zeros((8,15,15)), i % 225, 1.0 if i%2==0 else -1.0)

print(f"Buffer 大小: {len(buf)}")  # 100（环形覆盖）
planes, moves, outcomes = buf.sample(32)
print(f"采样: planes={planes.shape}, moves={moves.shape}")
```
