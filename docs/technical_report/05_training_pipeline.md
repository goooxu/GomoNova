# 第五章：训练管线

## 本章你将学到

- 一个训练样本长什么样
- 三相训练的设计：纯策略热身 → MCTS 自由规则 → MCTS 连珠规则
- 三种损失函数：结果加权交叉熵、KL 散度、价值 MSE
- 经验回放缓冲的预分配数组实现
- 优化器与学习率调度（warmup + 余弦退火）
- 评估晋升机制与对手多样性
- **大量真实的调参经验与取舍**

> 涉及源文件：`gomonova/training/pipeline.py`、`trainer.py`、`selfplay.py`、`replay.py`、`evaluator.py`、`gomonova/nn/losses.py`

第 04 章讲了数据怎么来（自博弈 + MCTS），本章讲数据怎么用（训练网络）。

## 5.1 一个训练样本长什么样

自博弈产生的每一手棋，都被整理成一个训练样本：

```python
(planes, move, outcome, mcts_policy)
```

| 字段 | 含义 | 形状 |
|------|------|------|
| `planes` | 落子前的棋盘编码 | `(16, 15, 15)` |
| `move` | 实际落下的位置 | 标量 (0–224) |
| `outcome` | 这手棋的最终结果（从落子方视角） | +1 赢 / -1 输 / 0 和 |
| `mcts_policy` | MCTS 访问分布（仅 MCTS 阶段有，否则为 None） | `(225,)` |

**outcome 的视角转换**很关键：一局棋黑胜（result=+1），那么黑棋的每手 outcome=+1，白棋的每手 outcome=-1（同一局棋，对赢家是好棋，对输家是坏棋）。

```python
def records_to_samples(records, augment=4):
    for rec in records:
        result = rec["result"]
        for planes, mcts_pol, move, player in rec["history"]:
            outcome = result if player == BLACK else -result   # 视角转换
            samples.append((planes, move, outcome, mcts_pol))
            # 对称增强：随机做一种对称变换，生成 augment-1 个额外样本
            for _ in range(augment - 1):
                t = random_transform()
                samples.append((transform(planes, t), transform(move, t), outcome, transform(mcts_pol, t)))
```

**对称增强（augment=4）**把每个样本扩充到 4 倍（原图 + 3 个随机对称变换），既增加数据量，又教会网络"棋形与方向无关"（第 02 章的 D4 变换）。

## 5.2 三相训练

GomoNova 的训练分为三个阶段，由配置中的 `mcts_start` 和 `renju_start` 控制：

```python
use_mcts  = iteration >= mcts_start      # 300
use_renju = iteration >= renju_start     # 2400
```

| 阶段 | 迭代范围 | MCTS | 规则 | 目的 |
|------|----------|------|------|------|
| ① 热身 | 0–300 | ❌ 纯策略 | 自由 | 快速建立基本棋感 |
| ② 主力 | 300–2400 | ✅ 25 次模拟 | 自由 | MCTS 引导学习战略 |
| ③ 禁手 | 2400–3000 | ✅ 25 次模拟 | 连珠 | 学会避免/利用禁手 |

**为什么要分相？**

- **阶段①（纯策略热身）：** 一开始网络完全随机，MCTS 用随机网络做搜索也搜不出什么好东西。不如先用纯策略快速对弈，让网络先建立"基本的棋感"（知道连子、知道堵），再引入 MCTS。这就像先学会基本规则，再学高级战术。

- **阶段②（MCTS + 自由规则）：** 网络有基本棋感后，MCTS 才能发挥"深思"的价值。这一阶段用**自由规则**（不判禁手），因为禁手检测很慢（第 02 章），会大幅拖慢训练。网络先专心学好"通用棋力"。

- **阶段③（MCTS + 连珠规则）：** 最后 600 轮切换到**连珠规则**，让 MCTS 在搜索时考虑禁手。网络由此学会"作为黑棋要避开禁手"和"作为白棋要逼对手走禁手"。因为对战时模型若下禁手直接判负（第 01 章），这一步必不可少。

> **调参经验：** `mcts_start=300` 是试出来的。太早（如 0）MCTS 质量差且拖慢训练；太晚（如 1000）网络在纯策略阶段养成坏习惯后难纠正。300 轮热身足够建立棋感，又不至于浪费太多时间。`renju_start=2400` 留 600 轮学禁手，实测足够——禁手本质是"某些位置不能下"，比学进攻防守简单。

## 5.3 损失函数

> 源文件：`gomonova/nn/losses.py`

总损失由三部分加权组成：

```python
loss = policy_loss + value_loss + l2_weight * l2
```

### ① 策略损失：两种来源

策略损失根据样本类型用不同公式：

**MCTS 样本 → KL 散度。** 让网络策略逼近 MCTS 访问分布：

```python
def policy_kl_divergence(logits, mcts_policy):
    log_probs = F.log_softmax(logits, dim=1)
    target = mcts_policy.clamp(min=1e-8)
    return (target * (target.log() - log_probs)).sum(dim=1).mean()
```

> **KL 散度**衡量两个概率分布的"差异"。这里让网络输出的分布尽量接近 MCTS 的访问分布——MCTS 认为该下哪（分布形状），网络就学着输出哪。

**纯策略样本 → 结果加权交叉熵。** 对没有 MCTS 信息的样本（阶段①和阶段②的中后盘），用"模仿实际落子 + 按胜负加权"：

```python
def policy_weighted_ce(logits, moves, outcomes, entropy_weight=0.005):
    log_probs = F.log_softmax(logits, dim=1)
    selected = log_probs.gather(1, moves.unsqueeze(1)).squeeze(1)  # 实际落子的 log 概率
    # 赢的棋步权重 1.0，输的 0.1，和棋 0.3
    weights = where(outcomes > 0.5, 1.0, where(outcomes < -0.5, 0.1, 0.3))
    ce_loss = -(selected * weights).mean()
    entropy = -(probs * log_probs).sum(dim=1).mean()   # 熵
    return ce_loss - entropy_weight * entropy
```

- **结果加权**：赢棋的招法多模仿（权重 1.0），输棋的少模仿（0.1）。这就是"自模仿学习"——只学自己赢的经验。
- **熵奖励**：鼓励策略分布保持一定"多样性"（不要太快收敛到只下一两个点），防止策略崩溃。

`total_loss` 自动判断每个样本属于哪种：

```python
def total_loss(logits, value_pred, moves, outcomes, model, mcts_policy=None, ...):
    if mcts_policy is not None:
        has_mcts = mcts_policy.sum(dim=1) > 0.5    # 有 MCTS 分布的样本
        kl = policy_kl_divergence(logits[has_mcts], mcts_policy[has_mcts])
        ce = policy_weighted_ce(logits[~has_mcts], moves[~has_mcts], outcomes[~has_mcts])
        p_loss = kl + ce
    else:
        p_loss = policy_weighted_ce(logits, moves, outcomes)
    v_loss = value_loss(value_pred, outcomes)
    return p_loss + v_loss + l2_weight * l2, p_loss, v_loss
```

### ② 价值损失：MSE

价值头预测的局面评估，向"最终胜负"这个 ground truth 靠拢：

```python
def value_loss(pred, target):
    return F.mse_loss(pred.squeeze(-1), target)   # target = outcome (+1/-1/0)
```

### ③ L2 正则

对所有权重做 L2 惩罚（`l2_weight=1e-4`），防止过拟合。

> **调参经验：KL 切换时的 loss 跳升。** 从阶段①进入阶段②（开始用 MCTS + KL 损失）时，会观察到 loss 突然跳升（如从 5.2 跳到 6.5），然后几轮内恢复并继续下降。**这是正常现象，不是 bug**——因为损失函数的定义变了（CE 换成 KL），数值尺度本就不同。只要之后能恢复下降趋势就说明训练健康。

## 5.4 经验回放缓冲

> 源文件：`gomonova/training/replay.py`

自博弈产生的样本不是用完就扔，而是存入一个**回放缓冲池**（容量 200 万），训练时从中随机抽样。这样每个样本能被反复利用，且打乱了样本间的相关性。

```python
class ReplayBuffer:
    def __init__(self, capacity=500_000):
        # 预分配 numpy 数组，而非 Python 列表
        self.planes = np.zeros((capacity, 16, 15, 15), dtype=np.float32)
        self.mcts_policy = np.zeros((capacity, 225), dtype=np.float32)
        self.moves = np.zeros(capacity, dtype=np.int64)
        self.outcomes = np.zeros(capacity, dtype=np.float32)
        self._size = 0; self._idx = 0

    def add(self, planes, move, outcome, mcts_policy=None):
        # 环形写入，满了就覆盖最旧的
        ...

    def sample(self, batch_size):
        indices = np.random.choice(self._size, size=batch_size, replace=False)
        return self.planes[indices], self.moves[indices], self.outcomes[indices], self.mcts_policy[indices]
```

> **性能经验：预分配数组 vs Python 列表。** 早期版本用 Python 列表存样本（每个样本一个独立的 numpy 数组），200 万样本会产生 200 万个 Python 对象，内存碎片化严重、采样慢。改成预分配的大数组后，内存连续、采样是简单的索引操作，速度和内存占用都大幅改善。

## 5.5 优化器与学习率调度

> 源文件：`gomonova/training/trainer.py`

```python
self.optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
```

用 **AdamW**（带权重衰减解耦的 Adam），这是训练深度网络的标准选择。

**学习率调度：warmup + 余弦退火。**

```python
def get_lr(self, iteration):
    if iteration < self.warmup_iters:
        return self.lr * (iteration + 1) / self.warmup_iters    # 线性升温
    progress = (iteration - self.warmup_iters) / (self.total_iters - self.warmup_iters)
    return self.lr_min + 0.5 * (self.lr - self.lr_min) * (1 + cos(progress * π))  # 余弦退火
```

```
学习率
  ↑
lr ┤      ╱‾‾‾‾‾╲
   │     ╱        ╲
   │    ╱          ╲___
   │   ╱               ‾‾╲___
lr_min┤________________________‾‾‾
   └──┬────┬──────────────────┬──→ 迭代
     0  warmup            total
```

- **Warmup（前 50 轮）**：学习率从 0 线性升到 `lr`。训练初期权重随机、梯度方向不稳，小学习率避免一开始就"跑偏"。
- **余弦退火**：之后学习率按余弦曲线平滑降到 `lr_min`。后期小学习率精细调整，稳定收敛。

> **调参经验：** `lr=1e-4`、`lr_min=1e-6`、`warmup=50`。62.7M 的大模型用 1e-4 比较稳；早期小模型曾用 3e-4。学习率太大训练会震荡甚至发散，太小则收敛慢。**batch_size=2048** 配合 1e-4 是常见搭配（大 batch 通常可配大 lr，但这里保守起见没放大）。

## 5.6 混合精度训练（BF16）

```python
with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=self.use_amp):
    logits, value_pred = self.network(x)
    loss, ... = total_loss(...)
loss.backward()
```

`autocast` 让前向/反向中的矩阵运算自动用 **BF16**（16 位浮点）而非 FP32，速度更快、显存更省。BF16 的指数位和 FP32 一样多（动态范围相同），所以**不需要 GradScaler**（不像 FP16 那样怕下溢）。

> 一个细节：BatchNorm 层保持 FP32（在 pipeline 里显式 `.float()`），因为归一化统计量对精度敏感。混合精度的完整原理见第 06 章。

## 5.7 评估与晋升

每隔一轮，让新模型和当前"最佳模型"对弈 100 局：

```python
result = play_match(network, best_network, device, num_games=100)
if result["winrate"] >= 0.55:        # 胜率 ≥ 55%
    best_network.load_state_dict(network.state_dict())   # 晋升
    save_checkpoint(best_path, network, ...)
```

**为什么是 55% 而不是 50%？** 50% 只是"打平"，要求 55% 确保新模型是**确实变强**了才晋升，避免噪声导致的虚假晋升。这个机制保证 `best.pt` 始终是历史最强版本。

`play_match` 用 lockstep 批处理同时下 100 局（第 06 章），很快。

## 5.8 对手多样性

为防止"策略循环"（网络只学会克制自己上一版，遇到不同风格就崩），引入**历史 checkpoint 池**：

```python
history_opponent_ratio: 0.5    # 50% 的对局用历史版本做对手
history_save_interval: 200     # 每 200 轮存一个历史版本
history_keep: 10               # 保留最近 10 个
```

每轮自博弈时，50% 的对局让当前模型对阵一个随机抽取的历史版本。这迫使网络学会应对**不同风格**的对手，而非只针对自己。

## 5.9 主循环

> 源文件：`gomonova/training/pipeline.py`

把以上组件串起来：

```python
for iteration in range(start_iter, total_iters):
    lr = trainer.update_lr(iteration)
    use_mcts  = iteration >= mcts_start
    use_renju = iteration >= renju_start

    # ① 自博弈生成数据
    samples = generate_games(network, device, num_games=512,
                             use_mcts=use_mcts, use_renju=use_renju, ...)
    replay.add_batch(samples)

    if len(replay) < batch_size * 2:    # 数据太少，先攒数据
        continue

    # ② 训练
    metrics = trainer.train_epoch(replay, batch_size=2048, steps=32)

    # ③ 评估 + 晋升（仅 rank 0）
    result = play_match(network, best_network, num_games=100)
    if result["winrate"] >= 0.55:
        best_network.load_state_dict(network.state_dict())
        save_checkpoint(...)
```

> **调参经验：`train_steps_per_iter=32`。** 每轮自博弈产生约 512×50×4 ≈ 10 万样本，但每轮只做 32 步梯度更新（每步 batch=2048，共 6.5 万样本）。**为什么不训更多步？** 因为实测发现**训练才是时间瓶颈**（占每轮 75% 时间），而 MCTS 数据质量高，不需要反复训练同一批数据。把步数从 64 减到 32，每轮省一半训练时间，棋力几乎不受影响。这是"数据质量高 → 少训几步也行"的典型例子。

## 5.10 多 GPU（DDP）

GomoNova 用 **DistributedDataParallel（DDP）** 在 4 张 GPU 上并行训练：

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/train_main.yaml
```

DDP 启动 4 个进程，每个进程占一张 GPU，各自独立做自博弈和训练，训练时通过 allreduce 同步梯度。

> **一个反直觉的教训：** 早期用 12.5M 小模型时，多 GPU **反而更慢**（通信开销 > 计算）。但换成 62.7M 大模型后，计算量上来了，DDP 才真正发挥作用。详见第 06 章的深入分析。

## 5.11 动手实验

```bash
# 用冒烟测试配置跑 2 轮，验证整个管线（约 1 分钟）
python scripts/train.py --config configs/train_smoke.yaml
```

观察输出中的 `Loss`、`P`(策略损失)、`V`(价值损失) 是否在下降。然后试着修改 `configs/train_smoke.yaml` 里的 `mcts_sims`、`lr`，观察 loss 曲线的变化，建立对超参数的直觉。

**下一步：** 第 06 章深入性能优化——如何把 MCTS 自博弈从"慢得无法忍受"优化到"可接受"，这是把整个方案从"理论可行"变成"实际能跑"的关键工程。
