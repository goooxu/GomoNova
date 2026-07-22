# 第七章：性能优化

## 本章你将学到

- GPU 批量推理的原理和加速效果
- 训练中的性能瓶颈分析
- 多 GPU 并行为什么对小模型反而更慢
- 开发机管理的实用经验

## 7.1 GPU 批量推理

### 核心原理

GPU 有几千个计算核心，擅长**同时处理大量相同操作**。一次处理 512 个棋盘和一次处理 1 个棋盘，耗时几乎相同：

| Batch Size | 前向推理耗时 | 每样本耗时 |
|-----------|-------------|-----------|
| 1 | 12ms | 12ms |
| 64 | 12ms | 0.19ms |
| 512 | ~15ms | 0.03ms |

**加速比：63×**（batch=64 vs batch=1）

### 在自博弈中的应用

```python
# 慢：逐局串行
for game in range(512):
    for move in range(50):
        policy = network(single_board)  # 512×50 = 25,600 次 GPU 调用

# 快：lockstep 并行
for move in range(50):
    policies = network(all_512_boards)  # 50 次 GPU 调用
```

从 25,600 次 GPU 调用减少到 50 次。

## 7.2 训练瓶颈分析

每轮迭代的时间分布（稳态）：

| 步骤 | 耗时 | 占比 | 瓶颈 |
|------|------|------|------|
| Self-Play | 4s | 19% | Python 游戏逻辑 |
| Train | 16s | 76% | GPU 计算 |
| Eval | 1s | 5% | GPU 计算 |
| **总计** | **21s** | 100% | |

**Self-Play 的瓶颈不在 GPU，在 Python：**
- 每步需要：更新棋盘、检查胜负、选子、禁手过滤
- 这些是 CPU 上的 Python 循环，GPU 在等待

**Train 的瓶颈在 GPU 计算：**
- 128 步 × batch=1024 × 12.5M 模型
- 每步：前向 + 反向 + 参数更新

## 7.3 自由规则加速

禁手检测非常慢（递归搜索活三/活四），每步需要对 ~200 个空位逐一检查。

**优化：训练时完全跳过禁手检测。**

| | 连珠规则 | 自由规则 |
|--|---------|---------|
| 每步耗时 | ~5ms（禁手检测） | ~0.01ms |
| 512局×50步 | ~128s | ~0.3s |

训练时用自由规则（所有空位都能下），推理时再加禁手过滤。网络学到的是"通用棋感"，禁手只是最后的合法性过滤。

## 7.4 批量评估（68s → 1s）

### 之前：逐局串行

```python
for game in range(50):
    while not game_over:
        move = network(single_board)  # 每步一次 GPU 调用
        board.play(move)
# 50局 × ~100步 = 5000 次 GPU 调用，每次 12ms = 60s
```

### 之后：lockstep 并行

```python
boards = [Board() for _ in range(50)]
while any(not done):
    # 收集所有需要 A 网络决策的棋盘
    a_boards = [boards[i] for i in active if is_a_turn(i)]
    a_moves = network(batch(a_boards))  # 一次调用

    # 收集所有需要 B 网络决策的棋盘
    b_boards = [boards[i] for i in active if is_b_turn(i)]
    b_moves = best_network(batch(b_boards))  # 一次调用
# ~100 步 × 2 次调用 = 200 次 GPU 调用 = 2.4s → 实测 1s
```

## 7.5 多 GPU 的陷阱

开发机有 4 张 GB300（284GB 显存），直觉上应该用多卡加速。实测：

| 配置 | 每轮耗时 | 说明 |
|------|---------|------|
| 单卡 | 21s | 基线 |
| 4卡 DataParallel | 94s | **反而慢 4.5 倍！** |

**为什么更慢？**

DataParallel 的工作方式：
1. 把 batch 拆成 4 份，分发到 4 张卡
2. 每张卡独立前向/反向
3. 把梯度收集回主卡，平均
4. 更新参数，再分发到 4 张卡

对于 12.5M 参数的小模型：
- 计算本身只要几毫秒
- 但分发/收集/同步的通信开销要几十毫秒
- **通信 > 计算**，多卡反而更慢

**结论：** 多 GPU 适合大模型（>100M 参数），小模型单卡最优。

## 7.6 开发机管理经验

### 磁盘空间

开发机的 home 目录通常很小（5GB），而 checkpoint 每个 73-145MB。

**规则：**
- 代码放 home（很小）
- 数据、checkpoint、日志、venv 全放 `/tmp`（通常挂载大磁盘）
- 每 30 分钟自动备份 checkpoint 到本地

### 机器过期

开发机（SLURM 作业）随时可能被回收。应对：

1. 训练在 `tmux` 中运行（SSH 断开不影响）
2. Checkpoint 定期备份到本地
3. 新机器部署流程：
   ```bash
   # 创建环境
   uv venv /tmp/gemsg/venv && uv pip install torch numpy ...
   # 同步代码
   rsync -az ./ user@new-host:~/gomonova/
   # 上传 checkpoint
   rsync -az checkpoints/best.pt user@new-host:/tmp/gemsg/checkpoints/
   # 启动训练（自动从 checkpoint 恢复）
   tmux new -s gomonova 'python scripts/train.py --config configs/train_main.yaml'
   ```

### 训练监控

```bash
# 查看最新进度
ssh user@host "tail -1 /tmp/gemsg/train.log"

# 检查进程是否存活
ssh user@host "tmux has-session -t gomonova"

# 检查磁盘
ssh user@host "df -h ~"
```

## 7.7 优化效果总结

| 优化 | 之前 | 之后 | 加速比 |
|------|------|------|--------|
| 批量自博弈 | 290s/4局 | 4s/512局 | ~9000× |
| 自由规则 | 5ms/步 | 0.01ms/步 | 500× |
| 批量评估 | 68s | 1s | 68× |
| BF16 | 22s/轮 | 21s/轮 | 1.05× |
| 多GPU | 21s/轮 | 94s/轮 | 0.22×（负优化）|
