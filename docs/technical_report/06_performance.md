# 第六章：性能优化

## 本章你将学到

- 性能优化的正确方法论：**先测量，再优化**
- GPU 批量推理的原理与实测吞吐
- 如何定位 MCTS 的真正瓶颈（不是 GPU，是 Python）
- 四个关键优化：批量 GPU 评估、扁平数组树、共享内存并行 MCTS、大模型 DDP
- 共享内存轮次同步的设计，以及为什么队列方案行不通
- 混合精度（BF16）的原理与陷阱
- 完整的基准数据

> 涉及源文件：`gomonova/mcts/flat_tree.py`、`gomonova/training/parallel_mcts.py`、`gomonova/training/pipeline.py`

第 04、05 章的方案在理论上很完美，但**直接实现会慢到无法训练**。MCTS 自博弈涉及海量 Python 循环，朴素实现下 512 局要跑 100 多秒，3000 轮训练要几个月。本章还原每一步优化的真实动机与数据——这是把方案从"理论可行"变成"实际能跑"的关键工程。

## 6.1 方法论：先测量，再优化

性能优化的第一原则是**不要凭直觉猜瓶颈**。GomoNova 的优化全程遵循一个循环：

```
① 计时（profile）   测量每个阶段的实际耗时
② 定位（locate）    找到占比最大的瓶颈
③ 优化（optimize）  针对瓶颈改进
④ 验证（verify）    重新计时，确认效果，回到 ①
```

> **反模式：** 看到代码"看起来慢"就优化。比如很多人会先去优化 GPU 推理，但实测发现 GPU 根本不是瓶颈——Python 循环才是。**测量永远先于优化。**

## 6.2 GPU 批量推理：理解硬件特性

### 核心原理

GPU 有几千个计算核心，擅长**同时处理大量相同操作**。关键洞察：**一次处理 512 个棋盘和一次处理 1 个棋盘，耗时几乎相同**——因为 GPU 的算力远超单个任务的需求，瓶颈在"启动一次计算"的固定开销。

实测数据（GB200 GPU，62.7M 模型）：

| Batch Size | 前向推理耗时 | 每样本耗时 |
|-----------|-------------|-----------|
| 1 | 9.2ms | 9.2ms |
| 512 | 25.4ms | 0.05ms |

**单样本吞吐 108 pos/s，批量吞吐 20,154 pos/s——差 186 倍。**

### 推论：一切都要批处理

这个特性决定了整个系统的设计哲学：**能批量就批量**。

```python
# 慢：逐局串行，25,600 次 GPU 调用
for game in range(512):
    for move in range(50):
        policy = network(single_board)

# 快：lockstep 并行，50 次批量调用
for move in range(50):
    policies = network(all_512_boards)   # 一次调用处理 512 局
```

自博弈、评估、MCTS 评估都遵循这个原则。

## 6.3 定位 MCTS 的瓶颈

引入 MCTS 后，先计时看瓶颈在哪。直觉可能认为是 GPU 评估慢，但实测**完全相反**：

对 MCTS 各操作单独计时：

| 操作 | 耗时 | 说明 |
|------|------|------|
| 创建 + 扩展一个节点 | **212μs** | Python 对象 + 字典操作 |
| best_child（PUCT 选择） | 38μs | Python 循环遍历子节点 |
| 一次 GPU 评估（单样本） | ~9ms | 但可批处理摊薄 |

**瓶颈是 Python 对象操作，不是 GPU。** 一次 MCTS 搜索（25 次模拟）要创建上千个 `MCTSNode` 对象，每个对象涉及 `__init__`、字典 `children`、Python float——这些在 Python 里都很慢。

> 用对象版 `MCTSNode` 跑 512 局 MCTS（25 次模拟/10 手）：**105.9 秒**。其中绝大部分时间花在 Python 树遍历上，GPU 大部分时间在空等。

## 6.4 优化①：批量 GPU 评估

第一个优化：把 MCTS 中分散的 GPU 评估**合并成批量调用**。

朴素实现里，每次模拟到达 leaf 节点就单独调用一次网络（单样本，9ms）。改成：收集一批 leaf 位置，一次性批量评估。

```python
# 收集所有需要评估的位置
planes = np.stack([board_to_planes(b) for _, _, b in expand_info])
x = torch.from_numpy(planes).to(device)
logits, values = network(x)        # 一次批量调用
policies = softmax(logits)
```

这把"上千次单样本调用"变成"少数几次批量调用"，GPU 利用率大幅提升。

## 6.5 优化②：扁平数组树

针对"节点创建 212μs"这个瓶颈，彻底重写树的数据结构：**用预分配的 numpy 数组替代 Python 对象**。

> 源文件：`gomonova/mcts/flat_tree.py`

### 对象版 vs 数组版

**对象版（MCTSNode）：** 每个节点是一个 Python 对象，子节点存在字典里。

```python
class MCTSNode:
    def __init__(self):
        self.children = {}        # 字典：动态分配，慢
        self.visit_count = 0      # Python int
        self.total_value = 0.0    # Python float
```

**数组版（FlatMCTSTree）：** 整棵树用几个预分配的大数组表示，节点 i 的子节点存在 `child_*[i*225 : i*225+n]` 这段连续内存里。

```python
class FlatMCTSTree:
    def __init__(self, max_nodes=64):
        total_slots = max_nodes * 225
        self.child_action = np.zeros(total_slots, dtype=np.int32)
        self.child_prior  = np.zeros(total_slots, dtype=np.float32)
        self.child_visit  = np.zeros(total_slots, dtype=np.int32)
        self.child_value  = np.zeros(total_slots, dtype=np.float64)
        self.num_children = np.zeros(max_nodes, dtype=np.int32)
        self.is_expanded  = np.zeros(max_nodes, dtype=np.bool_)
```

### 关键操作的向量化

`best_child` 从 Python 循环变成 numpy 向量运算：

```python
def best_child(self, node, c_puct):
    start = node * 225
    n = self.num_children[node]
    vc = self.child_visit[start:start+n].astype(np.float64)
    tv = self.child_value[start:start+n]
    prior = self.child_prior[start:start+n]
    sqrt_pv = math.sqrt(max(int(vc.sum()), 1))
    q = np.where(vc > 0, tv / np.maximum(vc, 1), 0.0)   # 向量化 Q 值
    u = c_puct * prior * sqrt_pv / (1.0 + vc)            # 向量化探索项
    best_i = int(np.argmax(q + u))                        # 一次 argmax
    return start + best_i, int(self.child_action[start + best_i])
```

**效果：节点创建从 212μs 降到约 5μs（40 倍）。** 因为 numpy 数组是连续内存、无 Python 对象开销，且操作是 C 层向量化执行。

> 512 局 MCTS：105.9s → **63.6s**（扁平树单独贡献 1.7×）。

## 6.6 优化③：共享内存并行 MCTS（核心优化）

扁平树解决了单节点的开销，但还有一个更大的问题：**整个 MCTS 搜索是单线程的**。512 局棋在一台 CPU 上串行遍历树，而开发机有 144 个 CPU 核——绝大部分 cores 在闲置。

### 目标架构

把 MCTS 拆成两部分，分别放到最合适的硬件上：

```
┌─────────────────────────────────────────────────────────┐
│  CPU（144 核并行）          GPU（集中评估）              │
│                                                         │
│  Worker 0: 遍历第 0-13 局的树                           │
│  Worker 1: 遍历第 14-27 局的树      ┌──────────────┐   │
│  Worker 2: 遍历第 28-41 局的树  ──→ │ 批量 NN 评估  │   │
│  ...                              │ （一次处理全部）│   │
│  Worker 35: 遍历第 498-511 局的树  └──────────────┘   │
│                                                         │
│  树遍历（CPU 密集）          评估（GPU 擅长）           │
└─────────────────────────────────────────────────────────┘
```

- **CPU workers**：每个 worker 负责一部分棋局的树遍历（选择、扩展、回溯）——这是 CPU 密集型，并行化到多核。
- **GPU 评估器**：所有 worker 需要评估时，把位置汇总到 GPU，**一次批量评估全部**——保持 GPU 大批量。

### 难点：如何同步？

挑战在于：worker 遍历树到 leaf 时需要 GPU 评估，评估完才能继续。这要求 CPU 和 GPU **协调节奏**。

**失败的尝试：用队列（Queue）。** 最初想用 `multiprocessing.Queue`，worker 把评估请求放进队列，GPU 评估后把结果放回。但实测有问题：
- 队列传输要序列化（pickle）每个位置（14KB），开销大
- worker 去同步化后，GPU 收到的是零散的小批量，失去批量优势

**成功的方案：共享内存 + 轮次同步。**

> 源文件：`gomonova/training/parallel_mcts.py`

核心思想：**所有 worker 按"轮"同步推进**。每一轮：

```
① 所有 worker 遍历各自的树一步，把需要评估的 leaf 位置写入共享内存
② worker 递增 ready_count，然后轮询等待
③ 主进程等 ready_count == 总局数（所有 worker 都写完了）
④ 主进程把全部位置一次送 GPU 批量评估
⑤ 主进程把结果写回共享内存，递增 gen（generation）
⑥ worker 检测到 gen 前进，读取结果，更新树，进入下一轮
```

```python
# 共享内存（所有进程可见）
planes_raw  = mp.RawArray("f", n * PLANE_SIZE)   # 位置缓冲
results_raw = mp.RawArray("f", n * RESULT_SIZE)  # 结果缓冲
ready_count = mp.Value("i", 0)                    # 就绪计数
gen         = mp.Value("i", 0)                    # 轮次代号

# Worker 端
def _sync(round_gen):
    with ready_count.get_lock():
        ready_count.value += n          # 我写完了
    while True:
        with gen.get_lock():
            if gen.value > round_gen:    # 等主进程评估完
                break
        time.sleep(0.00002)

# 主进程端
for rd in range(num_simulations + 1):
    while ready_count.value < n:         # 等所有 worker 写完
        time.sleep(0.00002)
    ready_count.value = 0
    # GPU 批量评估全部 n 个位置
    x = torch.from_numpy(planes_np.reshape(n, 16, 15, 15)).to(device)
    logits, values = network(x)
    results_np[:, :225] = softmax(logits)
    results_np[:, 225]  = values
    gen.value = rd + 1                    # 通知 worker 继续
```

**为什么共享内存比队列好？**

- **零序列化**：共享内存是同一块物理内存，worker 直接读写 numpy 视图，无需 pickle。
- **保持大批量**：轮次同步强制所有 worker 在同一时刻提交，GPU 每次评估的都是完整的 n 个位置（大批量），而非零散小批量。

### 效果

512 局 MCTS（25 次模拟 / 10 手）：

| 方案 | 耗时 | 加速比 |
|------|------|--------|
| 对象版串行 | 105.9s | 1× |
| 扁平树单进程 | 63.6s | 1.7× |
| **共享内存并行 MCTS** | **15.3s** | **6.9×** |

CPU 树遍历并行到多核 + GPU 保持大批量，两个瓶颈同时解决。

## 6.7 优化④：大模型 DDP

GomoNova 用 DDP 在 4 张 GPU 上并行。但这里有个**反直觉的教训**。

### 小模型时多 GPU 反而更慢

早期用 12.5M 小模型测试多卡：

| 配置 | 每轮耗时 |
|------|---------|
| 单卡 | 21s |
| 4 卡 DataParallel | **94s（慢 4.5 倍！）** |

**为什么？** 多卡要把 batch 拆分、分发、计算、收集梯度、同步。对小模型，计算本身只要几毫秒，但**通信开销（分发/收集/同步）要几十毫秒——通信 > 计算**，多卡反而拖累。

### 大模型时 DDP 才值得

换成 62.7M 大模型后，单卡计算量上来了（每步训练 ~0.5s），通信开销相对变小，DDP 的并行收益才超过通信成本。

> **经验法则：** 多 GPU 适合大模型（计算密集），小模型单卡最优。是否用多卡，要看"计算时间 vs 通信时间"的比例，不能一概而论。

DDP 还带来一个额外好处：4 个进程各自独立做自博弈，**每轮产生的训练数据是 4 倍**，相当于增大了有效 batch。

## 6.8 混合精度（BF16）

### 原理

浮点数有两种常见格式：

| 格式 | 总位数 | 指数位 | 尾数位 | 动态范围 |
|------|--------|--------|--------|---------|
| FP32 | 32 | 8 | 23 | 大 |
| BF16 | 16 | 8 | 7 | **同 FP32** |
| FP16 | 16 | 5 | 10 | 小（易下溢） |

**BF16 用 16 位达到了和 FP32 相同的动态范围**（指数位一样多），只是精度低些（尾数位少）。对神经网络训练，动态范围比精度更重要，所以 BF16 几乎不损失训练质量，但：

- 显存占用减半
- 矩阵运算速度更快（GPU 的 BF16 算力通常是 FP32 的 2 倍）

### 用法

```python
with torch.amp.autocast(device.type, dtype=torch.bfloat16):
    logits, value = network(x)     # 矩阵运算自动用 BF16
    loss = total_loss(...)
loss.backward()                     # 反向也用 BF16
```

`autocast` 自动把适合的运算（Conv、Linear）转成 BF16，其他保持 FP32。

### 两个陷阱

1. **BatchNorm 必须保持 FP32。** 归一化要计算均值/方差，对精度敏感。GomoNova 在创建网络后显式把 BN 层转回 FP32：

```python
network = GomoNovaNet(...).bfloat16().to(device)
for m in network.modules():
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.float()    # BN 保持 FP32
```

2. **BF16 不需要 GradScaler。** FP16 因为动态范围小，梯度容易下溢成 0，需要 GradScaler 放大损失。BF16 动态范围和 FP32 相同，**没有这个问题**，代码更简单。这是选 BF16 而非 FP16 的重要原因。

## 6.9 连珠阶段的代价

第 05 章提到训练分三相，最后阶段用连珠规则。这里有个性能代价：**禁手检测很慢**。

MCTS 扩展节点时要枚举合法落子，对黑棋需逐个检查是否禁手（第 02 章的递归判定）：

| 阶段 | 自博弈耗时（512 局） |
|------|---------------------|
| MCTS + 自由规则 | ~19s |
| MCTS + 连珠规则 | **~75s** |

禁手检测让自博弈慢了约 4 倍。因为连珠阶段只有最后 600 轮（占 20%），整体影响可控，但这是"规则正确性 vs 速度"的明确权衡——为了学会禁手，必须付出这个代价。

## 6.10 基准数据汇总

| 优化 | 之前 | 之后 | 加速比 |
|------|------|------|--------|
| GPU 批量推理 | 9.2ms/样本 | 0.05ms/样本 | 186× |
| 扁平数组树（节点创建） | 212μs | ~5μs | 40× |
| 批量 GPU 评估 + 扁平树（512 局 MCTS） | 105.9s | 63.6s | 1.7× |
| 共享内存并行 MCTS（512 局） | 63.6s | 15.3s | 4.2× |
| **MCTS 总计（vs 最初对象版）** | **105.9s** | **15.3s** | **6.9×** |

每轮迭代的稳态耗时（4× DDP）：

| 阶段 | 自博弈 | 训练 | 评估 | 合计 |
|------|--------|------|------|------|
| 纯策略热身 | ~6s | ~15s | ~1s | ~22s |
| MCTS 自由规则 | ~19s | ~16s | ~1s | ~36s |
| MCTS 连珠规则 | ~75s | ~19s | ~2s | ~96s |

3000 轮总训练时间约 **22–24 小时**（4× GPU）。

## 6.11 动手实验

```python
import time, torch
from gomonova.nn.network import GomoNovaNet
from gomonova.game.board import Board
from gomonova.nn.encoder import board_to_planes
import numpy as np

net = GomoNovaNet().bfloat16().cuda().eval()

# 对比单样本 vs 批量推理的吞吐
boards = [Board() for _ in range(512)]
planes = np.stack([board_to_planes(b) for b in boards])

# 批量（快）
x = torch.from_numpy(planes).cuda().bfloat16()
torch.cuda.synchronize(); t0 = time.time()
for _ in range(50):
    with torch.no_grad(): net(x)
torch.cuda.synchronize()
print(f"批量 512×50 次: {time.time()-t0:.2f}s")

# 单样本（慢）—— 取消注释体验差距
# x1 = torch.from_numpy(planes[:1]).cuda().bfloat16()
# torch.cuda.synchronize(); t0 = time.time()
# for _ in range(512):
#     with torch.no_grad(): net(x1)
# torch.cuda.synchronize()
# print(f"单样本 512 次: {time.time()-t0:.2f}s")
```

**下一步：** 第 07 章看训练好的模型如何用于对战——纯前向推理的实现，以及 CLI 和 Web 两个对弈界面。
