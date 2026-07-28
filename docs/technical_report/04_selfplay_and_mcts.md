# 第四章：自博弈与蒙特卡洛树搜索

## 本章你将学到

- 什么是自博弈（self-play），为什么"自己和自己下"能学会下棋
- 纯策略自博弈的局限：为什么光靠模仿自己远远不够
- 蒙特卡洛树搜索（MCTS）的完整原理——**无需任何强化学习背景**
- MCTS 的四个阶段：选择、扩展、评估、回溯
- PUCT 公式的直觉：如何在"利用已知"和"探索未知"之间平衡
- MCTS 如何生成训练目标（访问分布）
- **核心设计：为什么训练时用搜索、对战时不用**

> 涉及源文件：`gomonova/training/selfplay.py`、`gomonova/mcts/node.py`、`gomonova/mcts/search.py`

这是全书最重要的一章。如果你没有强化学习背景，不用担心——我们会从"自己和自己下棋"这个最朴素的直觉出发，一步步推到 MCTS。

## 4.1 什么是自博弈

### 一个朴素的类比

想象你刚学会五子棋的规则，但从来没见过任何棋谱，身边也没有对手。你怎么提高？

最自然的办法：**自己和自己下**。你执黑下一局，再执白下一局，下完后复盘——"这局黑棋赢了，那黑棋的哪些落子是好棋？白棋的哪些是坏棋？" 反复 thousands 局之后，你慢慢总结出"什么样的局面会赢"。

这就是**自博弈（self-play）**。GomoNova 的训练数据全部来自网络的自我对弈：

```
当前网络 ──执黑──┐
                 ├── 对弈一局 ──→ 产生胜负结果 ──→ 更新网络
当前网络 ──执白──┘
```

没有人类棋谱，没有外部对手，网络既是学生又是老师。

### 为什么这能行？

关键在于**胜负信号提供了客观的评价标准**。每一局棋都有明确的结局（黑胜/白胜/平局）。赢的那一方走过的棋步，整体上是"好的"；输的那一方走过的棋步，整体上是"坏的"。网络通过"多模仿赢棋、少模仿输棋"，逐渐向"会赢"的方向调整。

> 这就像学下棋时复盘：赢了的招法记下来多练，输了的招法反思少用。区别是人类复盘靠理解，网络靠梯度下降把"赢/输"这个信号转化为权重调整。

### 温度采样：探索 vs 利用

如果每步都选"当前认为最好的"，网络会陷入固定套路，无法发现新招法。所以自博弈时用**温度采样**引入随机性：

```python
def _sample_move(policy, temp):
    if temp < 1e-8:
        return np.argmax(policy)          # 温度≈0：选最优（利用）
    tempered = policy ** (1.0 / temp)     # 温度>0：按概率采样（探索）
    tempered /= tempered.sum()
    return np.random.choice(len(tempered), p=tempered)
```

- **温度高（如 1.0）**：概率被"拉平"，更可能尝试次优的招法 → **探索**
- **温度低（如 0.1）**：概率被"拉尖"，几乎只选最优 → **利用**

GomoNova 在开局前 20 手用较高温度（多探索），之后用低温度（多利用）。这保证了开局多样性，避免所有对局长得一样。

## 4.2 纯策略自博弈（lockstep 批处理）

最简单的自博弈：网络直接用自己的策略输出选子，一局一局下完。

### 开局：从搜索中学（无强制）

开局（前几手）完全由 **MCTS 引导**学习，不做任何强制。空盘黑先时，MCTS 靠价值评估自己发现天元（H8）是最优第一手；白棋应对天元时，MCTS 也搜出贴近天元的中心要点。这些正确开局选择作为 MCTS 访问分布目标蒸馏进网络，对战时纯推理自然复现——仍然不需要任何推理时规则干预。

> **历史教训（强制天元的弯路）：** 早期版本曾**强制**黑第一手下天元（每局固定 `board.play(TENGAN)` 开局），想把"黑第一手=天元"硬塞进训练分布。这只是给黑棋第一手打了补丁，没解决"开局走边"的根因；反而因为黑棋第一手被硬编码、白棋第一手仍由高温度噪声自博弈学习，造成训练分布不对称，白棋开局也跟着走边角。后来发现真正根因是 MCTS 实现本身有多处 bug（详见开发日志 V2.2），修好后 MCTS 能自己搜出天元与合理应对，强制开局便移除了。一课：**与其强制一个正确结果，不如修好产生结果的搜索。**

```python
def play_games_fast(network, device, num_games, ...):
    # 每局从空盘开始；开局由 MCTS（修正后）引导，无强制
    boards, histories = [], []
    for _ in range(num_games):
        boards.append(Board()); histories.append([])
    while active:
        # 把所有活跃棋盘打包成一个 batch，一次前向推理
        planes = np.stack([board_to_planes(b) for b in active_boards])
        policies = network(planes)                 # 一次 GPU 调用
        # 每局根据各自的策略采样一手
        for board, policy in zip(active_boards, policies):
            move = _sample_move(policy, temp)
            board.play(move)
```

**lockstep（同步步进）：** 不是下完一局再下一局，而是 512 局**同时**进行，每步把所有棋盘打包成一个 batch 送进 GPU。这把"512 局 × 50 步 = 25,600 次 GPU 调用"压缩成"50 次批量调用"，是巨大的加速（详见第 06 章）。

## 4.3 为什么纯自模仿不够

纯策略自博弈有个根本问题：**网络只能从自己的水平出发**。

- 如果网络还没学会"做双威胁"，它的自博弈里就**永远不会出现**双威胁的棋局，于是它**永远学不会**双威胁。
- 网络可能陷入"策略循环"：A 套路克制 B 套路，B 套路又克制 A 套路，网络在两者间反复横跳，棋力停滞。

这就像一个只和自己下棋的人，如果双方都不会某种战术，那这种战术永远不会被发现。**网络需要一个比"当前直觉"更可靠的老师，来告诉它"这步棋其实应该下在这里"。**

这个更可靠的老师，就是 **MCTS**。

## 4.4 什么是 MCTS（蒙特卡洛树搜索）

### 直觉：在脑中预演未来

高手下棋时会"算路"：**"如果我下这里，对手可能应那里，然后我再下那里……"** 他在脑中预演未来的几种可能，评估哪条路最好，然后选择。

MCTS 就是把这个"算路"过程系统化、算法化。它不靠直觉一步到位，而是**反复模拟未来**，统计"哪个落子最终赢的次数多"，用统计结果指导决策。

> 关键区别：网络的"直觉"是一次前向推理（快但浅）；MCTS 的"算路"是成百上千次模拟（慢但深）。

### 搜索树

MCTS 把"算路"组织成一棵**树**：

```
            [当前局面]  ← 根节点
           /    |    \
        落A   落B   落C   ← 第一手的候选
        /       |
     对手应   对手应       ← 第二手（对手视角）
      / \       |
    ...  ...   ...        ← 继续往下推演
```

- 每个**节点** = 一个棋盘局面
- 每条**边** = 一手落子
- 从根到叶的一条路径 = 一局"想象中的对弈"

MCTS 反复地从根节点往下"走"一局想象中的棋（叫一次**模拟/simulation**），走到某个位置后用网络评估"这个局面谁占优"，再把评估结果**回传**更新沿途的统计。模拟几百次后，统计就收敛了。

## 4.5 MCTS 的四个阶段

每一次模拟包含四个阶段：

```
① 选择 (Selection)   从根节点往下，用 PUCT 公式选最有潜力的分支，直到到达一个未扩展的节点
② 扩展 (Expansion)   在这个新节点，用网络策略生成所有合法落子作为子节点
③ 评估 (Evaluation)  用网络的价值头评估这个局面的优劣（谁占优）
④ 回溯 (Backup)      把评估值沿路径回传到根，更新沿途每个节点的统计
```

### ① 选择：用 PUCT 选分支

从根节点开始，每一层都选"得分最高"的子节点往下走，直到走到一个还没展开的节点。"得分"由 PUCT 公式计算（下节详述）。

### ② 扩展：生成子节点

到达一个新局面后，用网络的策略输出给所有合法落子打分（先验概率），为每个落子创建一个子节点。

```python
def expand(self, policy, legal_actions):
    total = policy[legal_actions].sum()
    for action in legal_actions:
        p = policy[action] / total          # 归一化的先验概率
        self.children[action] = MCTSNode(parent=self, action=action, prior=p)
    self.is_expanded = True
```

### ③ 评估：网络判断局面

用网络的价值头评估当前局面，得到一个 `[-1, 1]` 的值（从当前方视角）。

```python
def _evaluate(self, board):
    policy_logits, value = self.network(board_to_planes(board))
    policy = softmax(policy_logits)
    return policy, value.item()    # 策略 + 价值
```

### ④ 回溯：更新统计

把评估值沿路径回传到根，**每经过一个节点就更新它的统计**（访问次数 +1，累计价值 += value）。注意**每上一层要变号**——因为父节点和子节点是对手关系，对我好的对对手就是坏的。

```python
def backup(self, value):
    node = self
    while node is not None:
        node.visit_count += 1
        node.total_value += value
        value = -value          # 换对手，价值变号
        node = node.parent
```

## 4.6 PUCT 公式：探索与利用的平衡

选择阶段的核心是 **PUCT 公式**，它给每个子节点打分：

```
得分 = Q值 + c_puct × 先验概率 × √(父节点访问次数) / (1 + 自身访问次数)
       └──利用──┘   └──────────────── 探索 ────────────────┘
```

```python
def ucb_score(self, c_puct):
    parent_visits = self.parent.visit_count
    exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
    return self.q_value + exploration
```

两部分各司其职：

- **Q 值（利用）** = `total_value / visit_count`，即"这个分支历史上平均有多好"。Q 值高 → 多选它。
- **探索项** = 鼓励那些"还没怎么被探索"的分支。`visit_count` 越小（探索越少），探索项越大；`prior` 越高（网络越看好），探索项越大。

> **类比：** 这就像选餐厅。Q 值是"你去过的那几家的好评率"（利用已知），探索项是"那家新开的、网红推荐的店你还没去过，值得试试"（探索未知）。PUCT 自动在"吃老地方"和"尝鲜"之间平衡。`c_puct` 是旋钮：调大更爱尝鲜，调小更保守。

随着模拟次数增加，被选中的分支 `visit_count` 上升、探索项下降，搜索会逐渐聚焦到真正好的分支上。

## 4.7 MCTS 如何生成训练目标

模拟几百次后，根节点的每个子节点都有一个**访问次数**——被搜索得越多的落子，说明 MCTS 认为它越好。把访问次数归一化，就得到一个**访问分布（visit distribution）**：

```python
def get_visit_distribution(self, temperature=1.0):
    policy = np.zeros(225)
    for action, child in self.children.items():
        policy[action] = child.visit_count    # 访问次数
    # 归一化（可加温度）
    policy = policy ** (1.0 / temperature)
    policy /= policy.sum()
    return policy
```

**这个访问分布就是 MCTS 给网络的"标准答案"。** 它比网络自己的直觉更可靠，因为它是几百次深度模拟的统计结晶。

网络训练时，就学习让自己的策略输出**逼近这个访问分布**（用 KL 散度，第 05 章详述）。换句话说：

> **网络在学习"模仿一个比自己更聪明的搜索过程"。** 学会之后，网络不需要真的搜索，就能直接输出接近 MCTS 质量的落子。

## 4.8 核心设计：为什么训练用搜索、对战不用

这是整个系统最精妙的地方，值得反复理解：

| | 训练时 | 对战时 |
|--|--------|--------|
| 是否搜索 | **用 MCTS** | **不用，纯前向推理** |
| 速度 | 慢（每步几百次模拟） | 快（一次前向推理） |
| 质量 | 高（深度搜索的结晶） | 高（已学会逼近 MCTS） |
| 角色 | 生成"标准答案" | 直接输出落子 |

**逻辑链条：**

1. 训练时，MCTS 充当"超级老师"，为每个局面算出高质量的落子分布。
2. 网络反复学习"在这个局面下，MCTS 会怎么下"，把搜索的能力**内化**进权重。
3. 训练充分后，网络自己就能输出接近 MCTS 质量的落子——**搜索的知识已经被"蒸馏"进网络**。
4. 对战时，直接用网络一次前向推理，又快又强，无需搜索。

> **类比：** 这就像学生平时做题时可以参考详细的解题过程（MCTS 搜索），把解题思路吃透。考试时（对战）不能再翻参考，但因为平时已经内化了方法，直接就能做对。

这正是 AlphaZero 的核心思想。GomoNova 完整实现了它——唯一的区别是网络架构是原创的（MSAR-Net）。

## 4.9 选择性 MCTS 与 Dirichlet 噪声

### 选择性 MCTS

MCTS 很慢，没必要每步都搜索。GomoNova 只在**开局前几手**用 MCTS（`mcts_moves=5`），之后的棋步用纯策略快速下完。

为什么是开局？因为开局的选择影响全局战略，最值得"深思"；而中后盘战术相对直接，网络直觉就够用。这个设计大幅降低了自博弈的时间成本（详见第 06 章）。

```python
def play_games_with_mcts(network, device, num_games, mcts_searcher, mcts_moves=5, ...):
    # 阶段 A：前 mcts_moves 手用 MCTS
    for move_num in range(mcts_moves):
        visit_dists = mcts_searcher.flat_batch_search(active_boards)  # MCTS
        # 按访问分布采样落子
    # 阶段 B：剩余棋步用纯策略
    while not done:
        policies = network(boards)   # 纯前向推理
        # 按策略采样落子
```

### Dirichlet 噪声

为了让 MCTS 在根节点也保持探索（不要一开始就只盯着网络最看好的那一两手），在根节点的先验概率上加入 **Dirichlet 噪声**：

```python
def add_dirichlet_noise(self, alpha, epsilon):
    noise = np.random.dirichlet([alpha] * len(actions))
    child.prior = (1 - epsilon) * child.prior + epsilon * noise
```

这相当于"故意把一些注意力分给网络本来不看好的位置"，增加搜索的多样性，避免训练数据千篇一律。

## 4.10 代码导读

MCTS 有两个实现版本：

| 文件 | 版本 | 用途 |
|------|------|------|
| `mcts/node.py` + `mcts/search.py` | 对象版（MCTSNode） | 逻辑清晰，便于理解 |
| `mcts/flat_tree.py` | 扁平数组版（FlatMCTSTree） | 性能优化，实际训练用 |

对象版用 Python 对象和字典组织树，易读但慢；扁平数组版用预分配的 numpy 数组，快得多。两者的**算法完全相同**，只是数据结构不同。第 06 章会详细讲为什么要做这个优化。

`MCTSSearch` 的核心方法：

```python
class MCTSSearch:
    def _evaluate(self, board):        # 网络评估：返回 (策略, 价值)
    def _get_legal_actions(self, board):  # 合法落子（黑棋过滤禁手）
    def search(self, board):           # 单局 MCTS，返回根节点
    def flat_batch_search(self, boards):  # 批量 MCTS（多局并行评估）
    def get_move(self, board, temp):   # 搜索 + 按访问分布选子
```

## 4.11 动手实验

```python
import torch
from gomonova.game.board import Board, rc_to_pos
from gomonova.nn.network import GomoNovaNet
from gomonova.mcts.search import MCTSSearch

# 加载模型（或用随机初始化的小模型演示）
net = GomoNovaNet(channels=32, num_blocks=2, policy_channels=16, value_channels=8)
net.eval()
device = torch.device("cpu")

# 创建 MCTS 搜索器（10 次模拟，便于快速演示）
searcher = MCTSSearch(net, device, num_simulations=10, use_renju=False)

board = Board()
board.play(rc_to_pos(7, 7))   # 黑下天元
board.play(rc_to_pos(7, 8))   # 白应一手

# 运行 MCTS，看访问分布
root = searcher.search(board, add_noise=True)
dist = root.get_visit_distribution(temperature=1.0)
top3 = dist.argsort()[::-1][:3]
print("MCTS 推荐的前 3 个落子:")
for pos in top3:
    r, c = divmod(int(pos), 15)
    print(f"  ({r},{c})  访问占比 {dist[pos]:.2%}")
```

运行后你会看到 MCTS 把搜索资源集中在几个位置上（访问占比高），这就是它给网络的"标准答案"。

**下一步：** 第 05 章看训练管线如何把这些 MCTS 生成的"标准答案"转化为梯度更新，以及三相训练、损失函数、调参经验。
