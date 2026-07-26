# 第一章：项目概述

## 本章你将学到

- GomoNova 的设计约束，以及为什么选择这些约束
- 整体训练方案：自博弈 + MCTS 引导 → 训练 → 评估 → 晋升
- 项目目录结构和各模块职责
- 模块之间的依赖关系和一条关键的架构纪律

## 1.1 什么是 GomoNova

GomoNova 是一个五子棋（连珠规则）AI。它有四条硬性设计约束：

1. **不使用任何棋谱数据** — 不看人类怎么下棋，完全靠自己摸索
2. **原创网络架构** — 不复制 AlphaGo / AlphaZero / KataGo 的结构
3. **从零训练** — 不加载任何预训练权重
4. **对战时纯推理** — 落子 100% 由神经网络一次前向推理决定，不做任何搜索，也不加任何规则干预（模型若下在禁手位，直接判负）

需要特别澄清一个常见的误解：**"对战时不搜索"不等于"训练时也不搜索"**。GomoNova 在**训练阶段**大量使用蒙特卡洛树搜索（MCTS）来生成高质量的学习目标——这正是它棋力远超"纯自模仿"方案的关键。第 04 章会详细解释这个"训练用搜索、对战纯推理"的设计。

> 一句话概括：AI 唯一的"老师"是游戏规则本身，而 MCTS 是帮它把规则"想得更深"的放大镜。赢了好、输了坏，反复对弈中自己总结经验。

## 1.2 连珠规则简介

五子棋的基本规则很简单：15×15 棋盘，黑白交替落子，先连成五子者胜。

连珠（Renju）是五子棋的竞技规则，对黑棋有额外限制（**禁手**）：

| 禁手类型 | 含义 |
|----------|------|
| 三三禁手 | 一子落下同时形成两个"活三" |
| 四四禁手 | 一子落下同时形成两个"冲四/活四" |
| 长连禁手 | 形成六子或以上的连线 |

白棋无禁手，且长连（六子以上）也算胜。**唯一例外：** 黑棋若一子落下正好连成五子，即使同时触发其他禁手 pattern 也不算禁手——成五优先。

> 为什么需要禁手？因为黑棋先手优势太大，不加限制黑棋必胜。禁手平衡了先后手。

禁手的判定逻辑相当复杂（需要递归地判断"活三""冲四"），这是第 02 章的重点，也是后面性能优化绕不开的话题。

## 1.3 整体方案

GomoNova 的训练是一个循环，每一轮做三件事：

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   ① 自博弈 + MCTS          ② 训练            ③ 评估          │
│   网络自己和自己下棋，      用对弈产生的        新模型 vs       │
│   开局用 MCTS "深思"       数据更新权重        当前最佳模型    │
│   生成高质量学习目标                                          │
│                                                              │
│        │                     │                  │            │
│        ▼                     ▼                  ▼            │
│   经验回放池 ──────→ 梯度更新 ──────→ 胜率 ≥ 55%? ──→ 晋升    │
│                                                              │
└──────────────────────── 重复 3000 轮 ────────────────────────┘
```

**① 自博弈 + MCTS（Self-Play）：** 当前网络执黑和执白对弈。在开局的前几手，用 MCTS 做几十次模拟搜索，得到一个"访问分布"（哪些位置被搜索得最多）——这比网络自己的直觉更可靠，作为网络要学习的"标准答案"。之后的棋步用网络纯推理快速下完。每局产生约 50 个训练样本。

**② 训练（Train）：** 从经验回放池抽样。对 MCTS 生成的样本，用 **KL 散度**让网络逼近 MCTS 的访问分布；对纯推理的样本，用**结果加权交叉熵**（赢的棋步多学、输的少学）。同时用 MSE 训练价值网络判断局面优劣。

**③ 评估（Eval）：** 新模型和当前最佳模型对弈 100 局，胜率 ≥ 55% 则晋升为新的最佳模型。

这个循环重复 3000 轮，模型从完全随机逐渐学会进攻、防守、做杀。整个训练还分为**三个阶段**（先纯策略热身、再 MCTS+自由规则、最后 MCTS+连珠规则学禁手），详见第 05 章。

## 1.4 项目目录结构

```
gomonova/
├── configs/                  # 配置文件（YAML）
│   ├── base.yaml             # 基础配置（棋盘尺寸、规则）
│   ├── train_main.yaml       # 正式训练配置
│   ├── train_small.yaml      # 小规模调试配置
│   ├── train_smoke.yaml      # 冒烟测试配置（2 轮）
│   └── inference.yaml        # 推理/对战配置
│
├── gomonova/                 # 核心代码
│   ├── game/                 # 游戏引擎（纯逻辑，不依赖上层）
│   │   ├── board.py          # 棋盘状态管理 + Zobrist 哈希
│   │   ├── rules.py          # 连珠规则（禁手判定、胜负判定）
│   │   └── symmetry.py       # D4 对称变换（8 种）
│   │
│   ├── nn/                   # 神经网络
│   │   ├── encoder.py        # 棋盘 → 16 通道张量
│   │   ├── blocks.py         # MSAR 残差块 + SE 注意力
│   │   ├── network.py        # 完整网络（GomoNovaNet）
│   │   └── losses.py         # 损失函数（KL + CE + 价值 MSE）
│   │
│   ├── mcts/                 # 蒙特卡洛树搜索（仅训练用）
│   │   ├── node.py           # MCTS 节点（对象版）
│   │   ├── search.py         # MCTS 搜索 + 批量搜索
│   │   └── flat_tree.py      # 扁平数组树（性能优化版）
│   │
│   ├── training/             # 训练管线
│   │   ├── selfplay.py       # 自博弈生成数据（含 MCTS 集成）
│   │   ├── parallel_mcts.py  # 共享内存并行 MCTS
│   │   ├── replay.py         # 经验回放缓冲（预分配数组）
│   │   ├── trainer.py        # 梯度更新
│   │   ├── evaluator.py      # 模型评估（批量对弈）
│   │   └── pipeline.py       # 主循环（三相训练 + DDP）
│   │
│   ├── inference/            # 推理（无搜索）
│   │   └── player.py         # 纯前向推理选子
│   │
│   ├── cli/                  # 命令行对战
│   │   └── play.py           # curses 交互式界面
│   │
│   ├── web/                  # Web 对弈界面
│   │   ├── server.py         # FastAPI 后端
│   │   └── index.html        # Canvas 前端（单文件）
│   │
│   └── utils/                # 工具
│       ├── config.py         # 配置加载（含继承合并）
│       └── checkpoint.py     # 模型保存/加载
│
├── scripts/
│   ├── train.py              # 训练入口
│   └── hparam_search.py      # 超参数搜索工具
│
├── tests/                    # 单元测试
└── docs/
    ├── development_log.md    # 开发日志
    └── technical_report/     # 本技术报告
```

## 1.5 模块依赖关系

```
game/board.py ←── nn/encoder.py ←── mcts/search.py ←── training/selfplay.py
      ↑                                    ↓                     ↓
game/rules.py ←── inference/player.py   training/parallel_mcts.py
      ↑                                    ↓                     ↓
game/symmetry.py ←── training/selfplay  mcts/flat_tree.py   training/replay.py
                                                                 ↓
nn/network.py ←── training/pipeline.py ──→ training/trainer.py
(nn/blocks.py)        ↓                    (nn/losses.py)
                training/evaluator.py
```

**核心依赖规则：**

- `game` 层不依赖任何上层模块（纯游戏逻辑）
- `nn` 层只依赖 `game`（读取棋盘状态）
- `mcts` 层依赖 `game` + `nn`（搜索需要规则和网络评估）
- `training` 层依赖 `game` + `nn` + `mcts`（生成数据 + 训练网络）
- `inference` 层只依赖 `game` + `nn`，**绝不导入 training 或 mcts**

最后一条是一条刻意维护的**架构纪律**：推理路径（对战时走的代码）在物理上不可能调用搜索算法。这从代码结构上保证了"对战纯推理"的约束不会被意外破坏——不是靠自觉，而是靠 import 关系根本做不到。

## 1.6 动手实验

```python
# 创建一个棋盘，下几手棋，观察状态变化
from gomonova.game.board import Board, rc_to_pos

board = Board()
board.play(rc_to_pos(7, 7))   # 黑棋下天元 (H8)
board.play(rc_to_pos(7, 8))   # 白棋下 I8
print(board)                   # 打印棋盘（X=黑, O=白）
print(f"当前该谁下: {'黑' if board.current == 1 else '白'}")
print(f"已下 {board.move_count()} 手")

# 撤销一手
board.undo()
print(f"撤销后下了 {board.move_count()} 手")
```

运行后你会看到棋盘上天元位置有一个 `X`（黑子），以及当前轮到白棋。`play` / `undo` 是整个系统最基础的操作——无论自博弈、MCTS 还是评估，都建立在它们之上。

**下一步：** 第 02 章深入游戏引擎，看棋盘如何表示、禁手如何判定、对称变换如何把一局棋变成八局训练数据。
