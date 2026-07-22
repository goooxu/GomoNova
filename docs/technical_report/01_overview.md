# 第一章：项目概述

## 本章你将学到

- GomoNova 的设计约束和为什么选择这些约束
- 整体训练方案：自博弈 → 训练 → 评估 → 晋升
- 项目目录结构和模块职责
- 各模块之间的依赖关系

## 1.1 什么是 GomoNova

GomoNova 是一个五子棋（连珠规则）AI。它的特殊之处在于：

1. **不使用任何棋谱数据** — 不看人类怎么下棋，完全靠自己摸索
2. **对战时不搜索** — 不做 MCTS（蒙特卡洛树搜索），只靠神经网络一次前向推理就决定落子
3. **原创网络架构** — 不复制 AlphaGo/AlphaZero/KataGo 的结构
4. **从零训练** — 不使用任何预训练权重

这意味着 AI 唯一的"老师"就是游戏规则本身：赢了好，输了坏，反复对弈中自己总结经验。

## 1.2 连珠规则简介

五子棋的基本规则很简单：15×15 棋盘，黑白交替落子，先连成五子者胜。

连珠（Renju）是五子棋的竞技规则，对黑棋有额外限制（禁手）：

| 禁手类型 | 含义 |
|----------|------|
| 三三禁手 | 一子落下同时形成两个活三 |
| 四四禁手 | 一子落下同时形成两个冲四/活四 |
| 长连禁手 | 形成六子或以上的连线 |

白棋无禁手，且长连（六子以上）也算胜。

> 为什么需要禁手？因为黑棋先手优势太大，不加限制黑棋必胜。禁手平衡了先后手。

## 1.3 整体方案

GomoNova 的训练遵循一个循环：

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   ① 自博弈          ② 训练          ③ 评估     │
│   网络自己和自己    用对弈结果更新    新模型 vs   │
│   下 512 局棋      网络权重         最佳模型    │
│                                                 │
│        │                │               │       │
│        ▼                ▼               ▼       │
│   经验回放池 ──→ 梯度更新 ──→ 胜率≥55%? ──→ 晋升│
│                                                 │
└─────────────────────── 重复 3000 轮 ────────────┘
```

**① 自博弈（Self-Play）：** 当前网络执黑和执白各下一局，用温度采样增加探索。每局产生 ~50 个训练样本（每个落子位置是一个样本）。

**② 训练（Train）：** 从经验回放池中抽样，用"结果加权交叉熵"更新策略网络，用 MSE 更新价值网络。

**③ 评估（Eval）：** 新模型和当前最佳模型对弈 50 局。胜率 ≥ 55% 则晋升为新的最佳模型。

这个循环重复 3000 轮，模型从完全随机逐渐学会进攻、防守、做棋。

## 1.4 项目目录结构

```
gomonova/
├── configs/              # 配置文件（YAML）
│   ├── base.yaml         # 基础配置
│   ├── train_main.yaml   # 正式训练配置
│   └── inference.yaml    # 推理/对战配置
│
├── gomonova/             # 核心代码
│   ├── game/             # 游戏引擎
│   │   ├── board.py      # 棋盘状态管理
│   │   ├── rules.py      # 连珠规则（禁手判定）
│   │   └── symmetry.py   # D4 对称变换
│   │
│   ├── nn/               # 神经网络
│   │   ├── encoder.py    # 棋盘 → 8通道张量
│   │   ├── blocks.py     # MSAR 残差块 + SE 注意力
│   │   ├── network.py    # 完整网络（GomoNovaNet）
│   │   └── losses.py     # 损失函数
│   │
│   ├── training/         # 训练管线
│   │   ├── selfplay.py   # 自博弈生成数据
│   │   ├── replay.py     # 经验回放缓冲
│   │   ├── trainer.py    # 梯度更新
│   │   ├── evaluator.py  # 模型评估
│   │   └── pipeline.py   # 主循环
│   │
│   ├── inference/        # 推理（无搜索）
│   │   └── player.py     # 纯前向推理选子
│   │
│   ├── cli/              # 命令行对战
│   │   ├── play.py       # curses 交互式界面
│   │   └── render.py     # 棋盘渲染
│   │
│   └── utils/            # 工具
│       ├── config.py     # 配置加载
│       └── checkpoint.py # 模型保存/加载
│
├── scripts/
│   └── train.py          # 训练入口
│
└── tests/                # 单元测试
```

## 1.5 模块依赖关系

```
game/board.py ←── nn/encoder.py ←── training/selfplay.py
      ↑                                    ↓
game/rules.py ←── inference/player.py    training/replay.py
      ↑                                    ↓
game/symmetry.py ←── training/selfplay   training/trainer.py
                                           (nn/losses.py)
                                             ↓
nn/network.py ←── training/pipeline.py ──→ training/evaluator.py
(nn/blocks.py)
```

**核心依赖规则：**
- `game` 层不依赖任何上层模块（纯游戏逻辑）
- `nn` 层只依赖 `game`（读取棋盘状态）
- `training` 层依赖 `game` + `nn`（生成数据 + 训练网络）
- `inference` 层只依赖 `game` + `nn`（**绝不导入 training 或 mcts**）

最后一条规则保证了推理路径的纯净：对战时不可能意外调用搜索算法。

## 1.6 动手实验

```python
# 创建一个棋盘，下几手棋，观察状态变化
from gomonova.game.board import Board, rc_to_pos

board = Board()
board.play(rc_to_pos(7, 7))   # 黑棋下天元 (H8)
board.play(rc_to_pos(7, 8))   # 白棋下 I8
print(board)                   # 打印棋盘
print(f"当前该谁下: {'黑' if board.current == 1 else '白'}")
print(f"已下 {board.move_count()} 手")
```
