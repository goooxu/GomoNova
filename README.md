# GomoNova

从零训练的五子棋（连珠规则）AI。完全基于游戏规则，通过**自我对弈 + MCTS 引导训练**学习，不使用任何棋谱数据、不借用开源模型结构、不加载预训练权重。**对战时纯模型推理（无搜索）**——训练时用 MCTS 生成高质量学习目标，把搜索能力蒸馏进网络，对战时一次前向推理即可落子。

> 详细实现原理见 [`docs/technical_report/`](docs/technical_report/README.md)（面向 PyTorch 初学者、无强化学习背景）。

## 快速开始：和 AI 对战

### 1. 安装依赖

```bash
pip install torch numpy pyyaml rich click fastapi uvicorn
```

### 2a. Web 对战（推荐）

```bash
python -m gomonova.web.server --checkpoint checkpoints/best.pt --port 8000
# 浏览器打开 http://localhost:8000
```

图形化棋盘、落子动画、形势评估条、棋谱记录、AI 备选点展示。

### 2b. 命令行对战

```bash
python -m gomonova.cli.play --checkpoint checkpoints/best.pt
```

可选参数：
- `--device cuda` / `--device cpu` — 指定设备（默认自动检测）
- `--color w` — 执白（AI 先手）
- `--config configs/inference.yaml` — 模型配置文件

### 3. 操作方式

进入后是一个交互式棋盘界面，用键盘操作：

| 按键 | 功能 |
|------|------|
| `↑ ↓ ← →` 或 `W A S D` | 移动光标 |
| `Enter` 或 `Space` | 在光标处落子 |
| `u` | 悔棋（撤销你和 AI 各一手） |
| `h` | 提示（显示 AI 推荐的 top-3 位置） |
| `r` | 认输 |
| `n` | 新游戏 |
| `q` | 退出 |

棋盘上 `●` 为黑子，`○` 为白子，绿色闪烁 `◆` 为当前光标位置，最后一手有下划线标记。

## 训练

```bash
# 单 GPU
python scripts/train.py --config configs/train_main.yaml

# 4 GPU DDP（推荐）
torchrun --nproc_per_node=4 scripts/train.py --config configs/train_main.yaml
```

训练分三相：纯策略热身（0–300）→ MCTS 引导 + 自由规则（300–2400）→ MCTS + 连珠规则学禁手（2400–3000）。配置见 `configs/`，过程自动保存 checkpoint 到 `checkpoints/best.pt`。开发机可能过期，用 `scripts/sync_checkpoints.sh` 定时把 checkpoint 写回本机，配合断点续训跨机器衔接。

## 测试

```bash
python -m pytest tests/ -v
```

## 项目结构

```
gomonova/
├── configs/          # 训练/推理配置
├── gomonova/
│   ├── game/         # 棋盘、连珠规则（禁手）、对称变换
│   ├── nn/           # MSAR-Net 网络（多尺度注意力残差）、损失函数
│   ├── mcts/         # MCTS（训练用）：对象版节点、扁平数组树、搜索
│   ├── training/     # 自博弈、共享内存并行 MCTS、回放、训练器、评估器、管线
│   ├── inference/    # 纯前向推理（无搜索）
│   ├── cli/          # curses 交互式对战
│   ├── web/          # FastAPI 后端 + Canvas 前端对弈
│   └── utils/        # 配置加载、checkpoint
├── scripts/          # 训练入口、超参搜索、checkpoint 定时写回
├── tests/            # 单元测试
└── docs/             # 开发日志、技术报告（technical_report/）
```

## 规则说明

采用连珠规则：
- 黑棋先手，有禁手（三三禁手、四四禁手、长连禁手）
- 白棋无禁手，长连也算胜
- 先连成五子者胜
