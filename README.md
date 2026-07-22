# GomoNova

从零训练的五子棋（连珠规则）AI。完全基于游戏规则通过自我对弈学习，不使用任何棋谱数据，对战时纯模型推理（无搜索）。

## 快速开始：和 AI 对战

### 1. 安装依赖

```bash
pip install torch numpy pyyaml rich click
```

### 2. 启动对战

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
python scripts/train.py --config configs/train_main.yaml
```

训练配置见 `configs/` 目录。训练过程自动保存 checkpoint 到 `checkpoints/best.pt`。

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
│   ├── nn/           # MSAR-Net 网络（多尺度注意力残差）
│   ├── training/     # 自博弈、训练器、评估器、管线
│   ├── inference/    # 纯前向推理（无搜索）
│   ├── cli/          # curses 交互式对战
│   └── utils/        # 配置加载、checkpoint
├── scripts/          # 训练入口、同步脚本
├── tests/            # 单元测试
└── docs/             # 开发日志
```

## 规则说明

采用连珠规则：
- 黑棋先手，有禁手（三三禁手、四四禁手、长连禁手）
- 白棋无禁手，长连也算胜
- 先连成五子者胜
