# 第七章：推理与对弈界面

## 本章你将学到

- 训练好的模型如何用于对战：纯前向推理（无搜索）
- 落子选择的实现：贪心 / 温度采样、空位掩码
- "禁手判负"在哪里处理（不在推理层）
- CLI 命令行对弈界面
- Web 对弈界面的架构：FastAPI 后端 + Canvas 前端
- 无状态 API 的设计思路

> 涉及源文件：`gomonova/inference/player.py`、`gomonova/cli/play.py`、`gomonova/web/server.py`、`gomonova/web/index.html`

训练的最终目的是让模型能下棋。本章讲模型如何落地为可对弈的程序，以及两个对弈界面（CLI 和 Web）的实现。

## 7.1 纯前向推理：对战的核心

> 源文件：`gomonova/inference/player.py`

回顾第 04 章的核心设计：**训练时用 MCTS 搜索，对战时纯前向推理**。`InferencePlayer` 就是对战时的落子引擎，它的全部逻辑就是"把棋盘送进网络，取输出"：

```python
class InferencePlayer:
    @torch.no_grad()
    def get_move(self, board):
        planes = board_to_planes(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        with torch.amp.autocast(self.device.type, dtype=torch.bfloat16):
            logits, value = self.network(x)        # 一次前向推理
        policy = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
        v = value.item()

        # 只屏蔽已占位置（物理约束，非规则干预）
        empty = board.legal_moves()
        masked = np.zeros(225)
        masked[empty] = policy[empty]
        masked /= masked.sum()

        move = int(np.argmax(masked))              # 选概率最高的位置
        return move, masked, v
```

**注意这里没有任何搜索、没有任何规则判断。** 落子 100% 由网络输出的概率分布决定（选概率最高的位置）。MCTS 的搜索能力已经在训练时"蒸馏"进网络权重，对战时无需再搜索。

### 唯一的"掩码"：空位

代码里唯一的处理是**屏蔽已占位置**——不能把子下在已有棋子的地方。这是物理约束（那个位置已经被占了），不是规则干预。网络在训练时早已学会给空位高概率，这个掩码只是双保险。

> **关键区分：** "屏蔽已占位置"≠"屏蔽禁手"。GomoNova **不做禁手过滤**——如果网络执黑时把最高概率给了一个禁手位，它就真的下在那里，然后**被判负**。这是刻意为之：让模型对自己的输出完全负责（见 7.3）。

### 贪心 vs 温度采样

```python
if self.temperature < 1e-8:
    move = int(np.argmax(masked))         # 贪心：选最优
else:
    tempered = masked ** (1.0 / self.temperature)
    tempered /= tempered.sum()
    move = int(np.random.choice(225, p=tempered))   # 温度采样
```

对战默认用**贪心**（temperature=0，选概率最高的），追求最强着法。温度采样主要用于训练时的探索（第 04 章）。

## 7.2 为什么推理这么快

一次前向推理（62.7M 模型，单个棋盘）约 9ms。一盘棋约 50 手，纯推理总耗时不到 0.5 秒。这就是"对战不搜索"的直接收益——**MCTS 每步要几百次模拟（秒级），纯推理每步只要一次前向（毫秒级）**，快了几个数量级，而棋力因为训练时的蒸馏几乎不损失。

## 7.3 禁手判负：在哪里处理？

一个重要的架构决策：**禁手判定不在推理层，而在对弈界面层。**

```
推理层 (InferencePlayer)：纯模型输出，不管禁手
        ↓ 返回落子位置
对弈界面层 (CLI / Web)：检查这个位置是否禁手
        ↓ 若禁手 → 判模型负
```

**为什么这样分层？**

- 推理层保持"纯净"：它只负责"模型想下哪"，不掺杂规则逻辑。这让推理层可以被任何界面复用，也便于测试"模型本身的输出"。
- 规则判定是"裁判"的职责：禁手判负是游戏规则，属于对弈裁判（界面层）的工作，而非棋手（模型）的工作。

在 Web 后端，这个判定长这样：

```python
# AI 落子后检查
if ai_color == BLACK and is_forbidden(board, move):
    return _result(board, "ai_forbidden", ...)   # AI 下禁手 → 判负

# 人类落子前检查
if human_color == BLACK and is_forbidden(board, pos):
    return _result(board, "human_forbidden", ...)  # 人类下禁手 → 判负
```

**双方一致：** 无论人还是 AI，执黑下禁手都判负。这保证了规则的公平，也让模型必须真正学会避开禁手（而不是依赖过滤帮它挡）。

## 7.4 CLI 命令行对弈

> 源文件：`gomonova/cli/play.py`

CLI 用 `curses` 实现终端交互界面：

```bash
python -m gomonova.cli.play --checkpoint checkpoints/best.pt --color b
```

操作方式：

| 按键 | 功能 |
|------|------|
| 方向键 / WASD | 移动光标 |
| Enter / 空格 | 落子 |
| `h` | 提示（AI 推荐 top-3） |
| `u` | 悔棋 |
| `r` | 认输 |
| `n` | 新对局 |
| `q` | 退出 |

CLI 适合快速验证模型，但终端渲染棋盘不够直观。于是有了 Web 界面。

## 7.5 Web 对弈界面

> 源文件：`gomonova/web/server.py`（后端）、`gomonova/web/index.html`（前端）

Web 界面提供更好的视觉体验：木纹/石板棋盘、光泽棋子、落子动画、形势评估条、棋谱记录。

### 架构：前后端分离

```
浏览器（前端）                    服务器（后端）
┌─────────────────┐             ┌──────────────────┐
│ index.html      │  HTTP/JSON  │ FastAPI          │
│ - Canvas 棋盘   │ ──────────→ │ - /api/play      │
│ - 落子交互      │             │ - /api/hint      │
│ - 动画渲染      │ ←────────── │ - 游戏逻辑       │
│                 │             │ - 模型推理       │
└─────────────────┘             │ - 禁手判定       │
                                └──────────────────┘
```

- **后端（FastAPI）**：加载模型，处理游戏逻辑（重建棋盘、AI 推理、禁手判定、胜负判定），返回 JSON。
- **前端（单文件 HTML）**：Canvas 渲染棋盘和棋子，处理用户交互，调用后端 API。

### 无状态 API 设计

后端是**无状态**的：不保存任何对局状态。每次请求，前端把**完整的落子历史**发过来，后端从头重建棋盘：

```python
class PlayRequest(BaseModel):
    moves: list[int]            # 完整落子历史
    human_move: int | None      # 人类这步要下的位置
    human_color: int            # 人类执什么颜色

@app.post("/api/play")
def play(req: PlayRequest):
    board = _rebuild(req.moves)         # 从历史重建棋盘
    # 处理人类落子（检查禁手、判胜）
    # AI 推理落子（检查禁手、判胜）
    return _result(board, status, ai_move, black_value, top_k, win_line)
```

**为什么无状态？**

- **简单可靠**：后端无需管理会话，重启不丢状态，天然支持多用户。
- **悔棋免费**：前端把 `moves` 截短两个再发请求，就实现了悔棋，后端无需任何配合。
- **状态在客户端**：浏览器保存对局历史，后端只是"给定历史，返回下一步"的纯函数。

### 一次请求的完整流程

```python
def play(req):
    board = _rebuild(req.moves)

    # ① 人类落子（如果有）
    if req.human_move is not None:
        if req.human_color == BLACK and is_forbidden(board, pos):
            return _result(board, "human_forbidden")     # 人下禁手 → 人负
        board.play(pos)
        if check_winner_at(board, pos):
            return _result(board, "human_win", ...)      # 人赢了

    # ② AI 落子（纯模型输出）
    move, policy, value = _player.get_move(board)
    if ai_color == BLACK and is_forbidden(board, move):
        return _result(board, "ai_forbidden")            # AI 下禁手 → AI 负
    board.play(move)
    if check_winner_at(board, move):
        return _result(board, "ai_win", ...)             # AI 赢了

    # ③ 返回 AI 的落子 + top-k 备选 + 形势评估
    return _result(board, "playing", ai_move=move, black_value=..., top_k=...)
```

返回的 `top_k` 是 AI 考虑的前 3 个落子（含概率），前端把它们展示在棋盘上（第 2、3 选作为备选标记），让用户看到模型的"思路"。`black_value` 是形势评估（黑棋优势程度），前端用评估条展示。

### Canvas 渲染

前端用 HTML5 Canvas 绘制棋盘，几个要点：

- **棋盘纹理**：程序化生成（渐变 + 随机纹理线），而非贴图
- **光泽棋子**：径向渐变模拟球面反光，黑子带冷色描边、白子带柔光
- **落子动画**：新棋子从放大到正常尺寸（easeOutBack 缓动，带轻微回弹）
- **最后一手标记**：脉冲光圈
- **胜利连线**：五连的棋子间绘制发光连线
- **备选标记**：AI 的第 2、3 选用虚线圈 + 概率标注

```javascript
function drawStone(x, y, radius, color, alpha) {
  // 阴影
  ctx.arc(x + offset, y + offset, radius); ctx.fill();
  // 球面渐变
  const g = ctx.createRadialGradient(x - r*0.35, y - r*0.35, ...);
  g.addColorStop(0, highlight); g.addColorStop(1, shadow);
  ctx.arc(x, y, radius); ctx.fillStyle = g; ctx.fill();
  // 高光点
  ...
}
```

## 7.6 动手实验

```bash
# 启动 Web 对弈服务
python -m gomonova.web.server --checkpoint checkpoints/best.pt --port 8000

# 浏览器打开 http://localhost:8000
```

试着：
1. 执黑先行，观察 AI 的应对
2. 看 AI 每步的备选标记（第 2、3 选），体会模型的"犹豫"
3. 观察形势评估条随对局的变化
4. 故意把黑棋下成禁手形状，看 AI 是否会避开（或踩雷判负）

```python
# 也可以用 API 直接对弈（无需浏览器）
import requests
r = requests.post("http://localhost:8000/api/play",
                  json={"moves": [], "human_move": 112, "human_color": 1})
print(r.json())   # AI 的应对 + 形势评估 + 备选
```

**下一步：** 第 08 章总结整个项目的成果、完整超参参考、局限与未来方向。
