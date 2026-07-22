# 第六章：推理与对战

## 本章你将学到

- 纯前向推理的完整流程
- 推理时如何处理禁手
- 温度参数对棋风的影响
- curses 交互式界面的实现原理

## 6.1 纯前向推理

> 源文件：`gomonova/inference/player.py`

对战时，AI 的决策过程极其简单——**一次前向传播**：

```python
class InferencePlayer:
    @torch.no_grad()  # 不计算梯度，省内存省时间
    def get_move(self, board):
        # 1. 棋盘 → 张量
        planes = board_to_planes(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(device)

        # 2. 一次前向传播
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, value = self.network(x)

        # 3. logits → 概率分布
        policy = softmax(logits.float()).cpu().numpy()[0]  # 225 个概率

        # 4. 过滤禁手（黑棋）
        if board.current == BLACK:
            legal = [m for m in empty_positions if not is_forbidden(board, m)]
            policy[illegal_positions] = 0
            policy /= policy.sum()  # 重新归一化

        # 5. 选子
        if temperature == 0:
            move = argmax(policy)  # 贪心：选概率最高的
        else:
            move = random.choice(policy)  # 按概率采样

        return move, policy, value
```

**没有搜索、没有模拟、没有前瞻。** 网络直接输出"每个位置该下的概率"，选最高的那个。

### 为什么不需要搜索？

传统五子棋 AI（如 AlphaZero）用 MCTS 搜索几百步再决定。GomoNova 不搜索，靠的是：

1. **策略头的全局门控**：让网络自己做"战略规划"（见第三章）
2. **充分训练**：3000 轮自博弈让网络见过足够多的局面
3. **价值头辅助**：虽然不直接用于选子，但训练时帮助网络理解局面好坏

代价是棋力上限可能不如有搜索的 AI，但优势是**极快**（<50ms 一步）。

## 6.2 禁手过滤

训练时用自由规则（无禁手），但对战时必须遵守连珠规则：

```python
# 推理时：黑棋落子前检查禁手
if board.current == BLACK:
    legal = [m for m in board.legal_moves() if is_legal(board, m)]
    # is_legal = 位置为空 + 不是禁手
```

这意味着网络可能"想下"某个位置（概率最高），但如果是禁手，就选概率次高的合法位置。

## 6.3 温度参数

```python
player = InferencePlayer(network, device, temperature=0.0)
```

| 温度 | 行为 | 适用场景 |
|------|------|----------|
| 0 | 永远选概率最高的（贪心） | 正式对战 |
| 0.5 | 大概率选最优，偶尔选次优 | 让 AI 不那么"死板" |
| 1.0 | 完全按概率分布随机选 | 增加多样性 |
| 2.0 | 偏向低概率走法 | 故意下弱（让子） |

## 6.4 curses 交互式界面

> 源文件：`gomonova/cli/play.py`

### 为什么用 curses？

普通 `input()` 只能接收一行文字。curses 可以：
- 捕获单个按键（方向键、字母）
- 在固定位置刷新画面（不滚屏）
- 显示颜色和闪烁效果

### 核心循环

```python
def _game_loop(stdscr, player, human_color):
    board = Board()
    cursor_r, cursor_c = 7, 7  # 光标初始在天元

    while True:
        # 1. 绘制棋盘 + 光标
        _draw_board(stdscr, board, cursor_r, cursor_c, ...)

        # 2. 等待按键
        key = stdscr.getch()

        # 3. 处理按键
        if key == KEY_UP:    cursor_r -= 1
        elif key == KEY_DOWN:  cursor_r += 1
        elif key == KEY_LEFT:  cursor_c -= 1
        elif key == KEY_RIGHT: cursor_c += 1
        elif key == ENTER:
            # 落子 → AI 回应 → 检查胜负
            board.play(cursor_r * 15 + cursor_c)
            ai_move = player.get_move(board)
            board.play(ai_move)
        elif key == 'q':
            break
```

### 棋盘渲染

```python
def _draw_board(win, board, cursor_r, cursor_c, ...):
    for r in range(15):
        for c in range(15):
            if board.cells[r,c] == BLACK:
                win.addstr(y, x, "●", WHITE_BOLD)
            elif board.cells[r,c] == WHITE:
                win.addstr(y, x, "○", YELLOW_BOLD)
            elif r == cursor_r and c == cursor_c:
                win.addstr(y, x, "◆", GREEN_BLINK)  # 光标
            else:
                win.addstr(y, x, "·", DIM)
```

## 6.5 动手实验

```bash
# 在开发机上运行对战
cd ~/gomonova
/tmp/gemsg/venv/bin/python -m gomonova.cli.play \
    --checkpoint /tmp/gemsg/checkpoints/best.pt \
    --device cuda

# 操作：
# 方向键移动光标 → Enter 落子 → 观察 AI 回应
# h 键看 AI 的 top-3 建议
# u 键悔棋
# q 退出
```
