# 第二章：游戏引擎

## 本章你将学到

- 棋盘如何用 numpy 数组表示
- Zobrist 哈希的原理和用途
- 连珠禁手的检测算法
- D4 对称群如何用于数据增强

## 2.1 棋盘表示

> 源文件：`gomonova/game/board.py`

棋盘是一个 15×15 的网格，每个位置有三种状态：空(0)、黑(1)、白(2)。

```python
import numpy as np

BOARD_SIZE = 15
EMPTY, BLACK, WHITE = 0, 1, 2

class Board:
    def __init__(self):
        self.cells = np.zeros((15, 15), dtype=np.int8)  # 棋盘
        self.current = BLACK      # 当前该谁下
        self.history = []         # 所有落子记录
        self._hash = 0           # Zobrist 哈希
```

**为什么用 `int8`？** 一个字节就够了（只需存 0/1/2），省内存。15×15 = 225 字节表示整个棋盘。

**位置编号：** 用 0-224 的整数表示 225 个位置。`pos = row * 15 + col`。

```python
def pos_to_rc(pos):      # 整数 → (行, 列)
    return divmod(pos, 15)

def rc_to_pos(r, c):     # (行, 列) → 整数
    return r * 15 + c
```

### Zobrist 哈希

> 类比：给每个"某位置放了某颜色"的情况分配一个随机数。棋盘的哈希 = 所有已落子位置对应随机数的异或(XOR)。

```python
# 预计算随机表：[2种颜色, 15行, 15列]
_ZOBRIST = np.random.randint(0, 2**63, size=(2, 15, 15))

# 落子时：异或上对应的随机数
def play(self, pos):
    r, c = pos_to_rc(pos)
    self.cells[r, c] = self.current
    self._hash ^= _ZOBRIST[self.current - 1, r, c]  # O(1) 更新哈希
    self.history.append(pos)
    self.current = WHITE if self.current == BLACK else BLACK
```

**用途：** 快速判断两个棋盘状态是否相同（用于 MCTS 的去重，虽然当前训练不用 MCTS）。

## 2.2 连珠规则

> 源文件：`gomonova/game/rules.py`

### 胜负判定

检查最后落子位置是否在四个方向（横、竖、正斜、反斜）上连成五子：

```python
DIRS = ((0,1), (1,0), (1,1), (1,-1))  # →、↓、↘、↗

def _line_run(board, r, c, dr, dc, player):
    """计算经过 (r,c) 在方向 (dr,dc) 上的连续同色棋子数"""
    count = 1
    # 正方向延伸
    cr, cc = r + dr, c + dc
    while 0 <= cr < 15 and 0 <= cc < 15 and board.cells[cr, cc] == player:
        count += 1
        cr += dr; cc += dc
    # 反方向延伸
    cr, cc = r - dr, c - dc
    while 0 <= cr < 15 and 0 <= cc < 15 and board.cells[cr, cc] == player:
        count += 1
        cr -= dr; cc -= dc
    return count
```

- 黑棋：恰好 5 子为胜（6+ 是禁手）
- 白棋：5 子及以上都算胜

### 禁手检测

禁手检测是规则引擎最复杂的部分。核心思路：

**假设黑棋要下在位置 P，临时放上黑子，然后检查：**

1. **长连禁手：** 任何方向连线 ≥ 6？→ 禁手
2. **四四禁手：** 在 ≥ 2 个方向形成"四"？→ 禁手
3. **三三禁手：** 在 ≥ 2 个方向形成"活三"？→ 禁手

**例外：** 如果这手棋能连成恰好五子，则永远合法（五连优先于一切禁手）。

#### 什么是"四"？

"四"= 再下一手就能连成五。检测方法：扫描所有包含位置 P 的长度为 5 的窗口，如果窗口内有 4 颗黑子 + 1 个空位（无白子），那个空位就是"成五点"。

```python
def _four_threats_placed(board, pos, player, dr, dc):
    """在方向 (dr,dc) 上，找到能成五的空位"""
    threats = set()
    for start in range(-4, 1):  # 窗口起点偏移
        window = [P + start+i 方向上的位置 for i in range(5)]
        if 窗口内黑子==4 and 白子==0:
            threats.add(那个空位)
    return threats
```

- 1 个成五点 = 冲四（对手必须堵）
- 2 个成五点 = 活四（对手堵不住，必胜）

#### 什么是"活三"？

"活三"= 再下一手能变成活四。检测方法是递归的：

```python
def _is_open_three(board, pos, dr, dc):
    """位置 P 在方向 (dr,dc) 上是否形成活三？"""
    # 尝试在附近每个空位 E 放子
    for E in 方向上的空位:
        临时放子到 E
        if E 位置形成了活四（≥2 个成五点）:
            return True  # P 是活三
        撤回 E
    return False
```

> 这就是为什么禁手检测很慢——它需要递归地尝试多个位置。这也是为什么训练时跳过禁手检测（用自由规则）。

## 2.3 D4 对称变换

> 源文件：`gomonova/game/symmetry.py`

正方形有 8 种对称操作（D4 二面体群）：

```
恒等    旋转90°   旋转180°  旋转270°
水平翻转  垂直翻转  主对角线   副对角线
```

**为什么需要它？** 棋盘上同一个局面，旋转/翻转后本质相同。训练时随机应用对称变换，相当于把数据量扩大 8 倍。

```python
# 预计算 8 个置换表：每个表把位置 pos 映射到变换后的位置
TRANSFORMS = [_build_transform(t) for t in range(8)]

# 对策略向量应用变换
def transform_policy(policy, t):
    """把 225 维的策略向量按对称变换重映射"""
    perm = TRANSFORMS[t]
    out = np.empty_like(policy)
    out[perm] = policy  # 新位置[变换后] = 旧策略[变换前]
    return out
```

**在训练中的使用：**

```python
# selfplay.py 中的数据增强
for planes, move, outcome in game_samples:
    t = random.randint(1, 8)  # 随机选一种对称
    aug_planes = transform_board(planes, t)
    aug_move = transform_policy(one_hot(move), t).argmax()
    buffer.add(aug_planes, aug_move, outcome)
```

## 2.4 动手实验

```python
from gomonova.game.board import Board, rc_to_pos
from gomonova.game.rules import is_forbidden

# 构造一个三三禁手局面
board = Board()
board.cells[7, 6] = 1  # 黑子
board.cells[7, 8] = 1  # 黑子
board.cells[6, 7] = 1  # 黑子
board.cells[8, 7] = 1  # 黑子
board.current = 1  # 黑棋下

# (7,7) 同时形成横向活三和纵向活三 → 禁手
pos = rc_to_pos(7, 7)
print(f"(7,7) 是禁手吗？{is_forbidden(board, pos)}")  # True
```
