# 第二章：游戏引擎

## 本章你将学到

- 棋盘如何用 numpy 数组表示，以及为什么用 `int8`
- `play` / `undo` 的实现，以及 Zobrist 哈希的用途
- 胜负判定：黑棋"恰好五连"、白棋"五连以上"
- 连珠禁手的完整判定逻辑（长连、双四、双三）
- D4 对称变换如何把一局棋变成多局训练数据

> 涉及源文件：`gomonova/game/board.py`、`gomonova/game/rules.py`、`gomonova/game/symmetry.py`

游戏引擎是整个系统的地基。它必须**快**（自博弈和 MCTS 每秒要调用成千上万次）且**正确**（禁手判错会直接污染训练数据）。

## 2.1 棋盘表示

棋盘是一个 15×15 的 numpy 数组，用 `int8`（1 字节整数）存储每个交叉点的状态：

```python
EMPTY, BLACK, WHITE = 0, 1, 2

class Board:
    __slots__ = ("cells", "current", "history", "_hash")

    def __init__(self):
        self.cells = np.zeros((15, 15), dtype=np.int8)  # 0=空, 1=黑, 2=白
        self.current = BLACK        # 当前该谁下
        self.history = []           # 落子历史（位置列表）
        self._hash = 0              # Zobrist 哈希
```

**为什么用 `int8` 而不是 `int` 或布尔数组？**

- 比 `int64` 省 8 倍内存，缓存更友好（MCTS 要复制海量棋盘）
- 一个数组同时表达"谁占了这个位置"，比用两个布尔数组（is_black / is_white）更紧凑
- numpy 对 `int8` 的比较运算（`cells == BLACK`）有向量化加速

**位置编码：** 225 个交叉点用 0–224 的整数表示，`pos = r * 15 + c`。两个辅助函数负责转换：

```python
def pos_to_rc(pos):  return divmod(pos, 15)   # 一维 → (行, 列)
def rc_to_pos(r, c): return r * 15 + c         # (行, 列) → 一维
```

一维编码的好处：神经网络的策略输出是一个 225 维向量，每个分量直接对应一个位置，无需二维索引。

###落子与撤销

```python
def play(self, pos):
    r, c = pos_to_rc(pos)
    self.cells[r, c] = self.current          # 放上当前方的子
    self._hash ^= int(_ZOBRIST[self.current - 1, r, c])  # 更新哈希
    self.history.append(pos)
    self._hash ^= int(_ZOBRIST_TURN)         # 翻转回合标记
    self.current = WHITE if self.current == BLACK else BLACK

def undo(self):
    pos = self.history.pop()
    self._hash ^= int(_ZOBRIST_TURN)
    self.current = WHITE if self.current == BLACK else BLACK
    r, c = pos_to_rc(pos)
    self._hash ^= int(_ZOBRIST[self.current - 1, r, c])
    self.cells[r, c] = EMPTY
    return pos
```

`play` 和 `undo` 是严格互逆的：`undo` 精确地 reversing `play` 的每一步（包括哈希）。这一点至关重要——**MCTS 的核心操作就是"试走一步 → 评估 → 撤销"**，每局搜索要做几百次 `play`/`undo` 循环。

### Zobrist 哈希

`_hash` 是棋盘的 Zobrist 哈希值：为每个（颜色, 位置）组合预先随机分配一个 64 位整数，棋盘哈希 = 所有已落子位置对应随机数的异或（XOR）。

```python
_rng = np.random.RandomState(42)   # 固定种子，可复现
_ZOBRIST = _rng.randint(0, 2**63, size=(2, 15, 15), dtype=np.int64)
_ZOBRIST_TURN = _rng.randint(0, 2**63, dtype=np.int64)
```

XOR 的妙处在于它**可逆且可增量更新**：落子时 XOR 一次，撤销时再 XOR 一次同一个数就还原了（`a ^ b ^ b = a`）。所以 `play`/`undo` 只需两次 XOR 就能维护哈希，无需重新扫描整个棋盘。

> Zobrist 哈希的典型用途是检测"重复局面"（同一棋盘是否出现过）。GomoNova 当前没有用它做重复判定，但保留了这套机制，为将来扩展（如 MCTS 的转置表）留了接口。

## 2.2 胜负判定

```python
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))   # 横、竖、撇、捺 四个方向

def _line_run(board, r, c, dr, dc, player):
    """经过 (r,c)、沿 (dr,dc) 方向的 player 连子总数（含自身）。"""
    return 1 + _consecutive(...正向...) + _consecutive(...反向...)

def check_winner_at(board, pos):
    r, c = pos_to_rc(pos)
    player = int(board.cells[r, c])
    for dr, dc in DIRS:
        run = _line_run(board, r, c, dr, dc, player)
        if player == BLACK and run == 5:   return BLACK   # 黑棋恰好五连
        if player == WHITE and run >= 5:   return WHITE   # 白棋五连以上
    return None
```

注意黑白规则的**不对称**：

| | 黑棋 | 白棋 |
|--|------|------|
| 获胜条件 | **恰好** 5 连 | **5 连或以上** |
| 6 连（长连） | 不赢（还是禁手） | 赢 |

`check_winner_at` 只检查"刚落下那一子"是否形成连线，而不是扫描全盘——因为新连线必然经过最新落子，这样每步判定是 O(1) 的（只看 4 个方向），而非 O(全盘)。

## 2.3 禁手判定（连珠规则的核心难点）

禁手只约束黑棋。`is_forbidden` 的逻辑分三步：

```python
def is_forbidden(board, pos):
    r, c = pos_to_rc(pos)
    if board.cells[r, c] != 0:
        return False

    board.cells[r, c] = BLACK            # 临时放上黑子

    # ① 成五优先：若这一子正好连成五子，永远合法
    if makes_five(board, pos, BLACK):
        board.cells[r, c] = 0
        return False

    # ② 否则检查三种禁手
    forbidden = False
    if has_overline(board, pos, BLACK):              forbidden = True   # 长连（6+）
    if not forbidden and count_fours(board, pos, BLACK) >= 2:    forbidden = True   # 双四
    if not forbidden and count_open_threes(board, pos, BLACK) >= 2: forbidden = True   # 双三

    board.cells[r, c] = 0                # 恢复棋盘
    return forbidden
```

**关键设计：先放子、后还原。** 函数临时把黑子放上棋盘，判断完再移走，不破坏外部棋盘状态。这种"假设性落子"模式在整个禁手判定中反复出现。

**① 成五优先**是连珠规则的重要例外：哪怕这一子同时形成了双四或长连，只要它正好连成五子，就是合法的好棋。这避免了"明明能赢却被判禁手"的荒谬情况。

### 三种禁手

**长连（overline）：** 某方向连子数 ≥ 6。

```python
def has_overline(board, pos, player):
    for dr, dc in DIRS:
        if _line_run(board, r, c, dr, dc, player) >= 6:
            return True
    return False
```

**双四（double-four）：** 一子落下同时在 ≥2 个方向形成"四"。"四"指再下一子就能成五的棋形（含冲四、活四）。

`count_fours` 逐方向判断是否形成四。判断方法是扫描所有"包含该位置、长度为 5 的窗口"：若某窗口内有 4 颗己方子、1 个空位、没有对方子，那么这个空位就是"成五点"，说明形成了一个四。

```python
def count_fours(board, pos, player):
    board.cells[r, c] = player
    count = 0
    for dr, dc in DIRS:
        if _four_threats_placed(board, pos, player, dr, dc):  # 该方向有成五点
            count += 1
    board.cells[r, c] = 0
    return count
```

> 细节：对黑棋，`_four_threats_placed` 还会排除"成五后变成长连"的假四——因为长连不算赢，那种"四"是虚的。

**双三（double-three）：** 一子落下同时在 ≥2 个方向形成"活三"。"活三"指再下一子能变成"活四"（有两个成五点的四）的三。

```python
def count_open_threes(board, pos, player):
    board.cells[r, c] = player
    count = 0
    for dr, dc in DIRS:
        if _is_open_three_dir_placed(board, pos, player, dr, dc):
            count += 1
    board.cells[r, c] = 0
    return count
```

`_is_open_three_dir_placed` 的判定很巧妙：它**尝试在该方向附近的每个空位再下一子**，看能否形成"≥2 个成五点"的四。能，就说明原来这是个活三。

```python
def _is_open_three_dir_placed(board, pos, player, dr, dc):
    for offset in range(-5, 6):           # 尝试方向上附近的空位
        er, ec = r + offset*dr, c + offset*dc
        if board.cells[er, ec] != 0: continue
        board.cells[er, ec] = player       # 假设再下一子
        threats = _four_threats_placed(board, ..., dr, dc)
        board.cells[er, ec] = 0            # 还原
        if len(threats) >= 2:              # 能形成活四 → 原来是活三
            return True
    return False
```

这就是禁手判定**慢**的根源：判断一个位置是否禁手，要递归地"假设落子→扫描窗口→再假设落子"。后面第 06 章会看到，这正是 MCTS+连珠阶段的性能瓶颈，以及我们如何应对。

### 合法落子

```python
def is_legal(board, pos):
    if not board.is_empty(pos):
        return False
    if board.current == WHITE:
        return True                        # 白棋无禁手
    return not is_forbidden(board, pos)    # 黑棋需过禁手检查
```

## 2.4 对称变换（D4 群）

正方形棋盘有 8 种对称变换：恒等、3 种旋转（90°/180°/270°）、4 种翻转（水平、垂直、主对角、副对角）。这 8 种变换构成数学上的 **D4 二面体群**。

```python
def _build_transforms():
    # 预计算 8 张置换表，每张是 225 维的 pos → pos' 映射
    transforms.append(_to_perm(r_arr, c_arr))            # 恒等
    transforms.append(_to_perm(c_arr, n - r_arr))        # 旋转 90°
    transforms.append(_to_perm(n - r_arr, n - c_arr))    # 旋转 180°
    transforms.append(_to_perm(n - c_arr, r_arr))        # 旋转 270°
    transforms.append(_to_perm(r_arr, n - c_arr))        # 水平翻转
    transforms.append(_to_perm(n - r_arr, c_arr))        # 垂直翻转
    transforms.append(_to_perm(c_arr, r_arr))            # 主对角翻转
    transforms.append(_to_perm(n - c_arr, n - r_arr))    # 副对角翻转
```

**为什么需要它？** 数据增强。一局棋经过对称变换后，棋局本质相同（胜负、棋形都不变），但对神经网络来说是"新"的输入。把每个训练样本随机做一种对称变换，相当于把训练数据扩充到 8 倍，同时教会网络"棋形与绝对方向无关"——活三横着竖着斜着都是活三。

```python
def transform_policy(policy, t):
    """把 225 维策略向量按变换 t 重映射。"""
    perm = TRANSFORMS[t]
    out = np.empty_like(policy)
    out[perm] = policy
    return out
```

注意策略向量的变换用的是 `out[perm] = policy`（置换），保证变换后的策略仍然指向正确的棋盘位置。第 04 章会看到自博弈如何用这个函数做数据增强。

## 2.5 动手实验

```python
from gomonova.game.board import Board, rc_to_pos
from gomonova.game.rules import is_forbidden, check_winner_at

# 构造一个"双三禁手"局面
board = Board()
# 黑棋在 (7,7)(7,8) 和 (8,7)(9,7) 形成两个活三的雏形
for pos in [rc_to_pos(7,7), rc_to_pos(7,8), rc_to_pos(8,7), rc_to_pos(9,7)]:
    board.cells[pos // 15, pos % 15] = 1   # 直接摆黑子（跳过回合）

# 判断 (7,7) 附近某点是否禁手（需根据实际棋形调整位置）
test = rc_to_pos(7, 6)
print(f"位置 (7,6) 对黑棋是否禁手: {is_forbidden(board, test)}")

# 验证 Zobrist 哈希的可逆性
b = Board()
h0 = b.hash
b.play(rc_to_pos(7, 7)); b.play(rc_to_pos(8, 8))
b.undo(); b.undo()
print(f"play+undo 后哈希还原: {b.hash == h0}")   # True
```

**下一步：** 第 03 章看神经网络如何把棋盘（这 225 个 int8）编码成 16 通道张量，并输出"下哪里"和"谁占优"两个预测。
