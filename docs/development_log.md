# GomoNova 开发日志

## 项目概述

GomoNova 是一个从零训练的五子棋（连珠规则）AI，核心特点：
- 完全基于游戏规则，通过自我对弈强化学习训练，不使用任何棋谱数据
- 原创神经网络架构（MSAR-Net：多尺度注意力残差网络）
- 对战时落子完全依靠模型前向推理，不使用任何搜索算法
- 不使用任何预训练权重，所有参数从零开始训练
- 连珠规则：黑棋有三三禁手、四四禁手、长连禁手；白棋无禁手

## 架构设计

### MSAR-Net（Multi-Scale Attentive Residual Network）

设计动机：五子棋中的威胁跨越多个空间尺度——活三是局部的（3格），跳三是中程的（5格），而做杀（两个威胁的组合）是远程的（7格+）。传统单尺度 3×3 残差网络（如 AlphaZero 架构）需要很深的网络才能捕获远程关系。MSAR-Net 在每个残差块内并行处理三个尺度。

```
输入 (B, 8, 15, 15) — 8个二值通道（颜色相对编码）
  │
  ├─ Stem: Conv2d(8→C, k=3) + BN + Mish + 可学习位置编码
  │
  ├─ Tower: N × MSARBlock
  │    ├─ Branch-3×3: 局部战术（活三、冲四）
  │    ├─ Branch-5×5: 中程棋形（跳三、断连）
  │    ├─ Branch-7×7: 远程结构（做杀、双威胁关系）
  │    └─ Fuse: Concat → Conv1×1 → BN → SE注意力 → 残差连接
  │
  ├─ Policy Head: 局部logits + 全局上下文门控 → 225 logits
  │    （全局门控补偿无搜索的不足，让网络做全局战略决策）
  │
  └─ Value Head: Conv → GAP → FC → tanh → [-1, 1]
```

当前配置：C=96, N=8, 共 12,583,332 参数。

### 输入编码（8通道，颜色相对）

| 通道 | 内容 |
|------|------|
| 0 | 当前方棋子 |
| 1 | 对方棋子 |
| 2-3 | 当前方最近2手 |
| 4-5 | 对方最近2手 |
| 6 | 所有已占位置 |
| 7 | 当前方为黑棋则全1（turn bias） |

颜色相对编码使同一网络可执黑执白，无需两套权重。

### 连珠规则引擎

- 胜负判定：横/竖/斜四方向连成五子（白棋长连也算胜）
- 黑棋禁手：三三禁手、四四禁手、长连禁手
- 禁手判定算法：基于长度5窗口扫描 + 递归活三检测
- 训练时使用自由规则（无禁手）加速自博弈，推理时过滤禁手位置

## 训练方案演变

### 第一版：AlphaZero 式 MCTS 自博弈（已放弃）

最初设计为 AlphaZero 风格：MCTS 生成训练数据 → 网络学习 MCTS 的访问分布。

**放弃原因：** Python 实现的 MCTS 太慢。每步需要 400 次模拟，每次模拟复制棋盘 + 树搜索。512 局自博弈需要数小时，GPU 利用率接近 0%（瓶颈在 Python 游戏逻辑）。

### 第二版：纯自博弈 + REINFORCE（已放弃）

去掉 MCTS，直接用策略网络对弈，用 REINFORCE 策略梯度训练。

**放弃原因：** REINFORCE 方差太大，训练不稳定。策略损失从 -3 爆炸到 -289，即使加了 advantage 归一化和熵正则也无法稳定。

### 第三版：纯自博弈 + 结果加权自模仿（当前方案）

核心思想：只从胜利中学习。

- 自博弈：策略网络 + 温度采样（前30手 T=1.0 探索，之后 T=0.1 精确）
- 策略训练：结果加权交叉熵（赢的走法权重1.0，输的0.1，和棋0.3）
- 价值训练：MSE 预测对局结果
- 稳定性：熵正则（防崩溃）+ 梯度裁剪 + AdamW + cosine LR

**效果：** 训练完全稳定，损失从 3.8 平滑下降到 0.4。

## 性能优化历程

### 问题1：MCTS 太慢
- 原因：Python 逐局串行 + 每步复制棋盘
- 解决：放弃 MCTS，改用纯策略网络自博弈

### 问题2：自博弈仍然慢（290s/4局）
- 原因：禁手检测（is_forbidden）对每个空位做复杂模式分析
- 解决：训练时用自由规则（跳过禁手检测），推理时再过滤

### 问题3：游戏不结束
- 原因：训练用自由规则落子，但胜负判定仍用连珠规则（黑棋长连不判胜）
- 解决：训练时统一用自由规则判胜（5+连珠即胜）

### 问题4：Eval 占 75% 时间（68s/轮）
- 原因：评估对弈逐局串行
- 解决：改为 lockstep 批量并行（50局同时下，每步一次 GPU 调用）
- 效果：68s → 1s

### 问题5：磁盘满导致训练崩溃
- 原因：每10轮保存145MB checkpoint，home目录仅5GB
- 解决：checkpoint 存到大容量分区，改为每100轮保存，定期清理旧文件

## 当前训练配置

```yaml
model:
  channels: 96
  num_blocks: 8
  policy_channels: 48
  value_channels: 24
  # 总参数: 12,583,332

training:
  batch_size: 1024
  lr: 3e-4 (warmup 20轮 → cosine → 1e-5)
  train_steps_per_iter: 128
  total_iters: 3000
  use_amp: true  # fp16 混合精度

selfplay:
  games_per_iter: 512
  temp_threshold: 30  # 前30手高温度探索

eval:
  games: 50
  promote_winrate: 0.55
```

每轮耗时：Self-Play 4s + Train 19s + Eval 1s ≈ 24s
3000 轮预计总时间：~20 小时

## 训练进度记录

| 轮次 | 损失 | 策略损失 | 价值损失 | 备注 |
|------|------|----------|----------|------|
| 1 | 3.80 | 1.99 | 1.01 | 起始 |
| 86 | 0.80 | 0.22 | 0.47 | 快速下降期 |
| 160 | 0.55 | 0.21 | 0.29 | |
| 303 | 0.43 | 0.21 | 0.20 | |
| 346 | 0.56 | 0.20 | 0.33 | |
| 453 | 0.42 | 0.19 | 0.20 | 当前（BF16） |

## 精度方案演变

### 第一版：FP32 主权重 + FP16 autocast + GradScaler
标准 PyTorch AMP 方案。稳定运行，但显存占用较大（FP32 权重 + FP16 激活）。

### 第二版：BF16 主权重（当前方案）
- Conv/Linear 权重：BF16
- BatchNorm 权重/running stats：FP32（CUDA 内核强制要求）
- 训练计算：BF16 autocast
- 无需 GradScaler（BF16 指数范围 = FP32，不会下溢）
- Checkpoint 大小减半（73MB vs 145MB）
- 训练速度略快（21s/轮 vs 22s/轮）

**注意：** BF16 与 BatchNorm 不兼容（PyTorch 的 batch_norm CUDA 内核要求 FP32 权重）。
解决方案：`.bfloat16()` 后将 BN 层 `.float()` 转回 FP32。

### 多 GPU 并行结论
- DataParallel 对 12.5M 模型反而更慢（94s/轮 vs 22s/轮），通信开销大于计算收益
- DataParallel 与 BF16+BN 混合精度兼容（BN 是 FP32 所以不报错）
- 结论：当前模型规模下，单卡是最优选择
- 瓶颈在 Python 游戏逻辑（self-play），不在 GPU 计算

## CLI 方向键选子

重写了 CLI 为 curses 交互式界面：
- 方向键/WASD 移动光标（绿色闪烁 ◆）
- Enter/Space 落子
- u=悔棋, h=提示, r=认输, n=新游戏, q=退出
- 显示 AI top-3 候选和置信度
- 支持 --color w 执白

## 开发机管理经验

1. **Checkpoint 必须定期备份到本地：** 开发机随时可能过期，/tmp 会被清理
2. **Checkpoint 路径不要放在 home 目录：** home 通常只有 5GB，每 10 轮 145MB 很快写满
3. **设置定时备份：** 每 30 分钟自动 rsync best.pt 到本地
4. **训练日志也放在 /tmp：** 避免占用 home 空间
5. **恢复训练：** 用本地 best.pt 上传到新机器的 /tmp/checkpoints/ 即可继续

## 后续优化方向

**已完成（V2/V2.1）：**
- ✅ 引入 MCTS 引导训练（Python 优化版，共享内存并行，6.9× 加速）
- ✅ 更大模型（12.5M → 62.7M）
- ✅ 对手多样性（历史 checkpoint 池，50% 对局）
- ✅ 禁手感知训练（连珠阶段）
- ✅ 强制天元开局（修复走边）

**仍可探索：**
1. **C++/CUDA MCTS：** 当前 MCTS 是 Python 实现，C++ 重写可进一步加速，支持更多模拟次数
2. **开局深化：** 用更深 MCTS（几百次模拟）生成开局数据，强化开局定式（仍从规则生成，不用棋谱）
3. **课程学习：** 先在小棋盘（9×9）训练基本战术，再迁移到 15×15
4. **更大模型 + 更长训练：** 200M+ 模型、10000+ 轮，通常能显著提升棋力
5. **对战时轻量搜索：** 关键局面做极少量模拟补充（会突破"纯推理"约束，需权衡）

## 项目结构

```
gomonova/
├── configs/          # 训练/推理配置
├── gomonova/
│   ├── game/         # 棋盘、连珠规则、对称变换
│   ├── nn/           # MSAR-Net 网络定义、损失函数
│   ├── mcts/         # MCTS：node（对象版）、flat_tree（数组版）、search
│   ├── training/     # 自博弈、parallel_mcts（共享内存并行）、回放、训练器、评估器、管线
│   ├── inference/    # 纯前向推理（无搜索）
│   ├── cli/          # curses 命令行对战
│   ├── web/          # FastAPI 后端 + Canvas 前端对弈
│   └── utils/        # 配置加载、checkpoint
├── scripts/          # 训练入口、超参搜索、checkpoint 定时写回
├── tests/            # 单元测试（77个）
└── docs/             # 开发日志、技术报告（technical_report/）
```

## 使用方法

### 训练
```bash
# 单 GPU
python scripts/train.py --config configs/train_main.yaml

# 4 GPU DDP（推荐）
torchrun --nproc_per_node=4 scripts/train.py --config configs/train_main.yaml
```

### 对战（CLI）
```bash
python -m gomonova.cli.play --checkpoint checkpoints/best.pt
```

### 对战（Web）
```bash
python -m gomonova.web.server --checkpoint checkpoints/best.pt --port 8000
# 浏览器打开 http://localhost:8000
```

### 测试
```bash
python -m pytest tests/ -v
```

---

## V2 棋力提升改造

### 改造目标

将棋力从「初学者」提升到「能赢业余高手」，核心约束：
- 推理时 100% 纯模型输出（无搜索、无禁手过滤、无任何规则干预）
- 训练时允许使用 MCTS 等搜索算法
- 不使用棋谱、开源五子棋模型结构、预训练权重

### 架构改动

#### 模型扩容（12.5M → ~50M）
- channels: 96 → 192, num_blocks: 8 → 10
- policy_channels: 48 → 96, value_channels: 24 → 48
- Policy head 全局门控从标量改为逐位置（Conv1x1 产生 15×15 sigmoid 门）
- Value head 隐层从 128 扩到 256

#### 输入编码增强（8 → 16 通道）
- 通道 0-1：当前方/对方棋子
- 通道 2-7：当前方最近 6 手（原为 2 手）
- 通道 8-13：对方最近 6 手
- 通道 14：占位 | 通道 15：回合偏置

### 训练方法改动

#### MCTS 引导训练（核心改动）
- 重新启用 MCTS（`mcts/search.py`），添加 `use_renju` 参数支持自由/连珠规则
- 选择性 MCTS：仅对每局前 N 手使用 MCTS（默认 10 手），后续用纯策略
- 训练目标从「单个走法 + 交叉熵」升级为「MCTS 访问分布 + KL 散度」
- MCTS 步：`L = KL(mcts_policy ∥ network_policy)`
- 纯策略步：`L = outcome_weighted_CE(played_move)`（保持原有方式）

#### 三阶段训练
| 阶段 | 迭代 | MCTS | 规则 | 目的 |
|------|------|------|------|------|
| 1 热身 | 0-500 | ❌ | 自由 | 快速建立基本棋感 |
| 2 主力 | 500-2200 | ✅ 25 sims | 自由 | MCTS 引导学习战略 |
| 3 禁手 | 2200-3000 | ✅ 25 sims | 连珠 | 学会避免/利用禁手 |

#### 多 GPU DDP 训练
- 支持 `torchrun --nproc_per_node=4` 启动
- 每进程独立跑自博弈，训练时 DDP 同步梯度
- 评估仅在 rank 0 执行
- 单 GPU 也兼容（不用 torchrun 启动即可）

#### 对手多样性
- 每 200 轮保存历史 checkpoint，保留最近 10 个
- 50% 对局 vs 随机历史 checkpoint，防止策略循环

#### Replay Buffer 改造
- 预分配 numpy 数组替代 Python list（消除碎片化）
- 支持存储 MCTS 策略分布：`(planes, mcts_policy, move, outcome)`

### 推理层改动
- 移除黑棋禁手过滤（`is_legal` 屏蔽逻辑）
- 模型输出 → softmax → 仅屏蔽已占位置 → argmax → 直接落子
- 已占位置屏蔽是物理约束，不是规则干预

### 配置变更
```yaml
model: { channels: 192, num_blocks: 10, policy_channels: 96, value_channels: 48 }  # 62.7M 参数
training: { batch_size: 2048, lr: 1e-4, total_iters: 3000, train_steps_per_iter: 32 }
selfplay: { games_per_iter: 512, mcts_sims: 25, mcts_moves: 5 }
phases: { mcts_start: 300, renju_start: 2400 }
```

### 训练命令
```bash
# 单 GPU
python scripts/train.py --config configs/train_main.yaml

# 4 GPU DDP
torchrun --nproc_per_node=4 scripts/train.py --config configs/train_main.yaml

# 或使用 Makefile
make train-ddp NGPU=4
```

---

## V2.1 最终完善：开局修复、性能攻坚与训练完成

### 强制天元开局（修复"开局走边"）

**问题：** V2 模型开局经常下在边角（B1、N1），而非天元。这其实违反连珠规则（黑第一手必须下天元）。根因是纯自博弈无开局引导，走边习惯自我强化；位置编码也学不出足够强的中心偏好。

**解决：** 在 `selfplay.py` 的 `play_games_fast` 和 `play_games_with_mcts` 中，每局开始强制黑第一手下天元（H8 = pos 112），并作为训练样本（`mcts_policy=None`，走 outcome-weighted CE）。模型从训练分布学会"第一手=天元"，对战时纯推理自然下天元，**无需推理时规则干预**。这是"把规则约束转化为训练分布"的做法。

**决策：** 用户选择从头重训（方案 B），而非微调——旧权重已固化走边习惯，且几乎没学过天元开局的应对（微调还需 LR 重新升温）。旧走边模型保留为 `checkpoints/best_edge.pt`，历史快照归档到 `checkpoints/edge_v1/`。

### 性能优化（MCTS 自博弈 6.9× 加速）

V2 重新引入 MCTS 后，朴素实现慢到无法训练（512 局 MCTS 需 105.9s，GPU 大部分时间空等 Python 树遍历）。**关键洞察：瓶颈是 CPU 上的 Python 树遍历，不是 GPU。** 经过系统优化：

| 优化 | 手段 | 效果 |
|------|------|------|
| 批量 GPU 评估 | MCTS 叶节点合并成批量推理 | GPU 利用率大幅提升 |
| 扁平数组树 | `mcts/flat_tree.py`，预分配 numpy 数组替代 Python 对象 | 节点创建 212μs → ~5μs（40×） |
| 共享内存并行 MCTS | `training/parallel_mcts.py`，CPU 多核遍历树 + GPU 集中评估，轮次同步 | 512 局 105.9s → 15.3s |
| 大模型 DDP | 62.7M 模型用 4 卡 DDP（小模型 12.5M 时多卡反而慢 4.5×） | 每轮数据量 4× |

**共享内存轮次同步**是核心：所有 worker 按"轮"推进——每轮各自遍历树一步、把叶位置写入共享内存、递增 ready_count；主进程等所有 worker 就绪后一次批量 GPU 评估、写回结果、递增 gen。相比队列方案（序列化开销大、worker 去同步化导致 GPU 小批量），共享内存零序列化且保持 GPU 大批量。

### Web 对弈界面

新增 `gomonova/web/`：
- **后端**（`server.py`，FastAPI）：无状态 API。`/api/play` 从完整落子历史重建棋盘 → 人类落子禁手判负 → AI 纯推理落子 → AI 禁手判负 → 返回 AI 落子 + top-3 备选 + 形势评估 + 胜利连线。`/api/hint` 返回当前局面 top-3。
- **前端**（`index.html`，单文件 Canvas）：程序化石板棋盘、光泽棋子（径向渐变+高光）、落子动画（easeOutBack）、最后一手脉冲光圈、形势评估条、棋谱记录、AI 备选点虚线标记。

```bash
python -m gomonova.web.server --checkpoint checkpoints/best.pt --port 8000
```

### Checkpoint 自动写回（断点续训保障）

开发机随时可能过期（SLURM 作业结束）。`scripts/sync_checkpoints.sh` 用 `setsid` 后台运行，每 10 分钟把开发机 `/tmp/gomonova/checkpoints/` 的 `best.pt`（scp 到 .tmp 再 mv，**原子写入**避免半截文件）和 `model_*.pt` 拉回本机。配合 checkpoint 断点续训（pipeline 自动从 best.pt 的 iteration 恢复），训练可跨多次机器切换无缝衔接。

> 经验：开发机的 `/tmp` 在机器回收时**有时保留有时清空**（同一台机器重启后 `/tmp` 可能还在）。最可靠的持久存储是本机，故自动写回目标是本机 `checkpoints/`。

### 训练完成

完整训练 3000 轮，三相：

| 阶段 | 迭代 | 内容 |
|------|------|------|
| ① 热身 | 0–300 | 纯策略，自由规则 |
| ② 主力 | 300–2400 | MCTS（25 sims / 前 5 手），自由规则 |
| ③ 禁手 | 2400–3000 | MCTS，连珠规则 |

损失从 **5.97 降至 1.93**（策略 0.78 / 价值 0.65）。模型达到业余强手水平：开局符合天元规则，具备禁手意识，会进攻防守做杀。训练跨越多次开发机切换，靠自动写回 + 断点续训衔接。

### 技术报告

新增 `docs/technical_report/`（8 章 + 导读，面向 PyTorch 初学者、无 RL 背景）：
01 项目概述 · 02 游戏引擎 · 03 神经网络 · 04 自博弈与 MCTS（RL 扫盲）· 05 训练管线（调参经验）· 06 性能优化（实战）· 07 推理与界面 · 08 总结。每章含「本章你将学到」、类比、ASCII 图示、代码片段、「动手实验」。
