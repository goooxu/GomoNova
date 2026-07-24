#!/usr/bin/env bash
# Smoke test: 验证完整训练管线 + 性能基准
# 用法: bash scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "========================================="
echo "  GomoNova V2 Smoke Test"
echo "========================================="

# 1. 单元测试
echo ""
echo "[1/4] Running unit tests..."
python -m pytest tests/ -v --tb=short 2>&1 | tail -30

# 2. Smoke train (2 iterations, tiny model)
echo ""
echo "[2/4] Smoke training (2 iters, tiny model)..."
python scripts/train.py --config configs/train_smoke.yaml

# 3. 性能基准 (small model, 5 iters, 测量每阶段耗时)
echo ""
echo "[3/4] Performance benchmark (small model, 5 iters)..."
python scripts/train.py --config configs/train_small.yaml 2>&1 | head -20

# 4. 模型参数量检查
echo ""
echo "[4/4] Model param count..."
python -c "
from gomonova.nn.network import GomoNovaNet
from gomonova.nn.encoder import INPUT_CHANNELS
net = GomoNovaNet()
print(f'Input channels: {INPUT_CHANNELS}')
print(f'Model params: {net.num_params():,}')
print(f'Default config: 192ch / 10blocks / 96pc / 48vc')
"

echo ""
echo "========================================="
echo "  Smoke test complete!"
echo "========================================="
