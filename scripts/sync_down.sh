#!/usr/bin/env bash
set -euo pipefail
REMOTE="${1:?Usage: sync_down.sh <user@host> [remote_dir]}"
REMOTE_DIR="${2:-~/gomonova}"
mkdir -p "$(dirname "$0")/../checkpoints"
rsync -avz "$REMOTE:$REMOTE_DIR/checkpoints/" "$(dirname "$0")/../checkpoints/"
