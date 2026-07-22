#!/usr/bin/env bash
set -euo pipefail
REMOTE="${1:?Usage: sync_up.sh <user@host> [remote_dir]}"
REMOTE_DIR="${2:-~/gomonova}"
rsync -avz --delete \
    --exclude '.git' --exclude 'data/' --exclude 'checkpoints/' \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' \
    "$(dirname "$0")/../" "$REMOTE:$REMOTE_DIR/"
