#!/usr/bin/env bash
set -euo pipefail
PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_DIR="$PROJ_ROOT/backups"
mkdir -p "$BACKUP_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
tar -czf "$BACKUP_DIR/gomonova_ckpt_$TIMESTAMP.tar.gz" \
    -C "$PROJ_ROOT" checkpoints/ 2>/dev/null || echo "No checkpoints to backup"
echo "Backup saved to $BACKUP_DIR/gomonova_ckpt_$TIMESTAMP.tar.gz"
# Keep only the 5 most recent backups
ls -1t "$BACKUP_DIR"/gomonova_ckpt_*.tar.gz 2>/dev/null | tail -n +6 | xargs -r rm --
