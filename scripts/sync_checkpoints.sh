#!/bin/bash
# Periodically sync training checkpoints from the dev machine to local.
#
# best.pt is fetched atomically (scp to .tmp then mv) so a mid-write
# transfer never corrupts the local copy.  History snapshots (model_*.pt)
# are written once by the trainer, so a plain scp is safe.  Connection
# failures (machine down) are ignored and retried next cycle.
#
# Usage: scripts/sync_checkpoints.sh [dev_host] [interval_sec]
set -u
DEV="${1:-gemsg@10.85.120.20}"
INTERVAL="${2:-600}"
REMOTE_DIR=/tmp/gomonova/checkpoints
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/checkpoints"
SCP_OPTS="-o ConnectTimeout=15 -o StrictHostKeyChecking=no"

mkdir -p "$LOCAL_DIR"
echo "[$(date)] syncing $DEV:$REMOTE_DIR -> $LOCAL_DIR every ${INTERVAL}s"

while true; do
  # best.pt: atomic update
  if scp $SCP_OPTS "$DEV:$REMOTE_DIR/best.pt" "$LOCAL_DIR/.best.pt.tmp" 2>/dev/null; then
    mv "$LOCAL_DIR/.best.pt.tmp" "$LOCAL_DIR/best.pt"
  fi
  # history snapshots: written once, plain copy
  scp $SCP_OPTS "$DEV:$REMOTE_DIR"/model_*.pt "$LOCAL_DIR/" 2>/dev/null
  sleep "$INTERVAL"
done
