#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-h100}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/lrj/BIG/semantic-code-reuse/outputs/checkpoints}"
LOCAL_ROOT="${LOCAL_ROOT:-outputs/checkpoints}"
CHECKPOINT="${1:-graphcodebert_cls}"

if [[ "${CHECKPOINT}" == "all" ]]; then
  mkdir -p "${LOCAL_ROOT}"
  rsync -av "${REMOTE_HOST}:${REMOTE_ROOT}/" "${LOCAL_ROOT}/"
  echo "Synced all checkpoints from ${REMOTE_HOST}:${REMOTE_ROOT} to ${LOCAL_ROOT}"
else
  mkdir -p "${LOCAL_ROOT}/${CHECKPOINT}"
  rsync -av "${REMOTE_HOST}:${REMOTE_ROOT}/${CHECKPOINT}/" "${LOCAL_ROOT}/${CHECKPOINT}/"
  echo "Synced ${CHECKPOINT} from ${REMOTE_HOST}:${REMOTE_ROOT}/${CHECKPOINT} to ${LOCAL_ROOT}/${CHECKPOINT}"
fi
