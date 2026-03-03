#!/usr/bin/env bash
# Downloads training outputs from the specified remote host into the local workspace.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: download_from_server.sh sb-RL-172 20260107

Downloads /data/nvme_data/dzp_is_sb/AirGym-<date>/runs from the remote host into the current directory
as runs-<host>-<date>.
EOF
  exit 1
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
fi

if [ "$#" -ne 2 ]; then
  usage
fi

TARGET_HOST="$1"
RUN_TAG="$2"
case "$TARGET_HOST" in
  sb-RL-172)
    ;;
  *)
    echo "Unsupported target host: $TARGET_HOST" >&2
    exit 1
    ;;
esac

if ! [[ "$RUN_TAG" =~ ^[0-9]{8}$ ]]; then
  echo "Invalid run tag: $RUN_TAG (expected YYYYMMDD, e.g., 20260107)" >&2
  exit 1
fi

for cmd in ssh rsync; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command '$cmd' is not available. Please install it (e.g., openssh-client/rsync) or run this script on the host." >&2
    exit 1
  fi
done

REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/data/nvme_data/dzp_is_sb}"
REMOTE_RUN_DIR="${REMOTE_BASE_DIR%/}/AirGym-${RUN_TAG}/runs"
LOCAL_DIR="runs-${TARGET_HOST}-${RUN_TAG}"

if ! ssh "$TARGET_HOST" "[ -d '$REMOTE_RUN_DIR' ]"; then
  echo "Remote runs directory not found on ${TARGET_HOST}: ${REMOTE_RUN_DIR}" >&2
  echo "Run deployment/training before downloading." >&2
  exit 1
fi

rm -rf "$LOCAL_DIR"
mkdir -p "$LOCAL_DIR"

rsync -avz \
  "$TARGET_HOST:${REMOTE_RUN_DIR}/" \
  "$LOCAL_DIR/"

echo "Runs downloaded to $(pwd)/${LOCAL_DIR}"
