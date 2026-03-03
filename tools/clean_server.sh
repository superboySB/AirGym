#!/usr/bin/env bash
# Clean remote container/image for a tagged AirGym deployment.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: clean_server.sh sb-RL-172 20260107

Removes the remote AirGym container and image tagged with the given date.
USAGE
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

for cmd in ssh; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command '$cmd' is not available." >&2
    exit 1
  fi
done

IMAGE_NAME="airgym-image"
HOST_SLUG="$(echo "$TARGET_HOST" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_.-]/-/g')"
HOST_SLUG="${HOST_SLUG##[-.]}"
HOST_SLUG="${HOST_SLUG%%[-.]}"
if [ -z "$HOST_SLUG" ]; then
  HOST_SLUG="host"
fi

IMAGE_TAG="train-server-${HOST_SLUG}-${RUN_TAG}"
IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"
CONTAINER_NAME="airgym-sim-${RUN_TAG}"

ssh "$TARGET_HOST" "CONTAINER_NAME='${CONTAINER_NAME}' IMAGE_REF='${IMAGE_REF}' bash -s" <<'REMOTE'
set -euo pipefail

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  if docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
    docker stop "$CONTAINER_NAME" >/dev/null
    echo "已停止远端容器: $CONTAINER_NAME"
  fi
  docker rm "$CONTAINER_NAME" >/dev/null
  echo "已删除远端容器: $CONTAINER_NAME"
else
  echo "远端容器不存在: $CONTAINER_NAME"
fi

if docker images --format '{{.Repository}}:{{.Tag}}' | grep -Fxq "$IMAGE_REF"; then
  docker rmi "$IMAGE_REF" >/dev/null
  echo "已删除远端镜像: $IMAGE_REF"
else
  echo "远端镜像不存在: $IMAGE_REF"
fi
REMOTE
