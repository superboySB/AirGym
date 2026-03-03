#!/usr/bin/env bash
# Automates packaging the local airgym-sim container and deploying it to a remote host.
set -euo pipefail

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

usage() {
  cat <<'EOF'
Usage: deploy_on_server.sh sb-RL-172 20260107 [gpu-device]

Positional arguments:
  target-host   SSH host alias.
  run-tag       Date tag for container/image names (e.g., 20260107).
  gpu-device    Optional docker --gpus value (default: all). Examples: all, device=3,4.

Environment variables:
  LOCAL_PROJECT_PATH   Override the local project directory to sync (default: /home/dzp/projects/AirGym).
  LOCAL_TMP_DIR        Override the temporary directory root (default: /home/dzp/Public).
  REMOTE_BASE_DIR      Override remote base dir (default: /data/nvme_data/dzp_is_sb).
  REMOTE_AIRGYM_CACHE  Override remote AirGym cache root (default: /data/nvme_data/dzp_is_sb/docker/airgym).
EOF
  exit 1
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
fi

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  usage
fi

TARGET_HOST="$1"
RUN_TAG="$2"
GPU_DEVICE_SPEC="${3:-all}"

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

if [ -z "$GPU_DEVICE_SPEC" ]; then
  echo "Invalid gpu-device: cannot be empty" >&2
  exit 1
fi

if [[ "$GPU_DEVICE_SPEC" == device=* && "$GPU_DEVICE_SPEC" == *","* ]]; then
  if [[ "$GPU_DEVICE_SPEC" != \"*\" && "$GPU_DEVICE_SPEC" != \'*\' ]]; then
    GPU_DEVICE_SPEC="\"${GPU_DEVICE_SPEC}\""
  fi
fi

for cmd in docker ssh scp; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Command '$cmd' is required but not found in PATH." >&2
    exit 1
  fi
done

LOCAL_CONTAINER_NAME="airgym-sim"
REMOTE_CONTAINER_NAME="airgym-sim-${RUN_TAG}"
IMAGE_NAME="airgym-image"
HOST_SLUG="$(echo "$TARGET_HOST" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_.-]/-/g')"
HOST_SLUG="${HOST_SLUG##[-.]}"
HOST_SLUG="${HOST_SLUG%%[-.]}"
if [ -z "$HOST_SLUG" ]; then
  HOST_SLUG="host"
fi

IMAGE_TAG="train-server-${HOST_SLUG}-${RUN_TAG}"
IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"
IMAGE_ARCHIVE_NAME="${IMAGE_NAME}_${IMAGE_TAG}.tar"
PROJECT_ARCHIVE_NAME="airgym_project_${HOST_SLUG}_${RUN_TAG}.tar"
LOCAL_PROJECT_PATH="${LOCAL_PROJECT_PATH:-/home/dzp/projects/AirGym}"
PROJECT_DIR_NAME="$(basename "$LOCAL_PROJECT_PATH")"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/data/nvme_data/dzp_is_sb}"
REMOTE_PROJECT_DIR="AirGym-${RUN_TAG}"
REMOTE_IMAGE_ARCHIVE_PATH="${REMOTE_BASE_DIR%/}/${IMAGE_ARCHIVE_NAME}"
REMOTE_PROJECT_ARCHIVE_PATH="${REMOTE_BASE_DIR%/}/${PROJECT_ARCHIVE_NAME}"
REMOTE_AIRGYM_CACHE="${REMOTE_AIRGYM_CACHE:-/data/nvme_data/dzp_is_sb/docker/airgym}"

if [ ! -d "$LOCAL_PROJECT_PATH" ]; then
  echo "Local project path not found: $LOCAL_PROJECT_PATH" >&2
  exit 1
fi

log "Checking whether remote image ${IMAGE_REF} exists on ${TARGET_HOST}..."
REMOTE_IMAGE_PRESENT="$(ssh "$TARGET_HOST" "if docker images --format '{{.Repository}}:{{.Tag}}' | grep -Fxq '$IMAGE_REF'; then echo yes; else echo no; fi")"
UPLOAD_IMAGE="true"
if [ "$REMOTE_IMAGE_PRESENT" = "yes" ]; then
  UPLOAD_IMAGE="false"
  log "Remote image ${IMAGE_REF} already exists; will reuse it and skip image upload."
else
  log "Remote image ${IMAGE_REF} not found; will build and upload image."
fi

log "Preparing remote host ${TARGET_HOST}..."
REMOTE_PREP_OUTPUT="$(ssh "$TARGET_HOST" "IMAGE_ARCHIVE='${IMAGE_ARCHIVE_NAME}' PROJECT_ARCHIVE='${PROJECT_ARCHIVE_NAME}' PROJECT_DIR='${REMOTE_PROJECT_DIR}' REMOTE_BASE_DIR='${REMOTE_BASE_DIR}' UPLOAD_IMAGE='${UPLOAD_IMAGE}' bash -s" <<'REMOTE'
set -euo pipefail

BASE_DIR="${REMOTE_BASE_DIR%/}"
UPLOAD_IMAGE="${UPLOAD_IMAGE:-true}"
mkdir -p "$BASE_DIR"
PROJECT_PATH="${BASE_DIR}/$PROJECT_DIR"
IMAGE_ARCHIVE_PATH="${BASE_DIR}/$IMAGE_ARCHIVE"
PROJECT_ARCHIVE_PATH="${BASE_DIR}/$PROJECT_ARCHIVE"

if [ -e "$PROJECT_PATH" ]; then
  rm -rf "$PROJECT_PATH"
  echo "已删除远端目录: $PROJECT_PATH"
fi

if [ "$UPLOAD_IMAGE" = "true" ] && [ -e "$IMAGE_ARCHIVE_PATH" ]; then
  rm -f "$IMAGE_ARCHIVE_PATH"
  echo "已删除远端镜像归档: $IMAGE_ARCHIVE_PATH"
fi

if [ -e "$PROJECT_ARCHIVE_PATH" ]; then
  rm -f "$PROJECT_ARCHIVE_PATH"
  echo "已删除远端项目归档: $PROJECT_ARCHIVE_PATH"
fi
REMOTE
)"
if [ -n "$REMOTE_PREP_OUTPUT" ]; then
  printf '%s\n' "$REMOTE_PREP_OUTPUT"
fi

if [ "$UPLOAD_IMAGE" = "true" ]; then
  log "Validating that container ${LOCAL_CONTAINER_NAME} is running..."
  if ! docker ps --format '{{.Names}}' | grep -Fxq "$LOCAL_CONTAINER_NAME"; then
    echo "Container '${LOCAL_CONTAINER_NAME}' is not running. Start it before deploying." >&2
    exit 1
  fi

  log "Cleaning local docker image tag ${IMAGE_REF}..."
  if docker images --format '{{.Repository}}:{{.Tag}}' | grep -Fxq "$IMAGE_REF"; then
    docker rmi "$IMAGE_REF" >/dev/null || true
    log "Removed local image tag ${IMAGE_REF}."
  fi

  log "Cleaning dangling local docker images..."
  if docker images -q -f "dangling=true" >/dev/null 2>&1; then
    DANGLING_IDS="$(docker images -q -f "dangling=true")"
    if [ -n "$DANGLING_IDS" ]; then
      docker rmi $DANGLING_IDS >/dev/null || true
      log "Removed local dangling images."
    else
      log "No local dangling images to remove."
    fi
  fi
fi

TMP_BASE="${LOCAL_TMP_DIR:-/home/dzp/Public}"
mkdir -p "$TMP_BASE"
TMP_DIR="$(mktemp -d "${TMP_BASE%/}/deploy_airgym.XXXXXX")"
cleanup() {
  if [ -d "$TMP_DIR" ]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT

PROJECT_ARCHIVE_PATH="${TMP_DIR}/${PROJECT_ARCHIVE_NAME}"

if [ "$UPLOAD_IMAGE" = "true" ]; then
  IMAGE_ARCHIVE_PATH="${TMP_DIR}/${IMAGE_ARCHIVE_NAME}"
  log "Committing running container ${LOCAL_CONTAINER_NAME} to image ${IMAGE_REF}..."
  docker commit "$LOCAL_CONTAINER_NAME" "$IMAGE_REF" >/dev/null

  log "Saving image ${IMAGE_REF} to archive ${IMAGE_ARCHIVE_PATH}..."
  docker save -o "$IMAGE_ARCHIVE_PATH" "$IMAGE_REF"
fi

log "Packaging local project ${LOCAL_PROJECT_PATH} to archive ${PROJECT_ARCHIVE_PATH}..."
tar -cf "$PROJECT_ARCHIVE_PATH" -C "$(dirname "$LOCAL_PROJECT_PATH")" "$PROJECT_DIR_NAME"

if [ "$UPLOAD_IMAGE" = "true" ]; then
  log "Local 镜像归档位置: ${IMAGE_ARCHIVE_PATH}"
fi
log "Local 项目归档位置: ${PROJECT_ARCHIVE_PATH}"

if [ "$UPLOAD_IMAGE" = "true" ]; then
  log "Transferring image archive to ${TARGET_HOST}:${REMOTE_IMAGE_ARCHIVE_PATH}..."
  scp "$IMAGE_ARCHIVE_PATH" "${TARGET_HOST}:${REMOTE_IMAGE_ARCHIVE_PATH}"
else
  log "Skipping image upload (remote image exists)."
fi

log "Transferring project archive to ${TARGET_HOST}:${REMOTE_PROJECT_ARCHIVE_PATH}..."
scp "$PROJECT_ARCHIVE_PATH" "${TARGET_HOST}:${REMOTE_PROJECT_ARCHIVE_PATH}"

if [ "$UPLOAD_IMAGE" = "true" ]; then
  log "Remote 镜像归档位置: ${REMOTE_IMAGE_ARCHIVE_PATH}"
fi
log "Remote 项目归档位置: ${REMOTE_PROJECT_ARCHIVE_PATH}"

log "Loading image, unpacking project, and launching container on ${TARGET_HOST}..."
ssh "$TARGET_HOST" "IMAGE_ARCHIVE='${IMAGE_ARCHIVE_NAME}' PROJECT_ARCHIVE='${PROJECT_ARCHIVE_NAME}' PROJECT_DIR='${PROJECT_DIR_NAME}' REMOTE_PROJECT_DIR='${REMOTE_PROJECT_DIR}' IMAGE_REF='${IMAGE_REF}' REMOTE_BASE_DIR='${REMOTE_BASE_DIR}' REMOTE_CONTAINER_NAME='${REMOTE_CONTAINER_NAME}' GPU_DEVICE_SPEC='${GPU_DEVICE_SPEC}' UPLOAD_IMAGE='${UPLOAD_IMAGE}' REMOTE_AIRGYM_CACHE='${REMOTE_AIRGYM_CACHE}' bash -s" <<'REMOTE'
set -euo pipefail

CONTAINER_NAME="${REMOTE_CONTAINER_NAME}"
GPU_DEVICE_SPEC="${GPU_DEVICE_SPEC:-all}"
UPLOAD_IMAGE="${UPLOAD_IMAGE:-true}"
BASE_DIR="${REMOTE_BASE_DIR%/}"
REMOTE_AIRGYM_CACHE="${REMOTE_AIRGYM_CACHE%/}"

mkdir -p "$BASE_DIR"
mkdir -p "${REMOTE_AIRGYM_CACHE}/cache/pip" \
         "${REMOTE_AIRGYM_CACHE}/cache/ov" \
         "${REMOTE_AIRGYM_CACHE}/cache/glcache" \
         "${REMOTE_AIRGYM_CACHE}/cache/computecache" \
         "${REMOTE_AIRGYM_CACHE}/logs" \
         "${REMOTE_AIRGYM_CACHE}/data"

IMAGE_ARCHIVE_PATH="${BASE_DIR}/$IMAGE_ARCHIVE"
PROJECT_ARCHIVE_PATH="${BASE_DIR}/$PROJECT_ARCHIVE"
EXTRACTED_PROJECT_PATH="${BASE_DIR}/$PROJECT_DIR"
PROJECT_PATH="${BASE_DIR}/$REMOTE_PROJECT_DIR"

if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "远端容器已存在: $CONTAINER_NAME，正在删除以便更新..." >&2
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

if [ "$UPLOAD_IMAGE" = "true" ]; then
  if [ ! -f "$IMAGE_ARCHIVE_PATH" ]; then
    echo "远端镜像归档缺失: $IMAGE_ARCHIVE_PATH" >&2
    exit 1
  fi
  docker load -i "$IMAGE_ARCHIVE_PATH" >/dev/null
else
  if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -Fxq "$IMAGE_REF"; then
    echo "远端镜像不存在: $IMAGE_REF (请先上传镜像或运行 tools/clean_server.sh 清理后重试)" >&2
    exit 1
  fi
fi

rm -rf "$PROJECT_PATH"
tar -xf "$PROJECT_ARCHIVE_PATH" -C "$BASE_DIR"

if [ ! -d "$EXTRACTED_PROJECT_PATH" ]; then
  echo "远端解压后的目录缺失: $EXTRACTED_PROJECT_PATH" >&2
  exit 1
fi

if [ "$PROJECT_DIR" != "$REMOTE_PROJECT_DIR" ]; then
  rm -rf "$PROJECT_PATH"
  mv "$EXTRACTED_PROJECT_PATH" "$PROJECT_PATH"
fi

docker run --name "$CONTAINER_NAME" -itd --gpus "$GPU_DEVICE_SPEC" --network host --ipc=host --privileged \
  -v "${REMOTE_AIRGYM_CACHE}/cache/pip:/root/.cache/pip:rw" \
  -v "${REMOTE_AIRGYM_CACHE}/cache/ov:/root/.cache/ov:rw" \
  -v "${REMOTE_AIRGYM_CACHE}/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
  -v "${REMOTE_AIRGYM_CACHE}/cache/computecache:/root/.nv/ComputeCache:rw" \
  -v "${REMOTE_AIRGYM_CACHE}/logs:/root/.nvidia-omniverse/logs:rw" \
  -v "${REMOTE_AIRGYM_CACHE}/data:/root/.local/share/ov/data:rw" \
  -v "$PROJECT_PATH:/workspace/AirGym" \
  "$IMAGE_REF" /bin/bash

if [ "$UPLOAD_IMAGE" = "true" ]; then
  rm -f "$IMAGE_ARCHIVE_PATH"
fi
rm -f "$PROJECT_ARCHIVE_PATH"
REMOTE

log "Deployment to ${TARGET_HOST} completed successfully."
TMP_DIR_DISPLAY="$TMP_DIR"
cleanup
trap - EXIT
TMP_DIR=""
log "已移除本地临时目录: ${TMP_DIR_DISPLAY}"
log "完成了，可以 docker exec 进去跑代码了。"
