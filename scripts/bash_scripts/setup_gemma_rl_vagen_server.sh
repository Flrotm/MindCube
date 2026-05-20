#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f scripts/rl/patch_vagen_crossview_data_file.py ]]; then
  echo "[ERROR] Run this from the MindCube repo root." >&2
  exit 1
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-mindcube-gemma}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
RL_STACK_DIR="${RL_STACK_DIR:-/data/fuccelli/mindcube_runs/rl_stack}"
VERL_DIR="${VERL_DIR:-$RL_STACK_DIR/verl}"
VAGEN_DIR="${VAGEN_DIR:-$RL_STACK_DIR/VAGEN}"
PULL_LATEST="${PULL_LATEST:-1}"
INSTALL_RL_STACK="${INSTALL_RL_STACK:-1}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
INSTALL_SAFE_RL_DEPS="${INSTALL_SAFE_RL_DEPS:-1}"
VAGEN_PREFLIGHT_PACKAGES="${VAGEN_PREFLIGHT_PACKAGES:-vagen.env.crossview vagen.env.create_dataset vagen.server vagen.trainer}"
RESTORE_TORCH_STACK="${RESTORE_TORCH_STACK:-1}"
IMAGE_SOURCE="${IMAGE_SOURCE:-$PWD/data/other_all_image}"
IMAGE_LINK="${IMAGE_LINK:-$VAGEN_DIR/vagen/env/crossview/other_all_image}"

export HF_HOME="${HF_HOME:-/data/fuccelli/mindcube_cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/fuccelli/mindcube_cache/hf/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export TORCH_HOME="${TORCH_HOME:-/data/fuccelli/mindcube_cache/torch}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" "$RL_STACK_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]]; then
  if [[ ! -f "$CONDA_SH" ]]; then
    echo "[ERROR] Missing conda hook: $CONDA_SH" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV_NAME"
fi

if [[ "$PULL_LATEST" == "1" ]]; then
  git pull --ff-only origin kaggle-runner
fi

check_torch_stack() {
  python - <<'PY'
import json
import sys

try:
    import torch
    import torchvision
    import torchaudio
except Exception as exc:
    print(json.dumps({"torch_stack_error": str(exc)}, indent=2))
    raise

info = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torchaudio": torchaudio.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
}
print(json.dumps(info, indent=2))
if not torch.__version__.startswith("2.6.0+cu118"):
    sys.exit(42)
PY
}

restore_torch_stack() {
  echo "[INFO] Restoring server CUDA stack: torch 2.6.0+cu118"
  pip install --force-reinstall \
    torch==2.6.0+cu118 \
    torchvision==0.21.0+cu118 \
    torchaudio==2.6.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
}

clone_or_update() {
  local url="$1"
  local branch="$2"
  local dest="$3"
  if [[ -d "$dest/.git" ]]; then
    git -C "$dest" fetch origin "$branch"
    git -C "$dest" checkout "$branch"
    git -C "$dest" pull --ff-only origin "$branch"
  else
    git clone -b "$branch" "$url" "$dest"
  fi
}

echo "[INFO] Checking torch stack before RL stack install"
if ! check_torch_stack; then
  if [[ "$RESTORE_TORCH_STACK" == "1" ]]; then
    restore_torch_stack
    check_torch_stack
  else
    echo "[ERROR] Torch stack is not the known working cu118 build." >&2
    exit 1
  fi
fi

clone_or_update "https://github.com/JamesKrW/verl.git" "MindCube" "$VERL_DIR"
clone_or_update "https://github.com/mll-lab-nu/VAGEN.git" "MindCube" "$VAGEN_DIR"

if [[ "$INSTALL_RL_STACK" == "1" ]]; then
  if [[ "$INSTALL_DEPS" == "1" ]]; then
    echo "[INFO] Installing verl/VAGEN with their dependency metadata."
    pip install -e "$VERL_DIR"
    (cd "$VAGEN_DIR" && bash scripts/install.sh)
  else
    echo "[INFO] Installing verl/VAGEN editable with --no-deps to preserve torch."
    pip install --no-deps -e "$VERL_DIR"
    pip install --no-deps -e "$VAGEN_DIR"
  fi
fi

if [[ "$INSTALL_SAFE_RL_DEPS" == "1" ]]; then
  echo "[INFO] Installing small RL runtime deps without touching torch."
  pip install \
    "gym==0.26.2" \
    "gym-sokoban==0.0.6" \
    "gymnasium" \
    "qwen-vl-utils" \
    "mathruler" \
    "matplotlib" \
    "flask" \
    "together" \
    "hydra-core"
fi

echo "[INFO] Checking torch stack after RL stack install"
if ! check_torch_stack; then
  if [[ "$RESTORE_TORCH_STACK" == "1" ]]; then
    restore_torch_stack
    check_torch_stack
  else
    echo "[ERROR] Torch stack changed during setup." >&2
    exit 1
  fi
fi

# shellcheck disable=SC2206
PREFLIGHT_PACKAGE_ARRAY=($VAGEN_PREFLIGHT_PACKAGES)
python scripts/rl/preflight_vagen_imports.py \
  --vagen-root "$VAGEN_DIR" \
  --verl-root "$VERL_DIR" \
  --json-output "$RL_STACK_DIR/vagen_import_preflight.json" \
  --packages "${PREFLIGHT_PACKAGE_ARRAY[@]}"

python scripts/rl/patch_vagen_crossview_data_file.py --vagen-root "$VAGEN_DIR"

if [[ -d "$IMAGE_SOURCE" ]]; then
  mkdir -p "$(dirname "$IMAGE_LINK")"
  if [[ -e "$IMAGE_LINK" && ! -L "$IMAGE_LINK" ]]; then
    echo "[ERROR] Image link target exists and is not a symlink: $IMAGE_LINK" >&2
    exit 1
  fi
  ln -sfn "$IMAGE_SOURCE" "$IMAGE_LINK"
  echo "[INFO] Linked VAGEN images: $IMAGE_LINK -> $IMAGE_SOURCE"
else
  echo "[WARN] Image source not found yet: $IMAGE_SOURCE"
fi

DATA_DIR="$VAGEN_DIR/vagen/env/crossview/MindCube_RL_Data"
if [[ ! -f "$DATA_DIR/crossviewQA_train_cogmap_and_reasoning_plain.jsonl" ]]; then
  echo "[ERROR] Missing VAGEN MindCube RL data under $DATA_DIR" >&2
  exit 1
fi

echo "[INFO] VAGEN RL setup is ready."
echo "[INFO] VAGEN_DIR=$VAGEN_DIR"
echo "[INFO] VERL_DIR=$VERL_DIR"
