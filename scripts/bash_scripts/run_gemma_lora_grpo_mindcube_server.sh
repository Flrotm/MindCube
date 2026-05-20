#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f scripts/rl/train_gemma4_lora_grpo_mindcube.py ]]; then
  echo "[ERROR] Run this from the MindCube repo root." >&2
  exit 1
fi

MODE="${MODE:-smoke}" # smoke, full, eval
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mindcube-gemma}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CUDA_DEVICES="${CUDA_DEVICES:-2,3}"
RUN_ROOT="${RUN_ROOT:-/data/fuccelli/mindcube_runs/rl_runs}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-gemma4-lora-grpo-plain-cogmap-among-$MODE}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$RUN_ID}"
BASE_MODEL="${BASE_MODEL:-google/gemma-4-31B-it}"
SFT_LORA_PATH="${SFT_LORA_PATH:-/data/fuccelli/mindcube_runs/checkpoints/sft/gemma4/gemma4-31b-sft-plain-cgmap-ffr-out-native-thinking-1epoch-ebs128-gpu23}"
RL_STACK_DIR="${RL_STACK_DIR:-/data/fuccelli/mindcube_runs/rl_stack}"
VAGEN_DIR="${VAGEN_DIR:-$RL_STACK_DIR/VAGEN}"
DATA_DIR="${DATA_DIR:-$VAGEN_DIR/vagen/env/crossview/MindCube_RL_Data}"
IMAGE_ROOT="${IMAGE_ROOT:-$PWD/data}"
TEST_IDS_FILE="${TEST_IDS_FILE:-experiments/samples/MindCube_tinybench_raw_qa_representative_100_seed1337.jsonl}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/crossviewQA_train_cogmap_and_reasoning_plain_among.jsonl}"
EVAL_FILE="${EVAL_FILE:-$DATA_DIR/crossviewQA_tinybench_cogmap_and_reasoning_plain_rep100.jsonl}"

TOTAL_STEPS="${TOTAL_STEPS:-200}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1536}"
MAX_PIXELS="${MAX_PIXELS:-90000}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
KL_COEF="${KL_COEF:-0.001}"
SAVE_FREQ="${SAVE_FREQ:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-100}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
ADAPTER_PATH="${ADAPTER_PATH:-}"

export HF_HOME="${HF_HOME:-/data/fuccelli/mindcube_cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/fuccelli/mindcube_cache/hf/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export TORCH_HOME="${TORCH_HOME:-/data/fuccelli/mindcube_cache/torch}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

if [[ "$CUDA_DEVICES" == *" "* ]]; then
  echo "[ERROR] CUDA_DEVICES must be one comma-separated process, e.g. 2,3. Spaces create independent runs." >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV_NAME" ]]; then
  if [[ ! -f "$CONDA_SH" ]]; then
    echo "[ERROR] Missing conda hook: $CONDA_SH" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV_NAME"
fi

mkdir -p "$RUN_DIR"/{logs,checkpoints}

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[ERROR] Missing VAGEN data dir: $DATA_DIR" >&2
  echo "[ERROR] Run scripts/bash_scripts/setup_gemma_rl_vagen_server.sh once to clone VAGEN data." >&2
  exit 1
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
  python scripts/rl/filter_vagen_mindcube_among.py \
    --data-dir "$DATA_DIR" \
    --output "$TRAIN_FILE" \
    | tee "$RUN_DIR/logs/filter_among.json"
fi

if [[ ! -f "$EVAL_FILE" ]]; then
  EVAL_ARGS=(--data-dir "$DATA_DIR" --output "$EVAL_FILE" --limit "$EVAL_LIMIT")
  if [[ -n "$TEST_IDS_FILE" && -f "$TEST_IDS_FILE" ]]; then
    EVAL_ARGS+=(--ids-from "$TEST_IDS_FILE")
  fi
  python scripts/rl/make_vagen_eval_subset.py "${EVAL_ARGS[@]}" \
    | tee "$RUN_DIR/logs/make_eval_subset.json"
fi

ENABLE_THINKING_FLAG="--enable-thinking"
if [[ "$ENABLE_THINKING" == "0" || "$ENABLE_THINKING" == "false" || "$ENABLE_THINKING" == "False" ]]; then
  ENABLE_THINKING_FLAG="--no-enable-thinking"
fi

COMMON_ARGS=(
  --base-model "$BASE_MODEL"
  --image-root "$IMAGE_ROOT"
  --output-dir "$RUN_DIR/checkpoints"
  --quantization 8bit
  --dtype bf16
  --device-map auto
  --max-memory 0=42GiB
  --max-memory 1=42GiB
  --attn-implementation eager
  --max-response-length "$MAX_RESPONSE_LENGTH"
  --max-pixels "$MAX_PIXELS"
  --learning-rate "$LEARNING_RATE"
  --kl-coef "$KL_COEF"
  --save-freq "$SAVE_FREQ"
  --stop-sequences '</answer>'
  "$ENABLE_THINKING_FLAG"
)

case "$MODE" in
  smoke)
    TRAIN_LIMIT="${TRAIN_LIMIT:-1}"
    TOTAL_STEPS="${SMOKE_STEPS:-1}"
    NUM_GENERATIONS="${SMOKE_NUM_GENERATIONS:-2}"
    python scripts/rl/train_gemma4_lora_grpo_mindcube.py \
      --mode train \
      --sft-adapter-path "$SFT_LORA_PATH" \
      --train-file "$TRAIN_FILE" \
      --train-limit "$TRAIN_LIMIT" \
      --total-steps "$TOTAL_STEPS" \
      --train-batch-size "$TRAIN_BATCH_SIZE" \
      --num-generations "$NUM_GENERATIONS" \
      "${COMMON_ARGS[@]}" \
      2>&1 | tee "$RUN_DIR/logs/smoke.log"
    ;;
  full)
    python scripts/rl/train_gemma4_lora_grpo_mindcube.py \
      --mode train \
      --sft-adapter-path "$SFT_LORA_PATH" \
      --train-file "$TRAIN_FILE" \
      --total-steps "$TOTAL_STEPS" \
      --train-batch-size "$TRAIN_BATCH_SIZE" \
      --num-generations "$NUM_GENERATIONS" \
      "${COMMON_ARGS[@]}" \
      2>&1 | tee "$RUN_DIR/logs/full.log"
    ;;
  eval)
    if [[ -z "$ADAPTER_PATH" ]]; then
      echo "[ERROR] MODE=eval requires ADAPTER_PATH=/path/to/rl/checkpoint/global_step_N" >&2
      exit 1
    fi
    python scripts/rl/train_gemma4_lora_grpo_mindcube.py \
      --mode eval \
      --sft-adapter-path "$ADAPTER_PATH" \
      --eval-file "$EVAL_FILE" \
      --eval-limit "$EVAL_LIMIT" \
      --predictions-file "$RUN_DIR/predictions.jsonl" \
      --metrics-file "$RUN_DIR/metrics.json" \
      "${COMMON_ARGS[@]}" \
      2>&1 | tee "$RUN_DIR/logs/eval.log"
    ;;
  *)
    echo "[ERROR] MODE must be smoke, full, or eval. Got: $MODE" >&2
    exit 1
    ;;
esac

echo "[INFO] Done. Artifacts are in $RUN_DIR"
