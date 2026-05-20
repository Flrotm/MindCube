#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f scripts/rl/filter_vagen_mindcube_among.py ]]; then
  echo "[ERROR] Run this from the MindCube repo root." >&2
  exit 1
fi

MODE="${MODE:-smoke}" # smoke, full, eval
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mindcube-gemma}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CUDA_DEVICES="${CUDA_DEVICES:-2,3}"
PORT="${PORT:-5000}"
RL_STACK_DIR="${RL_STACK_DIR:-/data/fuccelli/mindcube_runs/rl_stack}"
VAGEN_DIR="${VAGEN_DIR:-$RL_STACK_DIR/VAGEN}"
RUN_ROOT="${RUN_ROOT:-/data/fuccelli/mindcube_runs/rl_runs}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-gemma4-rl-plain-cogmap-among-$MODE}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$RUN_ID}"
BASE_MODEL="${BASE_MODEL:-google/gemma-4-31B-it}"
MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-/data/fuccelli/mindcube_runs/checkpoints/rl_ready/gemma4-31b-sft-plain-cgmap-ffr-out-merged}"
SFT_CKPT_DIR="${SFT_CKPT_DIR:-/data/fuccelli/mindcube_runs/checkpoints/rl_ready/vagen_sft_dir}"
SFT_LORA_PATH="${SFT_LORA_PATH:-}"
SFT_LORA_GLOB="${SFT_LORA_GLOB:-/data/fuccelli/mindcube_runs/checkpoints/sft_full/gemma4-31b-sft-plain-cgmap-ffr-out-full-1epoch-ebs128-vision-safe-lora-*}"
IMAGE_SOURCE="${IMAGE_SOURCE:-$PWD/data/other_all_image}"
IMAGE_LINK="${IMAGE_LINK:-$VAGEN_DIR/vagen/env/crossview/other_all_image}"
TEST_IDS_FILE="${TEST_IDS_FILE:-experiments/samples/MindCube_tinybench_raw_qa_representative_100_seed1337.jsonl}"
EVAL_LIMIT="${EVAL_LIMIT:-100}"
EVAL_SETTINGS="${EVAL_SETTINGS:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1536}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
TEST_FREQ="${TEST_FREQ:-1000000}"
SAVE_FREQ="${SAVE_FREQ:-200}"
N_TRAJECTORY="${N_TRAJECTORY:-8}"
START_SERVER="${START_SERVER:-1}"
SERVER_WAIT_SECONDS="${SERVER_WAIT_SECONDS:-20}"
MERGE_IF_MISSING="${MERGE_IF_MISSING:-1}"
ENABLE_STOP_SEQUENCE_OVERRIDES="${ENABLE_STOP_SEQUENCE_OVERRIDES:-0}"
RESUME_PATH="${RESUME_PATH:-}"

export HF_HOME="${HF_HOME:-/data/fuccelli/mindcube_cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/fuccelli/mindcube_cache/hf/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
export TORCH_HOME="${TORCH_HOME:-/data/fuccelli/mindcube_cache/torch}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-20000000000}"

if [[ "$CUDA_DEVICES" == *" "* ]]; then
  echo "[ERROR] CUDA_DEVICES must be one comma-separated process, e.g. 2,3. Spaces create independent runs." >&2
  exit 1
fi

if [[ ! -d "$VAGEN_DIR" ]]; then
  echo "[ERROR] VAGEN_DIR not found: $VAGEN_DIR" >&2
  echo "[ERROR] Run scripts/bash_scripts/setup_gemma_rl_vagen_server.sh first." >&2
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

mkdir -p "$RUN_DIR"/{logs,data,configs}

python scripts/rl/patch_vagen_crossview_data_file.py --vagen-root "$VAGEN_DIR"

if [[ -d "$IMAGE_SOURCE" ]]; then
  mkdir -p "$(dirname "$IMAGE_LINK")"
  if [[ -e "$IMAGE_LINK" && ! -L "$IMAGE_LINK" ]]; then
    echo "[ERROR] Image link target exists and is not a symlink: $IMAGE_LINK" >&2
    exit 1
  fi
  ln -sfn "$IMAGE_SOURCE" "$IMAGE_LINK"
else
  echo "[ERROR] Missing image source: $IMAGE_SOURCE" >&2
  exit 1
fi

DATA_DIR="$VAGEN_DIR/vagen/env/crossview/MindCube_RL_Data"
AMONG_TRAIN_FILE="$DATA_DIR/crossviewQA_train_cogmap_and_reasoning_plain_among.jsonl"
EVAL_FILE="$DATA_DIR/crossviewQA_tinybench_cogmap_and_reasoning_plain_rep100.jsonl"

python scripts/rl/filter_vagen_mindcube_among.py \
  --data-dir "$DATA_DIR" \
  --output "$AMONG_TRAIN_FILE" \
  | tee "$RUN_DIR/logs/filter_among.json"

AMONG_COUNT="$(wc -l < "$AMONG_TRAIN_FILE" | tr -d ' ')"

EVAL_ARGS=(--data-dir "$DATA_DIR" --output "$EVAL_FILE" --limit "$EVAL_LIMIT")
if [[ -n "$TEST_IDS_FILE" ]]; then
  if [[ ! -f "$TEST_IDS_FILE" ]]; then
    echo "[ERROR] TEST_IDS_FILE not found: $TEST_IDS_FILE" >&2
    exit 1
  fi
  EVAL_ARGS+=(--ids-from "$TEST_IDS_FILE")
fi
if [[ -n "$EVAL_SETTINGS" ]]; then
  # shellcheck disable=SC2206
  SETTINGS_ARRAY=($EVAL_SETTINGS)
  EVAL_ARGS+=(--settings "${SETTINGS_ARRAY[@]}")
fi
python scripts/rl/make_vagen_eval_subset.py "${EVAL_ARGS[@]}" \
  | tee "$RUN_DIR/logs/make_eval_subset.json"

EVAL_COUNT="$(wc -l < "$EVAL_FILE" | tr -d ' ')"

if [[ "$MERGE_IF_MISSING" == "1" && ! -f "$MERGED_MODEL_DIR/config.json" ]]; then
  MERGE_ARGS=(
    --base-model "$BASE_MODEL" \
    --output-dir "$MERGED_MODEL_DIR" \
    --max-memory 0=38GiB \
    --max-memory 1=38GiB
  )
  if [[ -n "$SFT_LORA_PATH" ]]; then
    MERGE_ARGS+=(--adapter-path "$SFT_LORA_PATH")
  else
    MERGE_ARGS+=(--adapter-glob "$SFT_LORA_GLOB")
  fi
  python scripts/rl/merge_gemma_lora_for_rl.py "${MERGE_ARGS[@]}"
fi

if [[ ! -f "$MERGED_MODEL_DIR/config.json" ]]; then
  echo "[ERROR] Merged Gemma checkpoint is missing: $MERGED_MODEL_DIR" >&2
  exit 1
fi

mkdir -p "$SFT_CKPT_DIR/plain_cgmap_ffr_out"
CHECKPOINT_50="$SFT_CKPT_DIR/plain_cgmap_ffr_out/checkpoint-50"
if [[ -e "$CHECKPOINT_50" && ! -L "$CHECKPOINT_50" ]]; then
  echo "[ERROR] Expected checkpoint-50 to be a symlink or absent: $CHECKPOINT_50" >&2
  exit 1
fi
ln -sfn "$MERGED_MODEL_DIR" "$CHECKPOINT_50"
MODEL_PATH="$CHECKPOINT_50"

case "$MODE" in
  smoke)
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-crossview-cogmap_reasoning_plain-gemma-among-smoke}"
    TRAIN_SIZE="${TRAIN_SIZE:-8}"
    TEST_SIZE="${TEST_SIZE:-8}"
    STEPS="${STEPS:-1}"
    THIS_SAVE_FREQ="${THIS_SAVE_FREQ:-1}"
    VAL_ONLY="False"
    VAL_BEFORE_TRAIN="False"
    ;;
  full)
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-crossview-cogmap_reasoning_plain-gemma-among-full}"
    TRAIN_SIZE="${TRAIN_SIZE:-$AMONG_COUNT}"
    TEST_SIZE="${TEST_SIZE:-1}"
    STEPS="${STEPS:-$TOTAL_TRAINING_STEPS}"
    THIS_SAVE_FREQ="${THIS_SAVE_FREQ:-$SAVE_FREQ}"
    VAL_ONLY="False"
    VAL_BEFORE_TRAIN="False"
    ;;
  eval)
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-crossview-cogmap_reasoning_plain-gemma-among-final-eval}"
    TRAIN_SIZE="${TRAIN_SIZE:-1}"
    TEST_SIZE="${TEST_SIZE:-$EVAL_COUNT}"
    STEPS="${STEPS:-1}"
    THIS_SAVE_FREQ="${THIS_SAVE_FREQ:-1000000}"
    VAL_ONLY="True"
    VAL_BEFORE_TRAIN="True"
    if [[ -z "$RESUME_PATH" ]]; then
      RESUME_PATH="$VAGEN_DIR/checkpoints/vagen_crossview_gemma/crossview-cogmap_reasoning_plain-gemma-among-full/global_step_${TOTAL_TRAINING_STEPS}"
    fi
    if [[ ! -d "$RESUME_PATH" ]]; then
      echo "[ERROR] RESUME_PATH checkpoint not found: $RESUME_PATH" >&2
      exit 1
    fi
    ;;
  *)
    echo "[ERROR] MODE must be smoke, full, or eval. Got: $MODE" >&2
    exit 1
    ;;
esac

YAML_PATH="$RUN_DIR/configs/${MODE}_env_config.yaml"
TRAIN_PARQUET="$RUN_DIR/data/train.parquet"
TEST_PARQUET="$RUN_DIR/data/test.parquet"

cat > "$YAML_PATH" <<YAML
env1:
  env_name: crossview
  env_config:
    type: cogmap_and_reasoning_plain
    split: train
    reward_type: base
    data_file: $(basename "$AMONG_TRAIN_FILE")
  train_size: $TRAIN_SIZE
  test_size: 0
env2:
  env_name: crossview
  env_config:
    type: cogmap_and_reasoning_plain
    split: test
    reward_type: base
    data_file: $(basename "$EVAL_FILE")
  train_size: 0
  test_size: $TEST_SIZE
YAML

cat > "$RUN_DIR/gemma_generation_config.json" <<'JSON'
{
  "stop_sequences": ["</answer>"],
  "stop_sequence_window_tokens": 256
}
JSON

COMMON_ARGS=(
  "algorithm.adv_estimator=grpo"
  "algorithm.high_level_gamma=1.0"
  "data.train_files=$TRAIN_PARQUET"
  "data.val_files=$TEST_PARQUET"
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.max_prompt_length=$MAX_PROMPT_LENGTH"
  "data.max_response_length=$MAX_RESPONSE_LENGTH"
  "data.max_trajectory_length=3600"
  "data.image_key=images"
  "data.truncation=left"
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.actor.ppo_mini_batch_size=32"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.kl_loss_coef=0.001"
  "actor_rollout_ref.actor.kl_loss_type=mse"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.fsdp_config.param_offload=True"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=2"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.3"
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"
  "actor_rollout_ref.rollout.enforce_eager=False"
  "actor_rollout_ref.rollout.free_cache_engine=False"
  "actor_rollout_ref.rollout.n=1"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.rollout.top_p=0.95"
  "actor_rollout_ref.rollout.temperature=0.7"
  "critic.optim.lr=1e-5"
  "critic.model.use_remove_padding=True"
  "critic.model.path=$MODEL_PATH"
  "critic.model.enable_gradient_checkpointing=True"
  "critic.ppo_micro_batch_size_per_gpu=1"
  "critic.model.fsdp_config.param_offload=False"
  "critic.model.fsdp_config.optimizer_offload=False"
  "algorithm.kl_ctrl.kl_coef=0.001"
  "trainer.critic_warmup=0"
  "trainer.logger=['console']"
  "trainer.project_name=vagen_crossview_gemma"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  "trainer.n_gpus_per_node=2"
  "trainer.nnodes=1"
  "trainer.save_freq=$THIS_SAVE_FREQ"
  "trainer.test_freq=$TEST_FREQ"
  "trainer.total_training_steps=$STEPS"
  "trainer.remove_previous_ckpt_in_save=True"
  "rollout_manager.max_turns=1"
  "rollout_manager.window_size=5"
  "rollout_manager.use_multi_turn_reward=False"
  "rollout_manager.use_loss_mask=True"
  "rollout_manager.use_gae_mask=True"
  "trainer.val_before_train=$VAL_BEFORE_TRAIN"
  "trainer.val_generations_to_log_to_wandb=0"
  "trainer.val_only=$VAL_ONLY"
  "rollout_manager.n_trajectory=$N_TRAJECTORY"
)

if [[ "$ENABLE_STOP_SEQUENCE_OVERRIDES" == "1" ]]; then
  COMMON_ARGS+=(
    "+actor_rollout_ref.rollout.stop=['</answer>']"
    "+actor_rollout_ref.rollout.val_kwargs.stop=['</answer>']"
  )
fi

if [[ "$MODE" == "eval" ]]; then
  COMMON_ARGS+=("trainer.resume_mode=$RESUME_PATH")
fi

{
  echo "RUN_DIR=$RUN_DIR"
  echo "MODE=$MODE"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "VAGEN_DIR=$VAGEN_DIR"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "AMONG_COUNT=$AMONG_COUNT"
  echo "EVAL_COUNT=$EVAL_COUNT"
  echo "YAML_PATH=$YAML_PATH"
  printf 'TRAIN_COMMAND: python3 -m vagen.trainer.main_ppo'
  printf ' %q' "${COMMON_ARGS[@]}"
  printf '\n'
} | tee "$RUN_DIR/commands.txt"

cd "$VAGEN_DIR"

python -m vagen.env.create_dataset \
  --force_gen \
  --yaml_path "$YAML_PATH" \
  --train_path "$TRAIN_PARQUET" \
  --test_path "$TEST_PARQUET" \
  2>&1 | tee "$RUN_DIR/logs/create_dataset.log"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "$START_SERVER" == "1" ]]; then
  (
    python -m vagen.server.server "server.port=$PORT" use_state_reward=True
  ) > "$RUN_DIR/logs/server.log" 2>&1 &
  server_pid="$!"
  echo "$server_pid" > "$RUN_DIR/server.pid"
  echo "[INFO] Started VAGEN server PID $server_pid on port $PORT"
  sleep "$SERVER_WAIT_SECONDS"
fi

python3 -m vagen.trainer.main_ppo "${COMMON_ARGS[@]}" \
  2>&1 | tee "$RUN_DIR/logs/${MODE}.log"

echo "[INFO] Done. Artifacts are in $RUN_DIR"
