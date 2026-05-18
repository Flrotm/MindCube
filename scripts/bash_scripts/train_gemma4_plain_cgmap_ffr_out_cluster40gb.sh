#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f scripts/train_gemma_sft.py || ! -f scripts/convert_to_sft.py ]]; then
  echo "[ERROR] Run this from the MindCube repo root." >&2
  exit 1
fi

TASK_NAME="${TASK_NAME:-plain_cgmap_ffr_out}"
MODEL_ID="${MODEL_ID:-google/gemma-4-31B-it}"
PROCESSOR_ID="${PROCESSOR_ID:-$MODEL_ID}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
MAX_MEMORY="${MAX_MEMORY:-0=38GiB 1=38GiB}"

GENERAL_TRAIN_FILE="${GENERAL_TRAIN_FILE:-data/prompts/general/MindCube_train_${TASK_NAME}.jsonl}"
GEMMA_TRAIN_FILE="${GEMMA_TRAIN_FILE:-data/prompts/training/gemma4/MindCube_train_${TASK_NAME}_gemma_sft.json}"
IMAGE_ROOT="${IMAGE_ROOT:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft/gemma4/${TASK_NAME}}"
LOG_DIR="${LOG_DIR:-logs/sft_training}"
QUANTIZATION="${QUANTIZATION:-8bit}"
RUN_NAME="${RUN_NAME:-gemma4-31b-${TASK_NAME}-${QUANTIZATION}-lora-2x40gb}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
LLM_INT8_SKIP_MODULES="${LLM_INT8_SKIP_MODULES:-lm_head,model.lm_head,vision_tower,model.vision_tower,embed_vision,model.embed_vision,audio_tower,model.audio_tower,embed_audio,model.embed_audio}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-512}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MAX_PIXELS="${MAX_PIXELS:-90000}"
SAVE_STEPS="${SAVE_STEPS:-5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-12}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
REPORT_TO="${REPORT_TO:-none}"
PREFLIGHT="${PREFLIGHT:-1}"
PREFLIGHT_SAMPLES="${PREFLIGHT_SAMPLES:-0}"
LORA_SCOPE="${LORA_SCOPE:-language}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-auto}"
LANGUAGE_MODULE_MARKERS="${LANGUAGE_MODULE_MARKERS:-language_model,text_model,llm,decoder,model.layers}"
LANGUAGE_LORA_SUFFIXES="${LANGUAGE_LORA_SUFFIXES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
EXCLUDE_MODULE_MARKERS="${EXCLUDE_MODULE_MARKERS:-vision,visual,image,audio,projector,mm_projector,multi_modal}"
MODULES_TO_SAVE="${MODULES_TO_SAVE:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
NATIVE_THINKING_TARGETS="${NATIVE_THINKING_TARGETS:-1}"
APPEND_EOS_TOKEN="${APPEND_EOS_TOKEN:-1}"

mkdir -p "$(dirname "$GEMMA_TRAIN_FILE")" "$OUTPUT_DIR" "$LOG_DIR"

if [[ ! -f "$GEMMA_TRAIN_FILE" ]]; then
  if [[ ! -f "$GENERAL_TRAIN_FILE" ]]; then
    echo "[ERROR] Missing training prompt file: $GENERAL_TRAIN_FILE" >&2
    echo "[ERROR] Generate prompts first or set GENERAL_TRAIN_FILE=/path/to/file.jsonl." >&2
    exit 1
  fi

  echo "[INFO] Converting $GENERAL_TRAIN_FILE to Gemma SFT format..."
  python scripts/convert_to_sft.py \
    --input "$GENERAL_TRAIN_FILE" \
    --output "$GEMMA_TRAIN_FILE" \
    --model gemma4
fi

python - "$GEMMA_TRAIN_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"[ERROR] Converted training file does not exist: {path}")
with path.open("r", encoding="utf-8") as handle:
    data = json.load(handle)
if not isinstance(data, list) or not data:
    raise SystemExit(f"[ERROR] Converted training file must be a non-empty JSON list: {path}")
for index, record in enumerate(data[:10]):
    if not isinstance(record, dict) or "messages" not in record or "images" not in record:
        raise SystemExit(f"[ERROR] Converted record {index} is missing messages/images: {path}")
print(f"[INFO] Verified converted training file: {path} ({len(data)} records)")
PY

cmd=(
  python scripts/train_gemma_sft.py
  --task-name "$TASK_NAME"
  --train-file "$GEMMA_TRAIN_FILE"
  --image-root "$IMAGE_ROOT"
  --model-id "$MODEL_ID"
  --processor-id "$PROCESSOR_ID"
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
  --device-map "$DEVICE_MAP"
  --quantization "$QUANTIZATION"
  --llm-int8-skip-modules "$LLM_INT8_SKIP_MODULES"
  --attn-implementation "$ATTN_IMPLEMENTATION"
  --num-train-epochs "$NUM_TRAIN_EPOCHS"
  --learning-rate "$LEARNING_RATE"
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --max-length "$MAX_LENGTH"
  --max-pixels "$MAX_PIXELS"
  --save-steps "$SAVE_STEPS"
  --save-total-limit "$SAVE_TOTAL_LIMIT"
  --logging-steps "$LOGGING_STEPS"
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
  --report-to "$REPORT_TO"
  --lora-scope "$LORA_SCOPE"
  --lora-target-modules "$LORA_TARGET_MODULES"
  --language-module-markers "$LANGUAGE_MODULE_MARKERS"
  --language-lora-suffixes "$LANGUAGE_LORA_SUFFIXES"
  --exclude-module-markers "$EXCLUDE_MODULE_MARKERS"
  --modules-to-save "$MODULES_TO_SAVE"
)

add_bool_arg() {
  local flag="$1"
  local value="$2"
  if [[ "$value" == "1" || "$value" == "true" || "$value" == "True" || "$value" == "yes" || "$value" == "YES" ]]; then
    cmd+=("--${flag}")
  else
    cmd+=("--no-${flag}")
  fi
}

add_bool_arg "enable-thinking" "$ENABLE_THINKING"
add_bool_arg "native-thinking-targets" "$NATIVE_THINKING_TARGETS"
add_bool_arg "append-eos-token" "$APPEND_EOS_TOKEN"

if [[ "$PREFLIGHT" == "0" || "$PREFLIGHT" == "false" || "$PREFLIGHT" == "False" ]]; then
  cmd+=(--no-preflight)
else
  cmd+=(--preflight --preflight-samples "$PREFLIGHT_SAMPLES")
fi

for entry in $MAX_MEMORY; do
  cmd+=(--max-memory "$entry")
done

if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then
  cmd+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi

if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  cmd+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

echo "[INFO] Starting Gemma SFT"
echo "[INFO] GPUs: $GPU_DEVICES"
echo "[INFO] Model: $MODEL_ID"
echo "[INFO] Quantization: $QUANTIZATION"
echo "[INFO] 8-bit skip modules: $LLM_INT8_SKIP_MODULES"
echo "[INFO] Train file: $GEMMA_TRAIN_FILE"
echo "[INFO] Output dir: $OUTPUT_DIR"
echo "[INFO] LoRA scope: $LORA_SCOPE"
echo "[INFO] Enable thinking: $ENABLE_THINKING"
echo "[INFO] Native thinking targets: $NATIVE_THINKING_TARGETS"
echo "[INFO] Append EOS token: $APPEND_EOS_TOKEN"
echo "[INFO] Preflight: $PREFLIGHT (samples: $PREFLIGHT_SAMPLES; 0 means all)"
echo "[INFO] Log file: $LOG_FILE"

CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "${cmd[@]}" 2>&1 | tee "$LOG_FILE"
