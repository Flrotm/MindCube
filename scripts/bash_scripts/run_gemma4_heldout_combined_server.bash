#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f scripts/run_inference.py || ! -f scripts/prepare_heldout_prompts.py ]]; then
  echo "[ERROR] Run this from the MindCube repo root." >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-heldout-combined-gemma4}"
RUN_ROOT="${RUN_ROOT:-/data/fuccelli/mindcube_runs/experiments/runs}"
RUN_DIR="${RUN_ROOT%/}/${RUN_ID}"

export HF_HOME="${HF_HOME:-/data/fuccelli/mindcube_cache/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-/data/fuccelli/mindcube_cache/torch}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME"

HELDOUT_ROOT="${HELDOUT_ROOT:-/data/fuccelli/mindcube_runs/heldout/MindCube_heldout}"
SOURCE_INPUT="${SOURCE_INPUT:-$HELDOUT_ROOT/data/raw/MindCube_heldout.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-$HELDOUT_ROOT}"

MODEL_PATH="${MODEL_PATH:-google/gemma-4-31B-it}"
ROTATION_CONFIG_PATH="${ROTATION_CONFIG_PATH:-experiments/configs/gemma4_31b_cluster40gb_8bit_vision_safe_reasoning_greedy_inference.json}"
SFT_CONFIG_TEMPLATE="${SFT_CONFIG_TEMPLATE:-experiments/configs/gemma4_31b_sft_native_thinking_plain_cgmap_ffr_out_8bit_vision_safe_2gpu_inference.json}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-/data/fuccelli/mindcube_runs/checkpoints/sft/gemma4/gemma4-31b-sft-plain-cgmap-ffr-out-native-thinking-1epoch-ebs128-gpu23}"
SFT_PROCESSOR_PATH="${SFT_PROCESSOR_PATH:-$SFT_ADAPTER_PATH}"
ANSWER_REPAIR_MAX_NEW_TOKENS="${ANSWER_REPAIR_MAX_NEW_TOKENS:-128}"
ANSWER_REPAIR_ENABLE_THINKING="${ANSWER_REPAIR_ENABLE_THINKING:-true}"

MAX_NEW_TOKENS_ROTATION="${MAX_NEW_TOKENS_ROTATION:-6144}"
MAX_NEW_TOKENS_SFT="${MAX_NEW_TOKENS_SFT:-6144}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ROTATION_GPU_GROUPS="${ROTATION_GPU_GROUPS:-2,3}"
AROUND_GPU_GROUPS="${AROUND_GPU_GROUPS:-0,1}"
AMONG_GPU_GROUPS="${AMONG_GPU_GROUPS:-2,3}"
COMBINE_FALLBACK_POLICY="${COMBINE_FALLBACK_POLICY:-none}"

mkdir -p "$RUN_DIR"/{inputs,logs,configs}

if [[ ! -f "$SOURCE_INPUT" ]]; then
  echo "[ERROR] Missing heldout source input: $SOURCE_INPUT" >&2
  exit 1
fi
if [[ ! -d "$IMAGE_ROOT/other_all_image_v2" ]]; then
  echo "[ERROR] IMAGE_ROOT should contain other_all_image_v2/: $IMAGE_ROOT" >&2
  exit 1
fi
if [[ ! -f "$ROTATION_CONFIG_PATH" ]]; then
  echo "[ERROR] Missing rotation config: $ROTATION_CONFIG_PATH" >&2
  exit 1
fi
if [[ ! -f "$SFT_CONFIG_TEMPLATE" ]]; then
  echo "[ERROR] Missing SFT config template: $SFT_CONFIG_TEMPLATE" >&2
  exit 1
fi

ROTATION_RUNTIME_CONFIG_PATH="$RUN_DIR/configs/rotation_inference.json"
python - "$ROTATION_CONFIG_PATH" "$ROTATION_RUNTIME_CONFIG_PATH" "$ANSWER_REPAIR_MAX_NEW_TOKENS" "$ANSWER_REPAIR_ENABLE_THINKING" <<'PY'
import json
import sys
from pathlib import Path

template_path, output_path, repair_tokens, repair_thinking = sys.argv[1:]
config = json.loads(Path(template_path).read_text(encoding="utf-8"))
config["answer_repair_max_new_tokens"] = int(repair_tokens)
config["answer_repair_enable_thinking"] = repair_thinking.lower() in {"1", "true", "yes", "on"}
Path(output_path).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

SFT_CONFIG_PATH="$RUN_DIR/configs/sft_native_thinking_inference.json"
python - "$SFT_CONFIG_TEMPLATE" "$SFT_CONFIG_PATH" "$SFT_ADAPTER_PATH" "$SFT_PROCESSOR_PATH" "$AROUND_GPU_GROUPS" "$AMONG_GPU_GROUPS" "$ANSWER_REPAIR_MAX_NEW_TOKENS" "$ANSWER_REPAIR_ENABLE_THINKING" <<'PY'
import json
import sys
from pathlib import Path

(
    template_path,
    output_path,
    adapter_path,
    processor_path,
    around_gpu_groups,
    among_gpu_groups,
    repair_tokens,
    repair_thinking,
) = sys.argv[1:]


def visible_gpu_count(groups: str) -> int:
    counts = []
    for group in groups.split():
        devices = [device.strip() for device in group.split(",") if device.strip()]
        if devices:
            counts.append(len(devices))
    return max(counts or [1])


config = json.loads(Path(template_path).read_text(encoding="utf-8"))
config["peft_adapter_path"] = adapter_path
if processor_path:
    config["processor_path"] = processor_path
sft_gpu_count = max(visible_gpu_count(around_gpu_groups), visible_gpu_count(among_gpu_groups))
config["device_map"] = "auto" if sft_gpu_count > 1 else {"": 0}
config["max_memory"] = {str(index): "42GiB" for index in range(sft_gpu_count)}
config["answer_repair_max_new_tokens"] = int(repair_tokens)
config["answer_repair_enable_thinking"] = repair_thinking.lower() in {"1", "true", "yes", "on"}
Path(output_path).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
PY

python scripts/prepare_heldout_prompts.py \
  --input "$SOURCE_INPUT" \
  --output-dir "$RUN_DIR/inputs" \
  | tee "$RUN_DIR/logs/prepare_heldout_prompts.log"

rotation_input="$RUN_DIR/inputs/heldout_rotation_raw_qa.jsonl"
around_input="$RUN_DIR/inputs/heldout_around_plain_cgmap_ffr_out.jsonl"
among_input="$RUN_DIR/inputs/heldout_among_plain_cgmap_ffr_out.jsonl"

write_manifest() {
  python - "$RUN_ID" "$RUN_DIR" "$SOURCE_INPUT" "$IMAGE_ROOT" "$MODEL_PATH" "$ROTATION_RUNTIME_CONFIG_PATH" "$SFT_CONFIG_PATH" "$SFT_ADAPTER_PATH" "$ANSWER_REPAIR_MAX_NEW_TOKENS" "$ANSWER_REPAIR_ENABLE_THINKING" <<'PY'
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

(
    run_id,
    run_dir,
    source_input,
    image_root,
    model_path,
    rotation_config,
    sft_config,
    sft_adapter,
    repair_tokens,
    repair_thinking,
) = sys.argv[1:]
run_dir = Path(run_dir)

def git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return ""

manifest = {
    "run_id": run_id,
    "started_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
    "status": "running",
    "config_source": "generated by scripts/bash_scripts/run_gemma4_heldout_combined_server.bash",
    "git": {
        "commit": git("rev-parse", "--short", "HEAD") or "unknown",
        "branch": git("branch", "--show-current") or "unknown",
        "dirty": bool(git("status", "--short")),
    },
    "data": {
        "split": "heldout",
        "source_input": source_input,
        "image_root": image_root,
    },
    "answer_repair": {
        "max_new_tokens": int(repair_tokens),
        "enable_thinking": repair_thinking.lower() in {"1", "true", "yes", "on"},
    },
    "strategy": {
        "rotation": {
            "task": "raw_qa",
            "model": model_path,
            "config": rotation_config,
            "notes": "plain Gemma, matching the combined candidate rotation source",
        },
        "around": {
            "task": "plain_cgmap_ffr_out",
            "model": model_path,
            "config": sft_config,
            "adapter": sft_adapter,
            "notes": "fine-tuned Gemma SFT LoRA, matching the combined candidate around/among strategy",
        },
        "among": {
            "task": "plain_cgmap_ffr_out",
            "model": model_path,
            "config": sft_config,
            "adapter": sft_adapter,
            "notes": "fine-tuned Gemma SFT LoRA, matching the combined candidate around/among strategy",
        },
    },
    "artifacts": {
        "predictions": str(run_dir / "predictions.jsonl"),
        "submission": str(run_dir / "submission.jsonl"),
        "construction_report": str(run_dir / "construction_report.json"),
        "logs": str(run_dir / "logs"),
    },
}
(run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY
}

run_setting() {
  local setting="$1"
  local input_file="$2"
  local config_path="$3"
  local gpu_groups="$4"
  local max_new_tokens="$5"
  local setting_dir="$RUN_DIR/$setting"

  mkdir -p "$setting_dir"/{shards,logs}
  local total
  total="$(wc -l < "$input_file" | tr -d ' ')"
  if [[ "$total" -eq 0 ]]; then
    echo "[INFO] Skipping empty setting: $setting"
    : > "$setting_dir/predictions.jsonl"
    return
  fi

  read -r -a group_array <<< "$gpu_groups"
  local shard_count="${#group_array[@]}"
  if [[ "$shard_count" -lt 1 ]]; then
    echo "[ERROR] Empty GPU group list for $setting" >&2
    exit 1
  fi

  python - "$input_file" "$setting_dir/shards" "$shard_count" <<'PY'
import math
import sys
from pathlib import Path

input_path = Path(sys.argv[1])
shard_dir = Path(sys.argv[2])
shard_count = int(sys.argv[3])
lines = input_path.read_text(encoding="utf-8").splitlines()
per_shard = max(1, math.ceil(len(lines) / shard_count))
for index in range(shard_count):
    shard_lines = lines[index * per_shard : min(len(lines), (index + 1) * per_shard)]
    path = shard_dir / f"input_shard_{index:02d}.jsonl"
    path.write_text("\n".join(shard_lines) + ("\n" if shard_lines else ""), encoding="utf-8")
PY

  echo "[INFO] Running $setting: $total rows across $shard_count shard(s): $gpu_groups"
  local pids=()
  for index in "${!group_array[@]}"; do
    local shard_id
    shard_id="$(printf "%02d" "$index")"
    local shard_input="$setting_dir/shards/input_shard_${shard_id}.jsonl"
    local shard_output="$setting_dir/shards/predictions_shard_${shard_id}.jsonl"
    local shard_log="$setting_dir/logs/gpu_group_${shard_id}.log"
    local gpu_group="${group_array[$index]}"

    if [[ ! -s "$shard_input" ]]; then
      echo "[INFO] Skipping empty $setting shard $shard_id"
      continue
    fi

    (
      CUDA_VISIBLE_DEVICES="$gpu_group" python scripts/run_inference.py \
        --model-type gemma4 \
        --model-path "$MODEL_PATH" \
        --backend transformers \
        --input-file "$shard_input" \
        --output-file "$shard_output" \
        --image-root "$IMAGE_ROOT" \
        --batch-size "$BATCH_SIZE" \
        --max-new-tokens "$max_new_tokens" \
        --temperature 0.0 \
        --top-p 1.0 \
        --config "$config_path" \
        --multi-gpu
    ) > "$shard_log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "[ERROR] At least one $setting shard failed. Check $setting_dir/logs" >&2
    exit 1
  fi

  cat "$setting_dir"/shards/predictions_shard_*.jsonl > "$setting_dir/predictions.jsonl"
  local pred_count
  pred_count="$(wc -l < "$setting_dir/predictions.jsonl" | tr -d ' ')"
  echo "[INFO] Finished $setting: $pred_count/$total predictions"
  if [[ "$pred_count" -ne "$total" ]]; then
    echo "[ERROR] $setting prediction count mismatch: $pred_count/$total" >&2
    exit 1
  fi
}

write_manifest

run_setting "rotation" "$rotation_input" "$ROTATION_RUNTIME_CONFIG_PATH" "$ROTATION_GPU_GROUPS" "$MAX_NEW_TOKENS_ROTATION"
run_setting "around" "$around_input" "$SFT_CONFIG_PATH" "$AROUND_GPU_GROUPS" "$MAX_NEW_TOKENS_SFT"
run_setting "among" "$among_input" "$SFT_CONFIG_PATH" "$AMONG_GPU_GROUPS" "$MAX_NEW_TOKENS_SFT"

python scripts/combine_heldout_predictions.py \
  --source "$SOURCE_INPUT" \
  --predictions \
    "$RUN_DIR/rotation/predictions.jsonl" \
    "$RUN_DIR/around/predictions.jsonl" \
    "$RUN_DIR/among/predictions.jsonl" \
  --output-predictions "$RUN_DIR/predictions.jsonl" \
  --output-submission "$RUN_DIR/submission.jsonl" \
  --report "$RUN_DIR/construction_report.json" \
  --fallback-policy "$COMBINE_FALLBACK_POLICY" \
  | tee "$RUN_DIR/logs/combine_heldout_predictions.log"

python - "$RUN_DIR" <<'PY'
import datetime as dt
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
manifest_path = run_dir / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["status"] = "success"
manifest["completed_at"] = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

echo "[INFO] Done."
echo "[INFO] Run dir: $RUN_DIR"
echo "[INFO] Predictions: $RUN_DIR/predictions.jsonl"
echo "[INFO] Submission: $RUN_DIR/submission.jsonl"
