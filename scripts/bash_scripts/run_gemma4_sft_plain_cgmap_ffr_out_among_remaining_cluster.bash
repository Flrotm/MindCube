#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f scripts/run_inference.py || ! -f scripts/run_evaluation.py ]]; then
  echo "[ERROR] Run this from the MindCube repo root." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_RUN_SCRIPT="$SCRIPT_DIR/run_gemma4_noquant_raw_qa_50_cluster.bash"

if [[ ! -f "$BASE_RUN_SCRIPT" ]]; then
  echo "[ERROR] Missing base run script: $BASE_RUN_SCRIPT" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-gemma4-31b-sft-plain-cgmap-ffr-out-1epoch-ebs128-ft-cogmap-among-remaining-thinking-stop-answer-gpu23-long6144}"
RUN_ROOT="${RUN_ROOT:-/data/fuccelli/mindcube_runs/experiments/runs}"
RUN_DIR="${RUN_ROOT%/}/${RUN_ID}"

ALL_INPUT="${ALL_INPUT:-data/prompts/general/MindCube_tinybench_plain_cgmap_ffr_out.jsonl}"
SCAFFOLD_INPUT="${SCAFFOLD_INPUT:-data/scaffold/all/MindCube_tinybench.jsonl}"
FILTER_MODE="${FILTER_MODE:-remaining}" # remaining or all
PREVIOUS_RUN_ID="${PREVIOUS_RUN_ID:-gemma4-31b-sft-plain-cgmap-ffr-out-1epoch-ebs128-ft-cogmap-among57-thinking-stop-answer-gpu23-long6144}"
PREVIOUS_PREDICTIONS="${PREVIOUS_PREDICTIONS:-/data/fuccelli/mindcube_runs/experiments/runs/${PREVIOUS_RUN_ID}/predictions.jsonl}"

if [[ "$FILTER_MODE" != "remaining" && "$FILTER_MODE" != "all" ]]; then
  echo "[ERROR] FILTER_MODE must be 'remaining' or 'all', got: $FILTER_MODE" >&2
  exit 1
fi

if [[ ! -f "$ALL_INPUT" ]]; then
  if [[ -f "$SCAFFOLD_INPUT" ]]; then
    echo "[INFO] Missing $ALL_INPUT; generating it from $SCAFFOLD_INPUT"
    python scripts/generate_prompts.py \
      --input "$SCAFFOLD_INPUT" \
      --output "$ALL_INPUT" \
      --task plain_cgmap_ffr_out
  else
    echo "[ERROR] Missing source prompt file: $ALL_INPUT" >&2
    echo "[ERROR] Also missing scaffold fallback: $SCAFFOLD_INPUT" >&2
    exit 1
  fi
fi

if [[ "$FILTER_MODE" == "remaining" && ! -f "$PREVIOUS_PREDICTIONS" ]]; then
  echo "[ERROR] Remaining mode needs previous predictions to avoid skipping unfinished IDs." >&2
  echo "[ERROR] Missing previous predictions: $PREVIOUS_PREDICTIONS" >&2
  echo "[ERROR] Set FILTER_MODE=all to run all 600 among examples from scratch." >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
FILTERED_SOURCE="$RUN_DIR/source_among_${FILTER_MODE}.jsonl"
FILTER_SUMMARY="$RUN_DIR/source_filter_summary.json"

python - "$ALL_INPUT" "$PREVIOUS_PREDICTIONS" "$FILTERED_SOURCE" "$FILTER_SUMMARY" "$FILTER_MODE" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
previous_predictions_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
filter_mode = sys.argv[5]

def iter_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

def is_among(row):
    return "among" in str(row.get("id", "")).lower()

previous_ids = set()
if filter_mode == "remaining":
    previous_ids = {
        str(row.get("id", ""))
        for row in iter_jsonl(previous_predictions_path)
        if str(row.get("id", ""))
    }

source_total = 0
among_total = 0
selected_total = 0
skipped_previous = 0

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as output:
    for row in iter_jsonl(source_path):
        source_total += 1
        row_id = str(row.get("id", ""))
        if not is_among(row):
            continue
        among_total += 1
        if filter_mode == "remaining" and row_id in previous_ids:
            skipped_previous += 1
            continue
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
        selected_total += 1

summary = {
    "source": str(source_path),
    "previous_predictions": str(previous_predictions_path) if filter_mode == "remaining" else None,
    "output": str(output_path),
    "filter_mode": filter_mode,
    "source_total": source_total,
    "source_among_total": among_total,
    "previous_prediction_ids": len(previous_ids),
    "skipped_previous_prediction_ids": skipped_previous,
    "selected_total": selected_total,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

SAMPLE_COUNT="$(wc -l < "$FILTERED_SOURCE" | tr -d ' ')"
if [[ "$SAMPLE_COUNT" -eq 0 ]]; then
  echo "[ERROR] Filtered source is empty: $FILTERED_SOURCE" >&2
  exit 1
fi

export RUN_ID
export RUN_ROOT
export SOURCE_INPUT="$FILTERED_SOURCE"
export SAMPLE_COUNT
export IMAGE_ROOT="${IMAGE_ROOT:-./data/}"
export CONFIG_PATH="${CONFIG_PATH:-/data/fuccelli/mindcube_runs/configs/gemma4-31b-sft-plain-cgmap-ffr-out-1epoch-ebs128-ft-cogmap-among57-thinking-stop-answer-gpu23-long6144.json}"
export MODEL_PATH="${MODEL_PATH:-google/gemma-4-31B-it}"
export TASK_NAME="${TASK_NAME:-plain_cgmap_ffr_out}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-6144}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GPU_GROUPS="${GPU_GROUPS:-2,3}"
export RUN_DESCRIPTION="${RUN_DESCRIPTION:-Fine-tuned Gemma 4 31B LoRA with the best Among config, run on ${FILTER_MODE} tinybench Among examples for Plain-CGMap-FFR-Out.}"
export RUN_HYPOTHESIS="${RUN_HYPOTHESIS:-The best partial Among configuration should scale to the remaining tinybench Among set when deduplicated against previous predictions.}"
export RUN_TAGS="${RUN_TAGS:-finetuned,plain_cgmap_ffr_out,gemma4,31b,8bit-vision-safe,lora,among,${FILTER_MODE},thinking,stop-answer,gpu23,long6144}"

echo "[INFO] Launching base run script for $SAMPLE_COUNT Among examples"
bash "$BASE_RUN_SCRIPT"

if [[ "$FILTER_MODE" == "remaining" && -f "$PREVIOUS_PREDICTIONS" && -f "$RUN_DIR/predictions.jsonl" ]]; then
  MERGED_PREDICTIONS="$RUN_DIR/predictions_all_among_with_previous.jsonl"
  MERGED_EVALUATION="$RUN_DIR/evaluation_all_among_with_previous.json"
  MERGED_METRICS="$RUN_DIR/metrics_all_among_with_previous.json"

  cat "$PREVIOUS_PREDICTIONS" "$RUN_DIR/predictions.jsonl" > "$MERGED_PREDICTIONS"
  python scripts/run_evaluation.py \
    --input "$MERGED_PREDICTIONS" \
    --output "$MERGED_EVALUATION" \
    --task basic

  python - "$MERGED_EVALUATION" "$MERGED_METRICS" <<'PY'
import json
import sys
from pathlib import Path

evaluation = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
results = evaluation.get("results", {})
accuracy = results.get("gen_cogmap_accuracy")
settings = {}
for setting, values in results.get("settings", {}).items():
    setting_accuracy = values.get("gen_cogmap_accuracy")
    settings[setting] = {
        "total": values.get("total", 0),
        "correct": values.get("gen_cogmap_correct", 0),
        "accuracy": round(float(setting_accuracy) * 100, 4) if isinstance(setting_accuracy, (int, float)) else None,
    }
metrics = {
    "accuracy": round(float(accuracy) * 100, 4) if isinstance(accuracy, (int, float)) else None,
    "total": results.get("total", 0),
    "correct": results.get("gen_cogmap_correct", 0),
    "unfiltered_total": results.get("unfiltered_total", results.get("total", 0)),
    "settings": settings,
}
Path(sys.argv[2]).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
print(json.dumps(metrics, indent=2))
PY

  echo "[INFO] Merged all-among predictions: $MERGED_PREDICTIONS"
  echo "[INFO] Merged all-among metrics: $MERGED_METRICS"
fi
