# gemma_r

## Question
Evaluate the downloaded `gemma_r` notebook artifact even though the notebook timed out, treating the available predictions file as the complete run set.

## Setup
- Name: gemma_r
- Tags: baseline, raw_qa, gemma4, reasoning, timeout_artifact, partial_as_complete, transformers
- Model: gemma_r
- Backend: transformers
- Prompt/task: raw_qa
- Input: data/prompts/general/MindCube_tinybench_raw_qa.jsonl
- Predictions: `experiments/runs/gemma_r/predictions (1).jsonl`

## Result
Scored the 538 downloaded prediction rows as the full denominator.

- Overall accuracy: 20.82% (112/538)
- Among: 20.82% (112/538)
- Around: 0 rows
- Rotation: 0 rows
- Translation: 0 rows
- Extraction failures: 159/538

## Observations
- This report intentionally does not count the notebook-timeout remainder as missing or wrong.
- All available rows are `among` examples, so the dashboard is a report on the downloaded subset as a standalone set rather than a full tinybench category mix.
- Predicted answers are heavily skewed toward `D` plus missing extractions: A=100, B=43, C=30, D=205, E=1, missing=159.
- Ground truth is comparatively balanced across A=174, B=179, C=90, D=95.
- The largest quality issue is answer extraction/format compliance: 159 rows did not yield an A-E answer under the current parser.

## Decision
- Status: success
- Keep, retry, or discard: keep as a scored timeout artifact.
- Next experiment: rerun with stricter final-answer formatting and a checkpoint/resume path so timed-out notebooks can complete or merge cleanly.
