# 20260503-222914-gemma4-26b-a4b-4bit-t4x2-raw-qa-transformers

## Question
The MoE 26B-A4B Gemma 4 model should be the strongest Gemma candidate that can plausibly fit on Kaggle's two 16 GB T4 GPUs when loaded in 4-bit.

## Setup
- Name: gemma4-26b-a4b-4bit-t4x2-raw-qa-transformers
- Tags: baseline, raw_qa, gemma4, 26b-a4b, t4x2, 4bit, transformers
- Model: google/gemma-4-26B-A4B-it
- Backend: transformers
- Prompt/task: raw_qa
- Input: data/prompts/general/MindCube_tinybench_raw_qa.jsonl

## Result
Invalid run. The local dashboard/evaluation were generated, but all 1050
prediction rows are model-load error outputs rather than real answers.

- Overall accuracy: 0.00% (0/1050)
- Around: 0.00% (0/250)
- Rotation: 0.00% (0/200)
- Among: 0.00% (0/600)
- Extraction failures/errors: 1050/1050

## Observations
- The 4-bit 26B-A4B load path dispatched modules to CPU/disk, producing error
  strings in `answer` for every sample.
- The run folder is useful as evidence that this Kaggle T4 x2 + bitsandbytes
  path is not a viable baseline in the current runner.

## Decision
- Status: discard
- Keep, retry, or discard: discard for scoring; keep only as a failure record.
- Next experiment: Gemma 4 E4B fp16 split across T4 x2, or E2B with stricter
  final-answer formatting if E4B still does not fit.
