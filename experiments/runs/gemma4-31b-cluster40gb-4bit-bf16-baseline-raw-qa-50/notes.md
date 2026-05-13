# gemma4-31b-cluster40gb-4bit-bf16-baseline-raw-qa-50

## Question
With enough output budget and strict answer formatting, Gemma 4 31B reasoning should reduce missing final answers compared with the Kaggle timeout artifact while giving a stronger Gemma baseline than no-thinking direct answering.

## Setup
- Name: gemma4-31b-cluster40gb-4bit-bf16-baseline-raw-qa-50
- Tags: baseline, raw_qa, gemma4, 31b, bf16, 4bit, cluster40gb, 50-sample
- Model: google/gemma-4-31B-it
- Backend: transformers
- Prompt/task: raw_qa
- Input: experiments/subsets/gemma4-31b-cluster40gb-4bit-bf16-baseline-raw-qa-50/input_50.jsonl

## Result
- Accuracy: 20.0% (10/50).
- Evaluated subset: first 50 raw-QA tinybench rows, all `among` setting.
- Extraction failures: 5/50.

## Observations
- BF16 fixed the prior pad-only generation failure; responses now contain real reasoning text.
- The model often remains in a long `thought` response and sometimes does not produce the requested final answer format.
- Several responses describe the rendered views as blurry, distorted, or hard to read, so image resolution and prompt format are plausible failure factors.
- This is a small sanity run, not a full benchmark estimate, because it only covers the first 50 examples and only the `among` setting.

## Decision
- Status: success as a 50-sample baseline sanity check.
- Keep, retry, or discard: keep as baseline diagnostic; do not compare as full tinybench score.
- Next experiment: run the prompt-optimized raw-elimination 50-sample shard after confirming the baseline dashboard failure modes.
