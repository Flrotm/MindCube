# gemma4-31b-cluster40gb-4bit-reasoning-greedy-raw-qa-parallel

## Question
With enough output budget and strict answer formatting, Gemma 4 31B reasoning should reduce missing final answers compared with the Kaggle timeout artifact while giving a stronger Gemma baseline than no-thinking direct answering.

## Setup
- Name: gemma4-31b-cluster40gb-4bit-reasoning-greedy-raw-qa-transformers
- Tags: baseline, raw_qa, gemma4, 31b, reasoning, cluster40gb, 4bit, greedy, transformers
- Model: google/gemma-4-31B-it
- Backend: transformers
- Prompt/task: raw_qa
- Input: data/prompts/general/MindCube_tinybench_raw_qa.jsonl

## Result
Invalid output run. The merged prediction file contains 1050 rows, but every
`answer` field is repeated `<pad>` tokens with length 5120 and no extractable
A-D answer.

- Evaluator headline: 0.00% (0/1050)
- Extraction failures: 1050/1050
- Around: 0/250
- Rotation: 0/200
- Among: 0/600
- Translation: excluded by evaluator

## Observations
- The run completed mechanically across the two cluster shards, but the model
  generated only pad tokens rather than reasoning/final answers.
- The local prediction JSONL was repaired at two shard boundaries where a
  partial row had been concatenated with the next row. After repair, the file
  has 1050 parseable rows and 1050 unique IDs.
- The result should not be interpreted as a real 0% accuracy model result. It
  is an invalid decoding/configuration failure.
- Likely next debugging axis: generation/tokenizer configuration for Gemma 4
  31B 4-bit greedy decoding, especially pad/eos handling and decoding with
  `skip_special_tokens`.

## Decision
- Status: invalid_outputs
- Keep, retry, or discard: discard as a scored benchmark result; keep only as
  evidence that this Gemma 4 31B 4-bit greedy config can degenerate to pad-only
  outputs.
- Next experiment: run the raw-elimination strategy only after a small smoke
  test confirms non-pad generations and extractable `<answer>...</answer>`
  outputs.
