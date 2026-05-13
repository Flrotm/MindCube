# Experiment Summary

| Run | Status | Task | Backend | Accuracy | Correct/Total | Tags | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| 20260501-130930-qwen25vl-raw-qa-transformers | success | raw_qa | transformers | 38.0 | 399/1050 | baseline;raw_qa;qwen2.5-vl;transformers | [dashboard](experiments/runs/20260501-130930-qwen25vl-raw-qa-transformers/dashboard.html), [notes](experiments/runs/20260501-130930-qwen25vl-raw-qa-transformers/notes.md) |
| 20260503-222914-gemma4-26b-a4b-4bit-t4x2-raw-qa-transformers | invalid_outputs | raw_qa | transformers | 0.0 | 0/1050 | baseline;raw_qa;gemma4;26b-a4b;t4x2;4bit;transformers | [dashboard](experiments/runs/20260503-222914-gemma4-26b-a4b-4bit-t4x2-raw-qa-transformers/dashboard.html), [notes](experiments/runs/20260503-222914-gemma4-26b-a4b-4bit-t4x2-raw-qa-transformers/notes.md) |
| gemma_r | success | raw_qa | transformers | 20.8178 | 112/538 | baseline;raw_qa;gemma4;reasoning;timeout_artifact;partial_as_complete;transformers | [dashboard](experiments/runs/gemma_r/dashboard.html), [notes](experiments/runs/gemma_r/notes.md) |
| gemma4-31b-cluster40gb-4bit-reasoning-greedy-raw-qa-parallel | invalid_outputs | raw_qa | transformers | 0.0 | 0/1050 | baseline;raw_qa;gemma4;31b;reasoning;cluster40gb;4bit;greedy;transformers | [dashboard](experiments/runs/gemma4-31b-cluster40gb-4bit-reasoning-greedy-raw-qa-parallel/dashboard.html), [notes](experiments/runs/gemma4-31b-cluster40gb-4bit-reasoning-greedy-raw-qa-parallel/notes.md) |

## Reading Notes

- Compare runs that changed only one variable when possible.
- Treat accuracy changes on partial/subset runs as directional, not final.
- Write the decision in each run's `notes.md`: keep, retry, or discard.
