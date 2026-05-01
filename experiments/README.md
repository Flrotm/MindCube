# MindCube Experiment Workflow

Use this folder to keep experiment history separate from the large generated
data and model artifacts.

## Layout

```text
experiments/
  configs/             JSON configs for repeatable runs
  runs/                One folder per Kaggle/local experiment
  registry.csv         Compact table for comparing runs
  summary.md           Generated report-friendly summary
```

Each run folder contains:

```text
config.json            Exact config used for the run
manifest.json          Git commit, environment, commands, artifact paths
metrics.json           Compact metrics extracted from evaluation
analysis.md            Human-readable accuracy, error, and sample review
examples.csv           Per-example review table for spreadsheet analysis
notes.md               Human notes: what changed, what worked, what failed
report.md              Report-ready summary for that run
predictions.jsonl      Model outputs, ignored by git by default
evaluation.json        Full evaluator output
```

## Recommended Loop

1. Create or copy a config in `experiments/configs/`.
2. Push the code/config to GitHub.
3. Pull on Kaggle.
4. Run one experiment:

```bash
python scripts/run_experiment.py --config experiments/configs/qwen25vl_raw_qa_transformers.json
```

Use the `*_vllm.json` configs after installing vLLM on Kaggle. If vLLM is not
installed, the starter kit may fall back to the transformers backend.

5. Compare experiments:

```bash
python scripts/summarize_experiments.py
```

6. Open the run's `analysis.md` and `examples.csv` to inspect mistakes.
7. Edit the run's `notes.md` with observations and a decision.
8. Commit the small files: config, manifest, metrics, analysis, examples, notes,
   report, registry, and summary. Keep large outputs/checkpoints in Kaggle
   outputs or datasets.

When a run is worth submitting to EvalAI, convert its predictions:

```bash
python scripts/create_submission.py \
  --input experiments/runs/<run_id>/predictions.jsonl \
  --output experiments/runs/<run_id>/submission.jsonl
```

## What To Change Between Experiments

Change one main variable at a time when possible:

- model path, for example `Qwen/Qwen2.5-VL-7B-Instruct`
- prompt file, for example a custom generated JSONL under `data/prompts/...`
- prompt strategy, for example `raw_qa` vs `ff_rsn` vs cognitive-map variants
- decoding settings, especially `max_new_tokens`
- backend, for example `transformers` vs `vllm`

The registry should make it obvious which change helped and which one was noise.
