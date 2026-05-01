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

Each run folder can contain:

```text
config.json            Exact config used for the run
manifest.json          Git commit, environment, commands, artifact paths
predictions.jsonl      Model outputs from Kaggle, ignored by git by default
metrics.json           Compact metrics extracted from evaluation
analysis.md            Human-readable accuracy, error, and sample review
examples.csv           Per-example review table for spreadsheet analysis
notes.md               Human notes: what changed, what worked, what failed
report.md              Report-ready summary for that run
evaluation.json        Full evaluator output
```

## Recommended Loop

1. Create or copy a config in `experiments/configs/`.
2. Push the code/config to GitHub.
3. Pull on Kaggle and run inference only:

```bash
python scripts/run_experiment.py \
  --config experiments/configs/qwen25vl_raw_qa_transformers.json \
  --inference-only
```

Use the `*_vllm.json` configs after installing vLLM on Kaggle. If vLLM is not
installed, the starter kit may fall back to the transformers backend.

4. Zip and download the run folder from Kaggle:

```bash
RUN=$(ls -td experiments/runs/* | head -1)
zip -r /kaggle/working/mindcube_run.zip "$RUN"
```

5. Unzip the run folder locally under `experiments/runs/`.
6. Finalize the run locally:

```bash
python scripts/finalize_run.py --run-dir experiments/runs/<run_id>
```

7. Compare experiments locally:

```bash
python scripts/summarize_experiments.py
```

8. Open the run's `analysis.md` and `examples.csv` to inspect mistakes.
9. Edit the run's `notes.md` with observations and a decision.
10. Commit the small files: config, manifest, metrics, analysis, examples, notes,
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
