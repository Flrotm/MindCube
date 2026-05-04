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
dashboard.html         Self-contained review dashboard with charts and examples
notes.md               Human notes: what changed, what worked, what failed
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

Inference is fail-fast by default. If a model load or generation error appears
in a prediction, the run aborts instead of spending GPU time writing error rows.
Only use `--no-fail-fast` with `scripts/run_inference.py` when you intentionally
want to collect partial failures for debugging.

Use the `*_vllm.json` configs after installing vLLM on Kaggle. If vLLM is not
installed, the starter kit may fall back to the transformers backend.

For Gemma 4, install the newer multimodal Transformers stack in the Kaggle
notebook before running. Do not upgrade `torch`/`torchvision` on P100 runtimes;
newer CUDA wheels can drop support for the P100's `sm_60` architecture.

```bash
pip install -U -r requirements-gemma4.txt
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
  --config experiments/configs/gemma4_e2b_raw_qa_transformers.json \
  --inference-only
```

For the strongest Gemma baseline that should plausibly fit on Kaggle's T4 x2
runtime, use the 26B-A4B MoE model in 4-bit. This config intentionally uses
both 16 GB T4s with sequential placement so the multimodal front of the model
stays together and later layers can spill to the second GPU. It may still be
slow, so treat the E4B config as the reliable fallback.

Before a full Gemma 4 run, probe one sample from biggest to smallest. This
tries 26B-A4B 4-bit, E4B fp16, E4B 4-bit, then E2B, stopping at the largest
config that loads and writes a real prediction. The 31B 4-bit config exists
for larger GPU runtimes, but is intentionally not part of the default Kaggle
T4 x2 probe because it can destabilize the session after OOM.

```bash
python scripts/probe_gemma4_fit.py
```

The probe writes a summary under `experiments/probes/` with the exact full-run
command for the winning config.

```bash
pip install -U -r requirements-gemma4.txt
python scripts/run_experiment.py \
  --config experiments/configs/gemma4_26b_a4b_raw_qa_t4x2_4bit_transformers.json \
  --inference-only
```

For a Gemma 4 middle-ground reasoning run, prefer E4B in fp16 across both T4s.
This avoids the bitsandbytes 4-bit offload path that can produce error rows
instead of predictions.

```bash
python scripts/run_experiment.py \
  --config experiments/configs/gemma4_e4b_raw_qa_t4x2_fp16_reasoning_transformers.json \
  --inference-only
```

Fallback if the E4B fp16 run is too slow or runs out of memory:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
  --config experiments/configs/gemma4_e2b_raw_qa_answer_format_transformers.json \
  --inference-only
```

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

8. Open the run's `dashboard.html` to inspect mistakes, graphs, examples, logs,
   and run metadata.
9. Edit the run's `notes.md` with observations and a decision.
10. Commit the small files: config, manifest, metrics, dashboard, notes,
   registry, and summary. Keep large outputs/checkpoints in Kaggle outputs or
   datasets.

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
