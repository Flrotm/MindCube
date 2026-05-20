# Gemma MindCube RL Server Helpers

These helpers prepare and launch the VAGEN MindCube `cogmap_reasoning_plain`
RL run for the server layout used in this project.

## Server Setup

Run from `/home/fuccelli/mindcube/MindCube`:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mindcube-gemma
git pull --ff-only origin kaggle-runner

bash scripts/bash_scripts/setup_gemma_rl_vagen_server.sh
```

The setup script uses `/data/fuccelli/mindcube_runs/rl_stack/` for external
`verl` and `VAGEN` clones, exports the `/data` cache paths, installs editable
packages with `--no-deps` by default, installs
`requirements-vagen-rl-server.txt`, checks/restores the known
`torch==2.6.0+cu118` stack, runs an import preflight, patches VAGEN's crossview
env to support `data_file`, and symlinks `data/other_all_image`.

The import preflight checks base dependencies plus the VAGEN crossview,
dataset, server, and trainer modules. To sweep more modules, override
`VAGEN_PREFLIGHT_PACKAGES`, for example `VAGEN_PREFLIGHT_PACKAGES="vagen"`.

If VAGEN needs dependencies that are not already installed, rerun with:

```bash
INSTALL_DEPS=1 bash scripts/bash_scripts/setup_gemma_rl_vagen_server.sh
```

The script will restore the cu118 torch stack afterward unless
`RESTORE_TORCH_STACK=0`.

## Smoke, Full, Eval

Use one comma-separated 2-GPU process:

```bash
CUDA_DEVICES=2,3 MODE=smoke bash scripts/bash_scripts/run_gemma_rl_vagen_server.sh
```

Do not use `CUDA_DEVICES="2 3"` or `GPU_GROUPS="2 3"` for RL; spaces create
independent single-GPU runs in the existing inference helpers.

After the smoke run completes:

```bash
CUDA_DEVICES=2,3 MODE=full nohup bash scripts/bash_scripts/run_gemma_rl_vagen_server.sh \
  > /data/fuccelli/mindcube_runs/rl_runs/full_gemma_rl.out 2>&1 &
```

Then score the final checkpoint once on the 100-sample ID set:

```bash
CUDA_DEVICES=2,3 MODE=eval bash scripts/bash_scripts/run_gemma_rl_vagen_server.sh
```

By default the eval subset is built from
`experiments/samples/MindCube_tinybench_raw_qa_representative_100_seed1337.jsonl`
by selecting matching IDs from VAGEN's
`crossviewQA_tinybench_cogmap_and_reasoning_plain.jsonl`.

## Useful Overrides

```bash
TEST_IDS_FILE=/path/to/input_100.jsonl
EVAL_SETTINGS="among around"
SFT_LORA_PATH=/data/fuccelli/mindcube_runs/checkpoints/sft/gemma4/gemma4-31b-sft-plain-cgmap-ffr-out-native-thinking-1epoch-ebs128-gpu23
MERGED_MODEL_DIR=/data/fuccelli/mindcube_runs/checkpoints/rl_ready/custom-merged
SFT_LORA_GLOB='/data/fuccelli/mindcube_runs/checkpoints/sft_full/gemma4-31b-*'
ENABLE_STOP_SEQUENCE_OVERRIDES=0
```

Use `SFT_LORA_PATH` for an exact adapter directory. Use `SFT_LORA_GLOB` only
when you want the merge script to choose the latest matching adapter. Set
`MERGE_IF_MISSING=0` if the merged Gemma checkpoint already exists and you only
want to reuse it.
