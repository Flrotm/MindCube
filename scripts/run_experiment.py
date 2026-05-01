#!/usr/bin/env python3
"""Run and record one MindCube experiment.

The script is intentionally lightweight. On Kaggle, prefer `--inference-only`
so the GPU notebook only produces predictions and run metadata. After
downloading the run folder locally, use `scripts/finalize_run.py` to run
evaluation, dashboard generation, and registry updates.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "experiments" / "runs"
DEFAULT_REGISTRY = PROJECT_ROOT / "experiments" / "registry.csv"
REGISTRY_FIELDS = [
    "run_id",
    "started_at",
    "status",
    "name",
    "tags",
    "model",
    "backend",
    "split",
    "task",
    "accuracy",
    "total",
    "correct",
    "git_commit",
    "dashboard",
    "notes",
]


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "experiment"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def run_capture(args: List[str], cwd: Path = PROJECT_ROOT) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except FileNotFoundError:
        return ""
    return result.stdout.strip()


def git_info() -> Dict[str, Any]:
    commit = run_capture(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
    branch = run_capture(["git", "branch", "--show-current"]) or "unknown"
    status = run_capture(["git", "status", "--short"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": sys.platform,
        "hostname": socket.gethostname(),
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", ""),
        "kaggle_url_base": os.environ.get("KAGGLE_URL_BASE", ""),
    }
    try:
        import torch

        info["torch"] = getattr(torch, "__version__", "")
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # pragma: no cover - environment dependent
        info["torch_error"] = str(exc)

    nvidia = run_capture(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    if nvidia:
        info["nvidia_smi"] = nvidia.splitlines()
    return info


def make_run_id(config: Dict[str, Any], explicit: Optional[str]) -> str:
    if explicit:
        return slugify(explicit)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{slugify(config.get('name', 'experiment'))}"


def build_inference_command(config: Dict[str, Any], output_file: Path) -> List[str]:
    model = config.get("model", {})
    data = config.get("data", {})
    generation = config.get("generation", {})

    command = [
        "python",
        "scripts/run_inference.py",
        "--model-type",
        model.get("type", "qwen2.5vl"),
        "--model-path",
        model.get("path", "Qwen/Qwen2.5-VL-3B-Instruct"),
        "--backend",
        model.get("backend", "transformers"),
        "--input-file",
        data["input_file"],
        "--output-file",
        str(output_file),
        "--image-root",
        data.get("image_root", "./data/"),
        "--batch-size",
        str(generation.get("batch_size", 1)),
        "--max-new-tokens",
        str(generation.get("max_new_tokens", 512)),
        "--temperature",
        str(generation.get("temperature", 0.0)),
        "--top-p",
        str(generation.get("top_p", 1.0)),
    ]

    inference_config = model.get("inference_config")
    if inference_config:
        command.extend(["--config", inference_config])
    return command


def build_evaluation_command(config: Dict[str, Any], prediction_file: Path, evaluation_file: Path) -> List[str]:
    mode = config.get("evaluation", {}).get("mode", "auto")
    command = [
        "python",
        "scripts/run_evaluation.py",
        "--input",
        str(prediction_file),
        "--output",
        str(evaluation_file),
    ]
    if mode == "auto":
        command.append("--auto")
    else:
        command.extend(["--task", mode])
    return command


def build_analysis_command(run_dir: Path) -> List[str]:
    return [
        "python",
        "scripts/analyze_run.py",
        "--run-dir",
        str(run_dir),
    ]


def tee_subprocess(command: List[str], log_path: Path) -> int:
    exec_command = command.copy()
    if exec_command and exec_command[0] == "python":
        exec_command[0] = sys.executable
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(exec_command)}\n")
        log.flush()
        process = subprocess.Popen(
            exec_command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            console_encoding = sys.stdout.encoding or "utf-8"
            console_line = line.encode(console_encoding, errors="replace").decode(
                console_encoding,
                errors="replace",
            )
            print(console_line, end="")
            log.write(line)
        return process.wait()


def extract_metrics(evaluation_file: Path) -> Dict[str, Any]:
    evaluation = load_json(evaluation_file)
    results = evaluation.get("results", {})
    accuracy = results.get("gen_cogmap_accuracy")
    if isinstance(accuracy, (int, float)):
        accuracy_pct = round(float(accuracy) * 100, 4)
    else:
        accuracy_pct = None

    settings = {}
    for setting, values in results.get("settings", {}).items():
        setting_acc = values.get("gen_cogmap_accuracy")
        settings[setting] = {
            "total": values.get("total", 0),
            "correct": values.get("gen_cogmap_correct", 0),
            "accuracy": round(float(setting_acc) * 100, 4)
            if isinstance(setting_acc, (int, float))
            else None,
        }

    metrics: Dict[str, Any] = {
        "accuracy": accuracy_pct,
        "total": results.get("total", 0),
        "correct": results.get("gen_cogmap_correct", 0),
        "unfiltered_total": results.get("unfiltered_total", results.get("total", 0)),
        "settings": settings,
    }
    if "cogmap_similarity" in results:
        metrics["cogmap_similarity"] = results["cogmap_similarity"]
    return metrics


def write_notes_template(path: Path, config: Dict[str, Any], run_id: str) -> None:
    if path.exists():
        return
    tags = ", ".join(config.get("tags", []))
    content = f"""# {run_id}

## Question
{config.get("hypothesis", "What change is this experiment testing?")}

## Setup
- Name: {config.get("name", "")}
- Tags: {tags}
- Model: {config.get("model", {}).get("path", "")}
- Backend: {config.get("model", {}).get("backend", "")}
- Prompt/task: {config.get("data", {}).get("task", "")}
- Input: {config.get("data", {}).get("input_file", "")}

## Result
Fill this in after the run. Include the headline accuracy and any category that moved.

## Observations
- 

## Decision
- Status: undecided
- Keep, retry, or discard:
- Next experiment:
"""
    path.write_text(content, encoding="utf-8")


def write_report(path: Path, config: Dict[str, Any], manifest: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    setting_rows = []
    for setting, values in sorted(metrics.get("settings", {}).items()):
        setting_rows.append(
            f"| {setting} | {values.get('accuracy', '')} | {values.get('correct', '')} | {values.get('total', '')} |"
        )
    settings_table = "\n".join(setting_rows) if setting_rows else "| n/a |  |  |  |"

    content = f"""# {manifest["run_id"]}

## Summary
- Name: {config.get("name", "")}
- Hypothesis: {config.get("hypothesis", "")}
- Status: {manifest.get("status", "")}
- Accuracy: {metrics.get("accuracy", "")}
- Correct/total: {metrics.get("correct", "")}/{metrics.get("total", "")}
- Git: {manifest.get("git", {}).get("commit", "")} on {manifest.get("git", {}).get("branch", "")}

## Configuration
- Model: {config.get("model", {}).get("path", "")}
- Backend: {config.get("model", {}).get("backend", "")}
- Split: {config.get("data", {}).get("split", "")}
- Task/prompt: {config.get("data", {}).get("task", "")}
- Input file: {config.get("data", {}).get("input_file", "")}
- Max new tokens: {config.get("generation", {}).get("max_new_tokens", "")}
- Batch size: {config.get("generation", {}).get("batch_size", "")}

## Metrics By Setting
| Setting | Accuracy | Correct | Total |
| --- | ---: | ---: | ---: |
{settings_table}

## Artifacts
- Predictions: {manifest.get("artifacts", {}).get("predictions", "")}
- Evaluation: {manifest.get("artifacts", {}).get("evaluation", "")}
- Metrics: {manifest.get("artifacts", {}).get("metrics", "")}
- Log: {manifest.get("artifacts", {}).get("log", "")}

## Interpretation
See `notes.md` for observations, decision, and follow-up.
"""
    path.write_text(content, encoding="utf-8")


def read_registry(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_registry(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in REGISTRY_FIELDS})


def update_registry(path: Path, row: Dict[str, Any]) -> None:
    rows = read_registry(path)
    rows = [existing for existing in rows if existing.get("run_id") != row["run_id"]]
    rows.append(row)
    rows.sort(key=lambda item: item.get("started_at", ""))
    write_registry(path, rows)


def registry_row(
    run_id: str,
    started_at: str,
    status: str,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    git: Dict[str, Any],
    dashboard_path: Path,
    notes_path: Path,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "started_at": started_at,
        "status": status,
        "name": config.get("name", ""),
        "tags": ";".join(config.get("tags", [])),
        "model": config.get("model", {}).get("path", ""),
        "backend": config.get("model", {}).get("backend", ""),
        "split": config.get("data", {}).get("split", ""),
        "task": config.get("data", {}).get("task", ""),
        "accuracy": metrics.get("accuracy", ""),
        "total": metrics.get("total", ""),
        "correct": metrics.get("correct", ""),
        "git_commit": git.get("commit", ""),
        "dashboard": str(dashboard_path.relative_to(PROJECT_ROOT)),
        "notes": str(notes_path.relative_to(PROJECT_ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and record one MindCube experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    parser.add_argument("--run-id", help="Optional stable run id")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT), help="Directory for run folders")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="CSV registry path")
    parser.add_argument("--skip-inference", action="store_true", help="Use existing predictions file")
    parser.add_argument("--skip-evaluation", action="store_true", help="Use existing evaluation file")
    parser.add_argument("--inference-only", action="store_true", help="Run only model inference and metadata capture")
    parser.add_argument("--dry-run", action="store_true", help="Create metadata and print commands without running")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_json(config_path)
    run_id = make_run_id(config, args.run_id)
    run_dir = Path(args.run_root)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir = run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = now_utc()
    predictions = run_dir / "predictions.jsonl"
    evaluation = run_dir / "evaluation.json"
    metrics_path = run_dir / "metrics.json"
    notes_path = run_dir / "notes.md"
    dashboard_path = run_dir / "dashboard.html"
    log_path = run_dir / "logs" / "run.log"

    run_config_path = run_dir / "config.json"
    if config_path.resolve() != run_config_path.resolve():
        shutil.copy2(config_path, run_config_path)
    write_notes_template(notes_path, config, run_id)

    inference_command = build_inference_command(config, predictions)
    evaluation_command = build_evaluation_command(config, predictions, evaluation)
    analysis_command = build_analysis_command(run_dir)
    git = git_info()

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": started_at,
        "status": "created",
        "config_source": str(config_path.relative_to(PROJECT_ROOT)),
        "git": git,
        "environment": environment_info(),
        "commands": {
            "inference": inference_command,
            "evaluation": evaluation_command,
            "dashboard": analysis_command,
        },
        "artifacts": {
            "predictions": str(predictions.relative_to(PROJECT_ROOT)),
            "evaluation": str(evaluation.relative_to(PROJECT_ROOT)),
            "metrics": str(metrics_path.relative_to(PROJECT_ROOT)),
            "dashboard": str(dashboard_path.relative_to(PROJECT_ROOT)),
            "notes": str(notes_path.relative_to(PROJECT_ROOT)),
            "log": str(log_path.relative_to(PROJECT_ROOT)),
        },
    }
    write_json(run_dir / "manifest.json", manifest)

    if args.dry_run:
        print("Dry run created:")
        print(f"  {run_dir}")
        print("Inference command:")
        print("  " + " ".join(inference_command))
        print("Evaluation command:")
        print("  " + " ".join(evaluation_command))
        print("Analysis command:")
        print("  " + " ".join(analysis_command))
        if args.inference_only:
            print("Mode: inference only")
        return 0

    status = "success"
    if not args.skip_inference:
        print(f"Running inference for {run_id}")
        inference_code = tee_subprocess(inference_command, log_path)
        if inference_code != 0:
            status = "inference_failed"
    elif not predictions.exists():
        print(f"Missing predictions file: {predictions}", file=sys.stderr)
        status = "missing_predictions"

    if status == "success" and args.inference_only:
        status = "inference_complete"

    if status == "success" and not args.skip_evaluation:
        print(f"Running evaluation for {run_id}")
        evaluation_code = tee_subprocess(evaluation_command, log_path)
        if evaluation_code != 0:
            status = "evaluation_failed"
    elif status == "success" and not evaluation.exists():
        print(f"Missing evaluation file: {evaluation}", file=sys.stderr)
        status = "missing_evaluation"

    manifest["completed_at"] = now_utc()
    manifest["status"] = status
    write_json(run_dir / "manifest.json", manifest)

    metrics: Dict[str, Any] = {}
    if status == "success":
        metrics = extract_metrics(evaluation)
        write_json(metrics_path, metrics)
        print(f"Writing dashboard for {run_id}")
        analysis_code = tee_subprocess(analysis_command, log_path)
        if analysis_code != 0:
            status = "analysis_failed"
            manifest["status"] = status
            write_json(run_dir / "manifest.json", manifest)

    update_registry(
        Path(args.registry) if Path(args.registry).is_absolute() else PROJECT_ROOT / args.registry,
        registry_row(run_id, started_at, status, config, metrics, git, dashboard_path, notes_path),
    )

    print(f"Run status: {status}")
    if metrics:
        print(f"Accuracy: {metrics.get('accuracy')} ({metrics.get('correct')}/{metrics.get('total')})")
    print(f"Run folder: {run_dir}")
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
