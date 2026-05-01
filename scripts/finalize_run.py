#!/usr/bin/env python3
"""Evaluate and analyze a downloaded Kaggle run folder locally."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from run_experiment import (
    PROJECT_ROOT,
    DEFAULT_REGISTRY,
    build_analysis_command,
    build_evaluation_command,
    extract_metrics,
    git_info,
    load_json,
    now_utc,
    registry_row,
    tee_subprocess,
    update_registry,
    write_json,
    write_report,
)


def resolve_run_dir(value: str) -> Path:
    run_dir = Path(value)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    return run_dir


def resolve_prediction_path(run_dir: Path, manifest: Dict[str, Any]) -> Path:
    prediction_path = run_dir / "predictions.jsonl"
    if prediction_path.exists():
        return prediction_path

    artifact = manifest.get("artifacts", {}).get("predictions", "")
    if artifact:
        artifact_path = PROJECT_ROOT / artifact
        if artifact_path.exists():
            return artifact_path

    raise FileNotFoundError(f"Missing predictions file for {run_dir}")


def finalize_run(run_dir: Path, registry_path: Path, force: bool) -> str:
    manifest_path = run_dir / "manifest.json"
    config_path = run_dir / "config.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    manifest = load_json(manifest_path)
    config = load_json(config_path)
    run_id = manifest.get("run_id", run_dir.name)
    predictions = resolve_prediction_path(run_dir, manifest)
    evaluation = run_dir / "evaluation.json"
    metrics_path = run_dir / "metrics.json"
    analysis_path = run_dir / "analysis.md"
    examples_path = run_dir / "examples.csv"
    notes_path = run_dir / "notes.md"
    report_path = run_dir / "report.md"
    log_path = run_dir / "logs" / "finalize.log"

    manifest.setdefault("artifacts", {})
    manifest["artifacts"].update(
        {
            "predictions": str(predictions.relative_to(PROJECT_ROOT)),
            "evaluation": str(evaluation.relative_to(PROJECT_ROOT)),
            "metrics": str(metrics_path.relative_to(PROJECT_ROOT)),
            "analysis": str(analysis_path.relative_to(PROJECT_ROOT)),
            "examples": str(examples_path.relative_to(PROJECT_ROOT)),
            "notes": str(notes_path.relative_to(PROJECT_ROOT)),
            "report": str(report_path.relative_to(PROJECT_ROOT)),
        }
    )

    evaluation_command = build_evaluation_command(config, predictions, evaluation)
    analysis_command = build_analysis_command(run_dir)
    manifest.setdefault("commands", {})
    manifest["commands"]["local_evaluation"] = evaluation_command
    manifest["commands"]["local_analysis"] = analysis_command

    status = "success"
    if force or not evaluation.exists():
        print(f"Running local evaluation for {run_id}")
        eval_code = tee_subprocess(evaluation_command, log_path)
        if eval_code != 0:
            status = "evaluation_failed"
    else:
        print(f"Using existing evaluation: {evaluation}")

    metrics: Dict[str, Any] = {}
    if status == "success":
        metrics = extract_metrics(evaluation)
        write_json(metrics_path, metrics)

        if force or not analysis_path.exists() or not examples_path.exists():
            print(f"Writing local detailed analysis for {run_id}")
            analysis_code = tee_subprocess(analysis_command, log_path)
            if analysis_code != 0:
                status = "analysis_failed"
        else:
            print(f"Using existing analysis: {analysis_path}")

    manifest["finalized_at"] = now_utc()
    manifest["status"] = status
    manifest["local_git"] = git_info()
    write_json(manifest_path, manifest)

    if metrics:
        write_report(report_path, config, manifest, metrics)

    update_registry(
        registry_path,
        registry_row(
            run_id=run_id,
            started_at=manifest.get("started_at", ""),
            status=status,
            config=config,
            metrics=metrics,
            git=manifest.get("git", {}),
            analysis_path=analysis_path,
            notes_path=notes_path,
        ),
    )

    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize a downloaded Kaggle MindCube run locally")
    parser.add_argument("--run-dir", required=True, help="Downloaded run folder")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="Experiment registry CSV")
    parser.add_argument("--force", action="store_true", help="Re-run evaluation and analysis even if present")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = PROJECT_ROOT / registry_path

    status = finalize_run(run_dir, registry_path, args.force)
    print(f"Finalize status: {status}")
    print(f"Run folder: {run_dir}")
    print("Refresh comparison summary with:")
    print("  python scripts/summarize_experiments.py")
    return 0 if status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
