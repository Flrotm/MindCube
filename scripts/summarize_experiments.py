#!/usr/bin/env python3
"""Build a compact registry and Markdown summary from experiment runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "experiments" / "runs"
DEFAULT_REGISTRY = PROJECT_ROOT / "experiments" / "registry.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "experiments" / "summary.md"
FIELDS = [
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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel(path: Path) -> str:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def collect_runs(run_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        manifest_path = run_dir / "manifest.json"
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        if not manifest_path.exists() or not config_path.exists():
            continue
        manifest = load_json(manifest_path)
        config = load_json(config_path)
        metrics = load_json(metrics_path) if metrics_path.exists() else {}
        dashboard_path = run_dir / "dashboard.html"
        legacy_analysis_path = run_dir / "analysis.md"
        notes_path = run_dir / "notes.md"
        rows.append(
            {
                "run_id": manifest.get("run_id", run_dir.name),
                "started_at": manifest.get("started_at", ""),
                "status": manifest.get("status", ""),
                "name": config.get("name", ""),
                "tags": ";".join(config.get("tags", [])),
                "model": config.get("model", {}).get("path", ""),
                "backend": config.get("model", {}).get("backend", ""),
                "split": config.get("data", {}).get("split", ""),
                "task": config.get("data", {}).get("task", ""),
                "accuracy": metrics.get("accuracy", ""),
                "total": metrics.get("total", ""),
                "correct": metrics.get("correct", ""),
                "git_commit": manifest.get("git", {}).get("commit", ""),
                "dashboard": rel(dashboard_path) if dashboard_path.exists() else (rel(legacy_analysis_path) if legacy_analysis_path.exists() else ""),
                "notes": rel(notes_path) if notes_path.exists() else "",
            }
        )
    rows.sort(key=lambda row: row.get("started_at", ""))
    return rows


def write_registry(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def write_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment Summary",
        "",
        "| Run | Status | Task | Backend | Accuracy | Correct/Total | Tags | Notes |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        dashboard = row.get("dashboard", "")
        notes = row.get("notes", "")
        links = []
        if dashboard:
            links.append(f"[dashboard]({dashboard})")
        if notes:
            links.append(f"[notes]({notes})")
        review_links = ", ".join(links)
        correct_total = ""
        if row.get("correct") != "" or row.get("total") != "":
            correct_total = f"{row.get('correct', '')}/{row.get('total', '')}"
        lines.append(
            "| {run_id} | {status} | {task} | {backend} | {accuracy} | {correct_total} | {tags} | {notes} |".format(
                run_id=row.get("run_id", ""),
                status=row.get("status", ""),
                task=row.get("task", ""),
                backend=row.get("backend", ""),
                accuracy=row.get("accuracy", ""),
                correct_total=correct_total,
                tags=row.get("tags", ""),
                notes=review_links,
            )
        )
    lines.extend(
        [
            "",
            "## Reading Notes",
            "",
            "- Compare runs that changed only one variable when possible.",
            "- Treat accuracy changes on partial/subset runs as directional, not final.",
            "- Write the decision in each run's `notes.md`: keep, retry, or discard.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize MindCube experiment runs")
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    rows = collect_runs(run_root)
    registry_path = Path(args.registry)
    if not registry_path.is_absolute():
        registry_path = PROJECT_ROOT / registry_path
    summary_path = Path(args.summary)
    if not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path
    write_registry(registry_path, rows)
    write_summary(summary_path, rows)
    print(f"Wrote {len(rows)} runs to {args.registry} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
