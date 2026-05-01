#!/usr/bin/env python3
"""Create a detailed review view for a MindCube experiment run."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.core.extractors import extract_answer, get_setting_from_id  # noqa: E402


EXAMPLE_FIELDS = [
    "id",
    "setting",
    "gt_answer",
    "pred_answer",
    "is_correct",
    "extraction_failed",
    "question",
    "options",
    "raw_response",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def get_raw_response(item: Dict[str, Any]) -> str:
    for field in ("answer", "raw_response", "cogmap_gen_answer"):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def compact_text(value: Any, limit: int = 600) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def get_options(item: Dict[str, Any]) -> str:
    for field in ("options", "choices", "candidate_answers"):
        if field in item:
            return compact_text(item[field], 1000)
    return ""


def answer_distribution(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counter = Counter(row.get(key) or "missing" for row in rows)
    return dict(sorted(counter.items()))


def build_examples(predictions_path: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for item in iter_jsonl(predictions_path):
        raw_response = get_raw_response(item)
        pred_answer = extract_answer(raw_response)
        gt_answer = item.get("gt_answer")
        is_correct = bool(gt_answer and pred_answer == gt_answer)
        item_id = item.get("id", "")
        examples.append(
            {
                "id": item_id,
                "setting": get_setting_from_id(item_id),
                "gt_answer": gt_answer or "",
                "pred_answer": pred_answer or "",
                "is_correct": is_correct,
                "extraction_failed": not bool(pred_answer),
                "question": compact_text(item.get("question", "")),
                "options": get_options(item),
                "raw_response": compact_text(raw_response, 1200),
            }
        )
    return examples


def write_examples_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXAMPLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in EXAMPLE_FIELDS})


def setting_breakdown(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["setting"]].append(row)

    breakdown = []
    for setting, setting_rows in sorted(grouped.items()):
        total = len(setting_rows)
        correct = sum(1 for row in setting_rows if row["is_correct"])
        breakdown.append(
            {
                "setting": setting,
                "total": total,
                "correct": correct,
                "accuracy": round(correct / total * 100, 2) if total else 0.0,
                "extraction_failed": sum(1 for row in setting_rows if row["extraction_failed"]),
            }
        )
    return breakdown


def confusion_table(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        gt = row.get("gt_answer") or "missing"
        pred = row.get("pred_answer") or "missing"
        table[gt][pred] += 1
    return {gt: dict(sorted(preds.items())) for gt, preds in sorted(table.items())}


def sample_rows(rows: List[Dict[str, Any]], correct: bool, limit: int) -> List[Dict[str, Any]]:
    return [row for row in rows if row["is_correct"] is correct][:limit]


def markdown_example(row: Dict[str, Any]) -> str:
    return (
        f"- `{row['id']}` [{row['setting']}]: "
        f"gt `{row['gt_answer']}`, pred `{row['pred_answer'] or 'missing'}`. "
        f"Q: {row['question']} "
        f"Response: {row['raw_response']}"
    )


def table_lines(rows: List[Dict[str, Any]], columns: List[str]) -> List[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_analysis_md(
    path: Path,
    run_dir: Path,
    manifest: Dict[str, Any],
    metrics: Dict[str, Any],
    examples: List[Dict[str, Any]],
    sample_limit: int,
) -> None:
    wrong = [row for row in examples if not row["is_correct"]]
    extraction_failed = [row for row in examples if row["extraction_failed"]]
    breakdown = setting_breakdown(examples)
    pred_dist = answer_distribution(examples, "pred_answer")
    gt_dist = answer_distribution(examples, "gt_answer")
    confusion = confusion_table(examples)

    lines = [
        f"# Analysis: {manifest.get('run_id', run_dir.name)}",
        "",
        "## Headline",
        f"- Status: {manifest.get('status', '')}",
        f"- Accuracy: {metrics.get('accuracy', '')}",
        f"- Correct/total: {metrics.get('correct', '')}/{metrics.get('total', '')}",
        f"- Predictions reviewed: {len(examples)}",
        f"- Extraction failures: {len(extraction_failed)}",
        "",
        "## Breakdown By Setting",
        *table_lines(breakdown, ["setting", "accuracy", "correct", "total", "extraction_failed"]),
        "",
        "## Answer Distribution",
        f"- Ground truth: `{gt_dist}`",
        f"- Predicted: `{pred_dist}`",
        "",
        "## Confusion Table",
        "Rows are ground truth answers; nested counts are predicted answers.",
        "",
        "```json",
        json.dumps(confusion, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Error Samples",
    ]
    if wrong:
        lines.extend(markdown_example(row) for row in sample_rows(examples, False, sample_limit))
    else:
        lines.append("- No errors found in the reviewed predictions.")

    lines.extend(["", "## Correct Samples"])
    correct_samples = sample_rows(examples, True, sample_limit)
    if correct_samples:
        lines.extend(markdown_example(row) for row in correct_samples)
    else:
        lines.append("- No correct samples found in the reviewed predictions.")

    lines.extend(
        [
            "",
            "## Review Checklist",
            "- Are mistakes concentrated in one setting?",
            "- Is the model over-predicting one answer letter?",
            "- Are extraction failures caused by prompt format or answer parser limits?",
            "- Do wrong answers show visual confusion, spatial reasoning failure, or instruction-following failure?",
            "- What exact prompt/model/config change should be tried next?",
            "",
            "## Files",
            f"- Examples CSV: `{(run_dir / 'examples.csv').relative_to(PROJECT_ROOT)}`",
            f"- Full evaluation JSON: `{(run_dir / 'evaluation.json').relative_to(PROJECT_ROOT)}`",
            f"- Notes: `{(run_dir / 'notes.md').relative_to(PROJECT_ROOT)}`",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_run(run_dir: Path, sample_limit: int) -> Dict[str, Any]:
    manifest = load_json(run_dir / "manifest.json")
    metrics = load_json(run_dir / "metrics.json")
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        artifact_path = manifest.get("artifacts", {}).get("predictions", "")
        if artifact_path:
            predictions_path = PROJECT_ROOT / artifact_path
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found for {run_dir}")

    examples = build_examples(predictions_path)
    write_examples_csv(run_dir / "examples.csv", examples)
    write_analysis_md(run_dir / "analysis.md", run_dir, manifest, metrics, examples, sample_limit)

    return {
        "examples": len(examples),
        "errors": sum(1 for row in examples if not row["is_correct"]),
        "extraction_failed": sum(1 for row in examples if row["extraction_failed"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create detailed review files for a MindCube run")
    parser.add_argument("--run-dir", required=True, help="Run folder under experiments/runs")
    parser.add_argument("--sample-limit", type=int, default=20, help="Number of correct/error samples in Markdown")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    stats = analyze_run(run_dir, args.sample_limit)
    print(
        "Wrote analysis for {examples} examples: {errors} errors, {extraction_failed} extraction failures".format(
            **stats
        )
    )
    print(f"Analysis: {run_dir / 'analysis.md'}")
    print(f"Examples: {run_dir / 'examples.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
