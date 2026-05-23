#!/usr/bin/env python3
"""Stitch setting-specific heldout predictions and create submission JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.core.extractors import extract_answer  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc


def prediction_text(item: Dict) -> str:
    for field in ("answer", "raw_response", "cogmap_gen_answer"):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def available_letters(item: Dict) -> List[str]:
    question = str(item.get("question", ""))
    letters: List[str] = []
    for letter in ("A", "B", "C", "D", "E"):
        if f"{letter}." in question:
            letters.append(letter)
    return letters or ["A", "B", "C", "D"]


def fallback_answer(item: Dict, policy: str) -> Optional[str]:
    if policy == "none":
        return None
    letters = available_letters(item)
    if policy == "first":
        return letters[0]
    if policy == "hash":
        key = str(item.get("id") or item.get("question") or "")
        digest = hashlib.sha1(key.encode("utf-8", errors="replace")).hexdigest()
        return letters[int(digest, 16) % len(letters)]
    raise ValueError(f"Unsupported fallback policy: {policy}")


def load_predictions(paths: List[Path]) -> Dict[str, Dict]:
    predictions: Dict[str, Dict] = {}
    duplicates: List[str] = []
    for path in paths:
        for item in iter_jsonl(path):
            item_id = str(item.get("id", ""))
            if not item_id:
                continue
            if item_id in predictions:
                duplicates.append(item_id)
            predictions[item_id] = item
    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise ValueError(f"Duplicate prediction ids ({len(duplicates)}): {preview}")
    return predictions


def combine(
    source_path: Path,
    prediction_paths: List[Path],
    output_predictions: Path,
    output_submission: Path,
    report_path: Path,
    fallback_policy: str,
) -> Dict:
    source_rows = list(iter_jsonl(source_path))
    predictions_by_id = load_predictions(prediction_paths)

    missing_predictions: List[str] = []
    missing_answers: List[str] = []
    combined_rows: List[Dict] = []
    submission_rows: List[Dict[str, str]] = []

    for source in source_rows:
        item_id = str(source.get("id", ""))
        prediction = predictions_by_id.get(item_id)
        if prediction is None:
            missing_predictions.append(item_id)
            continue

        answer = extract_answer(prediction_text(prediction))
        if not answer:
            answer = fallback_answer(source, fallback_policy)
            if answer:
                prediction = dict(prediction)
                prediction["heldout_fallback_answer"] = answer
            else:
                missing_answers.append(item_id)
                continue

        combined_rows.append(prediction)
        submission_rows.append({"id": item_id, "answer": answer})

    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    with output_predictions.open("w", encoding="utf-8") as handle:
        for row in combined_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_submission.parent.mkdir(parents=True, exist_ok=True)
    with output_submission.open("w", encoding="utf-8") as handle:
        for row in submission_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "source": str(source_path),
        "prediction_paths": [str(path) for path in prediction_paths],
        "output_predictions": str(output_predictions),
        "output_submission": str(output_submission),
        "source_total": len(source_rows),
        "prediction_ids": len(predictions_by_id),
        "combined_total": len(combined_rows),
        "submission_total": len(submission_rows),
        "missing_predictions": missing_predictions,
        "missing_answers": missing_answers,
        "fallback_policy": fallback_policy,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if missing_predictions or missing_answers:
        raise RuntimeError(
            "Heldout combine incomplete: "
            f"{len(missing_predictions)} missing predictions, {len(missing_answers)} missing answers. "
            f"See {report_path}"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine MindCube heldout predictions into a submission")
    parser.add_argument("--source", required=True, help="Raw heldout JSONL, used for final row order")
    parser.add_argument("--predictions", nargs="+", required=True, help="Setting-specific prediction JSONLs")
    parser.add_argument("--output-predictions", required=True, help="Combined prediction JSONL")
    parser.add_argument("--output-submission", required=True, help="Submission JSONL")
    parser.add_argument("--report", required=True, help="Construction report JSON")
    parser.add_argument(
        "--fallback-policy",
        choices=["none", "first", "hash"],
        default="none",
        help="Fallback only when a prediction has no parseable answer",
    )
    args = parser.parse_args()

    report = combine(
        source_path=Path(args.source),
        prediction_paths=[Path(path) for path in args.predictions],
        output_predictions=Path(args.output_predictions),
        output_submission=Path(args.output_submission),
        report_path=Path(args.report),
        fallback_policy=args.fallback_policy,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
