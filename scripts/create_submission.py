#!/usr/bin/env python3
"""Convert MindCube prediction JSONL into EvalAI submission JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.core.extractors import extract_answer  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc


def get_raw_response(item: Dict) -> str:
    for field in ("answer", "raw_response", "cogmap_gen_answer"):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def convert(input_path: Path, output_path: Path, include_raw_response: bool) -> Dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    written = 0
    missing = 0

    with output_path.open("w", encoding="utf-8") as out:
        for item in iter_jsonl(input_path):
            total += 1
            item_id: Optional[str] = item.get("id")
            raw_response = get_raw_response(item)
            answer = extract_answer(raw_response)

            if not item_id or not answer:
                missing += 1
                continue

            submission_item = {"id": item_id, "answer": answer}
            if include_raw_response:
                submission_item["raw_response"] = raw_response
            out.write(json.dumps(submission_item, ensure_ascii=False) + "\n")
            written += 1

    return {"total": total, "written": written, "missing": missing}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create EvalAI JSONL submission from MindCube predictions")
    parser.add_argument("--input", required=True, help="Prediction JSONL from run_experiment.py or run_inference.py")
    parser.add_argument("--output", required=True, help="EvalAI submission JSONL path")
    parser.add_argument(
        "--no-raw-response",
        action="store_true",
        help="Do not include raw_response in submission rows",
    )
    args = parser.parse_args()

    stats = convert(
        Path(args.input),
        Path(args.output),
        include_raw_response=not args.no_raw_response,
    )
    print(
        "Wrote {written}/{total} submission rows to {output} ({missing} missing)".format(
            output=args.output,
            **stats,
        )
    )
    return 0 if stats["written"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
