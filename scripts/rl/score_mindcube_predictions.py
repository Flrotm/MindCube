#!/usr/bin/env python3
"""Score a MindCube prediction JSONL and report format failures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.core.extractors import extract_answer, extract_json_from_text, get_setting_from_id  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def raw_response(item: Dict[str, Any]) -> str:
    for key in ("answer", "raw_response", "cogmap_gen_answer", "llm_raw_response", "response"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def update_bucket(bucket: Dict[str, int], correct: bool, answer_ok: bool, cogmap_ok: bool) -> None:
    bucket["total"] = bucket.get("total", 0) + 1
    bucket["correct"] = bucket.get("correct", 0) + int(correct)
    bucket["invalid_answer"] = bucket.get("invalid_answer", 0) + int(not answer_ok)
    bucket["invalid_cogmap_json"] = bucket.get("invalid_cogmap_json", 0) + int(not cogmap_ok)


def finalize(bucket: Dict[str, int]) -> Dict[str, Any]:
    total = bucket.get("total", 0)
    return {
        **bucket,
        "accuracy": round(bucket.get("correct", 0) / total * 100, 4) if total else 0.0,
        "invalid_answer_rate": round(bucket.get("invalid_answer", 0) / total * 100, 4) if total else 0.0,
        "invalid_cogmap_json_rate": round(bucket.get("invalid_cogmap_json", 0) / total * 100, 4) if total else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    overall: Dict[str, int] = {}
    settings: Dict[str, Dict[str, int]] = {}
    for item in iter_jsonl(args.input):
        text = raw_response(item)
        predicted = extract_answer(text)
        gt_answer = item.get("gt_answer")
        correct = bool(predicted and gt_answer and predicted.upper() == str(gt_answer).upper())
        answer_ok = predicted is not None
        cogmap_ok = extract_json_from_text(text) is not None
        setting = str(item.get("setting_tag") or get_setting_from_id(str(item.get("id", ""))))
        update_bucket(overall, correct, answer_ok, cogmap_ok)
        update_bucket(settings.setdefault(setting, {}), correct, answer_ok, cogmap_ok)

    summary = {
        "input": str(args.input),
        "overall": finalize(overall),
        "settings": {setting: finalize(bucket) for setting, bucket in sorted(settings.items())},
    }
    text = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
