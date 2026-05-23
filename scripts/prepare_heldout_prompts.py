#!/usr/bin/env python3
"""Prepare model-ready MindCube heldout prompts.

The public prompt generator expects ground-truth answers because it was built
for train/eval splits. Heldout rows intentionally do not have answers, so this
script uses the same prompt templates directly and only writes inference fields.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_generation.templates import get_template  # noqa: E402
from src.scaffold_curation.cogmap.generators import CogMapGenerator  # noqa: E402


SETTINGS = ("rotation", "around", "among")


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc


def detect_setting(item: Dict) -> str:
    item_id = str(item.get("id", "")).lower()
    for setting in SETTINGS:
        if setting in item_id:
            return setting
    raise ValueError(f"Could not detect heldout setting from id={item.get('id')!r}")


def clean_for_inference(item: Dict) -> Dict:
    """Drop answer-only fields if a caller accidentally passes eval data."""
    result = copy.deepcopy(item)
    for field in ("gt_answer", "answer", "grounded_output"):
        result.pop(field, None)
    return result


def add_raw_prompt(item: Dict) -> Dict:
    result = clean_for_inference(item)
    result["input_prompt"] = get_template("raw_qa").generate_prompt(result)
    result["prompt_task"] = "raw_qa"
    return result


def add_plain_cogmap_prompt(item: Dict, generator: CogMapGenerator) -> Dict:
    result = clean_for_inference(item)
    original_meta_info = copy.deepcopy(result.get("meta_info"))
    result = generator.add_cogmap_to_item(result)
    if "plain_cogmap_gen_instruction" not in result:
        raise ValueError(f"Could not generate plain cogmap instruction for {result.get('id')}")
    if original_meta_info is not None:
        result["meta_info"] = original_meta_info
    result["input_prompt"] = get_template("plain_cgmap_ffr_out").generate_prompt(result)
    result["prompt_task"] = "plain_cgmap_ffr_out"
    return result


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(input_path: Path, output_dir: Path) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cogmap_generator = CogMapGenerator(format_type="full", suppress_warnings=True)

    rows_by_setting: Dict[str, List[Dict]] = {setting: [] for setting in SETTINGS}
    heldout_order: List[str] = []
    errors: List[Dict[str, str]] = []

    for item in iter_jsonl(input_path):
        item_id = str(item.get("id", ""))
        heldout_order.append(item_id)
        setting = detect_setting(item)
        try:
            if setting == "rotation":
                prepared = add_raw_prompt(item)
            else:
                prepared = add_plain_cogmap_prompt(item, cogmap_generator)
            prepared["heldout_setting"] = setting
            rows_by_setting[setting].append(prepared)
        except Exception as exc:
            errors.append({"id": item_id, "setting": setting, "error": str(exc)})

    if errors:
        error_path = output_dir / "prepare_errors.json"
        error_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise RuntimeError(f"Failed to prepare {len(errors)} rows; see {error_path}")

    output_paths = {
        "rotation": output_dir / "heldout_rotation_raw_qa.jsonl",
        "around": output_dir / "heldout_around_plain_cgmap_ffr_out.jsonl",
        "among": output_dir / "heldout_among_plain_cgmap_ffr_out.jsonl",
    }
    for setting, rows in rows_by_setting.items():
        write_jsonl(output_paths[setting], rows)

    order_path = output_dir / "heldout_order.json"
    order_path.write_text(json.dumps(heldout_order, indent=2) + "\n", encoding="utf-8")

    counts = Counter({setting: len(rows) for setting, rows in rows_by_setting.items()})
    report = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "total": len(heldout_order),
        "counts": dict(counts),
        "outputs": {setting: str(path) for setting, path in output_paths.items()},
        "order": str(order_path),
    }
    (output_dir / "prepare_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare MindCube heldout prompts for inference")
    parser.add_argument("--input", required=True, help="Raw heldout JSONL")
    parser.add_argument("--output-dir", required=True, help="Directory for prepared prompt JSONLs")
    args = parser.parse_args()

    report = prepare(Path(args.input), Path(args.output_dir))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
