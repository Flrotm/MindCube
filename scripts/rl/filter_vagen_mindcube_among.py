#!/usr/bin/env python3
"""Filter VAGEN MindCube RL JSONL files by setting tag.

The VAGEN MindCube branch ships the plain cogmap RL data as one mixed JSONL.
For the Gemma RL run we keep the official record shape but train only on
Among items, writing a sibling file instead of replacing the original.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_SOURCE = "crossviewQA_train_cogmap_and_reasoning_plain.jsonl"
DEFAULT_OUTPUT = "crossviewQA_train_cogmap_and_reasoning_plain_among.jsonl"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield item


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    return count


def resolve_path(value: str | None, data_dir: Path, fallback: str) -> Path:
    path = Path(value or fallback)
    if path.is_absolute():
        return path
    return data_dir / path


def filter_rows(rows: Iterable[Dict[str, Any]], setting: str) -> List[Dict[str, Any]]:
    wanted = setting.lower()
    return [
        item
        for item in rows
        if str(item.get("setting_tag", "")).lower() == wanted
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("vagen/env/crossview/MindCube_RL_Data"),
        help="Directory containing VAGEN MindCube_RL_Data JSONL files.",
    )
    parser.add_argument("--source", help=f"Source JSONL, default: {DEFAULT_SOURCE}")
    parser.add_argument("--output", help=f"Output JSONL, default: {DEFAULT_OUTPUT}")
    parser.add_argument("--setting", default="among", help="setting_tag value to keep.")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow writing an empty output file.",
    )
    args = parser.parse_args()

    source = resolve_path(args.source, args.data_dir, DEFAULT_SOURCE)
    output = resolve_path(args.output, args.data_dir, DEFAULT_OUTPUT)

    if source.resolve() == output.resolve():
        raise SystemExit("[ERROR] Refusing to overwrite the source JSONL.")
    if not source.exists():
        raise SystemExit(f"[ERROR] Source JSONL not found: {source}")

    all_rows = list(iter_jsonl(source))
    filtered = filter_rows(all_rows, args.setting)
    if not filtered and not args.allow_empty:
        raise SystemExit(
            f"[ERROR] No rows matched setting_tag={args.setting!r} in {source}"
        )

    written = write_jsonl(output, filtered)
    summary = {
        "source": str(source),
        "output": str(output),
        "setting": args.setting,
        "read": len(all_rows),
        "wrote": written,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
