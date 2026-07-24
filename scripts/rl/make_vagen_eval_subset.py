#!/usr/bin/env python3
"""Create a small VAGEN-compatible MindCube eval JSONL.

The preferred path keeps VAGEN's official RL record shape and selects records
by IDs from an existing MindCube test subset, preserving that subset's order.
If no ID file is provided, the script can take the first N records from the
source after optional setting filtering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_SOURCE = "crossviewQA_tinybench_cogmap_and_reasoning_plain.jsonl"
DEFAULT_OUTPUT = "crossviewQA_tinybench_cogmap_and_reasoning_plain_rep100.jsonl"


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


def load_ordered_ids(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    ids: List[str] = []
    seen = set()
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            candidates = data.get("ids") or data.get("examples") or data.get("records")
        else:
            candidates = data
        if not isinstance(candidates, list):
            raise ValueError(f"{path}: expected a JSON list or dict with ids/examples")
        for value in candidates:
            item_id = value.get("id") if isinstance(value, dict) else value
            if item_id is not None and str(item_id) not in seen:
                ids.append(str(item_id))
                seen.add(str(item_id))
        return ids

    for item in iter_jsonl(path):
        item_id = item.get("id")
        if item_id is not None and str(item_id) not in seen:
            ids.append(str(item_id))
            seen.add(str(item_id))
    return ids


def setting_of(item: Dict[str, Any]) -> str:
    value = item.get("setting_tag")
    if value:
        return str(value).lower()
    item_id = str(item.get("id", "")).lower()
    if item_id.startswith("among"):
        return "among"
    if item_id.startswith("around"):
        return "around"
    if item_id.startswith("rotation"):
        return "rotation"
    if item_id.startswith("translation"):
        return "translation"
    return "other"


def setting_allowed(item: Dict[str, Any], settings: Sequence[str]) -> bool:
    if not settings:
        return True
    wanted = {value.lower() for value in settings}
    return setting_of(item) in wanted


def select_by_ids(
    source_rows: List[Dict[str, Any]],
    ordered_ids: List[str],
    settings: Sequence[str],
) -> tuple[List[Dict[str, Any]], List[str]]:
    by_id = {str(item.get("id")): item for item in source_rows}
    selected: List[Dict[str, Any]] = []
    missing: List[str] = []
    for item_id in ordered_ids:
        item = by_id.get(item_id)
        if item is None:
            missing.append(item_id)
            continue
        if setting_allowed(item, settings):
            selected.append(item)
    return selected, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("vagen/env/crossview/MindCube_RL_Data"),
        help="Directory containing VAGEN MindCube_RL_Data JSONL files.",
    )
    parser.add_argument("--source", help=f"Source VAGEN JSONL, default: {DEFAULT_SOURCE}")
    parser.add_argument("--ids-from", type=Path, help="JSONL/JSON file with IDs to preserve.")
    parser.add_argument("--output", help=f"Output JSONL, default: {DEFAULT_OUTPUT}")
    parser.add_argument("--limit", type=int, default=100, help="Maximum rows to write. Use 0 for no limit.")
    parser.add_argument(
        "--settings",
        nargs="*",
        default=[],
        help="Optional settings to keep, for example: --settings among around",
    )
    parser.add_argument(
        "--allow-missing-ids",
        action="store_true",
        help="Do not fail when IDs from --ids-from are absent from the VAGEN source.",
    )
    args = parser.parse_args()

    source = resolve_path(args.source, args.data_dir, DEFAULT_SOURCE)
    output = resolve_path(args.output, args.data_dir, DEFAULT_OUTPUT)

    if source.resolve() == output.resolve():
        raise SystemExit("[ERROR] Refusing to overwrite the source JSONL.")
    if not source.exists():
        raise SystemExit(f"[ERROR] Source VAGEN JSONL not found: {source}")

    source_rows = list(iter_jsonl(source))
    missing: List[str] = []
    if args.ids_from:
        ordered_ids = load_ordered_ids(args.ids_from)
        selected, missing = select_by_ids(source_rows, ordered_ids, args.settings)
        if missing and not args.allow_missing_ids:
            preview = ", ".join(missing[:10])
            raise SystemExit(
                f"[ERROR] {len(missing)} IDs from {args.ids_from} were not found in {source}: {preview}"
            )
    else:
        selected = [item for item in source_rows if setting_allowed(item, args.settings)]

    if args.limit and args.limit > 0:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("[ERROR] No rows selected for eval subset.")

    written = write_jsonl(output, selected)
    counts: Dict[str, int] = {}
    for item in selected:
        counts[setting_of(item)] = counts.get(setting_of(item), 0) + 1

    summary = {
        "source": str(source),
        "ids_from": str(args.ids_from) if args.ids_from else None,
        "output": str(output),
        "limit": args.limit,
        "settings": args.settings,
        "source_rows": len(source_rows),
        "written": written,
        "missing_ids": len(missing),
        "settings_count": counts,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
