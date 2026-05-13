#!/usr/bin/env python3
"""Create a representative MindCube prompt sample.

The sampler preserves the source distribution hierarchically:
1. setting, inferred from the item id (among / around / rotation / ...)
2. subproblem, represented by the configured fields, by default type+category

This is meant for small diagnostic runs where taking the first N rows would
overrepresent whichever setting happens to appear first in the generated file.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def setting_from_id(item_id: str) -> str:
    lowered = (item_id or "").lower()
    if "around" in lowered:
        return "around"
    if "rotation" in lowered:
        return "rotation"
    if "translation" in lowered:
        return "translation"
    if "among" in lowered:
        return "among"
    return "other"


def canonical_value(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(canonical_value(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def subproblem_key(row: Mapping[str, Any], fields: List[str]) -> str:
    parts = []
    for field in fields:
        parts.append(f"{field}={canonical_value(row.get(field, ''))}")
    return ";".join(parts)


def largest_remainder_allocation(
    capacities: Mapping[str, int],
    target: int,
    *,
    min_one_when_possible: bool = True,
) -> Dict[str, int]:
    """Allocate target items proportionally, never exceeding capacities."""
    capacities = {key: int(value) for key, value in capacities.items() if int(value) > 0}
    target = min(int(target), sum(capacities.values()))
    if target <= 0 or not capacities:
        return {key: 0 for key in capacities}

    allocation = {key: 0 for key in capacities}
    if min_one_when_possible and target >= len(capacities):
        for key in capacities:
            allocation[key] = 1
        target -= len(capacities)

    remaining_capacity = {
        key: capacity - allocation[key]
        for key, capacity in capacities.items()
        if capacity > allocation[key]
    }
    if target <= 0 or not remaining_capacity:
        return allocation

    total_capacity = sum(remaining_capacity.values())
    expected = {
        key: (capacity / total_capacity) * target
        for key, capacity in remaining_capacity.items()
    }
    floors = {
        key: min(remaining_capacity[key], int(value))
        for key, value in expected.items()
    }
    for key, value in floors.items():
        allocation[key] += value

    remaining = target - sum(floors.values())
    order = sorted(
        remaining_capacity,
        key=lambda key: (
            expected[key] - int(expected[key]),
            remaining_capacity[key],
            key,
        ),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for key in order:
            if remaining <= 0:
                break
            if allocation[key] < capacities[key]:
                allocation[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return allocation


def count_by(rows: Iterable[Dict[str, Any]], key_name: str) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts[str(row[key_name])] += 1
    return counts


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Make a representative MindCube JSONL sample")
    parser.add_argument("--input", required=True, help="Source JSONL prompt file")
    parser.add_argument("--output", required=True, help="Output sampled JSONL file")
    parser.add_argument("--count", type=int, default=100, help="Number of examples to sample")
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["among", "around", "rotation"],
        help="Settings to include, or 'all'",
    )
    parser.add_argument(
        "--subproblem-fields",
        nargs="+",
        default=["type", "category"],
        help="Fields defining subproblem strata inside each setting",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Sampling seed")
    parser.add_argument("--summary-output", help="Optional JSON summary path")
    args = parser.parse_args()

    source = resolve_path(args.input)
    output = resolve_path(args.output)
    settings = {setting.lower() for setting in args.settings}
    include_all = "all" in settings
    rng = random.Random(args.seed)

    rows = []
    for index, row in enumerate(iter_jsonl(source), start=1):
        row = dict(row)
        row["_source_index"] = index
        setting = setting_from_id(str(row.get("id", "")))
        if include_all or setting in settings:
            row["_setting"] = setting
            row["_subproblem"] = subproblem_key(row, args.subproblem_fields)
            rows.append(row)

    if not rows:
        raise SystemExit(f"No rows matched settings {sorted(settings)} in {source}")

    target = min(args.count, len(rows))
    by_setting: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_setting[row["_setting"]].append(row)

    setting_alloc = largest_remainder_allocation(
        {setting: len(items) for setting, items in by_setting.items()},
        target,
        min_one_when_possible=True,
    )

    selected: List[Dict[str, Any]] = []
    subproblem_allocations: Dict[str, Dict[str, int]] = {}
    source_subproblem_counts: Dict[str, Dict[str, int]] = {}

    for setting, setting_rows in sorted(by_setting.items()):
        setting_target = setting_alloc.get(setting, 0)
        if setting_target <= 0:
            continue

        by_subproblem: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in setting_rows:
            by_subproblem[row["_subproblem"]].append(row)

        sub_alloc = largest_remainder_allocation(
            {key: len(items) for key, items in by_subproblem.items()},
            setting_target,
            min_one_when_possible=True,
        )
        subproblem_allocations[setting] = sub_alloc
        source_subproblem_counts[setting] = {
            key: len(items) for key, items in sorted(by_subproblem.items())
        }

        for subproblem, sub_target in sorted(sub_alloc.items()):
            if sub_target <= 0:
                continue
            candidates = list(by_subproblem[subproblem])
            selected.extend(rng.sample(candidates, sub_target))

    rng.shuffle(selected)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in selected:
            clean_row = {
                key: value
                for key, value in row.items()
                if key not in {"_source_index", "_setting", "_subproblem"}
            }
            handle.write(json.dumps(clean_row, ensure_ascii=False) + "\n")

    selected_setting_counts = Counter(row["_setting"] for row in selected)
    selected_subproblem_counts: Dict[str, Counter] = defaultdict(Counter)
    for row in selected:
        selected_subproblem_counts[row["_setting"]][row["_subproblem"]] += 1

    summary = {
        "input": str(source),
        "output": str(output),
        "seed": args.seed,
        "requested_count": args.count,
        "selected_count": len(selected),
        "settings": sorted(by_setting),
        "subproblem_fields": args.subproblem_fields,
        "source_setting_counts": {
            key: len(value) for key, value in sorted(by_setting.items())
        },
        "selected_setting_counts": dict(sorted(selected_setting_counts.items())),
        "source_subproblem_counts": source_subproblem_counts,
        "selected_subproblem_counts": {
            setting: dict(sorted(counter.items()))
            for setting, counter in sorted(selected_subproblem_counts.items())
        },
    }

    if args.summary_output:
        write_json(resolve_path(args.summary_output), summary)

    print(f"Wrote {len(selected)} examples to {output}")
    print("Setting allocation:")
    for setting, count in sorted(selected_setting_counts.items()):
        source_count = len(by_setting[setting])
        print(f"  {setting}: {count}/{len(selected)} sampled from {source_count}")
    if args.summary_output:
        print(f"Summary: {resolve_path(args.summary_output)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
