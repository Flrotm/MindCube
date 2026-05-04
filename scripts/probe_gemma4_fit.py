#!/usr/bin/env python3
"""Probe Gemma 4 configs from biggest to smallest on the current hardware.

This script runs one MindCube sample per candidate config using fail-fast
inference. It is meant for Kaggle GPU sessions, before spending hours on a full
1050-sample run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = [
    "experiments/configs/gemma4_26b_a4b_raw_qa_t4x2_4bit_transformers.json",
    "experiments/configs/gemma4_e4b_raw_qa_t4x2_fp16_reasoning_transformers.json",
    "experiments/configs/gemma4_e4b_raw_qa_t4x2_4bit_reasoning_transformers.json",
    "experiments/configs/gemma4_e2b_raw_qa_answer_format_transformers.json",
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def slugify(value: str) -> str:
    cleaned = []
    previous_dash = False
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
            previous_dash = False
        elif not previous_dash:
            cleaned.append("-")
            previous_dash = True
    return "".join(cleaned).strip("-") or "candidate"


def rel(path: Path) -> str:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def make_probe_input(source: Path, destination: Path, sample_index: int, dry_run: bool = False) -> None:
    if dry_run and not source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        placeholder = {"input_prompt": "dry-run placeholder", "images": []}
        destination.write_text(json.dumps(placeholder) + "\n", encoding="utf-8")
        return

    selected: Optional[str] = None
    with source.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            if idx == sample_index:
                selected = line
                break
            if selected is None:
                selected = line

    if selected is None:
        raise ValueError(f"No JSONL rows found in {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(selected if selected.endswith("\n") else selected + "\n", encoding="utf-8")


def build_command(config: Dict[str, Any], input_file: Path, output_file: Path, max_new_tokens: int) -> List[str]:
    model = config.get("model", {})
    data = config.get("data", {})
    generation = config.get("generation", {})
    command = [
        sys.executable,
        "scripts/run_inference.py",
        "--model-type",
        model.get("type", "gemma4"),
        "--model-path",
        model.get("path", ""),
        "--backend",
        model.get("backend", "transformers"),
        "--input-file",
        rel(input_file),
        "--output-file",
        rel(output_file),
        "--image-root",
        data.get("image_root", "./data/"),
        "--batch-size",
        "1",
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        str(generation.get("temperature", 1.0)),
        "--top-p",
        str(generation.get("top_p", 0.95)),
    ]
    inference_config = model.get("inference_config")
    if inference_config:
        command.extend(["--config", inference_config])
    return command


def tee_command(command: List[str], log_path: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        if dry_run:
            print("$ " + " ".join(command))
            return 0

        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def read_probe_answer(output_file: Path) -> Dict[str, Any]:
    if not output_file.exists():
        return {"rows": 0, "answer": "", "extracted_answer": None}

    rows = []
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    answer = str(rows[0].get("answer", "")) if rows else ""
    extracted = None
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from evaluation.core.extractors import extract_answer

        extracted = extract_answer(answer)
    except Exception:
        extracted = None

    return {"rows": len(rows), "answer": answer, "extracted_answer": extracted}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Gemma 4 fit from largest to smallest")
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
        help="Experiment config to probe. Repeat to override the default ladder.",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="JSONL sample index to probe")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Probe generation budget")
    parser.add_argument("--probe-root", default="experiments/probes", help="Where to write probe artifacts")
    parser.add_argument("--try-all", action="store_true", help="Do not stop at the first fitting config")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without loading models")
    args = parser.parse_args()

    candidate_paths = args.candidates or DEFAULT_CANDIDATES
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    probe_root = Path(args.probe_root)
    if not probe_root.is_absolute():
        probe_root = PROJECT_ROOT / probe_root
    probe_dir = probe_root / f"{timestamp}-gemma4-fit-probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    results = []
    winner: Optional[Dict[str, Any]] = None

    for candidate in candidate_paths:
        config_path = Path(candidate)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config = load_json(config_path)
        name = config.get("name", config_path.stem)
        candidate_dir = probe_dir / slugify(name)
        candidate_dir.mkdir(parents=True, exist_ok=True)

        source_input = PROJECT_ROOT / config.get("data", {}).get("input_file", "")
        probe_input = candidate_dir / "probe_input.jsonl"
        output_file = candidate_dir / "predictions.jsonl"
        log_path = candidate_dir / "probe.log"
        make_probe_input(source_input, probe_input, args.sample_index, dry_run=args.dry_run)

        command = build_command(config, probe_input, output_file, args.max_new_tokens)
        print(f"\n=== Probing {name} ===")
        print(f"Model: {config.get('model', {}).get('path', '')}")
        return_code = tee_command(command, log_path, args.dry_run)
        answer_info = read_probe_answer(output_file) if not args.dry_run else {
            "rows": 0,
            "answer": "",
            "extracted_answer": None,
        }

        if args.dry_run:
            status = "dry_run"
        else:
            status = "fits" if return_code == 0 and answer_info["rows"] > 0 else "failed"
        result = {
            "name": name,
            "config": rel(config_path),
            "model": config.get("model", {}).get("path", ""),
            "return_code": return_code,
            "status": status,
            "rows": answer_info["rows"],
            "extracted_answer": answer_info["extracted_answer"],
            "probe_dir": rel(candidate_dir),
            "full_run_command": (
                "python scripts/run_experiment.py --config "
                f"{rel(config_path)} --inference-only"
            ),
        }
        results.append(result)
        print(f"Probe status: {status}")
        if answer_info["extracted_answer"]:
            print(f"Extracted answer: {answer_info['extracted_answer']}")

        if status == "fits" and winner is None:
            winner = result
            if not args.try_all:
                break

    summary = {"probe_dir": rel(probe_dir), "winner": winner, "results": results}
    summary_path = probe_dir / "fit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nProbe summary: {summary_path}")
    if winner:
        print("Biggest fitting config:")
        print(f"  {winner['config']}")
        print("Full run command:")
        print(f"  {winner['full_run_command']}")
        return 0

    if args.dry_run:
        print("Dry run complete; no models were loaded.")
    else:
        print("No candidate fit on this hardware.")
    return 1 if not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
