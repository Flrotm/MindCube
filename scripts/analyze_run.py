#!/usr/bin/env python3
"""Create a standalone HTML dashboard for a MindCube experiment run."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.core.extractors import extract_answer, get_setting_from_id  # noqa: E402


OPTION_LABEL_RE = re.compile(r"(?<![A-Za-z0-9])([A-E])\.\s*")

REFERENCE_CATEGORY_TABLES = [
    {
        "title": "Table 1 | Performance of VLMs on MINDCUBE across categories (Part 1)",
        "columns": [
            ("overall", "Overall"),
            ("linear", "Linear"),
            ("perpendicular", "Perp."),
            ("self", "Self"),
            ("level1", "Level1"),
            ("level2", "Level2"),
        ],
        "groups": [
            ("Object Arrangement", [("linear", "Linear"), ("perpendicular", "Perp.")]),
            ("Perspective Taking", [("self", "Self"), ("level1", "Level1"), ("level2", "Level2")]),
        ],
        "rows": [
            {"model": "LLaVA-Video-7B-Qwen2", "overall": 41.96, "linear": 30.12, "perpendicular": 43.11, "self": 42.19, "level1": 60.76, "level2": 33.80},
            {"model": "Mantis(SigLip)", "overall": 41.04, "linear": 50.99, "perpendicular": 40.08, "self": 41.20, "level1": 54.43, "level2": 35.41},
            {"model": "GPT-4o", "overall": 38.81, "linear": 29.16, "perpendicular": 39.75, "self": 39.07, "level1": 46.20, "level2": 31.86},
            {"model": "Qwen2.5-VL-3B-Instruct", "overall": 33.21, "linear": 30.34, "perpendicular": 33.49, "self": 32.96, "level1": 46.84, "level2": 36.28},
            {"model": "LongVA-7B", "overall": 29.46, "linear": 24.88, "perpendicular": 29.91, "self": 28.81, "level1": 51.90, "level2": 39.83},
            {"model": "Qwen2.5-VL-7B-Instruct", "overall": 29.26, "linear": 21.35, "perpendicular": 30.02, "self": 28.77, "level1": 46.84, "level2": 36.81},
            {"model": "deepseek-vl2-small", "overall": 47.62, "linear": 26.91, "perpendicular": 49.63, "self": 48.32, "level1": 56.33, "level2": 31.11},
            {"model": "Robobrain", "overall": 37.38, "linear": 29.53, "perpendicular": 38.14, "self": 37.56, "level1": 55.06, "level2": 30.57},
            {"model": "Claude-sonnet-4", "overall": 44.75, "linear": 47.62, "perpendicular": 44.48, "self": 45.32, "level1": 49.38, "level2": 31.74},
            {"model": "Space-Mantis", "overall": 22.82, "linear": 29.32, "perpendicular": 22.19, "self": 22.15, "level1": 45.57, "level2": 33.48},
            {"model": "InternVL2-8B", "overall": 18.68, "linear": 13.11, "perpendicular": 19.22, "self": 17.89, "level1": 64.56, "level2": 27.99},
            {"model": "Space-Qwen", "overall": 33.28, "linear": 26.32, "perpendicular": 33.95, "self": 33.06, "level1": 46.84, "level2": 35.63},
            {"model": "LLaVA-Onevision-7B", "overall": 47.43, "linear": 44.09, "perpendicular": 47.75, "self": 48.04, "level1": 51.27, "level2": 33.48},
            {"model": "Spatial-MLLM", "overall": 32.06, "linear": 20.92, "perpendicular": 33.13, "self": 31.79, "level1": 46.84, "level2": 35.20},
            {"model": "mPLUG-Owl3-7B", "overall": 44.85, "linear": 26.91, "perpendicular": 46.59, "self": 45.15, "level1": 60.13, "level2": 35.74},
        ],
    },
    {
        "title": "Table 2 | Performance of VLMs on MINDCUBE across categories (Part 2)",
        "columns": [
            ("aa", "A-A"),
            ("ao", "A-O"),
            ("oo", "O-O"),
            ("rotation", "Rotation"),
            ("meanwhile", "Meanwhile"),
            ("sequence", "Sequence"),
        ],
        "groups": [
            ("Relation Pattern", [("aa", "A-A"), ("ao", "A-O"), ("oo", "O-O")]),
            ("Viewpoint Dynamics", [("rotation", "Rotation"), ("meanwhile", "Meanwhile"), ("sequence", "Sequence")]),
        ],
        "rows": [
            {"model": "LLaVA-Video-7B-Qwen2", "aa": 36.22, "ao": 57.61, "oo": 26.67, "rotation": 35.71, "meanwhile": 30.12, "sequence": 73.45},
            {"model": "Mantis(SigLip)", "aa": 23.78, "ao": 64.16, "oo": 25.24, "rotation": 37.65, "meanwhile": 24.99, "sequence": 82.74},
            {"model": "GPT-4o", "aa": 49.30, "ao": 48.38, "oo": 16.70, "rotation": 32.65, "meanwhile": 31.09, "sequence": 59.73},
            {"model": "Qwen2.5-VL-3B-Instruct", "aa": 37.85, "ao": 37.51, "oo": 20.65, "rotation": 37.37, "meanwhile": 27.88, "sequence": 46.05},
            {"model": "LongVA-7B", "aa": 19.72, "ao": 35.49, "oo": 25.58, "rotation": 35.89, "meanwhile": 24.67, "sequence": 40.50},
            {"model": "Qwen2.5-VL-7B-Instruct", "aa": 31.41, "ao": 34.67, "oo": 15.63, "rotation": 38.76, "meanwhile": 22.87, "sequence": 43.76},
            {"model": "deepseek-vl2-small", "aa": 43.98, "ao": 68.27, "oo": 25.33, "rotation": 37.00, "meanwhile": 32.97, "sequence": 87.13},
            {"model": "Robobrain", "aa": 30.94, "ao": 49.18, "oo": 27.37, "rotation": 35.80, "meanwhile": 28.79, "sequence": 59.66},
            {"model": "Claude-sonnet-4", "aa": 41.78, "ao": 67.25, "oo": 15.85, "rotation": 48.42, "meanwhile": 34.76, "sequence": 69.53},
            {"model": "Space-Mantis", "aa": 28.18, "ao": 17.03, "oo": 20.89, "rotation": 37.65, "meanwhile": 24.98, "sequence": 14.46},
            {"model": "InternVL2-8B", "aa": 15.67, "ao": 12.47, "oo": 24.58, "rotation": 36.45, "meanwhile": 21.78, "sequence": 7.36},
            {"model": "Space-Qwen", "aa": 31.59, "ao": 38.14, "oo": 26.13, "rotation": 38.02, "meanwhile": 28.51, "sequence": 44.58},
            {"model": "LLaVA-Onevision-7B", "aa": 42.28, "ao": 65.87, "oo": 29.79, "rotation": 36.45, "meanwhile": 33.80, "sequence": 84.38},
            {"model": "Spatial-MLLM", "aa": 27.72, "ao": 37.75, "oo": 25.80, "rotation": 38.39, "meanwhile": 26.84, "sequence": 44.19},
            {"model": "mPLUG-Owl3-7B", "aa": 47.80, "ao": 62.29, "oo": 18.83, "rotation": 37.84, "meanwhile": 31.02, "sequence": 81.55},
        ],
    },
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


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


def get_raw_response(item: Dict[str, Any]) -> str:
    for field in ("answer", "raw_response", "cogmap_gen_answer"):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def get_options(item: Dict[str, Any]) -> str:
    for field in ("options", "choices", "candidate_answers"):
        if field in item:
            return compact_text(item[field], 1000)
    return ""


def extract_choice_labels(question: str) -> List[str]:
    labels: List[str] = []
    for match in OPTION_LABEL_RE.finditer(question or ""):
        label = match.group(1)
        if label not in labels:
            labels.append(label)
    return labels


def random_chance(choice_count: int) -> float | None:
    if choice_count <= 0:
        return None
    return 100.0 / choice_count


def build_examples(predictions_path: Path) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for item in iter_jsonl(predictions_path):
        raw_response = get_raw_response(item)
        pred_answer = extract_answer(raw_response)
        gt_answer = item.get("gt_answer")
        is_correct = bool(gt_answer and pred_answer == gt_answer)
        item_id = item.get("id", "")
        question = str(item.get("question", ""))
        choice_labels = extract_choice_labels(question)
        choice_count = len(choice_labels)
        examples.append(
            {
                "id": item_id,
                "setting": get_setting_from_id(item_id),
                "category": item.get("category", ""),
                "type": item.get("type", ""),
                "gt_answer": gt_answer or "",
                "pred_answer": pred_answer or "",
                "is_correct": is_correct,
                "extraction_failed": not bool(pred_answer),
                "question": compact_text(question, 900),
                "options": get_options(item),
                "choice_labels": choice_labels,
                "choice_count": choice_count,
                "random_chance": round(random_chance(choice_count), 4) if choice_count else None,
                "raw_response": compact_text(raw_response, 1000),
            }
        )
    return examples


def answer_distribution(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counter = Counter(row.get(key) or "missing" for row in rows)
    return dict(sorted(counter.items()))


def setting_breakdown(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["setting"]].append(row)

    breakdown = []
    for setting, setting_rows in sorted(grouped.items()):
        total = len(setting_rows)
        correct = sum(1 for row in setting_rows if row["is_correct"])
        baseline = random_baseline(setting_rows)
        model_accuracy = round(correct / total * 100, 2) if total else 0.0
        random_accuracy = baseline.get("accuracy")
        breakdown.append(
            {
                "setting": setting,
                "total": total,
                "correct": correct,
                "wrong": total - correct,
                "accuracy": model_accuracy,
                "random_accuracy": random_accuracy,
                "random_expected_correct": baseline.get("expected_correct"),
                "random_parsed": baseline.get("parsed"),
                "delta_vs_random": round(model_accuracy - random_accuracy, 2)
                if random_accuracy is not None
                else None,
                "extraction_failed": sum(1 for row in setting_rows if row["extraction_failed"]),
            }
        )
    return breakdown


def random_baseline(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    parsed_rows = [row for row in rows if row.get("choice_count")]
    expected_correct = sum(1.0 / row["choice_count"] for row in parsed_rows)
    parsed = len(parsed_rows)
    total = len(rows)
    accuracy = expected_correct / parsed * 100 if parsed else None
    return {
        "total": total,
        "parsed": parsed,
        "unparsed": total - parsed,
        "expected_correct": round(expected_correct, 2),
        "accuracy": round(accuracy, 2) if accuracy is not None else None,
    }


def category_at(row: Dict[str, Any], index: int) -> str:
    category = row.get("category")
    if isinstance(category, list) and len(category) > index:
        return str(category[index]).strip().lower()
    return ""


def score_rows(rows: List[Dict[str, Any]], predicate: Any) -> Dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    total = len(selected)
    correct = sum(1 for row in selected if row["is_correct"])
    accuracy = correct / total * 100 if total else None
    return {
        "accuracy": round(accuracy, 2) if accuracy is not None else None,
        "correct": correct,
        "total": total,
    }


def category_scores(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    relation_aliases = {
        "aa": {"a-a", "aa", "p-p", "pp"},
        "ao": {"a-o", "ao", "p-o", "po"},
        "oo": {"o-o", "oo"},
    }
    return {
        "overall": score_rows(rows, lambda row: True),
        "linear": score_rows(rows, lambda row: category_at(row, 0) == "linear"),
        "perpendicular": score_rows(rows, lambda row: category_at(row, 0) == "perpendicular"),
        "self": score_rows(rows, lambda row: category_at(row, 3) == "self"),
        "level1": score_rows(rows, lambda row: category_at(row, 3) == "level1"),
        "level2": score_rows(rows, lambda row: category_at(row, 3) == "level2"),
        "aa": score_rows(rows, lambda row: category_at(row, 1) in relation_aliases["aa"]),
        "ao": score_rows(rows, lambda row: category_at(row, 1) in relation_aliases["ao"]),
        "oo": score_rows(rows, lambda row: category_at(row, 1) in relation_aliases["oo"]),
        "rotation": score_rows(rows, lambda row: category_at(row, 2) == "rotation"),
        "meanwhile": score_rows(rows, lambda row: category_at(row, 2) == "meanwhile"),
        "sequence": score_rows(rows, lambda row: category_at(row, 2) == "sequence"),
    }


def confusion_table(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        gt = row.get("gt_answer") or "missing"
        pred = row.get("pred_answer") or "missing"
        table[gt][pred] += 1
    return {gt: dict(sorted(preds.items())) for gt, preds in sorted(table.items())}


def get_labels(*distributions: Dict[str, int]) -> List[str]:
    labels = set()
    for distribution in distributions:
        labels.update(distribution.keys())
    preferred = ["A", "B", "C", "D", "E", "missing"]
    return [label for label in preferred if label in labels] + sorted(labels.difference(preferred))


def log_tail(run_dir: Path, limit: int = 120) -> str:
    candidates = [run_dir / "logs" / "finalize.log", run_dir / "logs" / "run.log"]
    lines: List[str] = []
    for path in candidates:
        if path.exists():
            lines.extend(path.read_text(encoding="utf-8", errors="replace").splitlines())
    return "\n".join(lines[-limit:])


def data_script(name: str, data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"<script id=\"{name}\" type=\"application/json\">{payload}</script>"


def render_kpis(metrics: Dict[str, Any], examples: List[Dict[str, Any]]) -> str:
    total = metrics.get("total", len(examples))
    correct = metrics.get("correct", sum(1 for row in examples if row["is_correct"]))
    accuracy = metrics.get("accuracy", 0.0)
    extraction_failed = sum(1 for row in examples if row["extraction_failed"])
    wrong = max(int(total or 0) - int(correct or 0), 0)
    cards = [
        ("Accuracy", f"{accuracy}%", "primary"),
        ("Correct", f"{correct}/{total}", "green"),
        ("Wrong", str(wrong), "red"),
        ("Extraction Failures", str(extraction_failed), "amber"),
    ]
    return "\n".join(
        f"""
        <section class="kpi kpi-{tone}">
          <span>{esc(label)}</span>
          <strong>{esc(value)}</strong>
        </section>
        """
        for label, value, tone in cards
    )


def fmt_pct(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.2f}%"


def fmt_num(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def hf_model_url(model_path: str) -> str:
    if "/" not in model_path or model_path.startswith((".", "/", "\\")):
        return ""
    return f"https://huggingface.co/{model_path}"


def render_key_value_grid(rows: List[tuple[str, Any]], raw_labels: set[str] | None = None) -> str:
    raw_labels = raw_labels or set()
    items = []
    for label, value in rows:
        rendered_value = str(value) if label in raw_labels else esc(value)
        items.append(
            f"""
        <div class="kv-item">
          <span>{esc(label)}</span>
          <strong>{rendered_value}</strong>
        </div>
        """
        )
    return "\n".join(items)


def render_command(command: Any) -> str:
    if isinstance(command, list):
        return " ".join(str(part) for part in command)
    if command is None:
        return ""
    return str(command)


def render_model_profile(config: Dict[str, Any], manifest: Dict[str, Any]) -> str:
    model = config.get("model", {})
    data = config.get("data", {})
    generation = config.get("generation", {})
    env = manifest.get("environment", {})
    git = manifest.get("git", {})
    local_git = manifest.get("local_git", {})
    commands = manifest.get("commands", {})

    model_path = str(model.get("path", ""))
    model_link = hf_model_url(model_path)
    model_name = (
        f"<a href=\"{esc(model_link)}\" target=\"_blank\" rel=\"noreferrer\">{esc(model_path)}</a>"
        if model_link
        else esc(model_path)
    )
    gpu_names = ", ".join(env.get("cuda_devices", [])) or "n/a"
    nvidia_smi = "; ".join(env.get("nvidia_smi", [])) or "n/a"
    cuda_status = "yes" if env.get("cuda_available") else "no"
    inference_command = render_command(commands.get("inference"))

    model_rows = [
        ("Model", model_name),
        ("Model Type", model.get("type", "")),
        ("Backend", model.get("backend", "")),
        ("Task", data.get("task", "")),
        ("Split", data.get("split", "")),
        ("Input File", data.get("input_file", "")),
        ("Image Root", data.get("image_root", "")),
    ]
    generation_rows = [
        ("Batch Size", generation.get("batch_size", "")),
        ("Max New Tokens", generation.get("max_new_tokens", "")),
        ("Temperature", generation.get("temperature", "")),
        ("Top-p", generation.get("top_p", "")),
    ]
    runtime_rows = [
        ("Kaggle GPU", gpu_names),
        ("NVIDIA SMI", nvidia_smi),
        ("CUDA Available", cuda_status),
        ("CUDA Devices", env.get("cuda_device_count", "")),
        ("Torch", env.get("torch", "")),
        ("Python", env.get("python", "")),
        ("Platform", env.get("platform", "")),
        ("Kernel Type", env.get("kaggle_kernel_run_type", "")),
    ]
    provenance_rows = [
        ("Kaggle Branch", git.get("branch", "")),
        ("Kaggle Commit", git.get("commit", "")),
        ("Kaggle Dirty", git.get("dirty", "")),
        ("Local Branch", local_git.get("branch", "")),
        ("Local Commit", local_git.get("commit", "")),
        ("Local Dirty", local_git.get("dirty", "")),
    ]

    return f"""
      <div class="model-profile">
        <article>
          <h3>Model</h3>
          <div class="kv-grid">{render_key_value_grid(model_rows, set(["Model"]))}</div>
        </article>
        <article>
          <h3>Generation</h3>
          <div class="kv-grid compact">{render_key_value_grid(generation_rows)}</div>
        </article>
        <article>
          <h3>Runtime</h3>
          <div class="kv-grid">{render_key_value_grid(runtime_rows)}</div>
        </article>
        <article>
          <h3>Code Version</h3>
          <div class="kv-grid compact">{render_key_value_grid(provenance_rows)}</div>
        </article>
      </div>
      <details class="command-block">
        <summary>Exact inference command</summary>
        <pre>{esc(inference_command)}</pre>
      </details>
    """


def render_setting_chart(breakdown: List[Dict[str, Any]]) -> str:
    rows = []
    for item in breakdown:
        accuracy = float(item.get("accuracy", 0.0))
        random_accuracy = item.get("random_accuracy")
        random_width = float(random_accuracy or 0.0)
        rows.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{esc(item["setting"])}</div>
              <div>
                <div class="bar-track"><div class="bar-fill" style="width:{accuracy:.2f}%"></div></div>
                <div class="bar-track baseline-track"><div class="bar-fill random-fill" style="width:{random_width:.2f}%"></div></div>
              </div>
              <div class="bar-value">
                {accuracy:.2f}% <span>{esc(item["correct"])}/{esc(item["total"])}</span><br>
                <small>random {fmt_pct(random_accuracy)}</small>
              </div>
            </div>
            """
        )
    return "\n".join(rows)


def render_random_baseline(overall_random: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    model_accuracy = float(metrics.get("accuracy") or 0.0)
    random_accuracy = overall_random.get("accuracy")
    delta = model_accuracy - random_accuracy if random_accuracy is not None else None
    delta_label = f"{delta:+.2f} pts" if delta is not None else "n/a"
    expected = overall_random.get("expected_correct")
    total = overall_random.get("total")
    parsed = overall_random.get("parsed")
    unparsed = overall_random.get("unparsed")
    return f"""
      <div class="baseline-hero">
        <div>
          <span>Random Total</span>
          <strong>{fmt_pct(random_accuracy)}</strong>
          <p>Uniform random over each question's parsed answer choices.</p>
        </div>
        <div>
          <span>Expected Correct</span>
          <strong>{fmt_num(expected)}/{esc(total)}</strong>
          <p>Choice counts parsed for {esc(parsed)}/{esc(total)} examples; unparsed {esc(unparsed)}.</p>
        </div>
        <div>
          <span>Model vs Random</span>
          <strong>{esc(delta_label)}</strong>
          <p>Model accuracy {fmt_pct(model_accuracy)} compared with random baseline.</p>
        </div>
      </div>
    """


def render_random_breakdown_table(breakdown: List[Dict[str, Any]]) -> str:
    rows = []
    for item in breakdown:
        rows.append(
            f"""
            <tr>
              <td>{esc(item["setting"])}</td>
              <td>{fmt_pct(item.get("accuracy"))}</td>
              <td>{fmt_pct(item.get("random_accuracy"))}</td>
              <td>{fmt_num(item.get("delta_vs_random"))} pts</td>
              <td>{fmt_num(item.get("random_expected_correct"))}/{esc(item.get("total"))}</td>
              <td>{esc(item.get("random_parsed"))}/{esc(item.get("total"))}</td>
            </tr>
            """
        )
    return f"""
    <table class="baseline-table">
      <thead>
        <tr>
          <th>Setting</th>
          <th>Model</th>
          <th>Random</th>
          <th>Delta</th>
          <th>Random Expected Correct</th>
          <th>Choice Count Coverage</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def render_current_category_cell(score: Dict[str, Any]) -> str:
    return f"""
      <td class="current-score">
        <strong>{fmt_pct(score.get("accuracy"))}</strong>
        <small>{esc(score.get("correct"))}/{esc(score.get("total"))}</small>
      </td>
    """


def render_benchmark_cell(value: Any) -> str:
    return f"<td>{fmt_pct(value)}</td>"


def render_category_benchmark_tables(current_scores: Dict[str, Dict[str, Any]]) -> str:
    blocks = []
    for table in REFERENCE_CATEGORY_TABLES:
        groups = table["groups"]
        columns = table["columns"]
        has_overall = columns[0][0] == "overall"
        top_header = ["<th rowspan=\"2\">Model</th>"]
        if has_overall:
            top_header.append("<th rowspan=\"2\">Overall</th>")
        for label, group_columns in groups:
            top_header.append(f"<th colspan=\"{len(group_columns)}\">{esc(label)}</th>")

        grouped_keys = {key for _, group_columns in groups for key, _ in group_columns}
        second_header = [
            f"<th>{esc(label)}</th>"
            for key, label in columns
            if key in grouped_keys
        ]

        current_cells = "".join(
            render_current_category_cell(current_scores.get(key, {}))
            for key, _ in columns
        )
        reference_rows = []
        for row in table["rows"]:
            cells = "".join(render_benchmark_cell(row.get(key)) for key, _ in columns)
            reference_rows.append(f"<tr><th>{esc(row['model'])}</th>{cells}</tr>")

        blocks.append(
            f"""
            <article class="benchmark-block">
              <h3>{esc(table["title"])}</h3>
              <div class="scroll-table benchmark-scroll">
                <table class="benchmark-table">
                  <thead>
                    <tr>{''.join(top_header)}</tr>
                    <tr>{''.join(second_header)}</tr>
                  </thead>
                  <tbody>
                    <tr class="current-run"><th>This run</th>{current_cells}</tr>
                    {''.join(reference_rows)}
                  </tbody>
                </table>
              </div>
            </article>
            """
        )
    return "\n".join(blocks)


def render_distribution_chart(gt_dist: Dict[str, int], pred_dist: Dict[str, int]) -> str:
    labels = get_labels(gt_dist, pred_dist)
    max_value = max([1] + [gt_dist.get(label, 0) for label in labels] + [pred_dist.get(label, 0) for label in labels])
    rows = []
    for label in labels:
        gt = gt_dist.get(label, 0)
        pred = pred_dist.get(label, 0)
        gt_width = gt / max_value * 100
        pred_width = pred / max_value * 100
        rows.append(
            f"""
            <div class="dist-row">
              <div class="dist-label">{esc(label)}</div>
              <div class="dist-bars">
                <div class="dist-line"><span>GT</span><div class="dist-track"><i class="gt" style="width:{gt_width:.2f}%"></i></div><b>{gt}</b></div>
                <div class="dist-line"><span>Pred</span><div class="dist-track"><i class="pred" style="width:{pred_width:.2f}%"></i></div><b>{pred}</b></div>
              </div>
            </div>
            """
        )
    return "\n".join(rows)


def heat_color(value: int, max_value: int) -> str:
    if max_value <= 0:
        return "rgba(255,255,255,0)"
    ratio = value / max_value
    alpha = 0.12 + ratio * 0.68
    return f"rgba(35, 116, 171, {alpha:.3f})"


def render_confusion_matrix(confusion: Dict[str, Dict[str, int]]) -> str:
    labels = get_labels(*[{key: 1 for key in confusion.keys()}], *[{key: 1 for row in confusion.values() for key in row.keys()}])
    max_value = max([0] + [value for row in confusion.values() for value in row.values()])
    header = "".join(f"<th>{esc(label)}</th>" for label in labels)
    body_rows = []
    for gt in labels:
        cells = []
        for pred in labels:
            value = confusion.get(gt, {}).get(pred, 0)
            tone = heat_color(value, max_value)
            cells.append(f"<td style=\"background:{tone}\"><strong>{value}</strong></td>")
        body_rows.append(f"<tr><th>{esc(gt)}</th>{''.join(cells)}</tr>")
    return f"""
    <table class="matrix">
      <thead><tr><th>GT \\ Pred</th>{header}</tr></thead>
      <tbody>{''.join(body_rows)}</tbody>
    </table>
    """


def render_sample_cards(rows: List[Dict[str, Any]], title: str, limit: int) -> str:
    selected = rows[:limit]
    if not selected:
        return f"<p class=\"muted\">No {esc(title.lower())} examples.</p>"
    cards = []
    for row in selected:
        outcome = "ok" if row["is_correct"] else "bad"
        cards.append(
            f"""
            <article class="sample {outcome}">
              <header>
                <code>{esc(row["id"])}</code>
                <span>{esc(row["setting"])}</span>
                <b>GT {esc(row["gt_answer"])} / Pred {esc(row["pred_answer"] or "missing")}</b>
              </header>
              <p>{esc(row["question"])}</p>
              <details>
                <summary>Response and options</summary>
                <div class="detail-block"><strong>Options</strong><br>{esc(row["options"])}</div>
                <div class="detail-block"><strong>Response</strong><br>{esc(row["raw_response"])}</div>
              </details>
            </article>
            """
        )
    return "\n".join(cards)


def render_metadata(config: Dict[str, Any], manifest: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    model = config.get("model", {})
    data = config.get("data", {})
    generation = config.get("generation", {})
    env = manifest.get("environment", {})
    git = manifest.get("git", {})
    local_git = manifest.get("local_git", {})
    rows = [
        ("Run ID", manifest.get("run_id", "")),
        ("Status", manifest.get("status", "")),
        ("Started", manifest.get("started_at", "")),
        ("Finalized", manifest.get("finalized_at", "")),
        ("Model", model.get("path", "")),
        ("Backend", model.get("backend", "")),
        ("Task", data.get("task", "")),
        ("Input", data.get("input_file", "")),
        ("Max New Tokens", generation.get("max_new_tokens", "")),
        ("Batch Size", generation.get("batch_size", "")),
        ("Kaggle GPU", ", ".join(env.get("cuda_devices", []))),
        ("Kaggle Torch", env.get("torch", "")),
        ("Kaggle Commit", git.get("commit", "")),
        ("Local Commit", local_git.get("commit", "")),
        ("Accuracy", metrics.get("accuracy", "")),
    ]
    return "\n".join(
        f"<tr><th>{esc(label)}</th><td>{esc(value)}</td></tr>" for label, value in rows
    )


def write_dashboard_html(
    path: Path,
    run_dir: Path,
    config: Dict[str, Any],
    manifest: Dict[str, Any],
    metrics: Dict[str, Any],
    examples: List[Dict[str, Any]],
    sample_limit: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    breakdown = setting_breakdown(examples)
    overall_random = random_baseline(examples)
    current_category_scores = category_scores(examples)
    pred_dist = answer_distribution(examples, "pred_answer")
    gt_dist = answer_distribution(examples, "gt_answer")
    confusion = confusion_table(examples)
    wrong = [row for row in examples if not row["is_correct"]]
    correct = [row for row in examples if row["is_correct"]]
    extraction_failures = [row for row in examples if row["extraction_failed"]]
    run_id = manifest.get("run_id", run_dir.name)
    log_text = log_tail(run_dir)

    dashboard_data = {
        "examples": examples,
        "breakdown": breakdown,
        "random_baseline": overall_random,
        "category_scores": current_category_scores,
        "reference_category_tables": REFERENCE_CATEGORY_TABLES,
        "gt_distribution": gt_dist,
        "pred_distribution": pred_dist,
        "confusion": confusion,
    }

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(run_id)} | MindCube Dashboard</title>
  <style>
    :root {{
      --bg: #f7f8f5;
      --paper: #ffffff;
      --ink: #1d2528;
      --muted: #637174;
      --line: #d9ded7;
      --teal: #227c72;
      --blue: #2374ab;
      --red: #b94d45;
      --amber: #b7791f;
      --green: #2f7d4f;
      --soft-blue: #d9ebf7;
      --soft-teal: #d8efea;
      --soft-red: #f5dfdc;
      --soft-amber: #f4e7c8;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }}
    .shell {{ max-width: 1400px; margin: 0 auto; padding: 28px; }}
    .topbar {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 20px;
      align-items: end;
      padding: 10px 0 22px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 16px; font-size: 18px; letter-spacing: 0; }}
    h3 {{ margin: 0 0 10px; font-size: 15px; letter-spacing: 0; }}
    .subtitle {{ color: var(--muted); margin: 0; max-width: 900px; }}
    .pill-row {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .pill {{ border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; background: #fff; color: var(--muted); font-size: 13px; }}
    .grid {{ display: grid; gap: 18px; margin-top: 18px; }}
    .kpis {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .two {{ grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); }}
    .wide-right {{ grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr); }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 1px 2px rgba(29, 37, 40, 0.04);
    }}
    .kpi {{
      min-height: 112px;
      border-radius: 8px;
      border: 1px solid var(--line);
      padding: 18px;
      background: var(--paper);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .kpi span {{ color: var(--muted); font-size: 13px; }}
    .kpi strong {{ font-size: 30px; letter-spacing: 0; }}
    .kpi-primary {{ background: var(--soft-blue); }}
    .kpi-green {{ background: #dceee1; }}
    .kpi-red {{ background: var(--soft-red); }}
    .kpi-amber {{ background: var(--soft-amber); }}
    .model-profile {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .model-profile article {{ border-top: 1px solid var(--line); padding-top: 14px; }}
    .kv-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
    .kv-grid.compact {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .kv-item {{ min-width: 0; }}
    .kv-item span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 2px; }}
    .kv-item strong {{ display: block; overflow-wrap: anywhere; font-size: 13px; font-weight: 650; }}
    .kv-item a {{ color: var(--blue); text-decoration: none; }}
    .kv-item a:hover {{ text-decoration: underline; }}
    .command-block {{ margin-top: 14px; }}
    .command-block summary {{ margin-bottom: 10px; }}
    .bar-row {{ display: grid; grid-template-columns: 110px minmax(160px, 1fr) 120px; gap: 12px; align-items: center; margin: 11px 0; }}
    .bar-label {{ color: var(--ink); font-weight: 650; }}
    .bar-track, .dist-track {{ height: 12px; background: #edf0eb; border-radius: 999px; overflow: hidden; border: 1px solid #dbe0d8; }}
    .bar-fill {{ height: 100%; background: linear-gradient(90deg, var(--teal), #55a68f); }}
    .baseline-track {{ height: 8px; margin-top: 5px; }}
    .random-fill {{ background: linear-gradient(90deg, var(--amber), #d9ad48); }}
    .bar-value {{ color: var(--ink); font-variant-numeric: tabular-nums; }}
    .bar-value span {{ color: var(--muted); }}
    .bar-value small {{ color: var(--muted); font-size: 12px; }}
    .baseline-hero {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }}
    .baseline-hero div {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #fbfcfa; }}
    .baseline-hero span {{ color: var(--muted); font-size: 13px; }}
    .baseline-hero strong {{ display: block; margin-top: 6px; font-size: 25px; }}
    .baseline-hero p {{ color: var(--muted); margin: 8px 0 0; font-size: 13px; }}
    .baseline-table, .benchmark-table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
    .baseline-table th, .baseline-table td, .benchmark-table th, .benchmark-table td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; }}
    .baseline-table th, .benchmark-table th {{ color: var(--muted); background: #f4f6f2; }}
    .benchmark-block + .benchmark-block {{ margin-top: 18px; }}
    .benchmark-scroll {{ max-height: 520px; }}
    .benchmark-table th, .benchmark-table td {{ white-space: nowrap; font-variant-numeric: tabular-nums; }}
    .benchmark-table .current-run th, .benchmark-table .current-run td {{ background: #e8f4ef; color: var(--ink); }}
    .benchmark-table .current-score strong {{ display: block; }}
    .benchmark-table .current-score small {{ color: var(--muted); }}
    .dist-row {{ display: grid; grid-template-columns: 56px minmax(0, 1fr); gap: 12px; margin: 12px 0; }}
    .dist-label {{ font-weight: 750; padding-top: 12px; }}
    .dist-line {{ display: grid; grid-template-columns: 42px minmax(0, 1fr) 52px; gap: 8px; align-items: center; margin: 5px 0; }}
    .dist-line span {{ color: var(--muted); font-size: 12px; }}
    .dist-line b {{ font-variant-numeric: tabular-nums; font-size: 12px; color: var(--muted); }}
    .dist-track i {{ display: block; height: 100%; }}
    .dist-track .gt {{ background: var(--blue); }}
    .dist-track .pred {{ background: var(--amber); }}
    .matrix {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
    .matrix th, .matrix td {{ border: 1px solid var(--line); padding: 9px; text-align: center; font-variant-numeric: tabular-nums; }}
    .matrix th {{ background: #f3f5f1; color: var(--muted); }}
    .matrix td strong {{ color: #11242c; }}
    .sample {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin: 10px 0; background: #fff; }}
    .sample.bad {{ border-left: 5px solid var(--red); }}
    .sample.ok {{ border-left: 5px solid var(--green); }}
    .sample header {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 8px; }}
    code {{ background: #eef1ed; border: 1px solid #dce1d9; border-radius: 6px; padding: 2px 5px; }}
    .sample header span {{ color: var(--muted); }}
    .sample p {{ margin: 0 0 8px; }}
    details {{ color: var(--muted); }}
    summary {{ cursor: pointer; color: var(--ink); font-weight: 650; }}
    .detail-block {{ margin: 8px 0; white-space: pre-wrap; }}
    .meta-table {{ width: 100%; border-collapse: collapse; }}
    .meta-table th, .meta-table td {{ border-bottom: 1px solid var(--line); padding: 9px 0; vertical-align: top; text-align: left; }}
    .meta-table th {{ width: 170px; color: var(--muted); font-weight: 600; }}
    .filters {{ display: grid; grid-template-columns: minmax(220px, 1fr) 160px 160px; gap: 10px; margin-bottom: 12px; }}
    input, select {{ width: 100%; padding: 9px 10px; border: 1px solid var(--line); border-radius: 6px; background: #fff; color: var(--ink); }}
    .examples-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .examples-table th, .examples-table td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; vertical-align: top; text-align: left; }}
    .examples-table th {{ color: var(--muted); background: #f4f6f2; position: sticky; top: 0; z-index: 1; }}
    .examples-table tbody tr.bad {{ background: #fff8f7; }}
    .examples-table tbody tr.ok {{ background: #f8fcf8; }}
    .scroll-table {{ max-height: 620px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }}
    .muted {{ color: var(--muted); }}
    pre {{ white-space: pre-wrap; max-height: 360px; overflow: auto; background: #242a2c; color: #eef3ef; border-radius: 8px; padding: 14px; font-size: 12px; }}
    .footer {{ color: var(--muted); font-size: 12px; margin: 28px 0 8px; }}
    @media (max-width: 900px) {{
      .shell {{ padding: 16px; }}
      .topbar, .two, .wide-right, .kpis {{ grid-template-columns: 1fr; }}
      .pill-row {{ justify-content: flex-start; }}
      .bar-row {{ grid-template-columns: 1fr; }}
      .filters {{ grid-template-columns: 1fr; }}
      .baseline-hero {{ grid-template-columns: 1fr; }}
      .model-profile, .kv-grid, .kv-grid.compact {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div>
        <h1>{esc(run_id)}</h1>
        <p class="subtitle">{esc(config.get("description", ""))}</p>
      </div>
      <div class="pill-row">
        <span class="pill">{esc(config.get("data", {}).get("task", ""))}</span>
        <span class="pill">{esc(config.get("model", {}).get("backend", ""))}</span>
        <span class="pill">{esc(manifest.get("status", ""))}</span>
      </div>
    </header>

    <section class="grid kpis">
      {render_kpis(metrics, examples)}
    </section>

    <section class="card grid">
      <div>
        <h2>Model Profile</h2>
        <p class="muted">Exact model, decoding, runtime, and code provenance captured for this experiment run.</p>
      </div>
      {render_model_profile(config, manifest)}
    </section>

    <section class="card grid">
      <div>
        <h2>Random Baseline</h2>
        <p class="muted">Computed as uniform random choice per question using parsed answer labels in the question text. Binary questions count as 50%; four-option questions count as 25%.</p>
      </div>
      {render_random_baseline(overall_random, metrics)}
      {render_random_breakdown_table(breakdown)}
    </section>

    <section class="card grid">
      <div>
        <h2>Category Benchmark Reference</h2>
        <p class="muted">Published MINDCUBE category scores with this run inserted at the top. This run shows percent plus correct/total counts; the downloaded category labels use P-P/P-O/O-O, mapped into the A-A/A-O/O-O relation columns used by the table.</p>
      </div>
      {render_category_benchmark_tables(current_category_scores)}
    </section>

    <section class="grid two">
      <article class="card">
        <h2>Accuracy By Setting</h2>
        <p class="muted">Top bar is model accuracy; smaller amber bar is random baseline.</p>
        {render_setting_chart(breakdown)}
      </article>
      <article class="card">
        <h2>Answer Distribution</h2>
        {render_distribution_chart(gt_dist, pred_dist)}
      </article>
    </section>

    <section class="grid wide-right">
      <article class="card">
        <h2>Confusion Matrix</h2>
        <p class="muted">Rows are ground truth answers; columns are predicted answers.</p>
        {render_confusion_matrix(confusion)}
      </article>
      <article class="card">
        <h2>Run Metadata</h2>
        <table class="meta-table"><tbody>{render_metadata(config, manifest, metrics)}</tbody></table>
      </article>
    </section>

    <section class="grid two">
      <article class="card">
        <h2>Error Samples</h2>
        {render_sample_cards(wrong, "error samples", sample_limit)}
      </article>
      <article class="card">
        <h2>Correct Samples</h2>
        {render_sample_cards(correct, "correct samples", sample_limit)}
      </article>
    </section>

    <section class="card grid">
      <div>
        <h2>Example Browser</h2>
        <p class="muted">Search by id, question, options, or response. The table renders from embedded run data.</p>
      </div>
      <div class="filters">
        <input id="search" placeholder="Search examples">
        <select id="settingFilter"><option value="">All settings</option></select>
        <select id="outcomeFilter">
          <option value="">All outcomes</option>
          <option value="wrong">Wrong only</option>
          <option value="correct">Correct only</option>
          <option value="missing">Extraction failures</option>
        </select>
      </div>
      <div class="scroll-table">
        <table class="examples-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Setting</th>
              <th>GT</th>
              <th>Pred</th>
              <th>Choices</th>
              <th>Random</th>
              <th>Question</th>
              <th>Response</th>
            </tr>
          </thead>
          <tbody id="examplesBody"></tbody>
        </table>
      </div>
      <p id="tableCount" class="muted"></p>
    </section>

    <section class="grid two">
      <article class="card">
        <h2>Review Checklist</h2>
        <ul>
          <li>Are mistakes concentrated in one setting?</li>
          <li>Is the model over-predicting one answer letter?</li>
          <li>Are extraction failures caused by prompt format or parser limits?</li>
          <li>Do wrong answers show visual confusion, spatial reasoning failure, or instruction-following failure?</li>
          <li>What exact prompt, model, or decoding change should be tried next?</li>
        </ul>
      </article>
      <article class="card">
        <h2>Log Tail</h2>
        <pre>{esc(log_text)}</pre>
      </article>
    </section>

    <p class="footer">
      Generated from {esc(rel(run_dir))}. Source files: predictions.jsonl, evaluation.json, metrics.json, manifest.json, config.json.
    </p>
  </main>

  {data_script("dashboard-data", dashboard_data)}
  <script>
    const data = JSON.parse(document.getElementById('dashboard-data').textContent);
    const examples = data.examples || [];
    const body = document.getElementById('examplesBody');
    const search = document.getElementById('search');
    const settingFilter = document.getElementById('settingFilter');
    const outcomeFilter = document.getElementById('outcomeFilter');
    const tableCount = document.getElementById('tableCount');
    const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (ch) => ({{
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }}[ch]));
    const truncate = (value, n) => {{
      const text = String(value ?? '');
      return text.length > n ? text.slice(0, n - 3) + '...' : text;
    }};
    const settings = [...new Set(examples.map((row) => row.setting).filter(Boolean))].sort();
    for (const setting of settings) {{
      const option = document.createElement('option');
      option.value = setting;
      option.textContent = setting;
      settingFilter.appendChild(option);
    }}
    function matches(row) {{
      const q = search.value.trim().toLowerCase();
      const setting = settingFilter.value;
      const outcome = outcomeFilter.value;
      if (setting && row.setting !== setting) return false;
      if (outcome === 'wrong' && row.is_correct) return false;
      if (outcome === 'correct' && !row.is_correct) return false;
      if (outcome === 'missing' && !row.extraction_failed) return false;
      if (!q) return true;
      return [row.id, row.setting, row.question, row.options, row.raw_response, row.gt_answer, row.pred_answer, row.choice_labels]
        .some((value) => String(value ?? '').toLowerCase().includes(q));
    }}
    function renderTable() {{
      const filtered = examples.filter(matches);
      const visible = filtered.slice(0, 300);
      body.innerHTML = visible.map((row) => `
        <tr class="${{row.is_correct ? 'ok' : 'bad'}}">
          <td><code>${{escapeHtml(row.id)}}</code></td>
          <td>${{escapeHtml(row.setting)}}</td>
          <td>${{escapeHtml(row.gt_answer)}}</td>
          <td>${{escapeHtml(row.pred_answer || 'missing')}}</td>
          <td>${{escapeHtml((row.choice_labels || []).join(''))}}</td>
          <td>${{escapeHtml(row.random_chance == null ? 'n/a' : row.random_chance.toFixed(2) + '%')}}</td>
          <td title="${{escapeHtml(row.question)}}">${{escapeHtml(truncate(row.question, 280))}}</td>
          <td title="${{escapeHtml(row.raw_response)}}">${{escapeHtml(truncate(row.raw_response, 220))}}</td>
        </tr>
      `).join('');
      tableCount.textContent = `Showing ${{visible.length}} of ${{filtered.length}} matching examples (${{examples.length}} total).`;
    }}
    search.addEventListener('input', renderTable);
    settingFilter.addEventListener('change', renderTable);
    outcomeFilter.addEventListener('change', renderTable);
    renderTable();
  </script>
</body>
</html>
"""
    path.write_text(html_doc, encoding="utf-8")


def analyze_run(run_dir: Path, sample_limit: int) -> Dict[str, Any]:
    manifest = load_json(run_dir / "manifest.json")
    config = load_json(run_dir / "config.json")
    metrics = load_json(run_dir / "metrics.json")
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        artifact_path = manifest.get("artifacts", {}).get("predictions", "")
        if artifact_path:
            predictions_path = PROJECT_ROOT / artifact_path
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found for {run_dir}")

    examples = build_examples(predictions_path)
    dashboard_path = run_dir / "dashboard.html"
    write_dashboard_html(dashboard_path, run_dir, config, manifest, metrics, examples, sample_limit)

    return {
        "examples": len(examples),
        "errors": sum(1 for row in examples if not row["is_correct"]),
        "extraction_failed": sum(1 for row in examples if row["extraction_failed"]),
        "dashboard": str(dashboard_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a standalone HTML dashboard for a MindCube run")
    parser.add_argument("--run-dir", required=True, help="Run folder under experiments/runs")
    parser.add_argument("--sample-limit", type=int, default=20, help="Number of correct/error samples in dashboard")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    stats = analyze_run(run_dir, args.sample_limit)
    print(
        "Wrote dashboard for {examples} examples: {errors} errors, {extraction_failed} extraction failures".format(
            **stats
        )
    )
    print(f"Dashboard: {stats['dashboard']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
