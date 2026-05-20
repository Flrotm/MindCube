#!/usr/bin/env python3
"""Preflight VAGEN/verl imports and report missing modules together."""

from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


MISSING_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")

BASE_CHECKS = [
    ("gym", "gym"),
    ("gym_sokoban", "gym-sokoban"),
    ("gymnasium", "gymnasium"),
    ("qwen_vl_utils", "qwen-vl-utils"),
    ("mathruler", "mathruler"),
    ("matplotlib", "matplotlib"),
    ("flask", "flask"),
    ("together", "together"),
    ("hydra", "hydra-core"),
    ("omegaconf", "omegaconf"),
    ("imageio", "imageio"),
    ("pyparsing", "pyparsing"),
    ("blinker", "blinker"),
    ("pydantic", "pydantic"),
    ("ray", "ray"),
    ("wandb", "wandb"),
    ("codetiming", "codetiming"),
    ("datasets", "datasets"),
    ("dill", "dill"),
    ("accelerate", "accelerate"),
    ("peft", "peft"),
    ("tensordict", "tensordict"),
    ("torchdata", "torchdata"),
    ("pylatexenc", "pylatexenc"),
    ("math_verify", "math-verify"),
    ("vllm", "vllm"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("vagen.env.create_dataset", "vagen.env.create_dataset"),
    ("vagen.server.server", "vagen.server.server"),
    ("vagen.trainer.main_ppo", "vagen.trainer.main_ppo"),
]


def add_path(path: Path | None) -> None:
    if path:
        resolved = path.resolve()
        if resolved.exists():
            sys.path.insert(0, str(resolved))


def module_error(exc: BaseException) -> Dict[str, str]:
    text = f"{exc.__class__.__name__}: {exc}"
    missing_match = MISSING_RE.search(str(exc))
    missing = missing_match.group(1) if missing_match else ""
    tb = traceback.format_exception_only(type(exc), exc)
    return {
        "error": text,
        "missing_module": missing,
        "short": "".join(tb).strip(),
    }


def import_one(module_name: str) -> Tuple[bool, Dict[str, str]]:
    try:
        module = importlib.import_module(module_name)
        return True, {
            "version": str(getattr(module, "__version__", "installed")),
        }
    except BaseException as exc:  # import-time failures are exactly what we need.
        return False, module_error(exc)


def iter_package_modules(package_name: str) -> Iterable[str]:
    ok, _ = import_one(package_name)
    if not ok:
        yield package_name
        return
    package = sys.modules.get(package_name)
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        yield package_name
        return
    yield package_name
    for info in pkgutil.walk_packages(package_path, prefix=f"{package_name}."):
        yield info.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vagen-root", type=Path, required=True)
    parser.add_argument("--verl-root", type=Path)
    parser.add_argument(
        "--packages",
        nargs="*",
        default=[
            "vagen.env.crossview",
            "vagen.env.create_dataset",
            "vagen.server",
            "vagen.trainer",
        ],
        help="Packages/modules to import recursively where possible.",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    add_path(args.verl_root)
    add_path(args.vagen_root)

    checks: List[Tuple[str, str]] = list(BASE_CHECKS)
    seen = {name for name, _ in checks}
    for package_name in args.packages:
        for module_name in iter_package_modules(package_name):
            if module_name not in seen:
                checks.append((module_name, module_name))
                seen.add(module_name)

    results = []
    missing_modules = set()
    failures = []
    for module_name, label in checks:
        ok, info = import_one(module_name)
        result = {"module": module_name, "label": label, "ok": ok, **info}
        results.append(result)
        if ok:
            print(f"[INFO] import ok: {label} ({info.get('version', 'installed')})")
        else:
            failures.append(result)
            if info.get("missing_module"):
                missing_modules.add(info["missing_module"].split(".")[0])
            print(f"[ERROR] import failed: {label}: {info['short']}")

    summary = {
        "checked": len(results),
        "failed": len(failures),
        "missing_modules": sorted(missing_modules),
        "failures": failures,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if failures:
        print("[ERROR] VAGEN import preflight failed.")
        if missing_modules:
            print("[ERROR] Missing top-level modules:")
            for module_name in sorted(missing_modules):
                print(f"  - {module_name}")
        raise SystemExit(1)

    print(f"[INFO] VAGEN import preflight passed ({len(results)} imports).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
