#!/usr/bin/env python3
"""Patch VAGEN MindCube crossview env to accept an explicit data_file.

The upstream MindCube branch chooses the RL JSONL from type+split. This patch
keeps that default behavior, but lets YAML configs pass:

  env_config:
    data_file: crossviewQA_train_cogmap_and_reasoning_plain_among.jsonl

Relative data_file values are resolved under vagen/env/crossview/MindCube_RL_Data.
Absolute data_file values are used directly.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def backup(path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".codex-rl.bak")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def patch_env_config(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text

    if "data_file:" not in text:
        if "Optional" not in text:
            text = re.sub(
                r"from typing import ([^\n]+)",
                lambda m: "from typing import "
                + (
                    m.group(1)
                    if "Optional" in m.group(1)
                    else m.group(1).rstrip() + ", Optional"
                ),
                text,
                count=1,
            )
        text = re.sub(
            r"(\n[ \t]*image_path:\s*str\s*=\s*[\"']\.[\"'])",
            "\n    data_file: Optional[str] = None\\1",
            text,
            count=1,
        )
        if text == original:
            text = re.sub(
                r"([ \t]+image_path:\s*str\s*=\s*[\"']\.[\"'])",
                "\n    data_file: Optional[str] = None\\1",
                text,
                count=1,
            )

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_env(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    if 'getattr(config, "data_file", None)' in text:
        return False

    pattern = re.compile(
        r"train_data_path\s*=\s*f[\"']crossviewQA_train_\{self\.type\}\.jsonl[\"']\s*"
        r"test_data_path\s*=\s*f[\"']crossviewQA_tinybench_\{self\.type\}\.jsonl[\"']\s*"
        r"self\.data_path\s*=\s*os\.path\.join\(\s*self\.script_dir\s*,\s*[\"']MindCube_RL_Data[\"']\s*,\s*"
        r"train_data_path\s*if\s*self\.split\s*==\s*[\"']train[\"']\s*else\s*test_data_path\s*\)",
        flags=re.MULTILINE,
    )
    replacement = """train_data_path = f"crossviewQA_train_{self.type}.jsonl"
        test_data_path = f"crossviewQA_tinybench_{self.type}.jsonl"
        selected_data_file = getattr(config, "data_file", None)
        if selected_data_file:
            self.data_path = (
                selected_data_file
                if os.path.isabs(selected_data_file)
                else os.path.join(self.script_dir, "MindCube_RL_Data", selected_data_file)
            )
        else:
            self.data_path = os.path.join(
                self.script_dir,
                "MindCube_RL_Data",
                train_data_path if self.split == "train" else test_data_path,
            )"""
    text, replacements = pattern.subn(replacement, text, count=1)
    if replacements != 1:
        raise RuntimeError(
            f"Could not find the crossview data_path block in {path}. "
            "The VAGEN file may have changed; patch it manually."
        )

    if text != original:
        backup(path)
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vagen-root",
        type=Path,
        required=True,
        help="Path to the cloned VAGEN repository.",
    )
    args = parser.parse_args()

    env_dir = args.vagen_root / "vagen" / "env" / "crossview"
    env_config = env_dir / "env_config.py"
    env = env_dir / "env.py"
    if not env_config.exists() or not env.exists():
        raise SystemExit(f"[ERROR] Could not find VAGEN crossview env under {env_dir}")

    changed = {
        "env_config.py": patch_env_config(env_config),
        "env.py": patch_env(env),
    }
    for file_name, did_change in changed.items():
        print(f"{file_name}: {'patched' if did_change else 'already patched'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
