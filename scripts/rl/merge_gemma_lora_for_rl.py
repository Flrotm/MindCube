#!/usr/bin/env python3
"""Merge a Gemma PEFT LoRA adapter into a standalone HF checkpoint for RL."""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_SKIP_MODULES = (
    "lm_head,model.lm_head,"
    "vision_tower,model.vision_tower,"
    "embed_vision,model.embed_vision,"
    "audio_tower,model.audio_tower,"
    "embed_audio,model.embed_audio"
)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def latest_path(pattern: str) -> Path:
    matches = [Path(item) for item in glob.glob(pattern)]
    if not matches:
        raise FileNotFoundError(f"No checkpoint matched: {pattern}")
    return max(matches, key=lambda item: item.stat().st_mtime)


def parse_max_memory(values: Iterable[str]) -> Optional[Dict[int, str]]:
    parsed: Dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--max-memory entries must look like 0=38GiB, got {value!r}")
        device, memory = value.split("=", 1)
        parsed[int(device)] = memory
    return parsed or None


def resolve_dtype(value: str) -> Any:
    import torch

    if value == "auto":
        return "auto"
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    raise ValueError(value)


def build_model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    from transformers import BitsAndBytesConfig

    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if args.device_map != "none":
        kwargs["device_map"] = args.device_map

    max_memory = parse_max_memory(args.max_memory)
    if max_memory:
        kwargs["max_memory"] = max_memory

    dtype = resolve_dtype(args.dtype)
    kwargs["dtype"] = dtype
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    if args.load_quantization == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=args.llm_int8_threshold,
            llm_int8_skip_modules=split_csv(args.llm_int8_skip_modules) or None,
        )

    return kwargs


def save_manifest(output_dir: Path, args: argparse.Namespace, adapter_path: Path) -> None:
    manifest = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "base_model": args.base_model,
        "adapter_path": str(adapter_path),
        "output_dir": str(output_dir),
        "dtype": args.dtype,
        "load_quantization": args.load_quantization,
        "llm_int8_skip_modules": split_csv(args.llm_int8_skip_modules),
        "max_memory": args.max_memory,
        "note": "Standalone merged checkpoint prepared for VAGEN/verl RL.",
    }
    (output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="google/gemma-4-31B-it")
    parser.add_argument(
        "--adapter-glob",
        default="/data/fuccelli/mindcube_runs/checkpoints/sft_full/"
        "gemma4-31b-sft-plain-cgmap-ffr-out-full-1epoch-ebs128-vision-safe-lora-*",
        help="Glob used to find the latest Gemma SFT LoRA adapter.",
    )
    parser.add_argument("--adapter-path", type=Path, help="Explicit PEFT adapter path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Standalone merged checkpoint directory to create.",
    )
    parser.add_argument("--processor-id", help="Processor source. Defaults to adapter path, then base model.")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default="auto", choices=["auto", "none"])
    parser.add_argument("--max-memory", action="append", default=[])
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--load-quantization", default="none", choices=["none", "8bit"])
    parser.add_argument("--llm-int8-threshold", type=float, default=6.0)
    parser.add_argument("--llm-int8-skip-modules", default=DEFAULT_SKIP_MODULES)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    adapter_path = args.adapter_path or latest_path(args.adapter_glob)
    if not adapter_path.exists():
        raise SystemExit(f"[ERROR] Adapter path not found: {adapter_path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise SystemExit(
            f"[ERROR] Output directory is not empty: {args.output_dir}. Use --force to overwrite."
        )
    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)

    summary = {
        "base_model": args.base_model,
        "adapter_path": str(adapter_path),
        "output_dir": str(args.output_dir),
        "dtype": args.dtype,
        "load_quantization": args.load_quantization,
        "max_memory": args.max_memory,
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return 0

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_kwargs = build_model_kwargs(args)
    try:
        model = AutoModelForImageTextToText.from_pretrained(args.base_model, **model_kwargs)
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForImageTextToText.from_pretrained(args.base_model, **model_kwargs)

    peft_model = PeftModel.from_pretrained(model, adapter_path)
    try:
        merged = peft_model.merge_and_unload(safe_merge=True)
    except TypeError:
        merged = peft_model.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    processor_source = args.processor_id or str(adapter_path)
    try:
        processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    except Exception:
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.save_pretrained(args.output_dir)
    save_manifest(args.output_dir, args, adapter_path)
    print(f"[INFO] Merged checkpoint written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
