#!/usr/bin/env python3
"""
Gemma multimodal SFT for MindCube prompt data.

This trainer consumes either:
  - Gemma SFT JSON produced by scripts/convert_to_sft.py --model gemma4
  - General MindCube JSONL records with input_prompt, grounded_output, images

The default path is tuned for plain_cgmap_ffr_out on a 2x40GB server using
one model-parallel process, 8-bit language-only LoRA, and Hugging Face TRL's
SFTTrainer.
"""

import argparse
import inspect
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLM_INT8_SKIP_MODULES = (
    "lm_head,model.lm_head,"
    "vision_tower,model.vision_tower,"
    "embed_vision,model.embed_vision,"
    "audio_tower,model.audio_tower,"
    "embed_audio,model.embed_audio"
)


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("train", "data", "records", "examples"):
            value = data.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(f"Unsupported training data shape in {path}")


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    if "messages" in record:
        normalized = dict(record)
        normalized.setdefault("images", collect_image_paths(record["messages"]))
        validate_record(normalized, -1)
        return normalized

    missing = [key for key in ("input_prompt", "grounded_output", "images") if key not in record]
    if missing:
        raise ValueError(f"Record is missing required fields: {missing}")

    images = list(record["images"])
    if not images:
        raise ValueError("Record has no images")
    content = [{"type": "image", "path": image_path} for image_path in images]
    content.append({"type": "text", "text": record["input_prompt"]})
    normalized = {
        "id": record.get("id", "unknown"),
        "images": images,
        "messages": [
            {"role": "user", "content": content},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": record["grounded_output"]}],
            },
        ],
    }
    validate_record(normalized, -1)
    return normalized


def collect_image_paths(messages: Sequence[Dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, dict):
            content = [content]
        if isinstance(content, str):
            continue
        for element in content:
            if not isinstance(element, dict):
                continue
            if element.get("type") != "image" and "image" not in element:
                continue
            value = element.get("path") or element.get("image") or element.get("url")
            if isinstance(value, str):
                paths.append(value)
    return paths


def iter_message_content(message: Dict[str, Any]) -> Iterable[Any]:
    content = message.get("content", [])
    if isinstance(content, list):
        return content
    return [content]


def content_has_text(message: Dict[str, Any]) -> bool:
    for element in iter_message_content(message):
        if isinstance(element, str) and element.strip():
            return True
        if isinstance(element, dict) and isinstance(element.get("text"), str) and element["text"].strip():
            return True
    return False


def validate_record(record: Dict[str, Any], index: int) -> None:
    prefix = f"record {index}" if index >= 0 else "record"
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{prefix}: messages must be a non-empty list")

    images = record.get("images") or collect_image_paths(messages)
    if not isinstance(images, list) or not images:
        raise ValueError(f"{prefix}: images must be a non-empty list")
    if not all(isinstance(image_path, str) and image_path.strip() for image_path in images):
        raise ValueError(f"{prefix}: every image path must be a non-empty string")

    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if "user" not in roles:
        raise ValueError(f"{prefix}: missing user message")
    if "assistant" not in roles:
        raise ValueError(f"{prefix}: missing assistant message")

    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    if not any(content_has_text(message) for message in assistant_messages):
        raise ValueError(f"{prefix}: assistant message has no target text")


def preflight_training_data(
    records: Sequence[Dict[str, Any]],
    image_root: Path,
    record_dir: Path,
    max_pixels: int,
    sample_limit: Optional[int],
) -> None:
    limit = len(records) if not sample_limit else min(sample_limit, len(records))
    if limit <= 0:
        raise ValueError("Preflight has no records to validate")

    print(f"Preflight: validating {limit}/{len(records)} records and their images...")
    for index, record in enumerate(records[:limit]):
        validate_record(record, index)
        load_record_images(record, image_root, record_dir, max_pixels)
        if (index + 1) % 500 == 0:
            print(f"  Preflight checked {index + 1}/{limit} records")
    print("Preflight passed.")


def template_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return messages with image payloads stripped to template-safe placeholders."""
    result: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, str):
            result.append({"role": message["role"], "content": content})
            continue

        if isinstance(content, dict):
            content = [content]

        converted = []
        for element in content:
            if isinstance(element, dict) and (element.get("type") == "image" or "image" in element):
                converted.append({"type": "image"})
            else:
                converted.append(element)
        result.append({"role": message["role"], "content": converted})
    return result


def prompt_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    last_assistant = None
    for index, message in enumerate(messages):
        if message.get("role") == "assistant":
            last_assistant = index
    if last_assistant is None:
        return list(messages)
    return list(messages[:last_assistant])


def resolve_image_path(image_path: str, image_root: Path, record_dir: Path) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    candidates = [
        image_root / image_path,
        PROJECT_ROOT / image_path,
        record_dir / image_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return image_root / image_path


def resize_to_max_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    if max_pixels <= 0 or image.width * image.height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / float(image.width * image.height))
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized = image.copy()
    resized.thumbnail(new_size, Image.Resampling.LANCZOS)
    return resized


def load_record_images(
    record: Dict[str, Any],
    image_root: Path,
    record_dir: Path,
    max_pixels: int,
) -> List[Image.Image]:
    image_paths = list(record.get("images") or collect_image_paths(record["messages"]))
    images: List[Image.Image] = []

    for image_path in image_paths:
        resolved = resolve_image_path(str(image_path), image_root, record_dir)
        if not resolved.exists():
            raise FileNotFoundError(f"Missing image: {image_path} (resolved as {resolved})")

        with Image.open(resolved) as image:
            converted = image.convert("RGB")
            converted.load()
        images.append(resize_to_max_pixels(converted, max_pixels))

    return images


def apply_chat_template(processor: Any, messages: Sequence[Dict[str, Any]], add_generation_prompt: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("add_generation_prompt", None)
        return processor.apply_chat_template(messages, **kwargs)


def collect_ignore_token_ids(tokenizer: Any, extra_ids: Iterable[int]) -> List[int]:
    token_ids = set()

    for attr in ("pad_token_id", "image_token_id", "boi_token_id", "eoi_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and value >= 0:
            token_ids.add(value)

    special_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    for key in ("image_token", "boi_token", "eoi_token"):
        token = special_map.get(key) or getattr(tokenizer, key, None)
        if token is None:
            continue
        try:
            value = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            continue
        if isinstance(value, int) and value >= 0:
            token_ids.add(value)

    for value in extra_ids:
        if isinstance(value, int) and value >= 0:
            token_ids.add(value)

    return sorted(token_ids)


class MindCubeGemmaCollator:
    def __init__(
        self,
        processor: Any,
        image_root: Path,
        record_dir: Path,
        max_pixels: int,
        max_length: Optional[int],
        train_on_prompt: bool,
        extra_ignore_token_ids: Sequence[int],
    ):
        self.processor = processor
        self.image_root = image_root
        self.record_dir = record_dir
        self.max_pixels = max_pixels
        self.max_length = max_length
        self.train_on_prompt = train_on_prompt
        self.ignore_token_ids = collect_ignore_token_ids(processor.tokenizer, extra_ignore_token_ids)

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts: List[str] = []
        prompt_texts: List[str] = []
        image_batches: List[List[Image.Image]] = []

        for example in examples:
            messages = template_messages(example["messages"])
            texts.append(apply_chat_template(self.processor, messages, add_generation_prompt=False).strip())
            prompt_texts.append(
                apply_chat_template(
                    self.processor,
                    template_messages(prompt_messages(example["messages"])),
                    add_generation_prompt=True,
                ).strip()
            )
            image_batches.append(
                load_record_images(example, self.image_root, self.record_dir, self.max_pixels)
            )

        batch = self._processor_call(texts, image_batches)
        labels = batch["input_ids"].clone()

        if not self.train_on_prompt:
            prompt_batch = self._processor_call(prompt_texts, image_batches)
            prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
            for row_index, prompt_length in enumerate(prompt_lengths):
                length = min(int(prompt_length), labels.shape[1])
                labels[row_index, :length] = -100

        for token_id in self.ignore_token_ids:
            labels[labels == token_id] = -100

        batch["labels"] = labels
        return batch

    def _processor_call(self, texts: Sequence[str], images: Sequence[Sequence[Image.Image]]) -> Dict[str, torch.Tensor]:
        kwargs: Dict[str, Any] = {
            "text": list(texts),
            "images": list(images),
            "return_tensors": "pt",
            "padding": True,
        }
        if self.max_length:
            kwargs["max_length"] = self.max_length
            kwargs["truncation"] = True

        try:
            return self.processor(**kwargs)
        except TypeError:
            kwargs.pop("max_length", None)
            kwargs.pop("truncation", None)
            return self.processor(**kwargs)


def parse_max_memory(values: Sequence[str]) -> Optional[Dict[int, str]]:
    if not values:
        return None
    parsed: Dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--max-memory entries must look like 0=38GiB, got {value!r}")
        device, memory = value.split("=", 1)
        parsed[int(device)] = memory
    return parsed


def resolve_torch_dtype(args: argparse.Namespace) -> Any:
    if args.dtype == "auto":
        return "auto"
    if args.dtype == "bf16":
        return torch.bfloat16
    if args.dtype == "fp16":
        return torch.float16
    if args.dtype == "fp32":
        return torch.float32
    return torch.bfloat16 if args.bf16 else torch.float16


def filtered_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in parameters}


def make_sft_config(args: argparse.Namespace) -> Any:
    from trl import SFTConfig

    config_kwargs = {
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": args.lr_scheduler_type,
        "optim": args.optim,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "dataloader_num_workers": args.dataloader_num_workers,
        "report_to": args.report_to,
        "remove_unused_columns": False,
        "dataset_text_field": "",
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_seq_length": args.max_length,
        "max_length": args.max_length,
    }

    return SFTConfig(**filtered_kwargs(SFTConfig, config_kwargs))


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def is_linear_like(module: torch.nn.Module) -> bool:
    class_name = module.__class__.__name__.lower()
    return isinstance(module, torch.nn.Linear) or "linear" in class_name


def is_language_module(name: str, language_markers: Sequence[str], exclude_markers: Sequence[str]) -> bool:
    normalized = name.lower()
    if any(marker and marker in normalized for marker in exclude_markers):
        return False
    return any(marker and marker in normalized for marker in language_markers)


def discover_language_lora_targets(model: torch.nn.Module, args: argparse.Namespace) -> List[str]:
    language_markers = [marker.lower() for marker in split_csv(args.language_module_markers)]
    exclude_markers = [marker.lower() for marker in split_csv(args.exclude_module_markers)]
    explicit_suffixes = split_csv(args.language_lora_suffixes)
    target_modules: List[str] = []

    for name, module in model.named_modules():
        if not name or not is_linear_like(module):
            continue
        if name.endswith("lm_head"):
            continue
        if explicit_suffixes and not any(name.endswith(suffix) for suffix in explicit_suffixes):
            continue
        if is_language_module(name, language_markers, exclude_markers):
            target_modules.append(name)

    if not target_modules:
        sample_modules = [
            name for name, module in model.named_modules()
            if name and is_linear_like(module)
        ][:40]
        sample_text = "\n    ".join(sample_modules)
        raise ValueError(
            "Could not find language-only LoRA targets. "
            "Adjust --language-module-markers/--language-lora-suffixes or use --lora-scope all. "
            f"Sample linear modules:\n    {sample_text}"
        )

    return target_modules


def resolve_lora_target_modules(model: torch.nn.Module, args: argparse.Namespace) -> Any:
    if args.lora_scope == "all":
        return "all-linear" if args.lora_target_modules == "auto" else args.lora_target_modules

    if args.lora_target_modules != "auto":
        return split_csv(args.lora_target_modules)

    target_modules = discover_language_lora_targets(model, args)
    print(f"Language-only LoRA target modules: {len(target_modules)}")
    for name in target_modules[:12]:
        print(f"  - {name}")
    if len(target_modules) > 12:
        print(f"  ... {len(target_modules) - 12} more")
    return target_modules


def make_lora_config(model: torch.nn.Module, args: argparse.Namespace) -> Any:
    from peft import LoraConfig

    modules_to_save = split_csv(args.modules_to_save)
    lora_kwargs = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "target_modules": resolve_lora_target_modules(model, args),
        "task_type": "CAUSAL_LM",
        "modules_to_save": modules_to_save or None,
        "ensure_weight_tying": True,
    }
    return LoraConfig(**filtered_kwargs(LoraConfig, lora_kwargs))


def load_model_and_processor(args: argparse.Namespace) -> Any:
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    dtype = resolve_torch_dtype(args)
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
    }

    if args.device_map == "local":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model_kwargs["device_map"] = {"": local_rank}
    elif args.device_map != "none":
        model_kwargs["device_map"] = args.device_map

    max_memory = parse_max_memory(args.max_memory)
    if max_memory:
        model_kwargs["max_memory"] = max_memory

    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    if args.quantization in {"4bit", "8bit"}:
        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
        if args.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=args.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_storage=compute_dtype,
            )
        else:
            int8_skip_modules = split_csv(args.llm_int8_skip_modules)
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=args.llm_int8_threshold,
                llm_int8_skip_modules=int8_skip_modules or None,
            )

    if dtype == "auto":
        model_kwargs["dtype"] = "auto"
    else:
        model_kwargs["dtype"] = dtype

    try:
        model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)

    processor_id = args.processor_id or args.model_id
    processor = AutoProcessor.from_pretrained(
        processor_id,
        trust_remote_code=args.trust_remote_code,
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"

    if hasattr(model, "config"):
        model.config.use_cache = False

    if args.quantization in {"4bit", "8bit"} and args.prepare_model_for_kbit_training:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    return model, processor


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma on MindCube with TRL and language-only LoRA.")

    parser.add_argument("--task-name", default="plain_cgmap_ffr_out")
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/prompts/training/gemma4")
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "checkpoints/sft/gemma4/plain_cgmap_ffr_out"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-samples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")

    parser.add_argument("--model-id", default="google/gemma-4-31B-it")
    parser.add_argument("--processor-id")
    parser.add_argument("--device-map", default="auto", choices=["auto", "local", "none"])
    parser.add_argument("--max-memory", action="append", default=[])
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--quantization", default="8bit", choices=["8bit", "4bit", "none"])
    parser.add_argument("--prepare-model-for-kbit-training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--llm-int8-threshold", type=float, default=6.0)
    parser.add_argument("--llm-int8-skip-modules", default=DEFAULT_LLM_INT8_SKIP_MODULES)
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--bnb-4bit-use-double-quant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-scope", default="language", choices=["language", "all"])
    parser.add_argument("--lora-target-modules", default="auto")
    parser.add_argument("--language-module-markers", default="language_model,text_model,llm,decoder,model.layers")
    parser.add_argument(
        "--language-lora-suffixes",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--exclude-module-markers",
        default="vision,visual,image,audio,projector,mm_projector,multi_modal",
    )
    parser.add_argument("--modules-to-save", default="")

    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-pixels", type=int, default=90000)
    parser.add_argument("--save-steps", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int, default=12)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--optim", default="adamw_torch_fused")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-on-prompt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--extra-ignore-token-id", action="append", type=int, default=[262144])
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume-from-checkpoint")

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    if args.train_file is None:
        args.train_file = args.data_dir / f"MindCube_train_{args.task_name}_gemma_sft.json"
    if args.run_name is None:
        args.run_name = f"gemma4-{args.task_name}-{args.quantization}-lora"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = [normalize_record(record) for record in load_records(args.train_file)]
    if args.max_train_samples:
        records = records[: args.max_train_samples]
    if not records:
        raise ValueError("No training records loaded")

    print("Gemma SFT configuration")
    print(f"  Train file: {args.train_file}")
    print(f"  Records: {len(records)}")
    print(f"  Image root: {args.image_root}")
    print(f"  Model: {args.model_id}")
    print(f"  Output: {args.output_dir}")
    print(f"  Device map: {args.device_map}")
    print(f"  Quantization: {args.quantization}")
    if args.quantization == "8bit":
        print(f"  8-bit skip modules: {args.llm_int8_skip_modules}")
    print(f"  LoRA scope: {args.lora_scope}")
    print(f"  Train on prompt: {args.train_on_prompt}")
    micro_batches = math.ceil(len(records) / args.per_device_train_batch_size)
    optimizer_steps = math.ceil(
        micro_batches * float(args.num_train_epochs) / args.gradient_accumulation_steps
    )
    print(f"  Micro-batches/epoch: {micro_batches}")
    print(f"  Estimated optimizer steps: {optimizer_steps}")

    sample = records[0]
    sample_images = load_record_images(sample, args.image_root, args.train_file.parent, args.max_pixels)
    print(f"  First record images: {len(sample_images)}")
    print(f"  First record id: {sample.get('id', 'unknown')}")

    if args.preflight:
        preflight_training_data(
            records=records,
            image_root=args.image_root,
            record_dir=args.train_file.parent,
            max_pixels=args.max_pixels,
            sample_limit=args.preflight_samples,
        )

    if args.dry_run:
        print("Dry run complete. No model was loaded.")
        return

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Refusing to start Gemma training on CPU.")

    try:
        from datasets import Dataset
        from trl import SFTTrainer
    except ImportError as exc:
        raise ImportError(
            "Gemma SFT requires datasets, trl, peft, bitsandbytes, and transformers. "
            "Install with: pip install -r requirements-gemma4.txt"
        ) from exc

    model, processor = load_model_and_processor(args)
    train_dataset = Dataset.from_list(records)
    data_collator = MindCubeGemmaCollator(
        processor=processor,
        image_root=args.image_root,
        record_dir=args.train_file.parent,
        max_pixels=args.max_pixels,
        max_length=args.max_length,
        train_on_prompt=args.train_on_prompt,
        extra_ignore_token_ids=args.extra_ignore_token_id,
    )
    training_args = make_sft_config(args)
    peft_config = make_lora_config(model, args)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
        "peft_config": peft_config,
        "processing_class": processor,
        "tokenizer": getattr(processor, "tokenizer", processor),
    }
    trainer = SFTTrainer(**filtered_kwargs(SFTTrainer.__init__, trainer_kwargs))

    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Training complete. Saved adapter and processor to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
