#!/usr/bin/env python3
"""Server-feasible Gemma 4 LoRA GRPO training for MindCube.

This keeps the MindCube/VAGEN RL target but avoids the full-weight VAGEN FSDP
path that does not fit Gemma4-31B on the 2x45GB server. The model is loaded with
the same vision-safe 8-bit base used by the successful SFT run, then the SFT
LoRA adapter is trained with a grouped policy-gradient objective.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.core.extractors import extract_answer, extract_json_from_text, get_setting_from_id  # noqa: E402


DEFAULT_SKIP_MODULES = (
    "lm_head,model.lm_head,"
    "vision_tower,model.vision_tower,"
    "embed_vision,model.embed_vision,"
    "audio_tower,model.audio_tower,"
    "embed_audio,model.embed_audio"
)


@dataclass
class Rollout:
    item: Dict[str, Any]
    prompt_inputs: Dict[str, Any]
    generated_ids: torch.Tensor
    response_text: str
    reward: float
    correct: bool
    answer_ok: bool
    cogmap_ok: bool


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_image_placeholders(prompt: str) -> str:
    lines = [
        line for line in prompt.splitlines()
        if line.strip().lower() not in {"<image>", "<image/>"}
    ]
    return "\n".join(lines).strip()


def resolve_image_path(image_path: str, image_root: Path, data_dir: Path) -> Path:
    candidate = Path(image_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for root in (image_root, PROJECT_ROOT / "data", data_dir.parent):
        resolved = root / image_path
        if resolved.exists():
            return resolved
    return image_root / image_path


def resize_to_max_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    if max_pixels <= 0 or image.width * image.height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / float(image.width * image.height))
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized = image.copy()
    resized.thumbnail(new_size, Image.Resampling.LANCZOS)
    return resized


def load_images(item: Dict[str, Any], image_root: Path, data_dir: Path, max_pixels: int) -> List[Image.Image]:
    images: List[Image.Image] = []
    for image_path in item.get("images", []):
        resolved = resolve_image_path(str(image_path), image_root=image_root, data_dir=data_dir)
        if not resolved.exists():
            raise FileNotFoundError(f"Missing image {image_path!r}, resolved as {resolved}")
        with Image.open(resolved) as image:
            converted = image.convert("RGB")
            converted.load()
        images.append(resize_to_max_pixels(converted, max_pixels=max_pixels))
    if not images:
        raise ValueError(f"Item {item.get('id')} has no images")
    return images


def make_messages(item: Dict[str, Any], images: Sequence[Image.Image]) -> List[Dict[str, Any]]:
    prompt = strip_image_placeholders(str(item.get("question_str", "")))
    content: List[Dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def move_inputs_to_device(inputs: Any, device: torch.device) -> Dict[str, Any]:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def detach_inputs_to_cpu(inputs: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            result[key] = value.detach().cpu()
        elif isinstance(value, list):
            result[key] = [
                item.detach().cpu() if torch.is_tensor(item) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def apply_chat_template(processor: Any, messages: Sequence[Dict[str, Any]], enable_thinking: bool) -> Dict[str, Any]:
    kwargs = {
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
    }
    try:
        return processor.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def decode_generated(processor: Any, generated_ids: torch.Tensor) -> Tuple[str, str]:
    raw = processor.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    clean = processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return raw, clean


def extract_final_channel(text: str) -> Optional[str]:
    marker = re.compile(r"<\|channel\|>\s*final\s*", re.IGNORECASE)
    matches = list(marker.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    next_marker = re.search(r"<\|channel\|>|<channel\|>", text[start:], re.IGNORECASE)
    end = start + next_marker.start() if next_marker else len(text)
    return text[start:end].strip()


def clean_response(raw: str, clean: str, stop_sequences: Sequence[str]) -> str:
    final = extract_final_channel(raw)
    text = final if final is not None else clean.strip() or raw.strip()
    text = re.sub(r"<\|channel\|>\s*(?:thought|final)\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("<channel|>", "").replace("<|channel|>", "")
    for sequence in stop_sequences:
        index = text.find(sequence)
        if index >= 0:
            text = text[: index + len(sequence)]
            break
    return text.strip()


def build_stop_criteria(processor: Any, stop_sequences: Sequence[str], input_len: int, window_tokens: int):
    if not stop_sequences:
        return None
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError:
        return None

    class StopOnDecodedSequence(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: Any, **kwargs: Any) -> bool:
            for row in input_ids:
                generated = row[input_len:]
                if generated.numel() <= 0:
                    continue
                suffix = generated[-window_tokens:]
                text = processor.decode(
                    suffix,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if any(sequence in text for sequence in stop_sequences):
                    return True
            return False

    return StoppingCriteriaList([StopOnDecodedSequence()])


def validate_position(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    return all(isinstance(coord, (int, float)) and 0 <= float(coord) <= 9 for coord in value)


def validate_cogmap_json(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    objects = value.get("objects")
    views = value.get("views")
    if not isinstance(objects, list) or not isinstance(views, list):
        return False
    if not objects or not views:
        return False
    for collection in (objects, views):
        for entry in collection:
            if not isinstance(entry, dict):
                return False
            if not isinstance(entry.get("name"), str) or not entry["name"].strip():
                return False
            if not validate_position(entry.get("position")):
                return False
            facing = entry.get("facing")
            if facing is not None and not isinstance(facing, str):
                return False
    return True


def score_response(text: str, gt_answer: str) -> Tuple[float, bool, bool, bool]:
    parsed_answer = extract_answer(text)
    answer_ok = parsed_answer is not None
    cogmap_ok = validate_cogmap_json(extract_json_from_text(text))
    format_ok = answer_ok and cogmap_ok
    correct = bool(parsed_answer and parsed_answer.upper() == str(gt_answer).upper())
    reward = (1.0 if format_ok else 0.0) + (5.0 if correct else 0.0)
    return reward, correct, answer_ok, cogmap_ok


def resolve_torch_dtype(value: str) -> Any:
    if value in ("auto", "", None):
        return "auto"
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(str(value).lower(), torch.bfloat16)


def parse_max_memory(values: Sequence[str]) -> Optional[Dict[int, str]]:
    parsed: Dict[int, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--max-memory entries must look like 0=42GiB, got {value!r}")
        device, memory = value.split("=", 1)
        parsed[int(device)] = memory
    return parsed or None


def load_model_and_processor(args: argparse.Namespace) -> Tuple[Any, Any]:
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    max_memory = parse_max_memory(args.max_memory)
    if max_memory:
        model_kwargs["max_memory"] = max_memory
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    dtype = resolve_torch_dtype(args.dtype)
    model_kwargs["dtype"] = dtype
    if args.quantization == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=args.llm_int8_threshold,
            llm_int8_skip_modules=split_csv(args.llm_int8_skip_modules) or None,
        )
    elif args.quantization != "none":
        raise ValueError(f"Unsupported quantization: {args.quantization}")

    try:
        base_model = AutoModelForImageTextToText.from_pretrained(args.base_model, **model_kwargs)
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        base_model = AutoModelForImageTextToText.from_pretrained(args.base_model, **model_kwargs)

    if args.prepare_model_for_kbit_training:
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    model = PeftModel.from_pretrained(base_model, args.sft_adapter_path, is_trainable=not args.eval_only)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    processor_source = args.processor_path or args.sft_adapter_path or args.base_model
    try:
        processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True, padding_side="left")
    except Exception:
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True, padding_side="left")
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    trainable = [param for param in model.parameters() if param.requires_grad]
    total_trainable = sum(param.numel() for param in trainable)
    print(f"[INFO] Trainable parameters: {total_trainable:,}")
    if hasattr(model, "hf_device_map"):
        print(f"[INFO] Device map: {model.hf_device_map}")
    return model, processor


def trainable_state(model: torch.nn.Module, clone_to_cpu: bool) -> Dict[str, torch.Tensor]:
    state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            value = param.detach().clone()
            state[name] = value.cpu() if clone_to_cpu else value
    return state


def copy_state_into_model(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state:
                param.copy_(state[name].to(device=param.device, dtype=param.dtype))


def sequence_mean_logprob(
    model: Any,
    prompt_inputs: Dict[str, Any],
    generated_ids: torch.Tensor,
    max_response_tokens: int = 0,
) -> torch.Tensor:
    device = model_device(model)
    prompt_inputs = move_inputs_to_device(prompt_inputs, device)
    if max_response_tokens and generated_ids.numel() > max_response_tokens:
        generated_ids = generated_ids[-max_response_tokens:]

    input_ids = prompt_inputs["input_ids"]
    attention_mask = prompt_inputs.get("attention_mask")
    prompt_len = int(input_ids.shape[-1])
    generated_ids = generated_ids.to(device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)
    full_input_ids = torch.cat([input_ids, generated_ids], dim=-1)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    full_attention_mask = torch.cat([attention_mask, torch.ones_like(generated_ids)], dim=-1)

    model_inputs = {
        key: value
        for key, value in prompt_inputs.items()
        if key not in {"input_ids", "attention_mask"}
    }
    model_inputs["input_ids"] = full_input_ids
    model_inputs["attention_mask"] = full_attention_mask

    outputs = model(**model_inputs, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    labels = full_input_ids[:, 1:]
    logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    response_mask = torch.zeros_like(labels, dtype=torch.bool)
    response_mask[:, max(prompt_len - 1, 0):] = True
    response_logprobs = logprobs[response_mask]
    if response_logprobs.numel() == 0:
        return logprobs.new_tensor(0.0)
    return response_logprobs.mean()


def reference_mean_logprob(
    model: Any,
    reference_state: Dict[str, torch.Tensor],
    prompt_inputs: Dict[str, Any],
    generated_ids: torch.Tensor,
    max_response_tokens: int = 0,
) -> torch.Tensor:
    current_state = trainable_state(model, clone_to_cpu=False)
    was_training = model.training
    copy_state_into_model(model, reference_state)
    try:
        model.eval()
        with torch.no_grad():
            value = sequence_mean_logprob(
                model,
                prompt_inputs,
                generated_ids,
                max_response_tokens=max_response_tokens,
            )
    finally:
        copy_state_into_model(model, current_state)
        if was_training:
            model.train()
    return value.detach()


def generate_one(
    model: Any,
    processor: Any,
    item: Dict[str, Any],
    image_root: Path,
    data_dir: Path,
    args: argparse.Namespace,
) -> Rollout:
    images = load_images(item, image_root=image_root, data_dir=data_dir, max_pixels=args.max_pixels)
    messages = make_messages(item, images)
    device = model_device(model)
    prompt_inputs = apply_chat_template(processor, messages, enable_thinking=args.enable_thinking)
    prompt_inputs = move_inputs_to_device(prompt_inputs, device)
    input_len = int(prompt_inputs["input_ids"].shape[-1])
    stopping_criteria = build_stop_criteria(
        processor,
        stop_sequences=args.stop_sequences,
        input_len=input_len,
        window_tokens=args.stop_sequence_window_tokens,
    )
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_response_length,
        "do_sample": args.temperature > 0,
        "top_p": args.top_p,
        "use_cache": True,
        "pad_token_id": getattr(getattr(processor, "tokenizer", None), "pad_token_id", None),
        "eos_token_id": getattr(getattr(processor, "tokenizer", None), "eos_token_id", None),
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
    if stopping_criteria is not None:
        generation_kwargs["stopping_criteria"] = stopping_criteria
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

    with torch.no_grad():
        output_ids = model.generate(**prompt_inputs, **generation_kwargs)
    generated_ids = output_ids[0, input_len:].detach()
    raw, clean = decode_generated(processor, generated_ids)
    response = clean_response(raw, clean, args.stop_sequences)
    reward, correct, answer_ok, cogmap_ok = score_response(response, str(item.get("gt_answer", "")))
    return Rollout(
        item=item,
        prompt_inputs=detach_inputs_to_cpu(prompt_inputs),
        generated_ids=generated_ids.cpu(),
        response_text=response,
        reward=reward,
        correct=correct,
        answer_ok=answer_ok,
        cogmap_ok=cogmap_ok,
    )


def compute_advantages(rewards: Sequence[float], mode: str) -> List[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    centered = [reward - mean for reward in rewards]
    if mode == "centered":
        return centered
    variance = sum(value * value for value in centered) / len(centered)
    std = math.sqrt(variance)
    if std < 1e-6:
        return centered
    return [value / std for value in centered]


def summarize_rollouts(rollouts: Sequence[Rollout]) -> Dict[str, Any]:
    total = len(rollouts)
    if not total:
        return {}
    by_setting: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rollout in rollouts:
        setting = str(rollout.item.get("setting_tag") or get_setting_from_id(str(rollout.item.get("id", ""))))
        bucket = by_setting[setting]
        bucket["total"] += 1
        bucket["correct"] += int(rollout.correct)
        bucket["invalid_answer"] += int(not rollout.answer_ok)
        bucket["invalid_cogmap_json"] += int(not rollout.cogmap_ok)

    def finalize(bucket: Dict[str, int]) -> Dict[str, Any]:
        total_items = bucket.get("total", 0)
        return {
            "total": total_items,
            "correct": bucket.get("correct", 0),
            "accuracy": bucket.get("correct", 0) / total_items if total_items else 0.0,
            "invalid_answer_rate": bucket.get("invalid_answer", 0) / total_items if total_items else 0.0,
            "invalid_cogmap_json_rate": bucket.get("invalid_cogmap_json", 0) / total_items if total_items else 0.0,
        }

    overall_counts: Dict[str, int] = defaultdict(int)
    for bucket in by_setting.values():
        for key, value in bucket.items():
            overall_counts[key] += value
    return {
        "overall": finalize(overall_counts),
        "settings": {setting: finalize(bucket) for setting, bucket in sorted(by_setting.items())},
    }


def save_checkpoint(model: Any, processor: Any, output_dir: Path, step: int) -> Path:
    checkpoint_dir = output_dir / f"global_step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    processor.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def run_train(args: argparse.Namespace, model: Any, processor: Any) -> None:
    train_items = list(iter_jsonl(args.train_file))
    if args.setting:
        train_items = [
            item for item in train_items
            if str(item.get("setting_tag", "")).lower() == args.setting.lower()
        ]
    if args.train_limit:
        train_items = train_items[: args.train_limit]
    if not train_items:
        raise ValueError("No training items loaded")

    print(f"[INFO] Train items: {len(train_items)}")
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    reference_state = trainable_state(model, clone_to_cpu=True) if args.kl_coef > 0 else {}
    data_dir = args.train_file.parent
    history_rows = []

    model.train()
    for step in range(1, args.total_steps + 1):
        batch_items = [train_items[(step - 1 + offset) % len(train_items)] for offset in range(args.train_batch_size)]
        all_rollouts: List[Rollout] = []
        loss_sum = 0.0
        loss_count = 0
        optimizer.zero_grad(set_to_none=True)
        loss_denominator = max(1, len(batch_items) * args.num_generations)

        model.eval()
        for item in batch_items:
            item_rollouts = [
                generate_one(model, processor, item, args.image_root, data_dir, args)
                for _ in range(args.num_generations)
            ]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model.train()
            advantages = compute_advantages([rollout.reward for rollout in item_rollouts], args.advantage_mode)
            for rollout, advantage in zip(item_rollouts, advantages):
                ref_logprob = None
                if args.kl_coef > 0:
                    ref_logprob = reference_mean_logprob(
                        model,
                        reference_state=reference_state,
                        prompt_inputs=rollout.prompt_inputs,
                        generated_ids=rollout.generated_ids,
                        max_response_tokens=args.max_train_response_tokens,
                    )
                actor_logprob = sequence_mean_logprob(
                    model,
                    rollout.prompt_inputs,
                    rollout.generated_ids,
                    max_response_tokens=args.max_train_response_tokens,
                )
                loss = -float(advantage) * actor_logprob
                if ref_logprob is not None:
                    loss = loss + args.kl_coef * (actor_logprob - ref_logprob)
                loss_sum += float(loss.detach().cpu())
                loss_count += 1
                (loss / loss_denominator).backward()
            all_rollouts.extend(item_rollouts)
            model.eval()

        if loss_count:
            model.train()
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            loss_value = loss_sum / loss_count
        else:
            loss_value = 0.0

        summary = summarize_rollouts(all_rollouts)
        reward_mean = sum(rollout.reward for rollout in all_rollouts) / max(len(all_rollouts), 1)
        row = {
            "step": step,
            "loss": loss_value,
            "reward_mean": reward_mean,
            **summary.get("overall", {}),
        }
        history_rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if args.save_freq and step % args.save_freq == 0:
            save_checkpoint(model, processor, args.output_dir, step)

    final_dir = save_checkpoint(model, processor, args.output_dir, args.total_steps)
    write_jsonl(args.output_dir / "train_history.jsonl", history_rows)
    print(f"[INFO] Final adapter saved to {final_dir}")


def run_eval(args: argparse.Namespace, model: Any, processor: Any) -> None:
    eval_items = list(iter_jsonl(args.eval_file))
    if args.eval_limit:
        eval_items = eval_items[: args.eval_limit]
    if not eval_items:
        raise ValueError("No eval items loaded")

    model.eval()
    rows = []
    rollouts: List[Rollout] = []
    data_dir = args.eval_file.parent
    for index, item in enumerate(eval_items, start=1):
        rollout = generate_one(model, processor, item, args.image_root, data_dir, args)
        rollouts.append(rollout)
        row = {
            **item,
            "response": rollout.response_text,
            "reward": rollout.reward,
            "correct": rollout.correct,
            "answer_ok": rollout.answer_ok,
            "cogmap_ok": rollout.cogmap_ok,
        }
        rows.append(row)
        print(json.dumps({"eval_index": index, "id": item.get("id"), "reward": rollout.reward, "correct": rollout.correct}))

    args.predictions_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.predictions_file, rows)
    summary = summarize_rollouts(rollouts)
    args.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="train", choices=["train", "eval"])
    parser.add_argument("--base-model", default="google/gemma-4-31B-it")
    parser.add_argument("--sft-adapter-path", required=True)
    parser.add_argument("--processor-path")
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--eval-file", type=Path)
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions-file", type=Path)
    parser.add_argument("--metrics-file", type=Path)
    parser.add_argument("--setting", default="among")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--quantization", default="8bit", choices=["8bit", "none"])
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory", action="append", default=[])
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--llm-int8-threshold", type=float, default=6.0)
    parser.add_argument("--llm-int8-skip-modules", default=DEFAULT_SKIP_MODULES)
    parser.add_argument("--prepare-model-for-kbit-training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--total-steps", type=int, default=200)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--kl-coef", type=float, default=0.001)
    parser.add_argument("--advantage-mode", default="group_norm", choices=["group_norm", "centered"])
    parser.add_argument("--save-freq", type=int, default=0)

    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=1536)
    parser.add_argument(
        "--max-train-response-tokens",
        type=int,
        default=512,
        help=(
            "Backprop only through this many generated response tokens. "
            "Generation can still use --max-response-length."
        ),
    )
    parser.add_argument("--max-pixels", type=int, default=90000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-sequences", nargs="*", default=["</answer>"])
    parser.add_argument("--stop-sequence-window-tokens", type=int, default=256)
    parser.add_argument("--eval-only", action="store_true")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.mode == "train" and not args.train_file:
        raise SystemExit("[ERROR] --train-file is required for training")
    if args.mode == "eval" and not args.eval_file:
        raise SystemExit("[ERROR] --eval-file is required for eval")
    args.eval_only = args.mode == "eval"
    args.predictions_file = args.predictions_file or (args.output_dir / "predictions.jsonl")
    args.metrics_file = args.metrics_file or (args.output_dir / "metrics.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_dump = vars(args).copy()
    config_dump["image_root"] = str(args.image_root)
    for key in ("train_file", "eval_file", "output_dir", "predictions_file", "metrics_file"):
        if config_dump.get(key) is not None:
            config_dump[key] = str(config_dump[key])
    (args.output_dir / "run_config.json").write_text(json.dumps(config_dump, indent=2) + "\n", encoding="utf-8")

    model, processor = load_model_and_processor(args)
    if args.mode == "train":
        run_train(args, model, processor)
    else:
        run_eval(args, model, processor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
