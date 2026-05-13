"""
Gemma 4 multimodal inference engine using the Transformers backend.
"""

import re
import traceback
from collections import Counter
from typing import Any, Dict, List

import torch
from PIL import Image

from ..base import BaseInferenceEngine
from ..utils import ConfigManager, ImageProcessor, ResponseProcessor


class GemmaInferenceEngine(BaseInferenceEngine):
    """Gemma 4 inference engine for image-text MindCube prompts."""

    def __init__(self, model_path: str, backend: str = "transformers", **kwargs):
        if backend != "transformers":
            print("Warning: Gemma 4 currently uses the transformers backend in this project")
        super().__init__(model_path, "gemma4", **kwargs)
        self.backend = "transformers"
        self.config.update(ConfigManager.get_default_config("gemma4"))
        self.config.update(kwargs)

    def load_model(self) -> None:
        """Load Gemma 4 model and processor."""
        try:
            from transformers import AutoProcessor

            try:
                from transformers import AutoModelForImageTextToText
            except ImportError as exc:
                raise ImportError(
                    "Gemma 4 image-text inference requires a recent Transformers version. "
                    "On Kaggle, run: pip install -U transformers accelerate torchvision"
                ) from exc

            dtype = self._resolve_torch_dtype(self.config.get("torch_dtype", "float16"))
            device_map = self.config.get("device_map")
            if not device_map:
                use_single_gpu = bool(self.config.get("single_gpu", True))
                device_map = {"": 0} if torch.cuda.is_available() and use_single_gpu else "auto"

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side=self.config.get("padding_side", "left"),
            )

            model_kwargs = {
                "device_map": device_map,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            quantization_config = self._build_quantization_config(self.config.get("quantization"))
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config

            max_memory = self._normalize_max_memory(self.config.get("max_memory"))
            if max_memory:
                model_kwargs["max_memory"] = max_memory

            for key in ("offload_folder", "offload_state_dict", "attn_implementation"):
                value = self.config.get(key)
                if value is not None:
                    model_kwargs[key] = value

            offload_folder = model_kwargs.get("offload_folder")
            if offload_folder:
                import os
                os.makedirs(offload_folder, exist_ok=True)

            if dtype == "auto":
                model_kwargs["dtype"] = "auto"
            else:
                model_kwargs["dtype"] = dtype

            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path,
                    **model_kwargs,
                )
            except TypeError:
                if "dtype" in model_kwargs:
                    model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path,
                    **model_kwargs,
                )

            print(f"Gemma 4 image-text model loaded successfully using {self.backend} backend")
            if hasattr(self.model, "hf_device_map"):
                print(f"Gemma 4 device map: {self.model.hf_device_map}")
        except Exception as exc:
            print(f"Error loading Gemma 4 model: {exc}")
            raise

    def process_input(self, prompt: str, image_paths: List[str], **kwargs) -> Dict[str, Any]:
        """Create Gemma 4 chat messages from a MindCube prompt and images."""
        if not self.validate_inputs(prompt, image_paths):
            raise ValueError("Invalid input data")

        final_answer_instruction = self.config.get("final_answer_instruction")
        if final_answer_instruction:
            prompt = f"{prompt.rstrip()}\n\n{final_answer_instruction}"

        image_payload_format = self.config.get("image_payload_format", "path")
        if image_payload_format == "path":
            # Hugging Face multimodal chat templates document local image files
            # with the "path" key. This lets the processor use its native loader.
            content = [{"type": "image", "path": image_path} for image_path in image_paths]
        else:
            images, errors = ImageProcessor.load_and_validate_images(image_paths)
            if errors:
                print(f"Image loading errors: {errors}")

            max_pixels = kwargs.get("max_pixels", self.config.get("max_pixels", 512 * 512))
            images = [self._resize_image(image.convert("RGB"), max_pixels) for image in images]
            content = [{"type": "image", "image": image} for image in images]

        content.append({"type": "text", "text": prompt})

        messages = []
        system_prompt = self.config.get("system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        return {"messages": messages}

    def generate_response(self, processed_input: Dict[str, Any], **kwargs) -> str:
        """Generate a response using Gemma 4."""
        try:
            if self.model is None or self.processor is None:
                self.load_model()

            device = self._model_device()
            template_kwargs = {
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "add_generation_prompt": True,
            }
            if "enable_thinking" in self.config:
                template_kwargs["enable_thinking"] = bool(
                    kwargs.get("enable_thinking", self.config["enable_thinking"])
                )

            try:
                inputs = self.processor.apply_chat_template(
                    processed_input["messages"],
                    **template_kwargs,
                )
            except TypeError:
                template_kwargs.pop("enable_thinking", None)
                inputs = self.processor.apply_chat_template(
                    processed_input["messages"],
                    **template_kwargs,
                )

            inputs = self._move_inputs_to_device(inputs, device)
            if self.config.get("log_input_tensors", False):
                print(f"Gemma 4 input tensors: {self._summarize_inputs(inputs)}")
            input_len = inputs["input_ids"].shape[-1]

            generation_config = self._generation_config(kwargs)
            outputs = self.model.generate(**inputs, **generation_config)
            generated_ids = outputs[0][input_len:]
            raw_response = self.processor.decode(
                generated_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            clean_response = self.processor.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            self._raise_if_degenerate_generation(generated_ids, raw_response, clean_response)

            final_response = self._extract_channel(raw_response, "final")
            if final_response is not None:
                response = final_response
            else:
                # Gemma's cleaned text can retain plain "thought"/"final" labels.
                # Use it for normal text, but inspect raw special tokens first so
                # a real final channel is not hidden by decoding.
                response = self._parse_response(clean_response)
                if not response.strip():
                    response = self._parse_response(raw_response)
                response = self._parse_response(response)
            response = self._strip_decode_artifacts(response)
            if not response.strip() and clean_response.strip():
                response = clean_response
            return ResponseProcessor.clean_response(response)
        except Exception as exc:
            print(f"Error in Gemma 4 generation: {exc}")
            print(traceback.format_exc())
            return f"Error: {str(exc)}"

    def _generation_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        config = self.config.get("generation_config", {}).copy()
        valid = {
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "do_sample",
            "num_beams",
            "repetition_penalty",
            "pad_token_id",
            "eos_token_id",
            "use_cache",
            "bad_words_ids",
        }
        config.update({key: value for key, value in kwargs.items() if key in valid})
        config.setdefault("max_new_tokens", self.config.get("max_new_tokens", 512))
        self._apply_token_defaults(config)

        temperature = float(config.get("temperature", 0.0) or 0.0)
        if temperature <= 0:
            config["do_sample"] = False
            config.pop("temperature", None)
            config.pop("top_p", None)
            config.pop("top_k", None)
        else:
            config["do_sample"] = True
        return config

    def _apply_token_defaults(self, config: Dict[str, Any]) -> None:
        tokenizer = self._tokenizer()
        if tokenizer is None:
            return

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

        if eos_token_id is not None:
            config.setdefault("eos_token_id", eos_token_id)
        if pad_token_id is not None:
            config.setdefault("pad_token_id", pad_token_id)
        elif eos_token_id is not None:
            config.setdefault("pad_token_id", eos_token_id)

        if not self.config.get("suppress_pad_token", True):
            return
        if pad_token_id is None or self._token_id_in(pad_token_id, config.get("eos_token_id")):
            return

        bad_words_ids = list(config.get("bad_words_ids") or [])
        if [pad_token_id] not in bad_words_ids:
            bad_words_ids.append([pad_token_id])
        config["bad_words_ids"] = bad_words_ids

    def _parse_response(self, response: str) -> str:
        if hasattr(self.processor, "parse_response"):
            try:
                parsed = self.processor.parse_response(response)
                if isinstance(parsed, str):
                    return parsed
                if isinstance(parsed, dict):
                    for key in ("answer", "final_answer", "final", "response", "content", "text"):
                        value = parsed.get(key)
                        if isinstance(value, str) and value.strip():
                            return value
                if parsed is not None:
                    return str(parsed)
            except Exception:
                pass

        final_response = self._extract_channel(response, "final")
        if final_response is not None:
            return final_response

        response = re.sub(r"<\|channel\|>\s*(?:thought|final)\s*", "", response)
        response = response.replace("<channel|>", "")
        response = response.replace("<|channel|>", "")
        return response

    def _extract_channel(self, response: str, channel: str) -> Any:
        marker = re.compile(rf"<\|channel\|>\s*{re.escape(channel)}\s*", re.IGNORECASE)
        matches = list(marker.finditer(response))
        if not matches:
            return None

        start = matches[-1].end()
        next_marker = re.search(r"<\|channel\|>|<channel\|>", response[start:], re.IGNORECASE)
        end = start + next_marker.start() if next_marker else len(response)
        return response[start:end].strip()

    def _raise_if_degenerate_generation(self, generated_ids: torch.Tensor, raw_response: str, clean_response: str) -> None:
        flat_ids = generated_ids.detach().flatten().cpu().tolist()
        if not flat_ids:
            raise RuntimeError("Gemma 4 generated zero new tokens")

        tokenizer = self._tokenizer()
        pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        if pad_token_id is not None and all(token_id == pad_token_id for token_id in flat_ids):
            raise RuntimeError(
                f"Gemma 4 generated only pad tokens: token_id={pad_token_id}, "
                f"count={len(flat_ids)}. The run is invalid; check token/generation config."
            )

        if clean_response.strip():
            return

        common_ids = Counter(flat_ids).most_common(5)
        raw_preview = raw_response.replace("\n", "\\n")[:120]
        raise RuntimeError(
            "Gemma 4 generated no text after removing special tokens. "
            f"common_token_ids={common_ids}; raw_preview={raw_preview!r}"
        )

    def _strip_decode_artifacts(self, response: str) -> str:
        tokenizer = self._tokenizer()
        if tokenizer is None:
            return response

        for attr in ("pad_token", "bos_token", "eos_token"):
            token = getattr(tokenizer, attr, None)
            if token:
                response = response.replace(token, "")
        return response

    def _tokenizer(self) -> Any:
        if self.processor is None:
            return None
        return getattr(self.processor, "tokenizer", self.processor)

    def _token_id_in(self, token_id: int, candidates: Any) -> bool:
        if candidates is None:
            return False
        if isinstance(candidates, (list, tuple, set)):
            return token_id in candidates
        return token_id == candidates

    def _resize_image(self, image: Image.Image, max_pixels: int) -> Image.Image:
        if max_pixels <= 0 or image.width * image.height <= max_pixels:
            return image
        resized = image.copy()
        scale = (max_pixels / float(image.width * image.height)) ** 0.5
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        resized.thumbnail(new_size, Image.Resampling.LANCZOS)
        return resized

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _move_inputs_to_device(self, inputs: Any, device: torch.device) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(device)
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _summarize_inputs(self, inputs: Any) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        items = inputs.items() if hasattr(inputs, "items") else []
        for key, value in items:
            if hasattr(value, "shape"):
                summary[key] = {
                    "shape": list(value.shape),
                    "dtype": str(getattr(value, "dtype", "")),
                    "device": str(getattr(value, "device", "")),
                }
            else:
                summary[key] = type(value).__name__
        return summary

    def _resolve_torch_dtype(self, value: Any) -> Any:
        if value in (None, "", "auto"):
            return "auto"
        if isinstance(value, torch.dtype):
            return value
        normalized = str(value).lower()
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(normalized, torch.float16)

    def _build_quantization_config(self, value: Any) -> Any:
        if not value:
            return None

        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "Gemma 4 quantized loading requires bitsandbytes. "
                "On Kaggle, run: pip install -U bitsandbytes"
            ) from exc

        if isinstance(value, str):
            normalized = value.lower()
            value = {
                "load_in_4bit": normalized in {"4bit", "nf4", "bnb4", "bitsandbytes-4bit"},
                "load_in_8bit": normalized in {"8bit", "int8", "bnb8", "bitsandbytes-8bit"},
            }
        elif value is True:
            value = {"load_in_4bit": True}

        if not isinstance(value, dict):
            raise ValueError(f"Unsupported quantization config: {value}")

        config = value.copy()
        for key in ("bnb_4bit_compute_dtype", "bnb_4bit_quant_storage"):
            if key in config:
                config[key] = self._resolve_quantization_dtype(config[key])
        return BitsAndBytesConfig(**config)

    def _resolve_quantization_dtype(self, value: Any) -> Any:
        resolved = self._resolve_torch_dtype(value)
        return torch.float16 if resolved == "auto" else resolved

    def _normalize_max_memory(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = {}
        for key, memory in value.items():
            try:
                normalized[int(key)] = memory
            except (TypeError, ValueError):
                normalized[key] = memory
        return normalized
