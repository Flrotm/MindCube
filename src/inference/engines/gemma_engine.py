"""
Gemma 4 multimodal inference engine using the Transformers backend.
"""

import re
import traceback
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
                from transformers import AutoModelForMultimodalLM
            except ImportError as exc:
                raise ImportError(
                    "Gemma 4 multimodal inference requires a recent Transformers version. "
                    "On Kaggle, run: pip install -U transformers accelerate torchvision"
                ) from exc

            dtype = self._resolve_torch_dtype(self.config.get("torch_dtype", "float16"))
            device_map = self.config.get("device_map")
            if not device_map:
                device_map = {"": 0} if torch.cuda.is_available() else "auto"

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )

            model_kwargs = {
                "device_map": device_map,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if dtype == "auto":
                model_kwargs["dtype"] = "auto"
            else:
                model_kwargs["dtype"] = dtype

            try:
                self.model = AutoModelForMultimodalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs,
                )
            except TypeError:
                if "dtype" in model_kwargs:
                    model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
                self.model = AutoModelForMultimodalLM.from_pretrained(
                    self.model_path,
                    **model_kwargs,
                )

            print(f"Gemma 4 model loaded successfully using {self.backend} backend")
        except Exception as exc:
            print(f"Error loading Gemma 4 model: {exc}")
            raise

    def process_input(self, prompt: str, image_paths: List[str], **kwargs) -> Dict[str, Any]:
        """Create Gemma 4 chat messages from a MindCube prompt and images."""
        if not self.validate_inputs(prompt, image_paths):
            raise ValueError("Invalid input data")

        images, errors = ImageProcessor.load_and_validate_images(image_paths)
        if errors:
            print(f"Image loading errors: {errors}")

        max_pixels = kwargs.get("max_pixels", self.config.get("max_pixels", 512 * 512))
        images = [self._resize_image(image, max_pixels) for image in images]

        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

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
                template_kwargs["enable_thinking"] = bool(self.config["enable_thinking"])

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
            input_len = inputs["input_ids"].shape[-1]

            generation_config = self._generation_config(kwargs)
            outputs = self.model.generate(**inputs, **generation_config)
            response = self.processor.decode(
                outputs[0][input_len:],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            response = self._parse_response(response)
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
        }
        config.update({key: value for key, value in kwargs.items() if key in valid})
        config.setdefault("max_new_tokens", self.config.get("max_new_tokens", 512))

        temperature = float(config.get("temperature", 0.0) or 0.0)
        if temperature <= 0:
            config["do_sample"] = False
            config.pop("temperature", None)
            config.pop("top_p", None)
            config.pop("top_k", None)
        else:
            config["do_sample"] = True
        return config

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

        response = re.sub(r"<\|channel\>thought\n.*?<channel\|>", "", response, flags=re.DOTALL)
        response = re.sub(r"<\|channel\>final\n?", "", response)
        response = response.replace("<channel|>", "")
        return response

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
