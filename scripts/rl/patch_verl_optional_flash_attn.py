#!/usr/bin/env python3
"""Patch VAGEN's verl fork for the server's Gemma RL runtime."""

from __future__ import annotations

import argparse
import py_compile
from pathlib import Path


OLD_IMPORT = "from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis"

OPTIONAL_IMPORT = '''try:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
except ModuleNotFoundError:
    from einops import rearrange

    def index_first_axis(hidden_states, indices):
        return hidden_states[indices]

    def unpad_input(hidden_states, attention_mask):
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        unpadded = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices)
        cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32),
            (1, 0),
        )
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        return unpadded, indices, cu_seqlens, max_seqlen_in_batch

    def pad_input(hidden_states, indices, batch, seqlen):
        output = hidden_states.new_zeros((batch * seqlen,) + hidden_states.shape[1:])
        output[indices] = hidden_states
        return rearrange(output, "(b s) ... -> b s ...", b=batch)
'''

OLD_TRANSFORMERS_IMPORT = (
    "from transformers import AutoModelForCausalLM, AutoConfig, "
    "AutoModelForVision2Seq, AutoModelForImageTextToText"
)

OPTIONAL_TRANSFORMERS_IMPORT = '''        from transformers import AutoModelForCausalLM, AutoConfig
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            AutoModelForVision2Seq = None
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            AutoModelForImageTextToText = None
'''

BROKEN_OPTIONAL_TRANSFORMERS_IMPORT = '''        from transformers import AutoModelForCausalLM, AutoConfig
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None
'''

OLD_AUTO_MODEL_SELECTION = '''            if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                actor_module_class = AutoModelForImageTextToText
            else:
                actor_module_class = AutoModelForCausalLM
'''

OPTIONAL_AUTO_MODEL_SELECTION = '''            if AutoModelForVision2Seq is not None and type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                actor_module_class = AutoModelForVision2Seq
            elif AutoModelForImageTextToText is not None and type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                actor_module_class = AutoModelForImageTextToText
            else:
                actor_module_class = AutoModelForCausalLM
'''


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if OPTIONAL_IMPORT in text:
        py_compile.compile(str(path), doraise=True)
        return False
    if OLD_IMPORT not in text:
        raise RuntimeError(f"Could not find flash-attn import in {path}")
    path.write_text(text.replace(OLD_IMPORT, OPTIONAL_IMPORT), encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    return True


def patch_fsdp_workers(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text

    old_import_line = f"        {OLD_TRANSFORMERS_IMPORT}"
    if BROKEN_OPTIONAL_TRANSFORMERS_IMPORT in text:
        text = text.replace(BROKEN_OPTIONAL_TRANSFORMERS_IMPORT, OPTIONAL_TRANSFORMERS_IMPORT)
    elif old_import_line in text:
        text = text.replace(old_import_line, OPTIONAL_TRANSFORMERS_IMPORT)
    if OLD_AUTO_MODEL_SELECTION in text:
        text = text.replace(OLD_AUTO_MODEL_SELECTION, OPTIONAL_AUTO_MODEL_SELECTION)

    text = text.replace(
        'attn_implementation="flash_attention_2"',
        'attn_implementation=os.getenv("VERL_HF_ATTN_IMPLEMENTATION", "sdpa")',
    )
    text = text.replace(
        "attn_implementation='flash_attention_2'",
        'attn_implementation=os.getenv("VERL_HF_ATTN_IMPLEMENTATION", "sdpa")',
    )

    changed = text != original
    if changed:
        path.write_text(text, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    if not changed:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-root", type=Path, required=True)
    args = parser.parse_args()

    actor_path = args.verl_root / "verl" / "workers" / "actor" / "dp_actor.py"
    fsdp_path = args.verl_root / "verl" / "workers" / "fsdp_workers.py"
    if not actor_path.exists():
        raise SystemExit(f"[ERROR] Missing verl actor file: {actor_path}")
    if not fsdp_path.exists():
        raise SystemExit(f"[ERROR] Missing verl FSDP worker file: {fsdp_path}")

    changed = patch_file(actor_path)
    if changed:
        print(f"[INFO] Patched optional flash-attn fallback in {actor_path}")
    else:
        print(f"[INFO] Optional flash-attn fallback already present in {actor_path}")

    changed = patch_fsdp_workers(fsdp_path)
    if changed:
        print(f"[INFO] Patched transformers/attention compatibility in {fsdp_path}")
    else:
        print(f"[INFO] Transformers/attention compatibility already present in {fsdp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
