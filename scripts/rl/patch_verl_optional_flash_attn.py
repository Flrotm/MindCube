#!/usr/bin/env python3
"""Patch VAGEN's verl fork so flash-attn padding helpers are optional."""

from __future__ import annotations

import argparse
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


def patch_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if OPTIONAL_IMPORT in text:
        return False
    if OLD_IMPORT not in text:
        raise RuntimeError(f"Could not find flash-attn import in {path}")
    path.write_text(text.replace(OLD_IMPORT, OPTIONAL_IMPORT), encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verl-root", type=Path, required=True)
    args = parser.parse_args()

    actor_path = args.verl_root / "verl" / "workers" / "actor" / "dp_actor.py"
    if not actor_path.exists():
        raise SystemExit(f"[ERROR] Missing verl actor file: {actor_path}")

    changed = patch_file(actor_path)
    if changed:
        print(f"[INFO] Patched optional flash-attn fallback in {actor_path}")
    else:
        print(f"[INFO] Optional flash-attn fallback already present in {actor_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
