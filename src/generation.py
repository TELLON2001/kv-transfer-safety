"""Shared generation for every condition (README "Experimental conditions").

All conditions decode with the SAME greedy loop, so the cache source is the only variable:
  native     : receiver prefills the prompt itself
  mapped     : sharer prefills, the mapper converts the cache, the receiver runs the last `hold_back`
               prompt tokens on top and decodes (kvtransfer CrossModelTransfer.handoff)
  sharer     : sharer prefills and decodes alone

The plan's "mismatched" condition (mapped cache from a different prompt) is not implemented: with
hold_back=1 the receiver's own tokens are only the final chat-template token, identical for every
prompt, so "prompt A on prompt B's mapped cache" is token-for-token the mapped condition for B.
A meaningful control needs a different token layout (see research-log 2026-10-01).
"""
from __future__ import annotations

import torch

CONDITIONS = ("native", "mapped", "sharer")


def chat_ids(tok, prompt: str, cfg: dict) -> torch.Tensor:
    """Tokenize one user turn through the chat template exactly as the config fixes it. [1, T]."""
    from kvtransfer import encode_prompt

    return encode_prompt(tok, prompt, chat=True, system=cfg["token_layout"]["system_prompt"],
                         enable_thinking=cfg["decoding"]["thinking_mode"])


def eos_token_id(tok) -> int:
    eos = tok.convert_tokens_to_ids("<|im_end|>")  # Qwen chat turn terminator
    if eos is None or eos == tok.unk_token_id:
        eos = tok.eos_token_id
    return eos


@torch.no_grad()
def start(cond: str, xfer, ids: torch.Tensor, hold_back: int = 1):
    """Return (model, last logits, cache, cache length) for a condition, ready for greedy_decode."""
    from kvtransfer.hf import prefill

    T = ids.shape[1]
    if cond == "native":
        logits, cache = prefill(xfer.target, ids.to(xfer.tgt_dev))
        return xfer.target, logits, cache, T
    if cond == "sharer":
        logits, cache = prefill(xfer.source, ids.to(xfer.src_dev))
        return xfer.source, logits, cache, T
    if cond == "mapped":
        logits, cache = xfer.handoff(ids, hold_back=hold_back)
        return xfer.target, logits, cache, T
    raise ValueError(f"unknown condition {cond!r}")


@torch.no_grad()
def greedy_decode(model, logits, cache, past_len: int, max_new_tokens: int, eos_id: int) -> list[int]:
    """Greedy-decode from (last-position logits, cache of length past_len). Shared by every condition."""
    from kvtransfer.hf import forward_with_cache

    dev = next(model.parameters()).device
    new: list[int] = []
    for _ in range(max_new_tokens):
        nxt = logits[:, -1].argmax(-1)
        new.append(int(nxt))
        if int(nxt) == eos_id:
            break
        out = forward_with_cache(model, cache, nxt[:, None].to(dev), past_len=past_len)
        past_len += 1
        logits, cache = out.logits, out.past_key_values
    return new
