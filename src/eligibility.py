"""Check that a sharer/receiver pair is eligible for KV-cache mapping.

The closed-form mapper (Heo et al.) needs the two models to share a tokenizer and
to have a matched KV-head count and head dimension. This module checks those from
the model configs alone (no weights loaded), so it is cheap to run as a guard
before a reproduction or study run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EligibilityReport:
    ok: bool
    checks: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    def raise_if_bad(self) -> "EligibilityReport":
        if not self.ok:
            raise ValueError("pair is not eligible for mapping:\n  " + "\n  ".join(self.problems))
        return self


def _head_dim(model_cfg: Any) -> int | None:
    # head_dim is explicit on newer configs; otherwise hidden_size / num_attention_heads.
    hd = getattr(model_cfg, "head_dim", None)
    if hd:
        return int(hd)
    hs = getattr(model_cfg, "hidden_size", None)
    nh = getattr(model_cfg, "num_attention_heads", None)
    if hs and nh:
        return int(hs // nh)
    return None


def check_pair(cfg: dict[str, Any]) -> EligibilityReport:
    """Load the two model configs and verify the eligibility constraints in `cfg`."""
    from transformers import AutoConfig, AutoTokenizer  # heavy import kept local

    sharer_id = cfg["sharer"]["model_id"]
    receiver_id = cfg["receiver"]["model_id"]
    want = cfg.get("eligibility", {})

    s_cfg = AutoConfig.from_pretrained(sharer_id)
    r_cfg = AutoConfig.from_pretrained(receiver_id)

    s_kv = getattr(s_cfg, "num_key_value_heads", None)
    r_kv = getattr(r_cfg, "num_key_value_heads", None)
    s_hd, r_hd = _head_dim(s_cfg), _head_dim(r_cfg)

    problems: list[str] = []

    if s_kv != r_kv:
        problems.append(f"num_key_value_heads differ: sharer={s_kv}, receiver={r_kv}")
    if s_hd != r_hd:
        problems.append(f"head_dim differ: sharer={s_hd}, receiver={r_hd}")

    # Optional expected values from the config act as a regression guard.
    exp_kv = want.get("num_key_value_heads")
    if exp_kv is not None and r_kv != exp_kv:
        problems.append(f"num_key_value_heads {r_kv} != expected {exp_kv}")
    exp_hd = want.get("head_dim")
    if exp_hd is not None and r_hd != exp_hd:
        problems.append(f"head_dim {r_hd} != expected {exp_hd}")

    if want.get("require_shared_tokenizer", True):
        s_tok = AutoTokenizer.from_pretrained(sharer_id)
        r_tok = AutoTokenizer.from_pretrained(receiver_id)
        if s_tok.get_vocab() != r_tok.get_vocab():
            problems.append("tokenizers do not share a vocab (mapper assumes a shared tokenizer)")

    checks = {
        "num_key_value_heads": {"sharer": s_kv, "receiver": r_kv},
        "head_dim": {"sharer": s_hd, "receiver": r_hd},
    }
    return EligibilityReport(ok=not problems, checks=checks, problems=problems)
