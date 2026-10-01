#!/usr/bin/env python
"""Pilot generation: every prompt under every condition, greedy, written one line at a time.

Resumable: re-running with the same --out skips (prompt id, condition) pairs already written, so an
interrupted multi-hour run continues where it stopped. Harmful and benign prompts are interleaved so
a partial run stays balanced across splits.

Outputs (git-ignored; contain harmful model text):
  results/pilot/<pair_id>/<mapper run>_<mapper dir>/generations.jsonl   one record per (prompt, condition)
  .../meta.json                                                          run settings, written once

Usage:
  python scripts/pilot_generate.py --config configs/qwen3-0p6b-to-1p7b.yaml \
      --mapper results/mappers/qwen3-0p6b-to-1p7b/k8_lam1.5e+03 --prompts <repo>/data/pilot/prompts.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.generation import CONDITIONS, chat_ids, eos_token_id, greedy_decode, start


def interleave(prompts: list[dict]) -> list[dict]:
    """harmful, benign, harmful, benign, ... so any prefix of the run is balanced."""
    h = [p for p in prompts if p["split"] == "harmful"]
    b = [p for p in prompts if p["split"] == "benign"]
    out = []
    for i in range(max(len(h), len(b))):
        out += h[i:i + 1] + b[i:i + 1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mapper", required=True)
    ap.add_argument("--prompts", required=True, help="data/pilot/prompts.jsonl from pilot_prompts.py")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--hold-back", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts (after interleaving)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from kvtransfer import CrossModelTransfer, Mapper, load_pair

    cfg = load_config(args.config)
    if cfg["decoding"]["do_sample"]:
        raise ValueError("pilot is greedy only; set decoding.do_sample: false")
    torch.manual_seed(cfg["decoding"]["seed"])
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    raw = Path(args.prompts).read_bytes()
    prompts = interleave([json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()])
    if args.limit:
        prompts = prompts[:args.limit]

    mp = Path(args.mapper)
    out_path = Path(args.out) if args.out else (
        Path("results/pilot") / cfg["pair_id"] / f"{mp.parent.name}_{mp.name}" / "generations.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["id"], r["condition"]))
    todo = [(p, c) for p in prompts for c in conditions if (p["id"], c) not in done]
    print(f"[pilot] {len(prompts)} prompts x {len(conditions)} conditions; {len(done)} done, "
          f"{len(todo)} to go -> {out_path}", flush=True)
    if not todo:
        return 0

    src, tgt, tok = load_pair(cfg["sharer"]["model_id"], cfg["receiver"]["model_id"])
    xfer = CrossModelTransfer(src, tgt, Mapper.load(mp))
    eos_id = eos_token_id(tok)

    meta_path = out_path.parent / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "pair_id": cfg["pair_id"], "sharer": cfg["sharer"]["model_id"],
            "receiver": cfg["receiver"]["model_id"], "mapper": str(mp), "conditions": conditions,
            "hold_back": args.hold_back, "max_new_tokens": args.max_new_tokens,
            "decoding": cfg["decoding"], "token_layout": cfg["token_layout"],
            "prompts_file": str(args.prompts), "prompts_sha256": hashlib.sha256(raw).hexdigest(),
            "templated_example": tok.decode(chat_ids(tok, prompts[0]["prompt"], cfg)[0]),
            "started": datetime.now().isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")

    t0 = time.time()
    cache_ids: dict[str, torch.Tensor] = {}
    with out_path.open("a", encoding="utf-8") as fh:
        for n, (p, cond) in enumerate(todo, 1):
            ids = cache_ids.setdefault(p["id"], chat_ids(tok, p["prompt"], cfg))
            t = time.time()
            model, logits, cache, past = start(cond, xfer, ids, args.hold_back)
            toks = greedy_decode(model, logits, cache, past, args.max_new_tokens, eos_id)
            fh.write(json.dumps({
                **p, "condition": cond,
                "response": tok.decode(toks, skip_special_tokens=True).strip(),
                "n_new_tokens": len(toks),
                "hit_limit": len(toks) >= args.max_new_tokens and toks[-1] != eos_id,
                "n_prompt_tokens": int(ids.shape[1]), "seconds": round(time.time() - t, 2),
            }, ensure_ascii=False) + "\n")
            fh.flush()
            if n % 25 == 0 or n == len(todo):
                rate = (time.time() - t0) / n
                print(f"[pilot] {n}/{len(todo)}  {rate:.1f}s each, ~{rate * (len(todo) - n) / 60:.0f} min left",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
