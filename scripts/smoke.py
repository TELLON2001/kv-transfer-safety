#!/usr/bin/env python
"""Smoke test: a handful of chat prompts, native vs mapped vs sharer, written out for hand reading.

Not a measurement. It answers one question before scaling up: does anything visibly change in the
receiver's behavior (refusals, coherence, over-refusal) when it decodes from a mapped cache?

Conditions (README "Experimental conditions"), all decoded with the SAME greedy loop below so the
cache source is the only variable:
  native  : receiver prefills the prompt itself
  mapped  : sharer prefills, mapper converts the cache, receiver runs the last `hold_back` prompt
            tokens on top and decodes (kvtransfer CrossModelTransfer.handoff)
  sharer  : sharer prefills and decodes alone (the floor)

Outputs go to results/smoke/<pair_id>/<mapper>/ (git-ignored: they may contain harmful completions):
  smoke.jsonl  one record per prompt with all three completions and empty label fields
  smoke.md     side-by-side view for hand reading and labeling (refuse / comply / incoherent)

Usage:
  python scripts/smoke.py --config configs/qwen3-1p7b-to-4b.yaml [--mapper DIR] [--max-new-tokens 256]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.generation import chat_ids, eos_token_id, greedy_decode, start

CONDITIONS = ("native", "mapped", "sharer")


def load_prompts(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [{"category": cat, "prompt": p} for cat, items in data.items() for p in items]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompts", default="configs/smoke_prompts.yaml")
    ap.add_argument("--mapper", default=None, help="mapper dir (default: mapper.cache_path)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--hold-back", type=int, default=1,
                    help="prompt tokens the receiver runs itself on top of the mapped cache")
    args = ap.parse_args()

    from kvtransfer import CrossModelTransfer, Mapper, load_pair

    cfg = load_config(args.config)
    dec, layout = cfg["decoding"], cfg["token_layout"]
    if dec["do_sample"]:
        raise ValueError("smoke test is greedy only; set decoding.do_sample: false")
    torch.manual_seed(dec["seed"])

    mapper_dir = args.mapper or cfg["mapper"]["cache_path"]
    src, tgt, tok = load_pair(cfg["sharer"]["model_id"], cfg["receiver"]["model_id"])
    xfer = CrossModelTransfer(src, tgt, Mapper.load(mapper_dir))
    eos_id = eos_token_id(tok)

    prompts = load_prompts(Path(args.prompts))
    mp = Path(mapper_dir)  # one folder per mapper so runs never overwrite each other
    out_dir = Path("results/smoke") / cfg["pair_id"] / f"{mp.parent.name}_{mp.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, item in enumerate(prompts):
        ids = chat_ids(tok, item["prompt"], cfg)
        T = ids.shape[1]
        rec = {"id": i, **item, "n_prompt_tokens": T, "n_mapped_tokens": T - args.hold_back,
               "receiver_runs": tok.decode(ids[0, T - args.hold_back:])}
        for cond in CONDITIONS:
            model, logits, cache, past = start(cond, xfer, ids, args.hold_back)
            toks = greedy_decode(model, logits, cache, past, args.max_new_tokens, eos_id)
            rec[cond] = tok.decode(toks, skip_special_tokens=True).strip()
            rec[f"{cond}_hit_limit"] = len(toks) >= args.max_new_tokens and toks[-1] != eos_id
            rec[f"label_{cond}"] = None  # fill by hand: refuse | comply | incoherent
        records.append(rec)
        print(f"[smoke] {i + 1}/{len(prompts)} {item['category']}: {item['prompt'][:60]}")

    first = chat_ids(tok, prompts[0]["prompt"], cfg)
    header = {"pair_id": cfg["pair_id"], "mapper": mapper_dir, "hold_back": args.hold_back,
              "max_new_tokens": args.max_new_tokens, "thinking_mode": dec["thinking_mode"],
              "system_prompt": layout["system_prompt"],
              "templated_example": tok.decode(first[0])}

    with (out_dir / "smoke.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"header": header}) + "\n")
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    lines = [f"# Smoke test: {cfg['pair_id']}", "",
             f"mapper `{mapper_dir}`, hold_back {args.hold_back}, greedy, "
             f"max_new_tokens {args.max_new_tokens}, thinking {dec['thinking_mode']}", "",
             "Token layout (first prompt, as templated):", "", "```", header["templated_example"], "```", ""]
    for r in records:
        lines += [f"## {r['id']}. [{r['category']}] {r['prompt']}", "",
                  f"{r['n_prompt_tokens']} prompt tokens, {r['n_mapped_tokens']} mapped, "
                  f"receiver runs `{r['receiver_runs']!r}`", ""]
        for cond in CONDITIONS:
            cut = " (hit max_new_tokens)" if r[f"{cond}_hit_limit"] else ""
            lines += [f"**{cond}**{cut}: label ____", "", "> " + r[cond].replace("\n", "\n> "), ""]
    (out_dir / "smoke.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[smoke] wrote {out_dir / 'smoke.jsonl'} and {out_dir / 'smoke.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
