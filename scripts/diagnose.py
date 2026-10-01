#!/usr/bin/env python
"""Held-out mapper diagnostics: is the mapped cache close to the receiver's own cache?

Runs kvtransfer.metrics.evaluate on FineWeb-Edu sequences that come AFTER the calibration
sequences in the stream. (The `kvtransfer eval` CLI at 04a8e99 takes the first N documents,
which overlap the calibration set, so its "held-out" numbers are in-sample.)

Read the output against the sharer baseline: mapped-vs-own top-1 agreement should sit well
above source-vs-target agreement, or the mapper is not carrying the receiver's state.

Usage:
  python scripts/diagnose.py --config configs/qwen3-1p7b-to-4b.yaml [--n-seqs 32]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mapper", default=None, help="mapper dir (default: mapper.cache_path)")
    ap.add_argument("--n-seqs", type=int, default=32)
    ap.add_argument("--suffix-len", type=int, default=32)
    args = ap.parse_args()

    from kvtransfer import Mapper, evaluate, load_pair
    from kvtransfer.data import iter_dataset, token_sequences

    cfg = load_config(args.config)
    cal = cfg["mapper"]["calibration"]
    mapper_dir = args.mapper or cfg["mapper"]["cache_path"]

    src, tgt, tok = load_pair(cfg["sharer"]["model_id"], cfg["receiver"]["model_id"])
    mapper = Mapper.load(mapper_dir)
    # same stream and length filter as calibration, so skipping num_sequences skips exactly
    # the calibration sequences
    skip, seq_len = cal["num_sequences"], cal["max_seq_len"]
    seqs = token_sequences(iter_dataset(cal["dataset"]), tok, seq_len=seq_len,
                           n_seqs=skip + args.n_seqs)[skip:]
    print(f"[diagnose] {len(seqs)} held-out seqs (skipped the first {skip}), mapper={mapper_dir}")

    rep = evaluate(src, tgt, mapper, seqs, prefix_len=seq_len - args.suffix_len,
                   suffix_len=args.suffix_len, progress=True)
    print(rep.summary())

    # name by mapper run + k (e.g. qwen3-0p6b-to-1p7b-n100_k8) + eval size, so runs never overwrite
    mp = Path(mapper_dir)
    out = Path("results/diagnostics") / cfg["pair_id"] / f"{mp.parent.name}_{mp.name}_n{args.n_seqs}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep.to_dict(), indent=2))
    print(f"[diagnose] written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
