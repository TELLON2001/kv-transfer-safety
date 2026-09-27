#!/usr/bin/env python
"""Phase 2: reproduce one accuracy number for a model pair before measuring anything new.

Flow:
  1. load the pair config
  2. assert eligibility (shared tokenizer, matched KV heads / head_dim)
  3. build or load the closed-form mapper
  4. run one capability benchmark under the *mapped* cache
  5. compare to the paper's reference within tolerance (the week-4 go/no-go gate)

The kvtransfer- and lm-eval-specific calls are marked INTEGRATE: their exact
signatures must be confirmed against the installed packages. Everything around them
(config, eligibility, gating, logging) is real and runnable.

Usage:
  python scripts/reproduce.py --config configs/qwen3-1p7b-to-4b.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# make src/ importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.eligibility import check_pair


def build_or_load_mapper(cfg: dict):
    """Return a fitted sharer->receiver mapper, loading a cached one if present."""
    cache_path = Path(cfg["mapper"]["cache_path"])
    if cache_path.exists():
        print(f"[mapper] loading cached mapper: {cache_path}")
        # INTEGRATE: kvtransfer load, e.g. kvtransfer.RidgeMapper.load(cache_path)
        raise NotImplementedError("wire up kvtransfer mapper load")
    print("[mapper] fitting mapper on calibration data "
          f"({cfg['mapper']['calibration']['num_sequences']} seqs from "
          f"{cfg['mapper']['calibration']['dataset']})")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # INTEGRATE: kvtransfer calibrate(sharer, receiver, dataset=..., n=..., strip_pe=...)
    #            then mapper.save(cache_path)
    raise NotImplementedError("wire up kvtransfer mapper calibration + save")


def run_benchmark(cfg: dict, mapper) -> float:
    """Run the reproduction task under the mapped cache and return accuracy (0..1)."""
    task = cfg["reproduction"]["task"]
    if not task:
        raise ValueError("set reproduction.task in the config (an lm-eval task name)")
    print(f"[eval] running task={task} under the mapped cache")
    # INTEGRATE: wrap the receiver so decode reads the mapped cache, expose it as an
    #            lm-eval model, and call lm_eval.simple_evaluate(model=..., tasks=[task],
    #            limit=cfg['reproduction'].get('limit')). Return the headline accuracy.
    raise NotImplementedError("wire up lm-eval with the mapped-cache receiver")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a pair config YAML")
    ap.add_argument("--skip-eligibility", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[config] pair={cfg['pair_id']}  "
          f"{cfg['sharer']['model_id']} -> {cfg['receiver']['model_id']}")

    if not args.skip_eligibility:
        report = check_pair(cfg)
        print("[eligibility] " + json.dumps(report.checks))
        report.raise_if_bad()
        print("[eligibility] OK")

    mapper = build_or_load_mapper(cfg)
    acc = run_benchmark(cfg, mapper)

    ref = cfg["reproduction"]["reference_accuracy"]
    tol = cfg["reproduction"]["tolerance_points"] / 100.0
    gap = None if ref is None else abs(acc - ref)
    passed = gap is not None and gap <= tol

    print(f"\n[result] accuracy={acc:.4f}  reference={ref}  "
          f"gap={gap}  tolerance={tol}  -> {'PASS' if passed else 'CHECK'}")
    print(f"[log] add to research-log.md on {date.today().isoformat()}: "
          f"task={cfg['reproduction']['task']} acc={acc:.4f} (ref {ref})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
