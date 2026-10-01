#!/usr/bin/env python
"""Phase 2: reproduce one accuracy number for a model pair before measuring anything new.

Flow:
  1. load the pair config
  2. assert eligibility (shared tokenizer, matched KV heads / head_dim)
  3. build or load the closed-form mapper (kvtransfer: calibrate -> Mapper.fit)
  4. run one capability benchmark under target-only, sharer-only and the mapped cache
     (kvtransfer.lm_eval_adapter.run_harness, which registers lm-eval model "kvtransfer")
  5. compare to the paper's reference within tolerance (the week-4 go/no-go gate)

kvtransfer is an independent reconstruction of Heo et al. (the paper released no code);
its API was read at commit 04a8e99 (see requirements.txt).

Usage:
  python scripts/reproduce.py --config configs/qwen3-1p7b-to-4b.yaml
  # quick L4 pass: short calibration (own stats/mapper paths) and a subsampled task
  python scripts/reproduce.py --config configs/qwen3-1p7b-to-4b.yaml --n-seqs 100 --task hellaswag --limit 250
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


def build_or_load_mapper(cfg: dict) -> str:
    """Return the directory of a fitted sharer->receiver mapper, fitting one if none is cached.

    The mapper is passed to lm-eval by path, so this returns the path rather than the object.
    """
    from kvtransfer import CalibrationStats, Mapper, calibrate, load_pair
    from kvtransfer.data import batches, iter_dataset, token_sequences

    mcfg = cfg["mapper"]
    cal = mcfg["calibration"]
    mapper_dir = Path(mcfg["cache_path"])
    if (mapper_dir / "mapper.json").exists():
        print(f"[mapper] loading cached mapper: {mapper_dir}")
        print(Mapper.load(mapper_dir).summary())
        return str(mapper_dir)

    key_space = "content" if mcfg["strip_positional_encoding"] else "rope"
    kinds = ("K", "V") if key_space == "content" else ("K", "V", "Krope")
    stats_dir = Path(mcfg["stats_path"])
    if (stats_dir / "meta.json").exists():
        # moments are reusable for any k, so a k sweep only pays for calibration once
        print(f"[mapper] loading cached calibration stats: {stats_dir}")
        stats = CalibrationStats.load(stats_dir, device=mcfg["stats_device"])
    else:
        print("[mapper] calibrating on "
              f"{cal['num_sequences']} x {cal['max_seq_len']} tokens from {cal['dataset']} "
              f"(stride {cal['stride']}, stats on {mcfg['stats_device']})")
        src, tgt, tok = load_pair(cfg["sharer"]["model_id"], cfg["receiver"]["model_id"])
        seqs = token_sequences(iter_dataset(cal["dataset"]), tok,
                               seq_len=cal["max_seq_len"], n_seqs=cal["num_sequences"])
        stats = calibrate(src, tgt, batches(seqs, cal["batch_size"]), stride=cal["stride"],
                          kinds=kinds, stats_device=mcfg["stats_device"],
                          source_name=cfg["sharer"]["model_id"],
                          target_name=cfg["receiver"]["model_id"], progress=True)
        stats_dir.mkdir(parents=True, exist_ok=True)
        stats.save(stats_dir)
        del src, tgt

    print(f"[mapper] fitting ridge: k={mcfg['k']} lam={mcfg['lam']} key_space={key_space}")
    mapper = Mapper.fit(stats, k=mcfg["k"], lam=mcfg["lam"], key_space=key_space)
    mapper.save(mapper_dir)
    print(mapper.summary())
    return str(mapper_dir)


def run_benchmark(cfg: dict, mapper_dir: str) -> dict:
    """Run the reproduction task for receiver, sharer and mapped cache; return the retention row."""
    from kvtransfer.lm_eval_adapter import run_harness

    rcfg = cfg["reproduction"]
    task = rcfg["task"]
    if not task:
        raise ValueError("set reproduction.task in the config (an lm-eval task name)")
    print(f"[eval] task={task} limit={rcfg.get('limit')}: receiver, sharer, mapped")
    out = run_harness(
        source=cfg["sharer"]["model_id"],
        target=cfg["receiver"]["model_id"],
        mapper=mapper_dir,
        tasks=[task],
        dtype=cfg["receiver"]["dtype"],
        limit=rcfg.get("limit"),
        num_fewshot=rcfg.get("num_fewshot"),
        metric=rcfg.get("metric"),
    )
    out_dir = Path("results/reproduction") / cfg["pair_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{task}.json").write_text(json.dumps(out["retention"], indent=2))
    print(f"[eval] retention rows written to {out_dir / (task + '.json')}")
    return out["retention"][0]


def apply_overrides(cfg: dict, args) -> None:
    """Apply CLI overrides for quick runs without editing the config file."""
    if args.task is not None:
        cfg["reproduction"]["task"] = args.task
    if args.limit is not None:
        cfg["reproduction"]["limit"] = int(args.limit) if args.limit >= 1 else args.limit
    if args.n_seqs is not None and args.n_seqs != cfg["mapper"]["calibration"]["num_sequences"]:
        m = cfg["mapper"]
        m["calibration"]["num_sequences"] = args.n_seqs
        m["stats_path"] = f"{m['stats_path']}-n{args.n_seqs}"
        mapper_dir = Path(m["cache_path"])
        m["cache_path"] = str(mapper_dir.parent.with_name(f"{mapper_dir.parent.name}-n{args.n_seqs}")
                              / mapper_dir.name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to a pair config YAML")
    ap.add_argument("--skip-eligibility", action="store_true")
    ap.add_argument("--task", default=None, help="override reproduction.task")
    ap.add_argument("--limit", type=float, default=None, help="override reproduction.limit")
    ap.add_argument("--n-seqs", type=int, default=None,
                    help="override calibration size; stats and mapper go to separate -n<N> paths")
    ap.add_argument("--fit-only", action="store_true", help="stop after the mapper is fitted")
    args = ap.parse_args()

    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    print(f"[config] pair={cfg['pair_id']}  "
          f"{cfg['sharer']['model_id']} -> {cfg['receiver']['model_id']}")

    if not args.skip_eligibility:
        report = check_pair(cfg)
        print("[eligibility] " + json.dumps(report.checks))
        report.raise_if_bad()
        print("[eligibility] OK")

    mapper_dir = build_or_load_mapper(cfg)
    if args.fit_only:
        return 0
    row = run_benchmark(cfg, mapper_dir)
    acc = row["transfer"]

    ref = cfg["reproduction"]["reference_accuracy"]
    tol = cfg["reproduction"]["tolerance_points"] / 100.0
    gap = None if ref is None else abs(acc - ref)
    passed = gap is not None and gap <= tol

    print(f"\n[result] metric={row['metric']}  receiver={row['target']:.4f}  "
          f"sharer={row['source']:.4f}  mapped={acc:.4f}  "
          f"retention={row['retention_pct']:.1f}%  floor-norm={row['normalized_retention_pct']:.1f}%")
    print(f"[result] reference={ref}  gap={gap}  tolerance={tol}  -> {'PASS' if passed else 'CHECK'}")
    print(f"[log] add to research-log.md on {date.today().isoformat()}: "
          f"task={cfg['reproduction']['task']} mapped={acc:.4f} receiver={row['target']:.4f} "
          f"(ref {ref})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
