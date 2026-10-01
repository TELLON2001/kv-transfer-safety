#!/usr/bin/env python
"""Refit the mapper from saved calibration statistics for several ridge lambdas (no models needed).

kvtransfer adds lambda to the UNNORMALIZED centered Gram (a sum over all calibration tokens), so a
fixed lambda regularizes less as the token count grows. This script reports lambda on a relative
scale, r = lambda / mean(diag(Gxx)) (mean per-feature centered sum of squares), and fits one mapper
per requested value. Run scripts/diagnose.py on each output directory to compare held out.

Usage:
  python scripts/refit.py --config configs/qwen3-0p6b-to-1p7b.yaml --rel 1e-4,1e-3,1e-2,1e-1
  python scripts/refit.py --config ... --lams 0.01,100        # absolute lambdas instead
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config


def mean_centered_diag(acc) -> float:
    """Mean of diag(Gxx) for the centered Gram, matching MomentAccumulator.centered."""
    dx = (acc.sum_x / acc.n).to(acc.dtype)
    return float((acc.gram.diagonal().double() - acc.n * dx.double() ** 2).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", default=None, help="override mapper.k")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--rel", help="comma list of relative lambdas r (lambda = r * mean diag of V Gram)")
    grp.add_argument("--lams", help="comma list of absolute lambdas")
    args = ap.parse_args()

    from kvtransfer import CalibrationStats, Mapper

    cfg = load_config(args.config)
    mcfg = cfg["mapper"]
    k = args.k or mcfg["k"]
    k = k if k == "all" else int(k)
    key_space = "content" if mcfg["strip_positional_encoding"] else "rope"

    stats = CalibrationStats.load(mcfg["stats_path"], device="cpu")
    scale = {kind: mean_centered_diag(acc) for kind, acc in stats.acc.items()}
    n_tok = stats.acc["V"].n
    print(f"[refit] {n_tok} tokens; mean centered diag: " +
          ", ".join(f"{kind}={v:.4g}" for kind, v in scale.items()))
    print(f"[refit] config lambda={mcfg['lam']} is r={mcfg['lam'] / scale['V']:.3g} relative to V "
          f"(r={mcfg['lam'] / scale['K']:.3g} relative to K)")

    if args.rel:
        lams = [float(r) * scale["V"] for r in args.rel.split(",")]
    else:
        lams = [float(x) for x in args.lams.split(",")]

    score = None
    base = Path(mcfg["cache_path"]).parent
    for lam in lams:
        print(f"[refit] k={k} lambda={lam:.4g} (r_V={lam / scale['V']:.3g}, r_K={lam / scale['K']:.3g})",
              flush=True)
        m = Mapper.fit(stats, k=k, lam=lam, score=score, key_space=key_space)
        score = np.asarray(m.meta["selection_score"])  # selection does not depend on lambda; reuse it
        out = base / f"k{k}_lam{lam:.3g}"
        m.save(out)
        norms = [float(w.norm()) for w in m.W_V]
        worst = int(np.argmax(norms))
        print(f"[refit]   saved {out}; in-sample R2 K={np.mean(m.fit_r2['K']):.3f} "
              f"V={np.mean(m.fit_r2['V']):.3f}; max ||W_V|| = {norms[worst]:.1f} (layer {worst})",
              flush=True)
        del m
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
