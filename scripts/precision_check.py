#!/usr/bin/env python
"""Is the mid-layer value collapse at lambda=0.01 a float32 accumulation effect or real overfitting?

kvtransfer accumulates the Gram/Cross moments in float32 (`stats_dtype` default) and solves in
float64. At lambda=0.01 the penalty is ~7e-9 of the Gram scale, below float32 rounding of the stored
entries (~6e-8 relative), so precision and regularization cannot be separated from the saved stats.
This script re-accumulates the VALUE moments in float64 with stock `kvtransfer.calibrate` (same
data, stride, batch size), refits V for each reference mapper at that mapper's own lambda and layer
selection, keeps that mapper's K maps unchanged, and compares against the float32 originals.

Phases (each skipped when its outputs exist, unless --force):
  calibrate  pre-check data identity on batch 1, accumulate V in float64, post-check, fit V per
             reference mapper, save new mappers + float64 centered Gram blocks for --layers
  compare    load the float32 V stats, compare the same blocks: perturbation norm vs lambda,
             spectra, condition numbers
  diagnose   scripts/diagnose.py on each new mapper (same 32 held-out passages as the sweep)

Run from the work dir (results/ relative), e.g. C:\\Users\\nadim\\kvt-work:
  python <repo>/scripts/precision_check.py --config <repo>/configs/qwen3-0p6b-to-1p7b.yaml
Close browsers and other heavy apps first: the float64 V moments alone are ~13.2 GB of RAM.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config import load_config

F64_GB = 13.2  # V gram + cross, 28672^2 float64 each


def log(msg: str) -> None:
    print(f"[precision {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_ram_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 1e9


def centered_lowmem(self, rows=None, cols=None):
    """MomentAccumulator.centered with gather indexing (gram[rows[:,None], rows]) instead of
    gram[rows][:, rows]. Identical values (pure gathers, same arithmetic), but it never materializes
    the [len(rows), p] intermediate, which is 1.9 GB per call in float64."""
    dx = (self.sum_x / self.n).to(self.dtype)
    dy = (self.sum_y / self.n).to(self.dtype)
    rows = torch.arange(self.gram.shape[0]) if rows is None else rows.to(self.device)
    cols = torch.arange(self.cross.shape[1]) if cols is None else cols.to(self.device)
    g = self.gram[rows[:, None], rows]
    c = self.cross[rows[:, None], cols]
    dxr, dyc = dx[rows], dy[cols]
    gxx = g - self.n * torch.outer(dxr, dxr)
    gxy = c - self.n * torch.outer(dxr, dyc)
    syy = (self.yy[cols] - self.n * dyc.double() ** 2).to(self.dtype)
    mx = dxr + self.shift_x[rows]
    my = dyc + self.shift_y[cols]
    return gxx, gxy, syy, mx, my


def mean_centered_diag(acc) -> float:
    """Same scale as scripts/refit.py: mean of diag(Gxx) for the full centered Gram."""
    dx = (acc.sum_x / acc.n).to(acc.dtype)
    return float((acc.gram.diagonal().double() - acc.n * dx.double() ** 2).mean())


def read_small(stats_dir: Path, kind: str) -> dict:
    """The small tensors of a saved accumulator, without loading the 6.6 GB matrices."""
    from safetensors import safe_open
    with safe_open(str(stats_dir / f"stats_{kind}.safetensors"), framework="pt") as f:
        return {k: f.get_tensor(k) for k in ("n", "shift_x", "shift_y", "sum_x", "sum_y")}


def rel_maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-30))


def spectral_norm_sym(m: torch.Tensor, iters: int = 100) -> float:
    """Largest |eigenvalue| of a symmetric matrix by power iteration (cheap vs a full eig)."""
    g = torch.Generator().manual_seed(0)
    v = torch.randn(m.shape[0], dtype=m.dtype, generator=g)
    v /= v.norm()
    s = 0.0
    for _ in range(iters):
        w = m @ v
        s = float(w.norm())
        if s == 0.0:
            return 0.0
        v = w / s
    return s


def spectrum(g: torch.Tensor, lam: float, noise: float | None = None) -> dict:
    e = torch.linalg.eigvalsh(g)
    emin, emax = float(e[0]), float(e[-1])
    out = {"eig_max": emax, "eig_min": emin, "n_negative": int((e < 0).sum()),
           "n_below_lam": int((e < lam).sum()),
           "cond_raw": emax / emin if emin > 0 else float("inf"),
           "cond_with_lam": (emax + lam) / (emin + lam) if emin + lam > 0 else float("inf")}
    if noise is not None:
        out["n_below_f32_noise"] = int((e < noise).sum())
    return out


# ---- phase 1: float64 calibration + V refit ---------------------------------------------------
def phase_calibrate(cfg, args, out_dir: Path, refs: list[Path], new_dirs: list[Path]) -> None:
    from kvtransfer import Mapper, calibrate, load_pair
    from kvtransfer.calibration import extract_content_kv
    from kvtransfer.data import batches, iter_dataset, token_sequences
    from kvtransfer.hf import model_spec
    from kvtransfer.rope import RopeCodec
    from safetensors.torch import save_file

    mcfg, cal = cfg["mapper"], cfg["mapper"]["calibration"]
    f32_dir = Path(mcfg["stats_path"])
    saved = read_small(f32_dir, "V")
    f32_meta = json.loads((f32_dir / "meta.json").read_text())
    for key, want in (("n_seqs", cal["num_sequences"]), ("seq_len", cal["max_seq_len"]), ("stride", cal["stride"])):
        if f32_meta[key] != want:
            raise SystemExit(f"float32 stats {key}={f32_meta[key]} != config {want}; not comparable")

    ram = free_ram_gb()
    log(f"free RAM {ram:.1f} GB (need >= {args.min_free_gb:.0f}; float64 V moments are ~{F64_GB} GB)")
    if ram < args.min_free_gb:
        raise SystemExit("not enough free RAM: close browsers/other heavy apps and retry")

    src, tgt, tok = load_pair(cfg["sharer"]["model_id"], cfg["receiver"]["model_id"])
    seqs = token_sequences(iter_dataset(cal["dataset"]), tok, seq_len=cal["max_seq_len"],
                           n_seqs=cal["num_sequences"])

    # Pre-check (seconds, before the long loop): kvtransfer sets shift_x/shift_y to the first batch's
    # feature means, so recomputing them from batch 1 tells us the data stream is the same one.
    with torch.no_grad():
        ids = next(batches(seqs, cal["batch_size"])).long()
        pos = torch.arange(ids.shape[1])
        so = src(input_ids=ids.to(src.device), use_cache=True)
        to = tgt(input_ids=ids.to(tgt.device), use_cache=True)
        x = extract_content_kv(src, RopeCodec.from_model(src), so.past_key_values,
                               model_spec(src).n_layers, pos.to(src.device), "V", cal["stride"])
        y = extract_content_kv(tgt, RopeCodec.from_model(tgt), to.past_key_values,
                               model_spec(tgt).n_layers, pos.to(tgt.device), "V", cal["stride"])
        dsx = rel_maxdiff(x.double().mean(0).cpu(), saved["shift_x"])
        dsy = rel_maxdiff(y.double().mean(0).cpu(), saved["shift_y"])
        del so, to, x, y
    log(f"pre-check batch 1: shift_x rel diff {dsx:.2e}, shift_y rel diff {dsy:.2e} (tol {args.tol:.0e})")
    if max(dsx, dsy) > args.tol:
        raise SystemExit("batch 1 does not match the float32 calibration data; stopping before the long run")

    t0 = time.time()
    log(f"calibrating V in float64 on {len(seqs)} x {cal['max_seq_len']} (stride {cal['stride']}, "
        f"batch {cal['batch_size']}, stats on cpu)")
    stats = calibrate(src, tgt, batches(seqs, cal["batch_size"]), stride=cal["stride"], kinds=("V",),
                      stats_device="cpu", stats_dtype=torch.float64,
                      source_name=cfg["sharer"]["model_id"], target_name=cfg["receiver"]["model_id"],
                      progress=True)
    log(f"calibration done in {(time.time() - t0) / 60:.1f} min")
    del src, tgt
    torch.cuda.empty_cache()

    acc = stats.acc["V"]
    acc.centered = types.MethodType(centered_lowmem, acc)
    # Post-check: same token count and the same per-feature means (sum/n) as the float32 run.
    if acc.n != int(saved["n"]):
        raise SystemExit(f"token count {acc.n} != float32 run {int(saved['n'])}")
    dmx = rel_maxdiff(acc.sum_x / acc.n + acc.shift_x, saved["sum_x"] / acc.n + saved["shift_x"].double())
    dmy = rel_maxdiff(acc.sum_y / acc.n + acc.shift_y, saved["sum_y"] / acc.n + saved["shift_y"].double())
    log(f"post-check: n={acc.n} matches; mean rel diff x {dmx:.2e}, y {dmy:.2e}")
    if max(dmx, dmy) > args.tol:
        raise SystemExit("feature means differ from the float32 run beyond tolerance; not comparable")
    if args.save_stats:
        stats.save(out_dir / "stats_f64")
        log(f"saved float64 stats to {out_dir / 'stats_f64'}")

    scale = mean_centered_diag(acc)
    hd = stats.target.head_dim
    ref_mappers = [Mapper.load(r) for r in refs]
    sel0 = ref_mappers[0].selected
    for m in ref_mappers[1:]:
        if not np.array_equal(m.selected, sel0):
            raise SystemExit("reference mappers use different layer selections; compare them separately")

    # float64 centered Gram blocks for the collapse layers, for the comparison phase
    blocks = {}
    for lt in args.layers:
        gxx = acc.centered(rows=stats.src_rows(sel0[lt].tolist()))[0]
        blocks[f"gxx.{lt}"] = gxx.contiguous()
    save_file(blocks, str(out_dir / "blocks_f64.safetensors"))
    del blocks
    log(f"saved float64 centered Gram blocks for layers {args.layers}")

    fits = {}
    for ref_dir, ref, new_dir in zip(refs, ref_mappers, new_dirs):
        lam = float(ref.lam)
        log(f"fitting V in float64 for {ref_dir.name}: lambda={lam:.4g} (r_V={lam / scale:.3g})")
        W_V, b_V, r2v = [], [], []
        for lt in range(stats.target.n_layers):
            w, b, _ = acc.solve(lam, rows=stats.src_rows(ref.selected[lt].tolist()), cols=stats.tgt_cols(lt))
            W_V.append(w.float().cpu())
            b_V.append(b.float().cpu())
            r2v.append(float(np.nanmean(acc.block_r2(*acc.last_column_residuals, block=hd))))
        meta = dict(ref.meta)
        meta["precision_check"] = {"V_stats_dtype": "float64", "K_from": str(ref_dir), "lam": lam,
                                   "r_V": lam / scale}
        new = Mapper(ref.source, ref.target, ref.selected, lam, W_K=ref.W_K, b_K=ref.b_K, W_V=W_V, b_V=b_V,
                     fit_r2={"K": ref.fit_r2.get("K", []), "V": r2v}, meta=meta, key_space=ref.key_space)
        new.save(new_dir)
        fits[ref_dir.name] = {
            "lam": lam, "r_V": lam / scale, "mapper_f64v": str(new_dir),
            "in_sample_r2_V": {"f32": ref.fit_r2.get("V", []), "f64": r2v},
            "W_V_norm": {"f32": [float(w.norm()) for w in ref.W_V], "f64": [float(w.norm()) for w in W_V]},
            "W_V_absmax": {"f32": [float(w.abs().max()) for w in ref.W_V],
                           "f64": [float(w.abs().max()) for w in W_V]},
        }
        for lt in args.layers:
            f = fits[ref_dir.name]
            log(f"  layer {lt}: in-sample V R2 f32 {f['in_sample_r2_V']['f32'][lt]:.3f} -> "
                f"f64 {r2v[lt]:.3f}; ||W_V|| {f['W_V_norm']['f32'][lt]:.1f} -> {f['W_V_norm']['f64'][lt]:.1f}; "
                f"max|w| {f['W_V_absmax']['f32'][lt]:.0f} -> {f['W_V_absmax']['f64'][lt]:.0f}")
        log(f"  saved {new_dir}")

    summary = {"n_tokens": acc.n, "mean_centered_diag_V_f64": scale, "layers": args.layers,
               "selected": sel0.tolist(), "precheck": {"shift_x": dsx, "shift_y": dsy},
               "postcheck": {"mean_x": dmx, "mean_y": dmy}, "fits": fits}
    (out_dir / "calibrate.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {out_dir / 'calibrate.json'}")


# ---- phase 2: float32 vs float64 Gram blocks ---------------------------------------------------
def phase_compare(cfg, args, out_dir: Path) -> None:
    from kvtransfer import CalibrationStats
    from safetensors.torch import load_file

    cal_sum = json.loads((out_dir / "calibrate.json").read_text())
    lams = sorted({f["lam"] for f in cal_sum["fits"].values()})
    sel = cal_sum["selected"]

    log("loading float32 V stats (~6.6 GB)")
    stats = CalibrationStats.load(cfg["mapper"]["stats_path"], device="cpu", kinds=["V"])
    acc = stats.acc["V"]
    acc.centered = types.MethodType(centered_lowmem, acc)
    scale32 = mean_centered_diag(acc)
    blocks = load_file(str(out_dir / "blocks_f64.safetensors"))

    rows_out = {}
    for lt in cal_sum["layers"]:
        g64 = blocks[f"gxx.{lt}"]
        # what the stock float32 pipeline hands the float64 solver
        g32 = acc.centered(rows=stats.src_rows(sel[lt]))[0].double()
        d = g32 - g64
        noise = spectral_norm_sym(d)
        row = {"rel_fro_err": float(d.norm() / g64.norm()),
               "max_abs_err_over_mean_diag": float(d.abs().max()) / scale32,
               "f32_noise_spectral": noise, "f32_noise_over_mean_diag": noise / scale32,
               "noise_over_lam": {f"{lam:.4g}": noise / lam for lam in lams}}
        if not args.no_eig:
            row["f64"] = {f"{lam:.4g}": spectrum(g64, lam, noise) for lam in lams[:1]}
            row["f32"] = {f"{lam:.4g}": spectrum(g32, lam) for lam in lams[:1]}
        rows_out[lt] = row
        msg = (f"layer {lt}: ||G32-G64||_F/||G64||_F {row['rel_fro_err']:.2e}; f32 noise (spectral) "
               f"{noise:.3g} = {noise / lams[0]:.3g} x lambda {lams[0]:.3g}")
        if not args.no_eig:
            s64, s32 = row["f64"][f"{lams[0]:.4g}"], row["f32"][f"{lams[0]:.4g}"]
            msg += (f"; eig_min f64 {s64['eig_min']:.3g} / f32 {s32['eig_min']:.3g}; "
                    f"cond f64 {s64['cond_raw']:.3g} / f32 {s32['cond_raw']:.3g}; "
                    f"f64 eigs below f32 noise: {s64['n_below_f32_noise']}/{g64.shape[0]}")
        log(msg)
        del g32, d

    (out_dir / "compare.json").write_text(json.dumps(
        {"mean_centered_diag_V_f32": scale32, "lams": lams, "layers": rows_out}, indent=2))
    log(f"wrote {out_dir / 'compare.json'}")


# ---- phase 3: held-out diagnostics -------------------------------------------------------------
def phase_diagnose(args, new_dirs: list[Path]) -> None:
    for d in new_dirs:
        log(f"diagnose {d}")
        cmd = [sys.executable, str(REPO / "scripts" / "diagnose.py"), "--config", args.config,
               "--mapper", str(d), "--n-seqs", str(args.n_seqs)]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            raise SystemExit(f"diagnose failed for {d} (exit {r.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--refs", default="k8,k8_lam1.5e+03",
                    help="reference float32 mapper dir names, siblings of mapper.cache_path")
    ap.add_argument("--layers", default="11,12,13,14,15", help="target layers for the Gram comparison")
    ap.add_argument("--phases", default="calibrate,compare,diagnose")
    ap.add_argument("--min-free-gb", type=float, default=16.0)
    ap.add_argument("--tol", type=float, default=1e-4, help="max relative diff for the data-identity checks")
    ap.add_argument("--n-seqs", type=int, default=32, help="held-out passages for diagnose.py")
    ap.add_argument("--save-stats", action="store_true", help="also save the full float64 V stats (~13 GB)")
    ap.add_argument("--no-eig", action="store_true", help="skip the eigendecompositions in compare")
    ap.add_argument("--force", action="store_true", help="rerun phases whose outputs exist")
    args = ap.parse_args()
    args.layers = [int(x) for x in args.layers.split(",")]
    phases = args.phases.split(",")

    cfg = load_config(args.config)
    pair = cfg["pair_id"]
    ref_parent = Path(cfg["mapper"]["cache_path"]).parent
    refs = [ref_parent / r for r in args.refs.split(",")]
    for r in refs:
        if not (r / "mapper.json").exists():
            raise SystemExit(f"reference mapper not found: {r}")
    new_dirs = [ref_parent.parent / f"{pair}-f64v" / r.name for r in refs]
    out_dir = Path("results/precision_check") / pair
    out_dir.mkdir(parents=True, exist_ok=True)

    if "calibrate" in phases:
        if (out_dir / "calibrate.json").exists() and not args.force:
            log("calibrate: outputs exist, skipping (--force to rerun)")
        else:
            phase_calibrate(cfg, args, out_dir, refs, new_dirs)
    if "compare" in phases:
        if (out_dir / "compare.json").exists() and not args.force:
            log("compare: outputs exist, skipping (--force to rerun)")
        else:
            phase_compare(cfg, args, out_dir)
    if "diagnose" in phases:
        phase_diagnose(args, new_dirs)
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
