# GPU setup

## What we have locally

RTX 5070 Laptop GPU, **8 GB** VRAM (driver 582.05). Two constraints follow:

1. **8 GB is not enough for the reproduction pair in bf16.** Qwen3-4B weights alone
   are about 8 GB, before the 1.7B sharer, both KV caches, and calibration
   activations. The plan's 24 GB target stands for the actual runs.
2. **Blackwell (sm_120) needs a very recent stack.** Stable CUDA wheels may not cover
   it yet; expect to need CUDA 12.8+ and possibly a torch nightly. Budget setup time.

So the laptop GPU is for **editing, plumbing, and tiny smoke tests** (a single small
model, short sequences, optionally 4-bit), not for reproduction or the study.

## Recommended: hybrid local + rented cloud GPU

Develop locally, run heavy jobs on a rented 24 GB box.

- **Dev loop:** edit on Windows, run smoke tests under **WSL2** (the ML stack, and
  likely kvtransfer, is Linux-first). `git push`, then `git pull` on the cloud box and
  run there.
- **Reproduction and study (24 GB):**
  - **Runpod / Vast.ai** RTX 3090 / 4090 / A5000, roughly $0.20 to $0.50 per hour.
    Cheapest and simplest for on-demand runs.
  - **Lambda / cloud VM spot instances** if you want a longer-lived box.
- **Free tiers to start:**
  - **Kaggle notebooks:** T4 16 GB, ~30 GPU-hours/week. Enough for the small pair.
  - **Colab** (free or Pro): easy, but ephemeral storage; push results to git/remote.

## Google Colab

Enough for reproduction and the pilot on the right tier; the free tier is marginal.

| Tier | GPU | Verdict |
| --- | --- | --- |
| Free | T4 16 GB | Marginal. Fits the pair only tightly, and T4 has no native bf16, so an fp16 run can shift the very number the week-4 gate checks. Fine for smoke tests and plumbing. |
| Pro | L4 24 GB | Recommended. Native bf16, comfortable for reproduction and the pilot. |
| Pro+ | A100 40 GB | Headroom for the larger within-family pairs in Phase 4. |

Two things to get right on Colab:

- **Use bf16-capable hardware (L4 or A100) for the actual reproduction.** The paper's
  numbers are bf16; the free T4 runs fp16 and can cost a few points on the comparison
  that decides the go/no-go. Keep the T4 for smoke tests.
- **Checkpoint to Google Drive.** Sessions disconnect (idle timeout, ~12 h cap,
  preemption), so persist state or you re-run calibration every reconnect:
  - HF cache: set `HF_HOME` to a Drive path
  - mapper: point `mapper.cache_path` at Drive (or copy the fitted mapper there)
  - results: write `results/` to Drive

Colab GPUs are datacenter cards (T4 / L4 / A100), so the Blackwell setup issue from the
local card does not apply, and it is Linux, so `kvtransfer` runs natively. This gets
painful only for the full Phase 4 matrix, where a persistent rented box is nicer.

See `notebooks/reproduce_colab.ipynb` for a first end-to-end smoke run (clone, install,
eligibility check).

## Practical notes

- Put the HF cache and datasets on the cloud box's local disk, not the repo
  (`.gitignore` already excludes `data/`, `results/`, `hf_cache/`).
- Pin the `kvtransfer` commit and, once reproduction works, `pip freeze >
  requirements.lock.txt` on the machine that produced the number.
- A 24 GB card runs the whole 1.7B -> 4B reproduction; larger within-family pairs in
  Phase 4 may want 40 to 48 GB (A6000 / A100 40 GB).
