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

Develop locally, run heavy jobs on a rented 24 GB box. Keep everything on **personal
accounts** (the IP clause).

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

## Practical notes

- Put the HF cache and datasets on the cloud box's local disk, not the repo
  (`.gitignore` already excludes `data/`, `results/`, `hf_cache/`).
- Pin the `kvtransfer` commit and, once reproduction works, `pip freeze >
  requirements.lock.txt` on the machine that produced the number.
- A 24 GB card runs the whole 1.7B -> 4B reproduction; larger within-family pairs in
  Phase 4 may want 40 to 48 GB (A6000 / A100 40 GB).
