# Research log

One line per run: what I tried, why, what happened. This becomes the methods
section. Newest entries at the bottom of each day.

## 2026-09-27
- Repo created. Structure scaffolded; plan mirrored to `docs/research-plan.md`.
- Phase 1 open: IP-clause check done; related-work table filled; GPU access next.
- Phase 2 started on Colab. Eligibility check passed for Qwen3-1.7B -> 4B:
  num_key_value_heads 8==8, head_dim 128==128, shared tokenizer. Pair is mappable.
  Locked these into configs/qwen3-1p7b-to-4b.yaml as guards.
- Reminder (Phase 2): once the mapper reproduces one accuracy number, pin the exact
  environment with `pip freeze > requirements.lock.txt` and commit it, alongside the
  `kvtransfer` commit SHA; that lock file is the reproducible env, `requirements.txt`
  stays the loose spec.
