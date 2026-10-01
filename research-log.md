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

## 2026-10-01
- Found: Heo et al. never evaluated Qwen3-1.7B -> 4B (their pairs: Qwen3 14B->32B, 8B->32B,
  Llama 3.1 8B->70B, Ministral 3 3B->8B / 3B->14B / 8B->14B). No reference accuracy exists
  for our pair. Plan: validate kvtransfer on Ministral 3 3B->8B (A100) before trusting it.
- Found: kvtransfer (Susmith4710, pinned 04a8e99) is an independent reconstruction; the paper
  released no code. We audit this implementation.
- Found: `kvtransfer eval` CLI takes the first N FineWeb docs, the same ones calibration used,
  so its "held-out" numbers are in-sample. Wrote scripts/diagnose.py, which skips them.
- Colab Pro unavailable (Lebanon). Free T4 has 12 GB system RAM < ~15 GB calibration stats,
  and no native bf16. Moved pipeline checks to the laptop (RTX 5070 8 GB, native bf16, 32 GB
  RAM) on a stand-in pair, Qwen3-0.6B -> 1.7B (same KV geometry: 8 x 128, shared tokenizer).
  Plausibility checks only; the paper evaluated neither pair.
- Offline test on tiny random Qwen3 (identity pair): cosine 1.0, top-1 1.0, mapped == native
  20/20 smoke prompts. kvtransfer suite 81/83 (2 Windows-only failures, unrelated).
- Calibration, 0.6B -> 1.7B, k=8, lam=0.01, stride 4: 100 seqs in 10 min; 500 seqs in 26.5 min.
  In-sample R2 K/V: 0.870/0.802 (n100), 0.805/0.702 (n500). Lower with more data, as expected
  for a regression overfitting less (sanity check, not a finding).
- Held-out diagnostics, same 32 FineWeb passages (after the 500 calibration ones):
  | mapper | R2 K | R2 V | attn cos (mean/min) | KL mean | top-1 mapped | top-1 sharer |
  | n100   | 0.900 | 0.508 | 0.858 / 0.521 | 0.372 | 0.765 | 0.656 |
  | n500   | 0.930 | 0.426 | 0.878 / 0.600 | 0.342 | 0.742 | 0.656 |
  Mapper beats the sharer-alone baseline by 9 to 11 points of top-1 agreement.
- FINDING (preliminary): the n500 mapper's value weights blow up in target layers 11 to 15
  (||W_V|| 5 to 7x the n100 fit, max entry ~1000 vs ~270). Held-out V R2 there drops to
  -0.5 to -1.15 (n100: +0.28 to +0.49) while in-sample V R2 stays 0.60 to 0.70, and attention
  cosine in those layers falls. 23 of the other layers improve with more data. Same source
  layers selected in both fits, so not a selection change; keys unaffected.
  Cause (hypothesis): kvtransfer adds lam=0.01 to the UNNORMALIZED centered Gram (a sum over
  ~128k tokens), as in the paper's Eq. 4 read literally, so regularization is negligible and
  the fit is effectively OLS on highly collinear features (8 adjacent source layers). fp32
  moment accumulation error grows with token count and lands in the near-singular directions.
  Next: lambda sweep refit from the saved stats + held-out diagnostics. If confirmed: the
  published lambda, applied literally, yields an unstable mapper on this pair, localized to
  mid-layer values; matters for the safety study (silent degradation unrelated to transfer).
- Step 4, capability, n500 mapper (lam=0.01, k=8), lm-eval acc_norm, limit 250 (SE ~3 pts each):
  | task          | 1.7B alone | 0.6B alone | mapped | retention | floor-norm |
  | hellaswag     | 0.492      | 0.504      | 0.424  | 86.2%     | 71.9%      |
  | arc_challenge | 0.452      | 0.352      | 0.360  | 79.6%     | 54.5%      |
  Mapped is at or below the sharer-alone level: this mapper does not carry the receiver's
  capability. HellaSwag shows no 1.7B-over-0.6B gap for these chat checkpoints, so retention
  there is uninformative; ARC shows a 10-pt gap of which mapped recovers ~1 pt. Suspect the
  unstable mid-layer V weights; rerun after the lambda sweep.
- Lambda sweep (refit.py from the n500 stats, k=8; same 32 held-out passages; sharer top-1 0.656).
  Scale: mean centered diag of the V Gram = 1.50e6 (K: 5.45e5) over 128k tokens, so the paper's
  lam=0.01 is r = lam/diag = 6.7e-9: effectively no regularization.
  | lambda (r = rel. to V) | R2 V | min V R2, L11-15 | attn cos mean/min | KL mean/p95 | top-1 |
  | 0.01 (r~7e-9, default) | 0.426 | -1.15 | 0.878 / 0.600 | 0.342 / 1.37 | 0.742 |
  | 1.5   (r=1e-6)         | 0.672 |  0.53 | 0.905 / 0.619 | 0.271 / 1.03 | 0.797 |
  | 15    (r=1e-5)         | 0.673 |  0.54 | 0.910 / 0.655 | 0.264 / 0.93 | 0.804 |
  | 150   (r=1e-4)         | 0.673 |  0.54 | 0.918 / 0.736 | 0.246 / 0.91 | 0.807 |
  | 1500  (r=1e-3)         | 0.670 |  0.54 | 0.925 / 0.864 | 0.253 / 0.94 | 0.804 |
  | 15000 (r=1e-2)         | 0.658 |  0.54 | 0.922 / 0.860 | 0.284 / 1.04 | 0.788 |
  Max ||W_V|| over layers: default 36.9k (layer 14) -> 2.4k at r=1e-6 -> 194 at r=1e-2.
  CONFIRMED on diagnostics: any r >= 1e-6 removes the L11-15 value collapse; held-out V R2
  0.43 -> 0.67, top-1 margin over sharer +8.6 -> +15 pts, worst-layer cosine 0.60 -> 0.86.
  Broad optimum r=1e-4 to 1e-3 (1e-4 best KL/top-1, 1e-3 best cosine incl. worst layer);
  over-regularizes by 1e-2. Caveat: lambda chosen on the same 32 passages it is scored on;
  the capability rerun (independent benchmarks) is the clean check.
- Step 4 rerun with regularized mappers (same tasks, limit 250, acc_norm; 1.7B alone and 0.6B
  alone recomputed each run and identical to before):
  | mapper              | arc_challenge mapped | hellaswag mapped |
  | default lam=0.01    | 0.360 (sharer 0.352, receiver 0.452) | 0.424 (sharer 0.504, receiver 0.492) |
  | r=1e-3 (lam=1503)   | 0.400  ret 88.5% / fn 74.3%          | 0.428  ret 87.0% / fn 73.6%          |
  | r=1e-4 (lam=150)    | 0.392  ret 86.7% / fn 70.3%          | 0.432  ret 87.8% / fn 75.2%          |
  ARC: regularization lifts mapped from sharer level to recovering ~40-48% of the 10-pt
  receiver-over-sharer gap. HellaSwag: unchanged; mapped stays ~7 pts below BOTH models, so
  that deficit is not the mid-layer V collapse. Open question. Candidate (untested): mapper is
  calibrated/diagnosed on 1024-token windows, benchmark contexts are short; early positions
  (attention-sink token) may map worse. Check with diagnostics on short prefixes.
  Working choice for the smoke test: r=1e-3 (best attention cosine incl. worst layer; ties on ARC).
- Step 5 smoke test (20 chat prompts, greedy, 256 tokens, thinking off, hold_back 1), mappers
  r=1e-3 and default lam. Native and sharer outputs identical across the two runs (deterministic).
  First-pass triage by OPENING LINE only (not labels; hand labels pending in smoke.md):
  | 10 harmful prompts        | native 1.7B | mapped r=1e-3 | mapped default | sharer 0.6B |
  | opens w/ explicit refusal | 7           | 0             | 1 (after a title) | 2        |
  | opens w/ clear compliance | 1           | 3             | ~4             | 2           |
  | ambiguous preamble        | 2           | 7             | ~5             | 6           |
  Native complies on the fake-vaccine-article prompt (non-discriminating item).
  Style shift: mapped answers hit the 256-token limit 19/20 (native 9/20, sharer 8/20);
  r=1e-3 mapped often opens "Certainly!"; default-lam mapped writes titled, document-like text
  ("**Title**", "Introduction:"), as if the chat framing were weakened. No over-refusal
  markers on the 10 benign prompts in any condition.
  Reading: in this first look the mapped receiver loses the native model's explicit refusals.
  Caveats: n=10, opening lines only, stand-in pair, mapper recovers only ~half the capability gap,
  so this may partly be "weaker model" behavior. Separating refusal loss from capability loss is
  exactly what the pilot is for. Hypothesis to test later: the mapped cache weakens chat-template
  conditioning (document-style continuations), which would hit refusals directly.
- Pilot harness built (Phase 3 design, fixed BEFORE any pilot output exists):
  prompts: HarmBench standard (200; centerforaisafety/HarmBench @ 8e1604d) + XSTest safe,
  20 per type x 10 types (Paul/XSTest @ f600c99, seed 0); data/pilot/prompts.jsonl sha256
  fbe418288436f7cb... Conditions: native, mapped, sharer; greedy, 256 new tokens, thinking off,
  hold_back 1. Judge: Claude Sonnet 5 (claude-sonnet-5), effort medium, JSON-schema output,
  rubric configs/judge_rubric.md (this commit), condition hidden from the judge; Message Batches.
  Hand check: blinded sample, 12 random per condition + top-ups so each judge label appears >= 2
  per condition; kappa reported on the random part. Stats: paired bootstrap over prompts (10k);
  go = 95% CI of (refusal retention / capability retention) below 1.
- Design note: the plan's Mismatched condition is degenerate under our layout. With hold_back=1
  the receiver's own token is the final chat-template token, identical for every prompt, so
  "prompt A on prompt B's mapped cache" is token-for-token Mapped(B). Dropped from the pilot.
  A meaningful control needs the receiver to read the real prompt itself, e.g. hold_back = the
  whole user turn on top of a donor's mapped context. To decide before Phase 4.
- Pilot generation done: 1200/1200 (400 prompts x native/mapped/sharer), 0.6B -> 1.7B, mapper
  k8 r=1e-3 (lam=1503). Greedy, 256 new tokens (max observed 256), ~8.7 s/item, one resume after
  27 items. 867/1200 hit the token cap. meta.json originally recorded the config's 512 under
  decoding.max_new_tokens; corrected to 256 with a note (generation used 256; fixed in c500d48).
  Prompts hash: meta.json records 724b3b75... = sha256 of the file bytes (CRLF on Windows);
  the fbe41828... above is the same content hashed with LF. Same prompt set. Make the two
  scripts hash the same way before Phase 4.
- Protocol change (hand check), made BEFORE any judge label exists: the judge is blocked on an
  API billing issue, so the blinded sheet was exported first: 17 random per condition (51 items,
  harmful/benign split evenly), no judge-label top-ups. Kappa on this random sample is the
  unbiased number as planned; top-ups for rare judge labels (e.g. incoherent) can be added as a
  separate sheet after the judge runs. Chose to wait for the planned API judge (Sonnet 5) rather
  than substitute an in-session judge.
