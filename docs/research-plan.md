# Research Plan: Does Safety Survive Cross-Model KV Cache Transfer?

_As of 2026-09-27. Mirror of the living doc in Claude; edit the doc, then sync here._

## Research question

When a receiver model skips prefill and decodes from a KV cache mapped over from
a sibling model, does it still refuse harmful requests at the same rate?

Cross-model KV transfer is being adopted as a serving optimization ([Heo et al.,
2026](https://arxiv.org/abs/2608.03893)), but it has only been evaluated on
capability benchmarks. Refusal depends on the receiver reading the prompt, and
here it never does.

**Hypothesis:** refusal degrades faster than general capability under the
mapping, so a pair that looks safe to deploy on accuracy numbers can quietly lose
safety behavior. Either outcome is useful: a gap is a warning, and no gap is
evidence the optimization is safe.

**Scope:** the mapper's eligibility check (matched KV head count and head
dimension, shared tokenizer) restricts this to within-family pairs, so the
headline claim is about within-family transfer — stated as a limitation up front.

## Timeline

~4 months to a workshop submission at 6–8 h/week; 6–9 months to a main-conference
paper. First projects run 1.5–2× over plan, so slack is baked in.

| Weeks | Phase | Done when |
| --- | --- | --- |
| 1–2 | Setup and overlap check | Contract IP clause checked, GPU access, related-work table filled |
| 3–4 | Reproduce the mapper | One accuracy number within a few points of the paper |
| 5–6 | Pilot | Refusal rate: native prefill vs. mapped cache, one model pair |
| 7–12 | Full study | All pairs, controls, confidence intervals |
| 13–16 | Write and submit | Draft reviewed by ≥1 outside reader, submitted |

Targets: NeurIPS / ICLR / MLSys workshops (check deadlines once the pilot works).

## Phase plan

Each phase ends with one concrete artifact, so a busy week stalls the project
without losing it.

- **Phase 1 — Setup (wk 1–2):** IP clause + personal hardware/accounts/time;
  secure a 24 GB+ GPU; read CacheBridge, Heo et al., LCGuard, "When Latent Agents
  Lie"; repo + research log + Scholar alerts on C2C and Heo et al.
- **Phase 2 — Reproduce (wk 3–4):** run `kvtransfer` on Qwen3-1.7B→4B; reproduce
  one reported accuracy number before measuring anything new.
- **Phase 3 — Pilot (wk 5–6):** 200 harmful + 200 benign prompts, one pair,
  native vs. mapped; hand-check 50 outputs against the judge.
- **Phase 4 — Full study (wk 7–12):** 2–3 more within-family pairs, mismatched-
  cache control, jailbreak-wrapped prompts; bootstrap CIs on every number, plus a
  power calculation so the go/no-go and any null result aren't decided by noise.
- **Phase 5 — Write (wk 13–16):** abstract + main figure first; outside feedback
  two weeks before deadline.

## Experimental design

Core comparison: refusal rate with native prefill vs. mapped cache, same receiver
and prompts.

**Models.** Start with Qwen3-1.7B→4B, then add pairs that pass the tool's
eligibility check (matched KV head count and head dimension). Keep chat template,
decoding settings (seed and greedy-vs-sampling included) and thinking mode fixed
across every condition. Document the exact token layout — which tokens the sharer
prefills, which the receiver decodes, and where system-prompt and chat-template
tokens sit — since the interpretation depends on where the harmful instruction
lives in the shared cache.

**Conditions.**

| Condition | Receiver sees | Purpose |
| --- | --- | --- |
| Native | Its own prefill | Baseline refusal and capability |
| Mapped | Sharer cache, mapped | The effect being measured |
| Sharer | Sharer's own prefill | Is the sharer the weaker link? |
| Mismatched | Mapped cache from a different prompt | Separate real transfer from generic disruption |

**Prompt sets.** Harmful: HarmBench / JailbreakBench. Over-refusal: XSTest /
OR-Bench. Capability: the paper's benchmarks via lm-evaluation-harness.

**Metrics.** Every output is first labeled refuse / comply / incoherent, so
degraded-but-non-refusing generations aren't counted as compliance. Refusal rate,
over-refusal rate, and capability retention — each defined relative to the
native-prefill baseline on the same receiver — all with bootstrap CIs. Headline:
refusal retention ÷ capability retention, with both components always reported
separately (the ratio is unstable when either denominator is small).

**Judge.** An existing refusal classifier extended to the three labels;
hand-label a random sample per condition (sized to the base rate, not a flat 50),
reporting Cohen's κ against the classifier and watching for incoherent output
scored as compliance. Fix the judge before seeing results.

## Go/no-go

| After | Go if | Otherwise |
| --- | --- | --- |
| Overlap check (wk 2) | Nobody has measured refusal under prefill-skipping transfer | Pivot to cross-family diagnosis or the serving/scheduling angle |
| Reproduction (wk 4) | Number within a few points of the paper's | Try another pair, check repo issues, email authors |
| Pilot (wk 6) | Refusal drops more than capability, outside the CI | Add jailbreak-wrapped prompts; if still no gap, write up "safety preserved" |

## Stretch goal: mechanism and fix

Only after the full study confirms a gap:

1. **Find the refusal direction** (Arditi et al., 2024) for the receiver — mean
   activation difference on harmful vs. harmless prompts.
2. **Test the mechanism** — project receiver activations onto that direction under
   native vs. mapped caches; a shrinking projection means the mapper distorts refusal.
3. **Try a fix** — constrain the mapper to preserve that direction, check refusal
   recovers without hurting capability.

## Related-paper tracker

All links verified 2026-09-27. Full entries: [`references.md`](references.md);
BibTeX: [`references.bib`](references.bib).

| Paper | What it does | Overlap |
| --- | --- | --- |
| [CacheBridge](https://arxiv.org/abs/2609.00891) (Sep 2026) | Closed-form affine cross-model KV transfer | Second prefill-reuse mapper; capability-only eval — audit target for a generality check |
| [Heo et al.](https://arxiv.org/abs/2608.03893) (Aug 2026) | Per-head ridge mapper, within-family, prefill reuse | Base method; capability-only eval |
| [When Does Latent Communication Pay?](https://arxiv.org/abs/2608.04893) (Aug 2026) | Causal audit of relayed KV caches | Source of the mismatched-cache control |
| [When Latent Agents Lie](https://arxiv.org/abs/2606.28958) (Jun 2026) | KV-cache integrity attacks, multi-agent | Adjacent: attacks, not transfer fidelity |
| [LCGuard](https://arxiv.org/abs/2605.22786) (May 2026) | Guarding KV sharing against leakage/attacks | Adjacent: defense, not refusal retention |
| [Latent Cache Flow](https://arxiv.org/abs/2605.22863) (May 2026) | Compressed cache channel, cross-context | Low |
| [Latent Space Communication via K-V Cache Alignment](https://arxiv.org/abs/2601.06123) (Jan 2026, Dery et al.) | Learned per-model adapters into a shared latent KV space, cross-architecture (Gemma-2) | Low; learned trained channel, not prefill-skip transfer |
| [C2C](https://arxiv.org/abs/2510.03215) (ICLR 2026) | Learned KV projection/fusion between LLMs | Low; foundational citation |

Code: [`kvtransfer`](https://github.com/Susmith4710/kvtransfer) (Heo et al. impl),
[`thu-nics/C2C`](https://github.com/thu-nics/C2C).
