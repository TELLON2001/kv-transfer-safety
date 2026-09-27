# Does Safety Survive Cross-Model KV Cache Transfer?

A safety audit of cross-model KV-cache transfer (a.k.a. cache-to-cache / prefill
reuse). When a **receiver** model skips prefill and decodes from a KV cache
**mapped over from a sibling model**, does it still refuse harmful requests at the
same rate?

**Hypothesis.** Refusal degrades faster than general capability under the
mapping, so a model pair that looks safe on accuracy numbers can quietly lose
safety behavior. Either outcome is publishable: a gap is a warning; no gap is
evidence the optimization is safe.

Full plan: [`docs/research-plan.md`](docs/research-plan.md) (mirror of the living
doc in Claude).

## Core comparison

Refusal rate with **native prefill** vs. with a **mapped cache**, on the same
receiver and the same prompts.

| Condition | Receiver decodes from | Purpose |
| --- | --- | --- |
| Native | Its own prefill | Baseline refusal + capability |
| Mapped | Sharer cache, mapped | The effect being measured |
| Sharer | Sharer's own prefill | Is the sharer the weaker link? |
| Mismatched | Mapped cache from a *different* prompt | Separate real transfer from generic disruption |

**Headline metric:** refusal retention ÷ capability retention (both components
also reported separately, each with bootstrap CIs).

## Output labels

Every generation is scored as **refuse / comply / incoherent** — the third label
keeps degraded-but-non-refusing output from being miscounted as compliance.

## Repo layout

```
configs/     experiment + model-pair configs
data/        prompt sets (not committed; see data/README.md)
notebooks/   exploration, figures
results/     metrics, run outputs (not committed)
scripts/     entry points (reproduce, pilot, full run)
src/         mapper, judge, metrics, harness
docs/        research plan + notes
```

## Status

Phase 1 — setup and overlap check. See [`research-log.md`](research-log.md).

## Provenance / ethics

- All work on personal hardware, accounts, and time (IP-clause check done).
- Harmful prompts come from public benchmarks (HarmBench, JailbreakBench). This
  audits an existing published optimization; it does not introduce a new attack.
- Findings about a specific published method will be shared with those authors
  before any public write-up.
