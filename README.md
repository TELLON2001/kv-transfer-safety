# Does Safety Survive Cross-Model KV Cache Transfer?

A safety audit of cross-model KV-cache transfer (a.k.a. cache-to-cache / prefill
reuse). When a **receiver** model skips prefill and decodes from a KV cache
**mapped over from a sibling model**, does it still refuse harmful requests at the
same rate?

**Hypothesis.** Refusal degrades faster than general capability under the
mapping, so a model pair that looks safe on accuracy numbers can quietly lose
safety behavior. Either outcome is publishable: a gap is a warning; no gap is
evidence the optimization is safe.

Full plan: [`docs/research-plan.md`](docs/research-plan.md)

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

## Experimental conditions

All four conditions run on the **same receiver model** and the **same prompts**;
the only thing that changes is the KV cache the receiver decodes from. Chat
template, decoding settings (seed, greedy-vs-sampling), and thinking mode are held
fixed across conditions, so the cache source is the sole variable.

- **Native:** receiver prefills the prompt itself. Baseline refusal and
  capability; every "retention" number is measured relative to this.
- **Mapped:** receiver decodes from the sharer's KV cache, mapped into the
  receiver's space. This is the deployed optimization and the effect under test.
- **Sharer:** sharer decodes from its own prefill. Isolates whether a safety drop
  originates in the sharer's cache rather than in the mapping: if the sharer itself
  fails to refuse, a Mapped failure isn't the mapper's doing.
- **Mismatched:** receiver decodes from a mapped cache built from a *different*
  prompt. Separates genuine information transfer from generic disruption: if Mapped
  and Mismatched behave alike, the receiver isn't actually reading the sharer's
  content. (Control design follows *When Does Latent Communication Pay?*,
  arXiv:2608.04893.)

Two comparisons carry the result:

- **Mapped vs. Native:** does the optimization change refusal at all.
- **Mapped vs. Mismatched:** is any change real transfer, or just cache corruption.

Interpretation depends on the **token layout** (which tokens the sharer prefills,
which the receiver decodes, and where the harmful instruction and chat-template
tokens sit in the shared cache), so it is recorded per model pair in `configs/`.

## Output labels

Every generation is scored as **refuse / comply / incoherent**. The third label
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

Phase 1: setup and overlap check. See [`research-log.md`](research-log.md).

## Provenance / ethics

- Harmful prompts come from public benchmarks (HarmBench, JailbreakBench). This
  audits an existing published optimization; it does not introduce a new attack.
- Findings about a specific published method will be shared with those authors
  before any public write-up.
