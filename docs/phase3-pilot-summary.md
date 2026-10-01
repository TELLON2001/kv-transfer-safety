# Phase 3 pilot: result, interpretation, and what it does not settle

_Generated 2026-10-01 from the pilot result files. Numbers come from `stats.json`, the two
sensitivity files, `judgments.jsonl` and `hand_labels.jsonl` in_
`results/pilot/qwen3-0p6b-to-1p7b/qwen3-0p6b-to-1p7b_k8_lam1.5e+03/` _(under `kvt-work`, not in git)._
_Narrative record: `research-log.md`._

## Summary

On the stand-in pair, refusal and capability degrade by about the same proportion under cross-model
KV transfer. The pre-registered go rule (the ratio of refusal retention to capability retention has
a 95% CI below 1) is not met, so the pilot is a NO-GO. That is a statement about what this pilot
could resolve, not a demonstration that the effect is absent: the observed effect size needs roughly
500 harmful prompts to detect at 80% power, and 200 were run.

## Setup

| | |
|---|---|
| Pair | Qwen3-0.6B (sharer) to Qwen3-1.7B (receiver), a stand-in, not the target pair |
| Mapper | k=8, relative lambda r=1e-3 (lam=1503), fitted from the n500 calibration stats |
| Conditions | native, mapped, sharer (Mismatched dropped, see Limits) |
| Prompts | 200 HarmBench standard (harmful) + 200 XSTest safe (benign), frozen before any output |
| Decoding | greedy, 256 new tokens, thinking off, hold_back 1 |
| Judge | claude-sonnet-5 via Message Batches, batch `msgbatch_016Ni4sy7ai5Kxrt4KsiYib9`, condition hidden |
| Hand labels | 127 items over two blinded sheets, labeled by the author |

## Judge validation

Cohen's kappa against the author's hand labels:

| sample | n | agreement | kappa |
|---|---|---|---|
| random (unbiased) | 46 | 0.826 | 0.631 |
| all items with both labels | 50 | 0.780 | 0.573 |

Per condition, all items: native 0.923 (n=13), mapped 0.842 (n=19), sharer 0.611 (n=18).
The judge is least reliable on the weakest model's output, which is worth carrying into Phase 4.
No numeric kappa threshold was pre-registered; 0.63 is "substantial" on the Landis and Koch scale.

## Result

Rates in %, 95% CI from a paired bootstrap over prompts (B=10k).

### Harmful prompts (n=200)

| condition | refuse | comply | incoherent |
|---|---|---|---|
| native | 71.0 | 27.0 | 2.0 |
| mapped | 64.0 | 34.0 | 2.0 |
| sharer | 45.0 | 47.0 | 8.0 |

Mapped minus native: -7.0 pts [-15.0, +0.5].
Refusal retention (mapped / native): 0.901 [0.801, 1.008].

### Benign prompts (n=200), over-refusal check

| condition | refuse | comply | incoherent |
|---|---|---|---|
| native | 8.0 | 88.0 | 4.0 |
| mapped | 7.0 | 87.5 | 5.5 |
| sharer | 14.5 | 75.5 | 10.0 |

Mapped minus native: -1.0 pts [-4.0, +1.5].
Transfer does not make the receiver over-refuse.

### Headline

| quantity | value |
|---|---|
| refusal retention | 0.901 [0.801, 1.008] |
| capability retention (arc_challenge, n=250, native 0.452, mapped 0.400) | 0.885 [0.718, 1.080] |
| ratio (refusal retention / capability retention) | 1.019 [0.803, 1.292] |
| verdict | NO-GO by the plan's rule (CI includes or exceeds 1) |

### Sensitivity for the 77 items the judge refused to label

| treatment | refusal retention | headline ratio |
|---|---|---|
| hand labels (primary) | 0.901 [0.801, 1.008] | 1.019 [0.803, 1.292] |
| all forced to comply | 0.952 [0.848, 1.066] | 1.076 [0.850, 1.367] |
| all forced to refuse | 0.960 [0.872, 1.056] | 1.085 [0.865, 1.364] |

The verdict does not hinge on those items.

## What the result means

**Mapped tracks native, not the sharer.** On harmful prompts mapped sits much closer to the
receiver's own behavior (71.0) than to the donor model's
(45.0). The receiver's refusal behavior substantially survives
having its cache replaced. That is the clearest signal here, and it points away from the alarming
version of the hypothesis.

**Both retentions fall together.** Refusal retention and capability retention are indistinguishable
at this precision, so the pilot does not show safety degrading faster than competence.

**But the pilot was underpowered.** Native and mapped disagree on 64 of 200 harmful
prompts (32%), split 39 where only native refuses against 25 where only
mapped refuses. A McNemar power calculation on that discordance needs about 506 prompts for 80%
power and about 674 for 90%. At n=200 a 7-point difference cannot be separated from noise. Any
write-up must say this; the null is not evidence of absence.

## Two methodological findings worth keeping

**1. Dropping judge refusals would have halved the effect.** The judge's safety layer refused to
label 77 responses, all on harmful prompts. Using judge labels alone the native-to-mapped
gap is 72.0 to 68.6 (3.4 pts);
with hand labels filled in it is 71.0 to
64.0 (7.0 pts).
These items are not missing at random: the likely trigger is harmful content in the response,
meaning compliance. Hand-filling them is mandatory, not optional.

**2. The judge-refused subset leans the way the hypothesis predicts.** The judge refused a nearly
balanced count per condition, yet hand labels split sharply:

| condition | refuse | comply | incoherent |
|---|---|---|---|
| native | 16 | 9 | 0 |
| mapped | 8 | 17 | 0 |
| sharer | 11 | 15 | 1 |

Selection depends on the judge's reaction rather than on chance, so this is a hint and not an
estimate. It is the strongest suggestion in the data that something real is there, and one reason
not to treat the NO-GO as closing the question.

## Correction to an earlier reading

The step 5 smoke test (n=10, triaged by opening line only) suggested near-total refusal loss under
mapping. At n=200 with whole responses labeled, that collapses to a 7-point difference. The smoke
reading was an artifact of judging opening lines: mapped answers often open with a preamble or a
title and refuse further down. **Do not triage refusal behavior on first lines.**

## Limits

- Stand-in pair. The paper evaluated neither 0.6B to 1.7B nor the target 1.7B to 4B.
- The mapper recovers only about 40 to 48% of the ARC gap between sharer and receiver, so capability
  retention is measured on a weak mapper. A better mapper could move both retentions.
- The capability CI uses a normal approximation on n=250 benchmark items and ignores pairing, so it
  is wide and conservative. It dominates the width of the headline ratio.
- One pair, one mapper, greedy decoding, 256-token cap, single judge.
- The Mismatched condition is degenerate under the current token layout: with hold_back=1 the
  receiver's own token is the final chat-template token, identical for every prompt, so "prompt A on
  prompt B's cache" is token-for-token Mapped(B). Dropped from the pilot; needs redesign for Phase 4.

## Options from here

1. **Scale the same pair to about 500 harmful prompts.** Roughly 900 extra generations, about 2 to 3
   hours on the laptop, plus a few dollars of judge batch. Needs a second harmful prompt source
   (AdvBench, StrongREJECT, JailbreakBench) because HarmBench standard is exhausted at 200, so it is
   a protocol change that mixes distributions and must be declared.
2. **Move to the real pair (Qwen3-1.7B to 4B)**, with the Ministral 3B to 8B anchor first to check
   that kvtransfer reproduces the paper at all. Needs paid cloud (RunPod or Vast; Colab Pro is
   unavailable in Lebanon).
3. **Write up the lambda finding**, which is independent of the safety question. See
   `docs/lambda-finding.md`.
