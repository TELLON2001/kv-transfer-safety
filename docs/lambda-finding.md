# The lambda finding: the published regularizer does nothing, and gets weaker with more data

_Written 2026-10-01. Status: confirmed on one pair (Qwen3-0.6B to Qwen3-1.7B), on held-out
diagnostics and on an independent benchmark. Not yet checked on a second pair._

_Implementation audited: `github.com/Susmith4710/kvtransfer`, pinned at `04a8e99`. This is an
independent reconstruction; Heo et al. (arXiv 2608.03893) released no code._

---

# Section 1: in plain terms

## The setup

Cross-model KV cache transfer lets a small "sharer" model read a prompt once, then hands its
internal cache to a bigger "receiver" model so the receiver can answer without re-reading the
prompt itself. The two models store that cache differently, so you need a translator between them.

That translator is fitted by linear regression: show both models the same text, record what each one
stores, and solve for the matrix that best converts one into the other. This is ridge regression,
and ridge regression has one knob, usually called lambda. Lambda decides how much the fit is allowed
to chase the training data versus staying conservative. Too low and the fit memorizes noise; too
high and it ignores the signal.

## What goes wrong

The paper specifies lambda = 0.01, and the implementation uses that as its default.

The problem is what lambda is being compared against. The regularizer is added to a quantity that is
a **running total over every calibration token**, not an average. With roughly 128,000 tokens of
calibration text, that total has a typical size of about 1,500,000. Adding 0.01 to 1,500,000 changes
essentially nothing. In relative terms the regularization strength is about 7 parts in a billion.

So the knob is set, but it is not connected to anything. The fit is effectively unregularized.

**And it gets worse the more data you use.** The total grows as you add calibration text, while
lambda stays fixed at 0.01. So doubling your calibration data halves the already negligible
regularization. This is backwards: adding data is supposed to make a fit more stable, and here it
makes the regularizer weaker.

## What that looks like in practice

We fitted the translator twice, once on 100 sequences and once on 500. The 500-sequence fit was
*worse* in a specific, localized way: in a band of middle layers, the part of the translator that
handles "value" vectors blew up. Its weights grew five to seven times larger, with individual
entries reaching about 1,000 instead of about 270.

The telltale sign of overfitting was there too. On the data it was fitted to, the 500-sequence fit
looked fine. On fresh text it was far worse than useless in those layers, scoring below what you
would get by ignoring the input entirely. Every other layer improved with more data, exactly as you
would expect. Only that middle band broke.

## The fix, and what it buys

Make lambda relative to the size of the thing it is added to, rather than an absolute number.

Any relative strength of about one part in a million or more removes the blow-up completely. The
useful range is broad, roughly one part in ten thousand to one part in a thousand, so this is not a
delicate tuning problem. With the fix:

- Quality of the value translation on fresh text goes from 0.43 to 0.67.
- In the worst affected layer, agreement between the translated cache and the real one goes from
  0.60 to 0.86.
- On a reasoning benchmark the model has never seen during fitting, accuracy rises from 0.360 to
  0.400. The receiver alone scores 0.452 and the sharer alone 0.352, so the fix takes the translator
  from "no better than the small model" to recovering about half the gap between them.

## Why this matters beyond one bug

This is a silent failure. Nothing crashes, nothing warns, and the fit reports a healthy score on its
own training data. You only see it if you check on held-out text, layer by layer. Anyone reproducing
the paper by following its equation literally would inherit a degraded translator and could easily
attribute the loss to the transfer method itself rather than to the regularizer.

For our safety audit it matters twice over, because a translator that is quietly broken would
degrade the receiver's behavior for reasons that have nothing to do with cross-model transfer, which
would contaminate any claim about safety surviving the handover.

---

# Section 2: for the paper

## Claim

In `kvtransfer` at `04a8e99`, the ridge penalty is added to an **unnormalized** centered Gram matrix.
Consequently the effective regularization strength is inversely proportional to the number of
calibration tokens, and at the published value of lambda = 0.01 it is numerically negligible. The
resulting solve is effectively ordinary least squares on strongly collinear features, which produces
a localized instability in mid-layer value maps. Setting lambda relative to the Gram scale removes
the instability and materially improves downstream accuracy.

## Mechanism

`MomentAccumulator` (`ridge.py`) streams sufficient statistics, accumulating

    gram  += xs^T xs      (float32)
    cross += xs^T ys      (float32)

over all calibration tokens, where `xs`, `ys` are shift-centered source and target features. The
centered Gram returned by `centered()` is

    Gxx = gram - n * outer(dx, dx)

which is a **sum over n tokens**, not a mean. The solve (`ridge.py:91`) is

    a = Gxx + lam * I
    W = solve(a, Gxy)

cast to float64. Since `Gxx` scales as O(n) while `lam` is a constant, the relative penalty

    r = lam / mean(diag(Gxx))

decreases as 1/n. The published Eq. 4 reads this way literally, so the implementation is a faithful
transcription; the issue is in the equation as stated, not only in the reconstruction.

**Measured scale** for Qwen3-0.6B to Qwen3-1.7B, k=8, 500 calibration sequences (about 128k tokens):
mean centered diagonal of the value Gram is 1.50e6 (keys: 5.45e5). Therefore

    r = 0.01 / 1.50e6 = 6.7e-9

The accumulation is in float32 while the solve is in float64, so accumulation error grows with token
count and lands preferentially in the near-singular directions that the absent penalty fails to
damp. The source features are 8 adjacent layers of the same model and are strongly collinear, so
those directions are not hypothetical.

## Evidence 1: the instability is data-dependent and localized

Comparing a 100-sequence fit against a 500-sequence fit, same k, same selected source layers:

| | 100 seqs | 500 seqs |
|---|---|---|
| held-out value R2, target layers 11 to 15 | +0.28 to +0.49 | -0.50 to -1.15 |
| in-sample value R2, same layers | 0.60 to 0.70 | 0.60 to 0.70 |
| norm of value weights, same layers | baseline | 5x to 7x larger |
| largest single weight entry | about 270 | about 1000 |

The remaining 23 target layers improve with more calibration data. Key maps are unaffected. The same
source layers are selected in both fits, so this is not a selection artifact. In-sample R2 is
unchanged while held-out R2 collapses: the signature of overfitting, and consistent with the penalty
being absent rather than merely small.

## Evidence 2: lambda sweep on held-out diagnostics

Refit from the saved 500-sequence statistics, k=8, evaluated on 32 FineWeb passages held out after
the calibration set. Sharer-alone top-1 agreement is 0.656 for reference.

| lambda | r | held-out R2 V | min V R2, layers 11 to 15 | attn cos mean / min | KL mean / p95 | top-1 |
|---|---|---|---|---|---|---|
| 0.01 (published) | 6.7e-9 | 0.426 | -1.15 | 0.878 / 0.600 | 0.342 / 1.37 | 0.742 |
| 1.5 | 1e-6 | 0.672 | 0.53 | 0.905 / 0.619 | 0.271 / 1.03 | 0.797 |
| 15 | 1e-5 | 0.673 | 0.54 | 0.910 / 0.655 | 0.264 / 0.93 | 0.804 |
| 150 | 1e-4 | 0.673 | 0.54 | 0.918 / 0.736 | 0.246 / 0.91 | **0.807** |
| 1500 | 1e-3 | 0.670 | 0.54 | **0.925 / 0.864** | 0.253 / 0.94 | 0.804 |
| 15000 | 1e-2 | 0.658 | 0.54 | 0.922 / 0.860 | 0.284 / 1.04 | 0.788 |

Maximum value-weight norm across layers: 36,900 at the published lambda (layer 14), 2,400 at r=1e-6,
194 at r=1e-2.

Any r >= 1e-6 removes the collapse. Held-out value R2 rises from 0.426 to about 0.67; top-1 margin
over the sharer-alone baseline widens from +8.6 to +15 points; worst-layer attention cosine rises
from 0.600 to 0.864. The optimum is broad, r = 1e-4 to 1e-3, with r=1e-2 beginning to over-regularize.

**Caveat:** lambda here is selected on the same 32 passages used to score it. The benchmark rerun
below is the independent check.

## Evidence 3: independent benchmarks

`lm-eval`, acc_norm, limit 250 (standard error about 3 points per cell). Receiver and sharer baselines
were recomputed each run and were identical across runs.

| mapper | arc_challenge | hellaswag |
|---|---|---|
| published lambda = 0.01 | 0.360 | 0.424 |
| r=1e-3 (lam=1503) | **0.400** | 0.428 |
| r=1e-4 (lam=150) | 0.392 | **0.432** |
| receiver alone (1.7B) | 0.452 | 0.492 |
| sharer alone (0.6B) | 0.352 | 0.504 |

On arc_challenge the receiver-over-sharer gap is 10.0 points. The published lambda recovers 0.8 of
those points, which is indistinguishable from the sharer-alone baseline: by this benchmark the
mapper carries none of the receiver's advantage. Regularized fits recover 4.0 to 4.8 points, that is
40 to 48% of the gap.

On hellaswag these chat checkpoints show no receiver-over-sharer gap at all (0.492 against 0.504), so
retention is uninformative there. Mapped sits about 7 points below both models under every lambda,
which the regularization does not fix. **That deficit is a separate open problem.** The leading
untested hypothesis is a context-length mismatch: the mapper is calibrated and diagnosed on
1024-token windows while benchmark contexts are short, so early positions, including the attention
sink token, may map poorly. Short-prefix diagnostics would test this.

## Recommended fix

Scale the penalty by the Gram, so that it is invariant to calibration set size. Either normalize the
Gram before solving,

    a = Gxx / n + (lam / n) * I          # equivalently, solve on per-token means

or express lambda relatively,

    a = Gxx + r * mean(diag(Gxx)) * I    # r dimensionless, default 1e-4 to 1e-3

The second is a one-line change at `ridge.py:91` and keeps the existing statistics format. Either
way the default should be a relative quantity. Reporting an absolute lambda without the token count
makes the published value unreproducible, since the same number means different things at different
calibration sizes.

Accumulating `gram` and `cross` in float64, or in float32 with compensated summation, would further
reduce the error term, though the sweep shows that adequate regularization alone is sufficient on
this pair.

## Scope and limitations

- One pair, one architecture family, one value of k. The effect should be confirmed on a second pair
  before being reported as general. The Ministral 3B to 8B anchor is the natural second case because
  the paper evaluated it.
- The relationship between r and the instability is established; the precise optimum is not, and the
  sweep that locates it is partly in-sample.
- All numbers are bf16 inference on a single RTX 5070 laptop GPU.
- Whether the authors' own unreleased implementation normalizes the Gram is unknown. The claim here
  is about the equation as published and about this reconstruction, which is the artifact available
  to anyone reproducing the work.

## Suggested disclosure

The finding is independent of our safety question and is useful to anyone using this code, so it is
worth an upstream issue against `kvtransfer` once a second pair confirms it. The issue should state
the scale measurement, the sweep, and the one-line fix, and should note explicitly that the
implementation follows the published equation, so the finding is about the method as specified rather
than a transcription error.