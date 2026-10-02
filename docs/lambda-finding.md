# The lambda finding: two silent defects in the mapper fit

_Written 2026-10-01, revised 2026-10-02 after float64 precision checks on values and keys. Status:
confirmed on one pair (Qwen3-0.6B to Qwen3-1.7B), on held-out diagnostics and on an independent
benchmark. Not yet checked on a second pair._

_Implementation audited: `github.com/Susmith4710/kvtransfer`, pinned at `04a8e99`. This is an
independent reconstruction; Heo et al. (arXiv 2608.03893) released no code._

_Revision note: the 2026-10-01 version attributed the mid-layer value collapse to overfitting
caused by an absent penalty. The float64 checks show the collapse is float32 rounding error. The
penalty finding stands, but its main effect is on keys and attention, not values._

---

# Section 1: in plain terms

## The setup

Cross-model KV cache transfer lets a small "sharer" model read a prompt once, then hands its
internal cache to a bigger "receiver" model so the receiver can answer without re-reading the
prompt itself. The two models store that cache differently, so you need a translator between them.

That translator is fitted by linear regression: show both models the same text, record what each one
stores, and solve for the matrix that best converts one into the other. This is ridge regression,
and ridge regression has one knob, usually called lambda. Lambda decides how much the fit is allowed
to chase the training data versus staying conservative. The translator has two halves, one for the
"key" vectors (which decide where the model looks) and one for the "value" vectors (which decide
what it reads once it looks there).

## Problem 1: the running totals are kept at too low a precision

The fit does not store the calibration data. It keeps running totals over every calibration token
and solves from those. The implementation keeps the totals in 32-bit floating point, which holds
about 7 significant digits.

Over 128,000 tokens, the totals become large, and the rounding in the last digits adds up. For the
value half of the translator, that rounding is enough to corrupt the weakest directions in the data:
quantities that must be positive by construction come out slightly negative. The solver then
divides by numbers near zero or of the wrong sign, and the weights explode.

That is what we saw in a band of middle layers: the value weights grew about 20 times larger than
they should be, and on fresh text the translator scored below what you would get by ignoring the
input entirely. Redoing exactly the same fit with 64-bit totals (about 16 significant digits), and
changing nothing else, made the problem disappear.

The key half has the same amount of rounding but is not damaged by it: there, the rounding falls on
the strong directions, where it is harmless. So the size of the rounding error alone does not tell
you whether a fit is broken.

## Problem 2: the regularization knob is not connected to anything

The paper specifies lambda = 0.01, and the implementation uses that as its default.

The regularizer is added to one of those running totals, not to an average. With roughly 128,000
tokens, the total has a typical size of about 1,500,000. Adding 0.01 to 1,500,000 changes
essentially nothing: the relative strength is about 7 parts in a billion. And it weakens as you add
data, since the total grows while lambda stays fixed.

Here the precision fix does not help. Even with 64-bit totals, the unregularized key translator
has weights about 50 times larger than a regularized one. Those weights score almost the same on a
direct reconstruction test (0.930 vs 0.932) but distort where the receiver looks: in the worst
layer, agreement between the receiver's real attention and its attention on the translated cache is
0.60 unregularized and 0.86 regularized.

## The fixes, and what they buy

1. Keep the running totals in 64-bit precision. This alone repairs the value half.
2. Make lambda relative to the size of what it is added to. This repairs the key half and the
   attention, and as a side effect also prevents the value collapse even at 32-bit precision.

The useful range for the relative strength is broad, roughly one part in ten thousand to one part in
a thousand, so this is not a delicate tuning problem. Compared with the published setting:

- Quality of the value translation on fresh text goes from 0.43 to 0.67 (either fix).
- Worst-layer attention agreement goes from 0.60 to 0.86 (relative lambda only).
- On a reasoning benchmark the model never saw during fitting, accuracy rises from 0.360 to
  0.400. The receiver alone scores 0.452 and the sharer alone 0.352, so the fix takes the
  translator from "no better than the small model" to recovering about half the gap between them.

## Why this matters beyond one bug

Both are silent failures. Nothing crashes, nothing warns, and the fit reports a healthy score on its
own training data. You only see them if you check on held-out text, layer by layer, and the second
one does not even show in the direct reconstruction score. Anyone reproducing the paper with this
code would inherit a degraded translator and could easily attribute the loss to the transfer method
itself.

For our safety audit it matters twice over, because a translator that is quietly broken would
degrade the receiver's behavior for reasons that have nothing to do with cross-model transfer, which
would contaminate any claim about safety surviving the handover.

---

# Section 2: for the paper

## Claims

In `kvtransfer` at `04a8e99`, on Qwen3-0.6B to Qwen3-1.7B:

1. **Float32 moment accumulation corrupts the value fit.** Rounding error in the float32 Gram makes
   the centered value Gram indefinite in the source-layer blocks that target layers 11 to 15 use.
   At the published lambda this produces the mid-layer value collapse. Accumulating in float64
   removes it with lambda unchanged. The key Gram is unaffected.
2. **The ridge penalty is numerically negligible.** It is added to an unnormalized Gram, so its
   relative strength falls as 1/n and is about 7e-9 at the published lambda = 0.01. A relative
   penalty is needed for the key maps independently of precision, and it improves worst-layer
   attention and next-token agreement beyond what float64 alone achieves.

## Mechanism

`MomentAccumulator` (`ridge.py`) streams sufficient statistics, accumulating

    gram  += xs^T xs      (stats_dtype, default float32)
    cross += xs^T ys      (stats_dtype, default float32)

over all calibration tokens, where `xs`, `ys` are features centered on the first batch's means.
The centered Gram returned by `centered()` is

    Gxx = gram - n * outer(dx, dx)

which is a **sum over n tokens**, not a mean. The solve (`ridge.py:91`) casts to float64 and computes

    a = Gxx + lam * I
    W = solve(a, Gxy)

Two consequences follow.

**Precision.** The float64 solve cannot recover what was lost when `gram` was summed in float32.
Relative rounding error in the stored entries is around 6e-8 or more, while the penalty is 6.7e-9 of
the Gram's scale. So at the published lambda the penalty is below the rounding noise of the
statistics it regularizes.

**Scale.** Since `Gxx` scales as O(n) while `lam` is a constant, the relative penalty

    r = lam / mean(diag(Gxx))

decreases as 1/n. The published Eq. 4 reads this way literally, so on this point the implementation
is a faithful transcription and the issue is in the equation as stated. The precision issue, by
contrast, is a property of this implementation's default.

**Measured scale** at k=8, 500 calibration sequences (128,000 tokens per kind): mean centered
diagonal of the value Gram is 1.50e6 (keys: 5.45e5), so r = 6.7e-9 for values and 1.8e-8 for keys.

## Evidence 1: the instability is data-dependent and localized

Comparing a 100-sequence fit against a 500-sequence fit, same k, same selected source layers,
float32 statistics, published lambda:

| | 100 seqs | 500 seqs |
|---|---|---|
| held-out value R2, target layers 11 to 15 | +0.28 to +0.49 | -0.50 to -1.15 |
| in-sample value R2, same layers | 0.60 to 0.70 | 0.60 to 0.70 |
| norm of value weights, same layers | baseline | 5x to 7x larger |
| largest single weight entry | about 270 | about 1000 |

The remaining 23 target layers improve with more calibration data, and key maps are unaffected.
Target layers 12 to 15 all select source layers 11 to 18 (in different orders) and target layer 11
selects 11 to 17 and 19, so the five affected target layers share two distinct Gram matrices.

## Evidence 2: float64 re-accumulation isolates precision

`scripts/precision_check.py` re-accumulates one kind's moments in float64 with the stock
`calibrate()` (`stats_dtype=torch.float64`; same 500 sequences, stride and batch size), refits that
kind at each reference mapper's lambda and layer selection, and keeps the other kind's maps from the
float32 reference. Data identity: first-batch feature means agree to 4e-8 (values) and 6e-8 (keys),
final means to 7e-8 and 7e-10, and both runs see 128,000 tokens.

Centered Gram blocks the solver receives, float32 vs float64:

| Kind | Target layers | f32 error, spectral norm | Error / lam | eig_min f64 | eig_min f32 | f64 eigenvalues below the f32 error |
|---|---|---|---|---|---|---|
| V | 11 | 15.4 | 1,540x | +0.272 | -0.524 | 728 / 8192 |
| V | 12 to 15 | 14.0 | 1,400x | +0.272 | -0.525 | 724 / 8192 |
| K | 11 | 21.7 | 2,170x | 1.6e-4 | 1.6e-4 | 712 / 8192 |
| K | 12 to 15 | 21.8 | 2,180x | 1.6e-4 | 1.6e-4 | 719 / 8192 |

Relative Frobenius error is about 4.3e-7 for both kinds. For values, the float32 Gram is indefinite
and lambda = 0.01 cannot restore positive definiteness. For keys, the error is larger but leaves the
bottom of the spectrum intact (condition number 1.3e12 in both precisions). The error's size relative
to lambda therefore does not predict damage; whether it moves the smallest eigenvalues does.

Refit at the published lambda, float32 to float64 statistics:

- Values, layers 11 to 15: weight norms 16 to 24x smaller (layer 14: 36,909 to 1,873; largest entry
  1,048 to 40); in-sample R2 about 0.02 lower, consistent with the float32 fit partly fitting
  rounding error.
- Keys, layers 11 to 15: weight norms change by under 0.02% (layer 11: 1,689.0 to 1,689.1); in-sample
  R2 identical.
- At r = 1e-3, float32 and float64 fits are identical to printed precision for both kinds.

## Evidence 3: held-out diagnostics

32 FineWeb passages held out after the calibration set, k=8, same selection throughout. Sharer-alone
top-1 agreement is 0.656.

Precision factorial:

| lambda | K stats | V stats | R2 K | R2 V | min V R2, L11 to 15 | attn cos mean / min | KL mean | top-1 |
|---|---|---|---|---|---|---|---|---|
| 0.01 | f32 | f32 | 0.930 | 0.426 | -1.15 | 0.878 / 0.600 | 0.342 | 0.742 |
| 0.01 | f64 | f32 | 0.930 | 0.426 | -1.15 | 0.878 / 0.600 | 0.342 | 0.744 |
| 0.01 | f32 | f64 | 0.930 | 0.671 | +0.53 | 0.902 / 0.600 | 0.277 | 0.789 |
| 0.01 | f64 | f64 | 0.930 | 0.671 | +0.53 | 0.902 / 0.600 | 0.276 | 0.790 |
| 1503 (r=1e-3) | f32 | f32 | 0.932 | 0.670 | +0.54 | 0.925 / 0.864 | 0.253 | 0.804 |
| 1503 (r=1e-3) | f64 | f32 | 0.932 | 0.670 | +0.54 | 0.925 / 0.864 | 0.252 | 0.802 |
| 1503 (r=1e-3) | f32 | f64 | 0.932 | 0.670 | +0.54 | 0.925 / 0.864 | 0.252 | 0.800 |

Top-1 differences of 0.004 or less are a few tokens out of 1,024, consistent with near-ties
(weights equal to printed precision).

Lambda sweep, float32 statistics:

| lambda | r | held-out R2 V | min V R2, layers 11 to 15 | attn cos mean / min | KL mean / p95 | top-1 |
|---|---|---|---|---|---|---|
| 0.01 (published) | 6.7e-9 | 0.426 | -1.15 | 0.878 / 0.600 | 0.342 / 1.37 | 0.742 |
| 1.5 | 1e-6 | 0.672 | 0.53 | 0.905 / 0.619 | 0.271 / 1.03 | 0.797 |
| 15 | 1e-5 | 0.673 | 0.54 | 0.910 / 0.655 | 0.264 / 0.93 | 0.804 |
| 150 | 1e-4 | 0.673 | 0.54 | 0.918 / 0.736 | 0.246 / 0.91 | **0.807** |
| 1500 | 1e-3 | 0.670 | 0.54 | **0.925 / 0.864** | 0.253 / 0.94 | 0.804 |
| 15000 | 1e-2 | 0.658 | 0.54 | 0.922 / 0.860 | 0.284 / 1.04 | 0.788 |

r is relative to the value Gram; for keys it is about 2.75x larger at the same lambda.

Reading the two tables together:

- **Values are a precision effect.** float64 alone takes held-out value R2 from 0.426 to 0.671, equal
  to the best regularized fit. Any r >= 1e-6 also fixes values at float32, because lambda = 1.5
  already exceeds the most negative float32 eigenvalue (about -0.52).
- **Keys and attention are a regularization effect.** Neither key precision nor value precision moves
  the worst-layer attention cosine off 0.600. It rises steadily with r up to 1e-3 (0.864), well past
  the point where values are fixed. The penalty shrinks key weight norms about 50x (layer 11: 1,689 to
  36) for a held-out key R2 change of only 0.930 to 0.932, so the damage is invisible in key R2.
- **Both fixes together are the best configuration.** float64 at the published lambda reaches top-1
  0.790; r = 1e-3 reaches 0.804. The optimum is broad, r = 1e-4 to 1e-3, with r = 1e-2 beginning to
  over-regularize.

**Caveat:** lambda here is selected on the same 32 passages used to score it. The benchmark rerun
below is the independent check.

## Evidence 4: independent benchmarks

`lm-eval`, acc_norm, limit 250 (standard error about 3 points per cell), float32 statistics.
Receiver and sharer baselines were recomputed each run and were identical across runs.

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
40 to 48% of the gap. The float64-at-published-lambda mapper has not been benchmarked.

On hellaswag these chat checkpoints show no receiver-over-sharer gap at all (0.492 against 0.504), so
retention is uninformative there. Mapped sits about 7 points below both models under every lambda,
which the regularization does not fix. **That deficit is a separate open problem.** The leading
untested hypothesis is a context-length mismatch: the mapper is calibrated and diagnosed on
1024-token windows while benchmark contexts are short, so early positions, including the attention
sink token, may map poorly. Short-prefix diagnostics would test this.

## Recommended fix

1. **Accumulate `gram` and `cross` in float64** (`stats_dtype=torch.float64`). This removes the value
   collapse at any lambda. Cost: memory doubles, to about 13.2 GB per kind on this pair (two
   28,672 x 28,672 matrices). Compensated float32 summation is a possible lower-memory alternative,
   untested here.
2. **Make the default penalty relative**, so it is invariant to calibration size:

       a = Gxx + r * mean(diag(Gxx)) * I    # r dimensionless, default 1e-4 to 1e-3

   This is a one-line change at `ridge.py:91` and keeps the existing statistics format. An equivalent
   alternative is to apply an absolute lambda to the per-token Gram, `a = Gxx / n + lam * I` (with
   `Gxy / n`), which also fixes the scale; lambda then needs re-tuning for that normalization.
   Reporting an absolute lambda without the token count makes the published value unreproducible,
   since the same number means different things at different calibration sizes.

Fix 2 alone gives the best held-out numbers on this pair and also prevents the value collapse in
float32. Fix 1 removes a failure mode that otherwise depends on lambda being set large enough, at a
memory cost.

## Scope and limitations

- One pair, one architecture family, one value of k. The effect should be confirmed on a second pair
  before being reported as general. The Ministral 3B to 8B anchor is the natural second case because
  the paper evaluated it.
- The relationship between r and the instability is established; the precise optimum is not, and the
  sweep that locates it is partly in-sample.
- The float64 checks were run at 500 calibration sequences only, not at 100.
- All numbers are bf16 inference on a single RTX 5070 laptop GPU; moments accumulated on CPU.
- Whether the authors' own unreleased implementation normalizes the Gram, or accumulates in higher
  precision, is unknown. The penalty-scale claim is about the equation as published; the precision
  claim is about this reconstruction's default, which is the artifact available to anyone
  reproducing the work.

## Suggested disclosure

Both findings are independent of our safety question and useful to anyone using this code, so they
are worth an upstream issue against `kvtransfer` once a second pair confirms them.
`docs/ridge-penalty-scale.md` is the draft for that issue. It should state the precision finding, the
scale measurement, the factorial and sweep, and both fixes. It should note that the penalty-scale
issue follows the published equation (so it concerns the method as specified), while the precision
issue is an implementation default.
