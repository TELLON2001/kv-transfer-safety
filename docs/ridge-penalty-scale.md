# Float32 moment accumulation and ridge penalty scale in kvtransfer

_Nadim Tello, 2026-10-02 (draft). Pair: Qwen3-0.6B → Qwen3-1.7B. Implementation: `github.com/Susmith4710/kvtransfer` at commit `04a8e99`, an independent reconstruction of Heo et al. (arXiv 2608.03893)._

## Summary

Two related issues affect the closed-form ridge fit in this implementation.

1. **Moments are accumulated in float32.** The Gram and cross-moment matrices are summed in float32 (`stats_dtype` default) and only the solve is float64. On this pair, float32 rounding makes the centered value Gram indefinite: its smallest eigenvalue moves from +0.27 to −0.52. With 500 calibration sequences, this causes the mid-layer value maps to collapse on held-out text (held-out R² as low as −1.15). Accumulating the same moments in float64, with λ unchanged, removes the collapse completely.
2. **λ is applied to a summed, not averaged, Gram.** At λ = 0.01 the penalty is about 7 × 10⁻⁹ of the Gram's scale, so the solve is effectively unregularized, and the ratio shrinks as calibration data grows. A penalty set relative to the Gram scale also removes the collapse, because it is large enough to restore positive definiteness. It is needed separately for keys: float64 leaves the key maps unchanged, and the worst-layer attention cosine rises from 0.60 to 0.86 only when the penalty is raised.

The two fixes address different maps, and the best result needs both.

## Setup

- Source layers per target layer: k = 8
- Calibration: 500 FineWeb-Edu sequences × 1,024 tokens, stride 4 (128,000 tokens per kind), with a 100-sequence comparison fit
- Held-out evaluation: 32 FineWeb-Edu passages that come after the calibration sequences in the stream
- Inference: bf16, single RTX 5070 laptop GPU; moments accumulated on CPU

## Scale of the penalty

The mapper is the ridge solution

$$
W = \left(G_{xx} + \lambda I\right)^{-1} G_{xy},
\qquad
G_{xx} = \sum_{i=1}^{n} x_i x_i^\top - n\,\mu\mu^\top ,
$$

where $G_{xx}$ grows linearly with the token count $n$. The relevant quantity is the relative penalty

$$
r = \frac{\lambda}{\operatorname{mean}\!\left(\operatorname{diag}(G_{xx})\right)} .
$$

| Gram | mean diagonal | r at λ = 0.01 |
|---|---|---|
| Values | 1.50 × 10⁶ | 6.7 × 10⁻⁹ |
| Keys | 5.45 × 10⁵ | 1.8 × 10⁻⁸ |

Because $G_{xx}$ scales with $n$ while $\lambda$ is fixed, $r$ falls roughly as $1/n$ as calibration data grows.

## Symptom: value maps collapse at the larger calibration size

Same k and same selected source layers; target layers 11–15, value maps, λ = 0.01, float32 moments:

| | 100 sequences | 500 sequences |
|---|---|---|
| Held-out value R² | +0.28 to +0.49 | −0.50 to −1.15 |
| In-sample value R² | 0.60 to 0.70 | 0.60 to 0.70 |
| Value weight norm | baseline | 5–7× larger |
| Largest weight entry | ~270 | ~1000 |

The other 23 target layers improve with more data, and key maps are unaffected. In-sample R² is unchanged while held-out R² collapses.

Target layers 12–15 all select the same eight source layers (11–18, in different orders), and target layer 11 selects 11–17 and 19. The five affected target layers therefore share two distinct Gram matrices, not five.

## Cause: float32 accumulation makes the Gram indefinite

We re-accumulated the value moments in float64 using the stock `calibrate()` with `stats_dtype=torch.float64`. Everything else was unchanged: the same 500 sequences, stride and batch size. The data matched the float32 run (first-batch feature means agree to 4 × 10⁻⁸, final means to 7 × 10⁻⁸, and both runs see 128,000 tokens). We then compared the centered Gram blocks that the solver receives for the selected source layers.

| Target layers | Float32 error, spectral norm | Error / λ (0.01) | Smallest eigenvalue, float64 | Smallest eigenvalue, float32 | Float64 eigenvalues below the float32 error |
|---|---|---|---|---|---|
| 11 | 15.4 | 1,540× | +0.272 | −0.524 | 728 of 8,192 |
| 12–15 | 14.0 | 1,400× | +0.272 | −0.525 | 724 of 8,192 |

The overall relative error is small (‖G₃₂ − G₆₄‖_F / ‖G₆₄‖_F = 4.2 × 10⁻⁷), but it is about 1,400 times larger than λ. In 724 to 728 directions, the true Gram signal is smaller than the rounding error. The float64 Gram is positive definite with condition number 8.3 × 10⁸; the float32 Gram has negative eigenvalues. Adding λ = 0.01 cannot restore positive definiteness, since the most negative eigenvalue is about −0.52. The solve therefore divides by near-zero or negative directions, which produces the large weights.

Refitting the value maps from the float64 moments at the same λ = 0.01, with the same source layers:

| Target layer | Value weight norm, float32 → float64 | Largest weight entry | In-sample value R² |
|---|---|---|---|
| 11 | 17,689 → 749 | 323 → 11 | 0.698 → 0.681 |
| 12 | 25,846 → 1,411 | 378 → 19 | 0.631 → 0.613 |
| 13 | 30,943 → 1,579 | 495 → 21 | 0.636 → 0.617 |
| 14 | 36,909 → 1,873 | 1,048 → 40 | 0.603 → 0.582 |
| 15 | 22,662 → 1,403 | 168 → 5 | 0.606 → 0.598 |

Weights shrink 16–24×. In-sample R² falls by about 0.02, consistent with the float32 fit partly fitting rounding error.

Held-out results on the 32 passages. The value maps come from the moments listed; the key maps are taken unchanged from the float32 fit at the same λ:

| Mapper | Value moments | Held-out value R² | Worst value R², layers 11–15 | Worst-layer attention cosine | Top-1 agreement |
|---|---|---|---|---|---|
| λ = 0.01 | float32 | 0.426 | −1.15 | 0.600 | 0.742 |
| λ = 0.01 | float64 | 0.671 | +0.53 | 0.600 | 0.789 |
| r = 10⁻³ | float32 | 0.670 | +0.54 | 0.864 | 0.804 |
| r = 10⁻³ | float64 | 0.670 | +0.54 | 0.864 | 0.800 |

With float64 moments and the default λ, held-out value R² matches the best-regularized fit (0.671 vs 0.670). At r = 10⁻³ the float32 and float64 fits are identical to printed precision, so precision stops mattering once the penalty is large enough. The 0.804 vs 0.800 top-1 difference is 4 of 1,024 tokens, consistent with near-ties.

The attention cosine does not change with the value moments (0.600 in both λ = 0.01 rows). It depends on the key maps, which these mappers take unchanged from the float32 fit.

## Keys

We repeated the procedure for the key moments (content-space keys, RoPE stripped). Data identity again held: first-batch means agree to 6 × 10⁻⁸, final means to 7 × 10⁻¹⁰, and both runs see 128,000 tokens.

| Target layers | Float32 error, spectral norm | Error / λ (0.01) | Smallest eigenvalue, float64 | Smallest eigenvalue, float32 | Condition number, float64 / float32 |
|---|---|---|---|---|---|
| 11 | 21.7 | 2,170× | 1.6 × 10⁻⁴ | 1.6 × 10⁻⁴ | 1.3 × 10¹² / 1.3 × 10¹² |
| 12–15 | 21.8 | 2,180× | 1.6 × 10⁻⁴ | 1.6 × 10⁻⁴ | 1.3 × 10¹² / 1.3 × 10¹² |

The key Gram has a larger float32 error than the value Gram and is far worse conditioned, yet its smallest eigenvalues survive float32 intact. The error must lie mainly along high-variance directions. Refitting the keys from float64 moments at λ = 0.01 changes the key weight norms by under 0.02% (layer 11: 1,689.0 → 1,689.1), with identical in-sample R². So the size of the rounding error alone does not predict damage; what matters is whether rounding changes the smallest eigenvalues, which it does for values and not for keys.

Held-out results, all at the same layer selection:

| Mapper | Key moments | Value moments | Held-out key R² | Held-out value R² | Attention cosine, mean / worst layer | Logit KL | Top-1 agreement |
|---|---|---|---|---|---|---|---|
| λ = 0.01 | float32 | float32 | 0.930 | 0.426 | 0.878 / 0.600 | 0.342 | 0.742 |
| λ = 0.01 | float64 | float32 | 0.930 | 0.426 | 0.878 / 0.600 | 0.342 | 0.744 |
| λ = 0.01 | float32 | float64 | 0.930 | 0.671 | 0.902 / 0.600 | 0.277 | 0.789 |
| λ = 0.01 | float64 | float64 | 0.930 | 0.671 | 0.902 / 0.600 | 0.276 | 0.790 |
| r = 10⁻³ | float32 | float32 | 0.932 | 0.670 | 0.925 / 0.864 | 0.253 | 0.804 |
| r = 10⁻³ | float64 | float32 | 0.932 | 0.670 | 0.925 / 0.864 | 0.252 | 0.802 |

Key precision changes no metric. With both kinds in float64 at the default λ, the worst-layer attention cosine stays at 0.600 and top-1 agreement at 0.790, against 0.864 and 0.804 with r = 10⁻³. The remaining gap is a regularization effect on the keys: the penalty shrinks the key weight norms about 50× (layer 11: 1,689 → 36) at a cost of about 0.01 in-sample R², and held-out key R² barely moves (0.930 → 0.932). The large unregularized key weights hurt attention without showing up in key R².

## λ sweep (float32 moments)

Refit from the saved 500-sequence float32 statistics. Sharer-alone top-1 agreement is 0.656. r is computed against the value Gram; for keys, r is about 2.75× larger at the same λ.

| λ | r | Held-out value R² | Worst value R², layers 11–15 | Worst-layer attention cosine | Top-1 agreement |
|---|---|---|---|---|---|
| 0.01 (default) | 6.7e-9 | 0.426 | −1.15 | 0.600 | 0.742 |
| 1.5 | 1e-6 | 0.672 | 0.53 | 0.619 | 0.797 |
| 15 | 1e-5 | 0.673 | 0.54 | 0.655 | 0.804 |
| 150 | 1e-4 | 0.673 | 0.54 | 0.736 | 0.807 |
| 1500 | 1e-3 | 0.670 | 0.54 | 0.864 | 0.804 |
| 15000 | 1e-2 | 0.658 | 0.54 | 0.860 | 0.788 |

Any r ≥ 10⁻⁶ removes the value collapse. This fits the cause above: λ = 1.5 already exceeds the most negative float32 eigenvalue (about −0.52), so the regularized system is positive definite again. The worst-layer attention cosine keeps improving up to r = 10⁻³, well past the point where values are fixed; this is the key-side effect described above. Caveat: λ is selected on the same held-out passages used to score it.

## Suggested change

1. **Accumulate the moments in float64** (`stats_dtype=torch.float64`). On this pair, this alone fixes the value maps at the default λ. Cost: memory doubles, to about 13.2 GB per kind here (Gram plus cross-moments, 28,672² each).
2. **Express the penalty relative to the Gram scale**, which makes it invariant to calibration size:

$$
W = \Big(G_{xx} + r \cdot \operatorname{mean}\!\big(\operatorname{diag}(G_{xx})\big)\, I\Big)^{-1} G_{xy},
\qquad r \approx 10^{-4} \text{ to } 10^{-3}.
$$

This is still needed with float64 moments: it is what improves the keys and the worst-layer attention. With float32 moments it also prevents the value collapse, as long as the penalty exceeds the most negative eigenvalue that rounding introduces. That makes float64 optional on this pair once r is in range, but it removes a failure mode that otherwise depends on how λ happens to be set.

## Open questions

- Whether the original implementation normalizes the Gram (e.g. per-token means) or scales features before the solve.
- Whether the effect holds on a second pair; not yet tested.
- The float64 check was run at 500 sequences only, not at 100.
