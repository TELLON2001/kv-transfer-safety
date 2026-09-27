# References

All entries verified against source on **2026-09-27**. BibTeX in
[`references.bib`](references.bib).

## Cross-model KV transfer (the methods under audit)

### Heo et al. 2026: the base method
**Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for
Prefill Reuse.** Taekyung Heo, Rasoul Shafipour, Ritchie Zhao, Maximilian Golub,
Mohammad Mahdi Kamani, Ritika Borkar, Makesh Tarun Chandran, Pantea Zardoshti,
Bita Darvish Rouhani. arXiv:[2608.03893](https://arxiv.org/abs/2608.03893) (Aug 2026).
Ridge-regression mapper that lets a smaller model prefill and a larger sibling
decode without re-prefilling. Selects predictive source layers per target layer,
strips positional encoding, trains on ~500 calibration sequences. Retains **73–98%
of standalone-prefill accuracy** on four pairs; **2.7–25× faster** than prefill.
Requires same family, matched KV dims, **shared tokenizer**.
Code: [`Susmith4710/kvtransfer`](https://github.com/Susmith4710/kvtransfer) (Apache-2.0).

### CacheBridge 2026
**CacheBridge: Efficient Cross-Model KV Cache Transfer.** Xingyu Qu, Siyuan Lu,
Zhiyu Chen, Sheng Wang, Tao Lin. arXiv:[2609.00891](https://arxiv.org/abs/2609.00891) (Sep 2026).
Closed-form affine interface for online deployment; restricts each target head to a
single source head, weights reconstruction error by attention sensitivity. Improves
on Full-Head Mapping in speed and storage.

### C2C 2026: foundational citation
**Cache-to-Cache: Direct Semantic Communication Between Large Language Models.**
Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, Yu Wang.
arXiv:[2510.03215](https://arxiv.org/abs/2510.03215), ICLR 2026.
LLMs exchange information directly via KV-cache (learned projection + fusion)
instead of text. Reports 8.5–10.5% accuracy over individual models, ~2.0× latency
speedup. Code: [`thu-nics/C2C`](https://github.com/thu-nics/C2C).

### Latent Cache Flow 2026
**Latent Cache Flow: Model-to-Model Communication Without Text.** Maximillian Rossi,
Prajwal Raghunath, Eugene Wu. arXiv:[2605.22863](https://arxiv.org/abs/2605.22863) (May 2026).
Compressed latent channel that transmits a summary of information the target lacks,
handling differing contexts between models. _Overlap: low._

### Dery et al. 2026
**Latent Space Communication via K-V Cache Alignment.** Lucio M. Dery, Zohar Yahav,
Henry Prior, Qixuan Feng, Jiajun Shen, Arthur Szlam.
arXiv:[2601.06123](https://arxiv.org/abs/2601.06123) (Jan 2026).
Augments each model with adapters that translate its KV state into and out of a
shared latent space, giving a high-bandwidth cross-model channel without changing
pre-trained weights; demonstrated on Gemma-2, including soft-prompt/skill transfer.
_Overlap: low; a learned, trained channel across architectures, not the prefill-skip
transfer this project audits._

## Auditing / integrity / defense of shared caches

### When Does Latent Communication Pay? (source of the mismatched-cache control)
**A Causal Audit of Relayed KV Caches in Multi-Agent LLMs.** Jiaming Cheng,
Subhransu Das, Rajiv Ramnath. arXiv:[2608.04893](https://arxiv.org/abs/2608.04893) (Aug 2026).
Tests whether gains depend on cache *content* vs. mere presence, using mismatched,
zeroed, and random cache substitutions across families and benchmarks. This is where
the project's **mismatched-cache control** comes from.

### When Latent Agents Lie 2026
**KV-Cache Integrity in Multi-Agent LLM Collaboration.** Luís Brito, Carlos Baquero.
arXiv:[2606.28958](https://arxiv.org/abs/2606.28958) (Jun 2026).
Tampering with hidden KV state can collapse performance while the visible commitment
still looks plausible; proposes HMAC-SHA256 authentication. _Adjacent: attacks, not
transfer fidelity._

### LCGuard 2026
**Latent Communication Guard for Safe KV Sharing in Multi-Agent Systems.** Sadia Asif,
Mohammad Mohammadi Amiri, Momin Abbas, Prasanna Sattigeri, Karthikeyan Natesan
Ramamurthy. arXiv:[2605.22786](https://arxiv.org/abs/2605.22786) (May 2026).
Adversarial training of cache transformations that preserve task semantics while
reducing reconstructable sensitive input. _Adjacent: defense, not refusal retention._

## Safety mechanism (stretch goal)

### Arditi et al. 2024
**Refusal in Language Models Is Mediated by a Single Direction.** Andy Arditi, Oscar
Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda.
arXiv:[2406.11717](https://arxiv.org/abs/2406.11717) (Jun 2024; v3 Oct 2024).
Refusal in chat models is largely carried by a single direction in activation space.
Basis for the stretch goal's mechanism test and fix.

## Software

| Repo | Implements | License |
| --- | --- | --- |
| [`Susmith4710/kvtransfer`](https://github.com/Susmith4710/kvtransfer) | Heo et al. closed-form ridge mapper; calibrate/eval/benchmark/generate; HF DynamicCache-compatible | Apache-2.0 |
| [`thu-nics/C2C`](https://github.com/thu-nics/C2C) | Cache-to-Cache projectors + training/inference (ICLR 2026 official) | see repo |
