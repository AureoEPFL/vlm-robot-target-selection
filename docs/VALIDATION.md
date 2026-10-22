# Completion evidence

This ledger distinguishes implementation from executed evidence.

| Requirement | Evidence | Status |
|---|---|---|
| Official data provenance and partitions | Actual archive SHA-256 and counts in DATA_CARD.md; remote provenance.json | Verified |
| Target-independent candidate construction | Six local protocol tests, including shared rendering for different targets | Verified |
| Image/scene separation | Category holdout and exact-pixel audit; no physical session IDs available | Verified within documented metadata limits |
| Answer-only loss and LoRA gradients | Remote smoke: 1,638,400 trainable parameters, finite nonzero gradients, masked prefixes | Verified in smoke |
| Adapter save and independent-process reload | Remote smoke: 8/8 exact outputs reproduced in a new process | Verified in smoke |
| Pretrained and adapted evaluations | Seed 42 testA 31.0% vs 29.0% baseline; testB 27.5% vs 17.5% baseline | Verified |
| Validation-based checkpoint selection | Seed 42 selected step 250: 53/128 (41.41%), baseline 32/128 | Verified |
| Robustness and uncertainty | Paraphrase, permutation, visual variants with paired intervals and consistency rates | Verified; limitations documented |
| Colab execution | Source bundle and smoke protocol checked; clean rerun limited by Colab quota | Partially verified |
| Interactive demo | Actual widget selection matched the persisted testB prediction for a checked example | Verified |
| Public documentation and archive | README, model/data cards, protocol CI, results, and source archive | Ready |

The self-contained notebook currently has 23 cells. Its schema, Python syntax, embedded-source checksum, and synchronization with repository modules are checked locally. This is not a substitute for the pending full notebook execution.

The substantive measurements were executed on a free Colab T4. A second GPU allocation was refused by Colab usage limits, so the final clean-kernel smoke attempt could not complete after the import repair; no score is inferred from that attempt. Existing training, evaluation, reload, and interactive-demo evidence remain the authoritative results.
