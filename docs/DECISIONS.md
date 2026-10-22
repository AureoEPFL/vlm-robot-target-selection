# Decisions and evidence

## 2026-10-22 — initial protocol

- Use the original SmolVLM-500M checkpoint at an immutable revision; language-model attention projections receive LoRA adapters. Keep the visual encoder frozen initially.
- Use supplied candidate boxes. Predict only a candidate ID. This measures conditional target selection, not autonomous detection or grasp execution.
- Candidate sets must be constructed for an image independently of which referring expression is queried. All annotations for that image contribute candidate boxes. Exclude images with too many candidates instead of using a target-dependent shortlist.
- Randomize candidate order deterministically by seed and example, with fresh training permutations. Preserve the mapping to the underlying object for evaluation of permutation consistency.
- Preserve official RoboRefIt test partitions. Do not train on the `Zray26/roborefit` Hub mirror: its metadata exposes only a test split and does not establish original partition provenance.
- Author website links a deleted archive. Current official repository links Google Drive file `1pdGF1HaU_UiKfh5Z618hy3nRjVbq_VuW`.
- Dataset issue reports describe ambiguous or wrong labels. Keep these limitations explicit; never repair or exclude test labels based on the model's predictions.
- No cloud training results exist yet. The dependency lock and empirical conclusions will be filled from actual runtime evidence.

## Sources

- https://github.com/luyh20/VL-Grasp
- https://luyh20.github.io/RoboRefIt.github.io/
- https://github.com/luyh20/VL-Grasp/issues/3
- https://github.com/luyh20/VL-Grasp/issues/15
- https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct
- https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
# Execution updates

- Main seed42 completed: step250 selected at53/128 validation correct (41.41%), versus49/128 at125 and32/128 pretrained. Training604.09s, peak allocated3,174,146,048bytes. Independent-process reload reproduced128/128 raw outputs exactly. Test metrics have not yet been observed. The model and protocol are frozen for final test evaluation; seed43 remains a repeatability check only.

- Main run at step125: validation49/128 (38.28%) versus pretrained32/128 (25%), with no invalid answers. Continue the prespecified250steps and select between125/250 using validation only. Given this functioning adaptation and the limited compute scope, prioritize a second training seed over adding a new visual architecture or an unnecessary hyperparameter sweep. Seed42 remains the primary model; seed43 is a variability check on fixed manifests and evaluation numbering.

- The loss-mask audit revealed that the chat template inserts whitespace before its answer digit. The final secondary digit-ranking diagnostic therefore evaluates causal logits at the canonical digit position, after that fixed whitespace, rather than ranking digits at the very first generated position. A fixed dummy answer of 1 locates the position; future tokens are causally masked and the true target is never provided. Primary free-generation predictions and checkpoint-selection criteria are unchanged. Final evaluations must be rerun with this diagnostic implementation.

- Smoke v2 completed on a new free T4: 4 optimizer steps, 4.81 seconds training, 1.37 GB peak allocated training memory. Nonzero finite adapter gradients and loss-mask assertions passed. Separate-process loading reproduced all 8 validation outputs exactly. Smoke scores are not a substantive performance estimate.
- Since memory use was small, use batch 8 with accumulation 1 for the main experiment, preserving the planned effective batch size of 8. Keep 250 optimizer steps and 2,000 training examples; validate at steps 125 and 250.

- The initial strict parser treated `2.` as invalid. Inspection of smoke validation outputs showed terminal punctuation was the only format issue. Before any test-set evaluation, define parsing as one in-range integer with an optional terminal full stop; no prose, extraction from explanations, or target-dependent correction. Re-evaluate the baseline under this frozen protocol.
- Separate training, data-selection and evaluation-permutation seeds. Repeating a training seed must keep the same data manifests and evaluation numbering.
- Initial FP16 backward verification failed before any optimizer update. Reduce AMP's initial scale from its default 65,536 to 128 and preserve a gradient diagnostic before asserting finite nonzero gradients. The numerical cause is not yet confirmed; no successful training is claimed at this point.
- A source replacement upload stalled until the free Colab runtime expired. Recovered a new free T4 session. Use uniquely named uploads and observe the upload asynchronously to avoid blocking on an overwrite dialog. The new session redownloads and checksum-verifies the archive.

- 2026-10-22: Allocated a free Colab Tesla T4 (15,360 MiB); Python 3.13.15, torch 2.11.0+cu130, Transformers 5.20.1, PEFT 0.21.2. No model execution takes place on the local machine.
- The initial pretrained smoke run returned no valid integer on eight validation images. This is a format-compliance observation on a tiny smoke subset, not a grounding-performance conclusion.
- Added a secondary restricted-next-token diagnostic over the candidate digit tokens, with the same prompt/image and no extra target input. It separates number-format compliance from relative candidate preference. Free-generation accuracy remains the primary metric. Restricted probabilities are normalized among digit choices and are not calibrated confidence.
- PEFT 0.21.2 rejected preinstalled torchao 0.10.0. Torchao is not used by this full-precision LoRA experiment; remove that optional incompatible package in the disposable Colab environment. This fix must appear in the final notebook setup.
