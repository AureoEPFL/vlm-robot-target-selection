# Adapter model card

## Task and input contract

The model selects one supplied candidate object from an RGB scene using a referring expression. Draw all candidate boxes uniformly, assign numbers with the documented permutation, and use the prompt in `render.py`. A valid output is one number in range, optionally followed by a full stop. Map this number back through the candidate permutation to obtain the selected box.

The adapter is not an object detector or an action policy. It does not predict grasps, trajectories, depth, or motor commands. Use outside the supplied-candidate protocol is unmeasured.

## Base model and adaptation

- Base: `HuggingFaceTB/SmolVLM-500M-Instruct`.
- Immutable revision: `a7da5b986cb59b408707209984f360a5f4ad7e47`.
- Trainable parameters: 1,638,400 language-attention LoRA parameters.
- Target projections: q, k, v, o in the language backbone; frozen vision encoder and original backbone weights.
- Rank 8, alpha 16, dropout 0.05; AdamW, learning rate 0.0002, weight decay 0.01.
- Effective batch size 8; 250 optimizer steps over 2,000 sampled training images.
- FP16 with loss scaling, SDPA attention, 512-pixel longest image side, image splitting disabled.
- The processor and tokenizer are loaded from the pinned base revision; only the adapter weights and configuration are needed from the checkpoint archive.

## Selection and evidence

Checkpoints at steps 125 and 250 are compared on 128 validation images from the held-out shelf category. Both training seeds selected step 250. Seed 42 is the prespecified primary model; seed 43 is a repeatability check. No test result selects the seed or configuration.

Consult `RESULTS.md` and `results/` for measured performance, uncertainty, resource usage, per-example outputs, robustness checks and training logs. Validation results are involved in selection and are not an unbiased final performance estimate.

## Limitations

Candidate boxes come from annotations, so results assume candidate coverage. Filtering changes the evaluation population. Scene metadata describes support categories rather than physical capture sessions; exact pixel hashes cannot rule out near-duplicate images or broader scene overlap. Dataset authors' issue reports describe annotation noise. Spatial and similar-object strata are automatic proxies. Two seeds cannot estimate training variability reliably.

Free-generation accuracy is the primary metric. Normalized candidate-digit probabilities are a diagnostic, not calibrated confidence. The model is not validated for autonomous robotic operation or safety-critical decisions.

## Licenses

Original project code: MIT. Base model: Apache-2.0. Dataset-image redistribution terms were not established from the archive; no source images are included in the repository. The notebook retrieves the author-provided dataset and renders examples in the user's runtime. See `DATA_CARD.md` for provenance.
