# Measured results

Generated from saved per-example predictions. These are fixed filtered subsets of RoboRefIt, not full benchmark scores.

Hardware: **Tesla T4**. Selection used validation only; selected optimizer step: **250**.

| Split | Model | n | Accuracy | Wilson 95% CI | Invalid | Random expectation |
|---|---|---:|---:|---|---:|---:|
| val | pretrained | 128 | 25.0% | 18.3%–33.2% | 0.0% | 20.0% |
| val | LoRA | 128 | 41.4% | 33.2%–50.1% | 0.0% | 20.0% |
| testA | pretrained | 200 | 29.0% | 23.2%–35.6% | 0.0% | 22.7% |
| testA | LoRA | 200 | 31.0% | 25.0%–37.7% | 0.0% | 22.7% |
| testB | pretrained | 200 | 17.5% | 12.9%–23.4% | 0.0% | 20.5% |
| testB | LoRA | 200 | 27.5% | 21.8%–34.1% | 0.0% | 20.5% |

The paired accuracy differences (LoRA minus pretrained) and bootstrap intervals are saved in comparison.json. Validation scores participate in checkpoint selection and should not be treated as an unbiased final estimate.

| Split | Model | Mean latency (s) | p95 latency (s) | Peak allocated GPU (GiB) |
|---|---|---:|---:|---:|
| val | pretrained | 0.298 | 0.311 | 1.01 |
| val | LoRA | 0.340 | 0.362 | 1.01 |
| testA | pretrained | 0.299 | 0.317 | 1.01 |
| testA | LoRA | 0.339 | 0.362 | 1.01 |
| testB | pretrained | 0.301 | 0.317 | 1.01 |
| testB | LoRA | 0.337 | 0.358 | 1.01 |

![Accuracy and uncertainty](accuracy.png)

![Training loss](training.png)

Training time excluding validation: 604.1 s. Peak training allocated GPU memory: 2.96 GiB.

Intervals measure uncertainty on the selected examples; they do not include domain shift or variation across training seeds. See strata.csv for support counts by scene, candidate count, spatial keywords and repeated annotated classes. Small strata are descriptive only.

## Controlled robustness

| Split | Variant | n | Accuracy | Change from original | Same valid object |
|---|---|---:|---:|---:|---:|
| testA | paraphrase | 200 | 32.5% | +1.5% | 85.0% |
| testA | permutation | 200 | 35.5% | +4.5% | 46.0% |
| testA | visual | 200 | 32.0% | +1.0% | 82.0% |
| testB | paraphrase | 200 | 27.0% | -0.5% | 84.0% |
| testB | permutation | 200 | 26.0% | -1.5% | 39.0% |
| testB | visual | 200 | 25.0% | -2.5% | 78.0% |

Paraphrase changes only the instruction wrapper. Permutation renumbers the same boxes. Visual perturbation reduces image brightness and contrast before labels are drawn. Consistency compares underlying object indices, not displayed candidate numbers. These are limited controlled tests, not comprehensive robustness guarantees.

## Representative cases

The first two examples in each outcome category are selected deterministically. Images are rendered from downloaded source data inside the notebook.

| Split | Outcome | Example | Instruction | Expected ID | Base | LoRA |
|---|---|---|---|---:|---|---|
| val | fixed | train:9364 | I would like to grasp the shampoo in a cylindrical bottle on the middle shelf | 3 | 5. | 3 |
| val | fixed | train:5380 | I want to get the white bottled body milk on the bottom right shelf | 1 | 5. | 1 |
| val | regressed | train:9183 | grasp the facial cleanser in a white cylindrical bottle on the middle shelf | 2 | 2. | 4 |
| val | regressed | train:8733 | could you fetch me the white bottled facial cleanser on the bottom shelf | 5 | 5. | 4 |
| val | both_wrong | train:5467 | could you give me the facial cleanser with a black handy pump on the middle shelf | 5 | 1. | 1 |
| val | both_wrong | train:5358 | will you please give me the large bottled mouthwash on the middle shelf | 3 | 5. | 4 |
| val | both_correct | train:4594 | could you pass me the standing facial cleanser on the bottom shelf near the left | 2 | 2. | 2 |
| val | both_correct | train:9591 | I would like to grasp the pepsi cola can on the bottom shelf near the right | 4 | 4. | 4 |
| testA | fixed | testA:1642 | I need to pick up the small blue cuboid box on the top shelf | 2 | 1. | 2 |
| testA | fixed | testA:6415 | would you bring me the screwdriver | 3 | 1. | 3 |
| testA | regressed | testA:5045 | fetch the white toy polar bear | 1 | 1. | 4 |
| testA | regressed | testA:6813 | I hope to grab the green and white medicine box | 1 | 1. | 4 |
| testA | both_wrong | testA:6476 | will you please bring me the sachima which is a cuboid dessert | 2 | 4. | 3 |
| testA | both_wrong | testA:5265 | would you please fetch me the black wireless mouse at the top | 2 | 3. | 3 |
| testA | both_correct | testA:4456 | I want to grab the purple vitamin bottle | 1 | 1. | 1 |
| testA | both_correct | testA:7956 | grasp the body milk with a red bottle cap on the right | 1 | 1. | 1 |
| testB | fixed | testB:3034 | could you find the light brown lion | 3 | 1. | 3 |
| testB | fixed | testB:2481 | grab the screwdriver on the left | 3 | 1. | 3 |
| testB | regressed | testB:492 | I want to grasp the yellow glue stick | 1 | 1. | 4 |
| testB | regressed | testB:1725 | will you please hand me the coconut juice on the top shelf near the left | 5 | 5. | 3 |
| testB | both_wrong | testB:2881 | bring me the toy camel | 5 | 1. | 4 |
| testB | both_wrong | testB:456 | I would like to get the blue umbrella | 2 | 1. | 1 |
| testB | both_correct | testB:777 | please pass me the blue cone | 5 | 5. | 5 |
| testB | both_correct | testB:2669 | I hope to get the thermometer | 1 | 1. | 1 |

## Limits

Candidates are supplied from annotations; no autonomous detection or control is demonstrated. Validation holds out the shelf category. Exact pixel duplicates are audited, but physical scene independence cannot be guaranteed by the available metadata. Annotation errors and candidate filtering affect the distribution. Negative results are retained.
