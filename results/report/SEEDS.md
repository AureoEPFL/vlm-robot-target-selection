# Training-seed repeat

The data manifests and evaluation numbering are fixed. Seed 42 remains the primary reported model; seed 43 is a variability check, not an opportunity to choose a seed after inspecting the tests.

| Split | Training seed | n | Accuracy | Invalid |
|---|---:|---:|---:|---:|
| testA | 42 | 200 | 31.0% | 0.0% |
| testA | 43 | 200 | 30.0% | 0.0% |
| testB | 42 | 200 | 27.5% | 0.0% |
| testB | 43 | 200 | 25.0% | 0.0% |

Two seeds provide a limited repeatability check. They do not establish a reliable distribution of training outcomes.