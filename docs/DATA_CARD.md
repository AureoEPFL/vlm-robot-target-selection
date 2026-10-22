# Data provenance and scope

The authoritative source is the [VL-Grasp author repository](https://github.com/luyh20/VL-Grasp). Its RoboRefIt download points to Google Drive file ID `1pdGF1HaU_UiKfh5Z618hy3nRjVbq_VuW`. The older dataset homepage link returned 404 during access on 2026-10-22. A Hugging Face mirror containing only 2,000 test rows was rejected because it did not preserve verifiable training partitions.

The downloaded `final_dataset.rar` is **6,084,791,269 bytes**, SHA-256:

```text
26cbcb6b5a1b80965d68a59b20cd1fcbd0726f909199119245cdeb1b2f67394e
```

Inspection of the actual JSON files found:

| Original split | Images | Referring expressions |
|---|---:|---:|
| train | 7,929 | 36,915 |
| testA | 1,859 | 8,523 |
| testB | 1,083 | 5,320 |

These counts come from the archive, not from reconciling inconsistent headline totals in secondary descriptions. Annotation fields include `num`, `text`, `bbox`, `rgb_path`, `depth_path`, `mask_path`, `scene`, and `class`. RGB paths use Windows separators; the loader normalizes them for Colab. Depth and mask data are not used.

The archive listing contained three JSON files and no separate README, text license, or dataset terms file. The project website's source license and upstream software license do not establish a license for every image. This repository therefore distributes code, configurations, identifiers and aggregate measurements; it downloads source data from the author rather than repackaging it. Image galleries are generated within the notebook from the user's downloaded copy.

## Known limitations

The authors' issue tracker reports reversed bounding-box coordinates and some incorrect or ambiguous expressions. Coordinate repairs are mechanical and logged. Filtering is fixed before evaluating model predictions; test failures are not removed after observing performance.

Object candidates come from annotations. Unannotated distractors are not candidates. The candidate-count filter changes the evaluated distribution. Exact pixel hashing detects exact duplicates only, not nearby camera views or related capture sessions. A held-out `shelf` validation category avoids category sharing between training and validation, but does not prove physical independence in the official test partitions.

The manifests and `data_audit.json` are the definitive records of the subset actually used, its exclusions, corrections, and split assignment.
