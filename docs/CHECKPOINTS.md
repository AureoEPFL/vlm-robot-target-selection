# Trained adapters

The two adapter ZIPs are delivered separately from the source archive. They contain no base-model weights or source images. Seed 42 is the primary checkpoint; seed 43 is a repeatability check.

## Use in Colab

1. Open the main notebook and choose a T4 GPU.
2. Set `MODE = "inference"` and run all cells.
3. Upload `robot-target-seed42-adapter.zip` when requested. The notebook loads the pinned public base model and applies the exported LoRA weights.

Alternatively set `ADAPTER_ZIP_URL` to a release asset URL and set `ADAPTER_ZIP_SHA256` to the matching value below. No public release URL exists yet because the destination repository has not been selected. When publishing, attach these ZIPs as release assets and record their verified URLs here; keep weights out of Git.

## Verified files

### robot-target-seed42-adapter.zip

Size: 6,071,257 bytes.

Archive SHA-256:

```text
ff145e1ca11bd27584ffa5402745ed0ea8b6893c0189663932c60e6c8c470cb9
```

Contained `adapter/adapter_model.safetensors` SHA-256:

```text
936a598e098f057619bc3ffddd5a2a8013a4179a9496c4f89d1ee9a341802d86
```

### robot-target-seed43-adapter.zip

Size: 6,072,418 bytes.

Archive SHA-256:

```text
e44cc19751ffc99c28538628559068d229ea56d8b7ebf0debc545d946df077d8
```

Contained `adapter/adapter_model.safetensors` SHA-256:

```text
effe6109c1175934db1b84d2219c2e9626e2234b1819059cf0c42f2bbf7a3945
```

The contained tensor hashes were checked against the hashes recorded by the final evaluation processes. The archive includes `experiment_config.json`; the processor is reconstructed from the immutable base revision. See [MODEL_CARD.md](MODEL_CARD.md) and [the complete export manifest](../results/export_manifest.json).
