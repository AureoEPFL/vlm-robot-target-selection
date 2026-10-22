"""Build static, auditable reports from persisted predictions, never invented scores."""
from __future__ import annotations

import csv
import html
import json
import random
from pathlib import Path

from .metrics import summarize


def paired_interval(base: list, adapted: list, repeats=2000) -> dict:
    left = {r["id"]: r for r in base}
    right = {r["id"]: r for r in adapted}
    assert left.keys() == right.keys()
    differences = [int(right[key]["correct"]) - int(left[key]["correct"]) for key in sorted(left)]
    rng = random.Random(2026)
    estimates = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(repeats))
    return {"paired_accuracy_difference": sum(differences) / len(differences),
            "paired_bootstrap_95": [estimates[int(.025*repeats)], estimates[int(.975*repeats)]]}


def build_report(baseline_dir, adapted_dir, train_dir, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline_dir, adapted_dir, train_dir, output = map(Path, (baseline_dir, adapted_dir, train_dir, output))
    output.mkdir(parents=True, exist_ok=True)
    measurements, comparisons, cases, strata_rows, resources = [], {}, [], [], []
    for split in ("val", "testA", "testB"):
        paired = {}
        for model_name, directory in (("pretrained", baseline_dir), ("LoRA", adapted_dir)):
            path = directory / split / "original/predictions.json"
            if not path.exists():
                continue
            rows = json.loads(path.read_text())
            metrics = summarize(rows)
            stored = json.loads(path.with_name("metrics.json").read_text())
            for key in ("n", "correct", "accuracy", "invalid_rate", "random_expected_accuracy"):
                assert metrics[key] == stored[key], f"Saved metric mismatch: {path} / {key}"
            measurements.append(dict(split=split, model=model_name, **metrics))
            resources.append({"split": split, "model": model_name, "latency_mean_s": stored["latency_mean_s"],
                              "latency_p95_s": stored["latency_p95_s"],
                              "peak_gpu_allocated_GiB": stored["peak_gpu_allocated_bytes"]/2**30})
            for field, groups in stored["strata"].items():
                for group, group_metrics in groups.items():
                    strata_rows.append({"split": split, "model": model_name, "field": field,
                                        "group": group, "n": group_metrics["n"],
                                        "accuracy": group_metrics["accuracy"],
                                        "invalid_rate": group_metrics["invalid_rate"]})
            paired[model_name] = rows
        if len(paired) == 2:
            comparisons[split] = paired_interval(paired["pretrained"], paired["LoRA"])
            bmap = {r["id"]: r for r in paired["pretrained"]}
            for category, condition in (("fixed", lambda a, b: a and not b),
                                        ("regressed", lambda a, b: b and not a),
                                        ("both_wrong", lambda a, b: not a and not b),
                                        ("both_correct", lambda a, b: a and b)):
                selected = [r for r in paired["LoRA"] if condition(r["correct"], bmap[r["id"]]["correct"])][:2]
                cases.extend(dict(split=split, category=category, **row,
                                  baseline_output=bmap[row["id"]]["raw_output"]) for row in selected)
    assert measurements, "No saved predictions exist."
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measurements[0]))
        writer.writeheader(); writer.writerows(measurements)
    (output / "comparison.json").write_text(json.dumps({"measurements": measurements, "paired_comparisons": comparisons}, indent=2))
    (output / "representative_cases.json").write_text(json.dumps(cases, indent=2))
    for filename, table in (("strata.csv", strata_rows), ("resources.csv", resources)):
        with (output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(table)
    robustness = []
    for split in ("testA", "testB"):
        original_path = adapted_dir / split / "original/predictions.json"
        if not original_path.exists():
            continue
        original = json.loads(original_path.read_text())
        original_by_id = {r["id"]: r for r in original}
        for variant in ("paraphrase", "permutation", "visual"):
            path = adapted_dir / split / variant / "predictions.json"
            if not path.exists():
                continue
            perturbed = json.loads(path.read_text())
            scores = summarize(perturbed)
            consistency = sum(r["predicted_object"] is not None and r["predicted_object"] == original_by_id[r["id"]]["predicted_object"] for r in perturbed) / len(perturbed)
            robustness.append({"split": split, "variant": variant, **scores,
                               "same_valid_object_rate": consistency, **paired_interval(original, perturbed)})
    (output / "robustness.json").write_text(json.dumps(robustness, indent=2))
    splits = sorted({m["split"] for m in measurements})
    figure, axes = plt.subplots(1, len(splits), figsize=(4 * len(splits), 4), squeeze=False)
    for axis, split in zip(axes[0], splits, strict=True):
        selected = [m for m in measurements if m["split"] == split]
        axis.bar([m["model"] for m in selected], [m["accuracy"] for m in selected], color=["#727e8b", "#197c80"][:len(selected)])
        for i, m in enumerate(selected):
            low, high = m["wilson_95"]
            axis.errorbar(i, m["accuracy"], yerr=[[m["accuracy"]-low], [high-m["accuracy"]]], color="black", capsize=5)
        axis.axhline(selected[0]["random_expected_accuracy"], color="#c86e28", linestyle="--", label="Uniform random")
        axis.set(ylim=(0, 1), title=f'{split} · n={selected[0]["n"]}', ylabel="Selection accuracy")
        axis.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(output / "accuracy.png", dpi=180); plt.close(figure)
    training = json.loads((train_dir / "training_summary.json").read_text())
    training_log = json.loads((train_dir / "training_log.json").read_text())
    figure, axis = plt.subplots(figsize=(7, 3))
    axis.plot([r["step"] for r in training_log], [r["loss"] for r in training_log], color="#197c80")
    axis.set(xlabel="Optimizer step", ylabel="Answer-token cross-entropy", title="Observed training loss")
    figure.tight_layout(); figure.savefig(output / "training.png", dpi=180); plt.close(figure)
    environment = json.loads((train_dir / "environment.json").read_text())
    lines = ["# Measured results", "", "Generated from saved per-example predictions. These are fixed filtered subsets of RoboRefIt, not full benchmark scores.", "",
             f'Hardware: **{environment["gpu"]}**. Selection used validation only; selected optimizer step: **{training["selected_step"]}**.', "",
             "| Split | Model | n | Accuracy | Wilson 95% CI | Invalid | Random expectation |", "|---|---|---:|---:|---|---:|---:|"]
    for row in measurements:
        lo, hi = row["wilson_95"]
        lines.append(f'| {row["split"]} | {row["model"]} | {row["n"]} | {row["accuracy"]:.1%} | {lo:.1%}–{hi:.1%} | {row["invalid_rate"]:.1%} | {row["random_expected_accuracy"]:.1%} |')
    lines += ["", "The paired accuracy differences (LoRA minus pretrained) and bootstrap intervals are saved in comparison.json. Validation scores participate in checkpoint selection and should not be treated as an unbiased final estimate.", "",
              "| Split | Model | Mean latency (s) | p95 latency (s) | Peak allocated GPU (GiB) |", "|---|---|---:|---:|---:|"]
    for row in resources:
        lines.append(f'| {row["split"]} | {row["model"]} | {row["latency_mean_s"]:.3f} | {row["latency_p95_s"]:.3f} | {row["peak_gpu_allocated_GiB"]:.2f} |')
    lines += ["", "![Accuracy and uncertainty](accuracy.png)", "", "![Training loss](training.png)", "",
              f'Training time excluding validation: {training["training_seconds_excluding_validation"]:.1f} s. Peak training allocated GPU memory: {training["peak_training_gpu_allocated_bytes"]/2**30:.2f} GiB.', "",
              "Intervals measure uncertainty on the selected examples; they do not include domain shift or variation across training seeds. See strata.csv for support counts by scene, candidate count, spatial keywords and repeated annotated classes. Small strata are descriptive only.", ""]
    if robustness:
        lines += ["## Controlled robustness", "", "| Split | Variant | n | Accuracy | Change from original | Same valid object |", "|---|---|---:|---:|---:|---:|"]
        for row in robustness:
            lines.append(f'| {row["split"]} | {row["variant"]} | {row["n"]} | {row["accuracy"]:.1%} | {row["paired_accuracy_difference"]:+.1%} | {row["same_valid_object_rate"]:.1%} |')
        lines += ["", "Paraphrase changes only the instruction wrapper. Permutation renumbers the same boxes. Visual perturbation reduces image brightness and contrast before labels are drawn. Consistency compares underlying object indices, not displayed candidate numbers. These are limited controlled tests, not comprehensive robustness guarantees.", ""]
    lines += [
              "## Representative cases", "", "The first two examples in each outcome category are selected deterministically. Images are rendered from downloaded source data inside the notebook.", "",
              "| Split | Outcome | Example | Instruction | Expected ID | Base | LoRA |", "|---|---|---|---|---:|---|---|"]
    for row in cases:
        clean = lambda text: str(text).replace("|", "/").replace("\n", " ")
        lines.append(f'| {row["split"]} | {row["category"]} | {row["id"]} | {clean(row["instruction"])} | {row["answer"]} | {clean(row["baseline_output"])} | {clean(row["raw_output"])} |')
    lines += ["", "## Limits", "", "Candidates are supplied from annotations; no autonomous detection or control is demonstrated. Validation holds out the shelf category. Exact pixel duplicates are audited, but physical scene independence cannot be guaranteed by the available metadata. Annotation errors and candidate filtering affect the distribution. Negative results are retained.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    return measurements


def gallery(manifests, predictions, data_root, output, config, limit=12, baseline=None):
    """Generate a private/local image gallery, excluded from the public source bundle."""
    from .render import make_input
    from PIL import Image, ImageDraw
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = {r["id"]: r for r in json.loads(Path(manifests).read_text())}
    predictions = json.loads(Path(predictions).read_text())
    baseline = {r["id"]: r for r in json.loads(Path(baseline).read_text())} if baseline else {}
    failures = [r for r in predictions if not r["correct"]][:limit//2]
    successes = [r for r in predictions if r["correct"]][:limit-len(failures)]
    body = ["<!doctype html><meta charset='utf-8'><title>Grounding examples</title>",
            "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto}section{display:inline-block;vertical-align:top;width:48%;padding:1%}img{width:100%}</style>",
            "<h1>Selected successes and failures</h1><p>Candidate numbers are model inputs. Expected and predicted IDs below are evaluation annotations.</p>"]
    for i, prediction in enumerate(failures + successes):
        row = rows[prediction["id"]]
        image, _, answer, _ = make_input(row, data_root, config.get("eval_seed", 42), size=config["image_size"])
        panels = [("Pretrained", baseline[prediction["id"]]), ("LoRA", prediction)] if baseline else [("LoRA", prediction)]
        canvas = Image.new("RGB", (image.width * len(panels), image.height + 38), "white")
        with Image.open(Path(data_root) / row["image_path"]) as original:
            scale = config["image_size"] / max(original.size)
        for column, (name, result) in enumerate(panels):
            panel = image.copy()
            if result["predicted_object"] is not None:
                box = [round(v * scale) for v in row["boxes"][result["predicted_object"]]]
                ImageDraw.Draw(panel).rectangle(box, outline="#00a8ff", width=5)
            canvas.paste(panel, (column * image.width, 38))
            ImageDraw.Draw(canvas).text((column * image.width + 8, 10),
                                       f'{name}: {result["prediction"]} | expected: {answer} | correct: {result["correct"]}', fill="black")
        filename = f"example_{i:02d}.png"
        canvas.save(output / filename)
        baseline_text = f'; pretrained {baseline[prediction["id"]]["prediction"]}' if baseline else ""
        body.append(f'<section><img src="{filename}"><p>{html.escape(prediction["instruction"])}</p><p>Expected {answer}{baseline_text}; LoRA {html.escape(str(prediction["prediction"]))}; correct={prediction["correct"]}. Blue outlines show model selections.</p></section>')
    (output / "index.html").write_text("\n".join(body))
    return failures + successes


def seed_report(directories: dict[int, Path], output: Path):
    records = []
    lines = ["# Training-seed repeat", "", "The data manifests and evaluation numbering are fixed. Seed 42 remains the primary reported model; seed 43 is a variability check, not an opportunity to choose a seed after inspecting the tests.", "",
             "| Split | Training seed | n | Accuracy | Invalid |", "|---|---:|---:|---:|---:|"]
    for split in ("testA", "testB"):
        for seed, folder in sorted(directories.items()):
            predictions = json.loads((Path(folder) / split / "original/predictions.json").read_text())
            metrics = summarize(predictions)
            records.append({"split": split, "training_seed": seed, **metrics})
            lines.append(f'| {split} | {seed} | {metrics["n"]} | {metrics["accuracy"]:.1%} | {metrics["invalid_rate"]:.1%} |')
    lines += ["", "Two seeds provide a limited repeatability check. They do not establish a reliable distribution of training outcomes."]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "seeds.json").write_text(json.dumps(records, indent=2))
    (output / "SEEDS.md").write_text("\n".join(lines))
    return records
