"""Remote GPU experiment runner with persisted predictions and adapter checkpoints."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import random
import time
from pathlib import Path

import torch

from .metrics import parse_answer, stratify, summarize
from .modeling import add_lora, encode, load_model, predict, seed_everything, to_device
from .render import make_input


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def environment() -> dict:
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "Pillow", "numpy", "gdown", "huggingface-hub"):
        packages[name] = importlib.metadata.version(name)
    return {"python": platform.python_version(), "packages": packages,
            "process_id": os.getpid(), "parent_process_id": os.getppid(),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "latency_protocol": "batch size 1, greedy generation, one warmup, CUDA synchronize; processor+generation included; image rendering, disk IO and separately timed canonical-digit diagnostic excluded",
            "restricted_diagnostic": "candidate digit logits after canonical assistant whitespace; fixed dummy digit used only to locate its causally preceding position"}


def evaluate(model, processor, rows, data_root, config, output: Path, variant="original") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    evaluation_seed = config.get("eval_seed", 42)
    image, prompt, _, _ = make_input(rows[0], data_root, evaluation_seed, size=config["image_size"], variant=variant)
    predict(model, processor, image, prompt, config["max_new_tokens"])
    torch.cuda.reset_peak_memory_stats()
    records = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        image, prompt, answer, order = make_input(row, data_root, evaluation_seed, size=config["image_size"], variant=variant)
        torch.cuda.synchronize()
        tic = time.perf_counter()
        text, diagnostic = predict(model, processor, image, prompt, config["max_new_tokens"], candidates=len(order))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - tic
        prediction = parse_answer(text, len(order))
        records.append({"id": row["id"], "image_id": row["image_id"], "scene": row["scene"],
                        "n_candidates": len(order), "spatial": row["spatial"], "similar_objects": row["similar_objects"],
                        "variant": variant, "instruction": row["text"], "candidate_order": order,
                        "answer": answer, "raw_output": text, "prediction": prediction,
                        "predicted_object": order[prediction-1] if prediction else None,
                        "correct": prediction == answer, "latency_s": elapsed - diagnostic["diagnostic_latency_s"],
                        "restricted_correct": diagnostic["restricted_prediction"] == answer, **diagnostic})
        if (index + 1) % 25 == 0:
            print(f"evaluate {variant}: {index+1}/{len(rows)}", flush=True)
    summary = summarize(records)
    summary.update({"peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "evaluation_seconds": time.perf_counter() - started,
                    "strata": stratify(records)})
    save_json(output / "predictions.json", records)
    save_json(output / "metrics.json", summary)
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({k: v for k, v in summary.items() if k != "strata"}), flush=True)
    return summary


def run_evaluation(config, manifests, data_root, output, adapter=None, splits=("val",), variants=("original",)):
    from .data import sha256
    seed_everything(config["seed"])
    model, processor = load_model(config, adapter)
    output = Path(output)
    save_json(output / "environment.json", environment())
    save_json(output / "config.json", config)
    save_json(output / "model_provenance.json", {
        "base_model": config["model_id"], "base_revision": config["model_revision"],
        "adapter_sha256": sha256(Path(adapter) / "adapter_model.safetensors") if adapter else None,
    })
    for split in splits:
        rows = json.loads((Path(manifests) / f"{split}.json").read_text())
        for variant in variants:
            evaluate(model, processor, rows, data_root, config, output / split / variant, variant)


def train(config, manifests, data_root, output):
    seed_everything(config["seed"])
    output, manifests = Path(output), Path(manifests)
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((manifests / "train.json").read_text())
    val_rows = json.loads((manifests / "val.json").read_text())
    model, processor = load_model(config)
    model, parameter_audit = add_lora(model, config)
    save_json(output / "config.json", config)
    save_json(output / "environment.json", environment())
    save_json(output / "parameter_audit.json", parameter_audit)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"], weight_decay=.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.)
    rng, indices, cursor, epoch = random.Random(config["seed"]), list(range(len(rows))), 0, 0
    rng.shuffle(indices)
    best_score, best_step, logs = -1., 0, []
    gradient_audit = None
    started, training_seconds, peak_training = time.perf_counter(), 0., 0
    for step in range(1, config["max_steps"] + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(config["gradient_accumulation"]):
            images, prompts, answers = [], [], []
            for _ in range(config["batch_size"]):
                if cursor == len(indices):
                    cursor, epoch = 0, epoch + 1
                    rng.shuffle(indices)
                row = rows[indices[cursor]]
                cursor += 1
                image, prompt, answer, _ = make_input(row, data_root, config["seed"], epoch, config["image_size"])
                images.append(image); prompts.append(prompt); answers.append(answer)
            batch = to_device(encode(processor, images, prompts, answers))
            if step == 1 and not losses:
                save_json(output / "loss_mask_audit.json", {
                    "supervised_text": [processor.tokenizer.decode(ids[ids != -100], skip_special_tokens=True)
                                        for ids in batch["labels"]],
                    "supervised_token_counts": (batch["labels"] != -100).sum(dim=1).tolist(),
                    "total_token_counts": batch["attention_mask"].sum(dim=1).tolist(),
                    "prefix_image_padding_excluded": True,
                })
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            losses.append(float(loss.detach()))
            scaler.scale(loss / config["gradient_accumulation"]).backward()
        scaler.unscale_(optimizer)
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if gradient_audit is None:
            gradients = {n: float(p.grad.float().norm()) for n, p in trainable if p.grad is not None}
            save_json(output / "gradient_probe.json", {"loss": losses, "scaler": scaler.get_scale(),
                                                       "gradient_norms": gradients})
            assert gradients and any(value > 0 for value in gradients.values()), "No nonzero LoRA gradients"
            assert all(math_value >= 0 and math_value < float("inf") for math_value in gradients.values())
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            gradient_audit = {"nonzero_lora_gradients": sum(v > 0 for v in gradients.values()),
                              "frozen_base_has_no_gradients": True, "gradient_norms": gradients}
            save_json(output / "gradient_audit.json", gradient_audit)
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - step_start
        peak_training = max(peak_training, torch.cuda.max_memory_allocated())
        log = {"step": step, "loss": sum(losses)/len(losses), "gradient_norm": float(norm),
               "epoch": epoch, "training_seconds": training_seconds}
        logs.append(log)
        if step == 1 or step % 10 == 0:
            print(json.dumps(log), flush=True)
        if step % config["eval_every"] == 0 or step == config["max_steps"]:
            scores = evaluate(model, processor, val_rows, data_root, config, output / f"validation_step_{step}")
            log["validation_accuracy"] = scores["accuracy"]
            if scores["accuracy"] > best_score:
                best_score, best_step = scores["accuracy"], step
                model.save_pretrained(output / "adapter", safe_serialization=True)
                processor.save_pretrained(output / "adapter")
                save_json(output / "reload_expected.json", json.loads((output / f"validation_step_{step}" / "predictions.json").read_text()))
            save_json(output / "training_log.json", logs)
    report = {"best_validation_accuracy": best_score, "selected_step": best_step,
              "training_seconds_excluding_validation": training_seconds,
              "wall_seconds_including_loading_excluded": time.perf_counter() - started,
              "peak_training_gpu_allocated_bytes": peak_training, "optimizer_steps": config["max_steps"]}
    save_json(output / "training_summary.json", report)
    print("TRAINING_COMPLETE", json.dumps(report), flush=True)


def verify_reload(config, manifests, data_root, run_dir):
    """CLI invocation starts a new Python process, so no trained weights survive."""
    run_dir = Path(run_dir)
    run_evaluation(config, manifests, data_root, run_dir / "reload", str(run_dir / "adapter"))
    expected = json.loads((run_dir / "reload_expected.json").read_text())
    actual = json.loads((run_dir / "reload/val/original/predictions.json").read_text())
    mismatches = [a["id"] for a, b in zip(expected, actual, strict=True)
                  if a["id"] != b["id"] or a["raw_output"] != b["raw_output"]]
    proof = {"new_process": True, "examples": len(actual), "exact_output_mismatches": mismatches,
             "passed": not mismatches, "reload_process_id": os.getpid(),
             "training_process_id": json.loads((run_dir / "environment.json").read_text()).get("process_id")}
    if proof["training_process_id"] is not None:
        assert proof["training_process_id"] != proof["reload_process_id"], "Reload must run in a separate process."
    save_json(run_dir / "reload_verification.json", proof)
    assert not mismatches, f"Adapter reload output mismatch: {mismatches}"
    print("RELOAD_VERIFIED", json.dumps(proof), flush=True)
