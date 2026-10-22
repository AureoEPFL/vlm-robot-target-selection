"""Recompute released metrics and audit cross-file identities without a GPU."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_target.metrics import parse_answer, summarize


def main() -> None:
    results = ROOT / "results/experiment"
    assert results.is_dir(), "Extract the measured experiment artifacts first."
    checked = 0
    for directory in ("final-baseline", "final-adapted", "final-repeat"):
        for path in sorted((results / directory).rglob("predictions.json")):
            split = path.parent.parent.name
            manifest = json.loads((results / "main-manifests" / f"{split}.json").read_text())
            examples = {row["id"]: row for row in manifest}
            rows = json.loads(path.read_text())
            assert len(rows) == len(examples)
            assert {row["id"] for row in rows} == examples.keys()
            for row in rows:
                example = examples[row["id"]]
                order = row["candidate_order"]
                assert sorted(order) == list(range(len(example["boxes"])))
                assert row["answer"] == order.index(example["target"]) + 1
                assert row["prediction"] == parse_answer(row["raw_output"], len(order))
                assert row["correct"] == (row["prediction"] == row["answer"])
                predicted_object = order[row["prediction"] - 1] if row["prediction"] else None
                assert row["predicted_object"] == predicted_object
            actual = summarize(rows)
            saved = json.loads(path.with_name("metrics.json").read_text())
            for field in ("n", "correct", "accuracy", "invalid_rate", "random_expected_accuracy", "wilson_95", "image_bootstrap_95"):
                left = actual[field] if isinstance(actual[field], list) else [actual[field]]
                right = saved[field] if isinstance(saved[field], list) else [saved[field]]
                assert all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
                           for a, b in zip(left, right, strict=True)), (str(path), field)
            checked += len(rows)
    for directory in ("main-lora", "main-repeat"):
        run = results / directory
        summary = json.loads((run / "training_summary.json").read_text())
        history = json.loads((run / "training_log.json").read_text())
        candidates = [row for row in history if "validation_accuracy" in row]
        selected = max(candidates, key=lambda row: row["validation_accuracy"])
        assert selected["step"] == summary["selected_step"]
        assert selected["validation_accuracy"] == summary["best_validation_accuracy"]
        proof = json.loads((run / "reload_verification.json").read_text())
        assert proof["new_process"] and proof["passed"] and not proof["exact_output_mismatches"]
    print(f"Verified {checked} predictions, saved metrics, selected checkpoints and reload evidence.")


if __name__ == "__main__":
    main()
