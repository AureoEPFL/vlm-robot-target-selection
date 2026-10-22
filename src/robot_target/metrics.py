"""Metrics recomputable from saved per-example predictions; no ML dependencies."""
from __future__ import annotations

import math
import random
import re
from collections import defaultdict


def parse_answer(text: str, count: int) -> int | None:
    """Accept one number, optionally with a terminal full stop; no explanations."""
    value = text.strip()
    if not re.fullmatch(r"[1-9][0-9]*\.?", value):
        return None
    number = int(value.rstrip("."))
    return number if 1 <= number <= count else None


def wilson(correct: int, total: int) -> list[float]:
    if not total:
        return [0.0, 1.0]
    z, p = 1.959963984540054, correct / total
    denominator = 1 + z * z / total
    middle = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [max(0.0, middle - radius), min(1.0, middle + radius)]


def summarize(rows: list[dict], bootstrap: int = 1000) -> dict:
    if not rows:
        return {"n": 0}
    correct = sum(bool(r["correct"]) for r in rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_id"]].append(int(row["correct"]))
    rng = random.Random(1701)
    clusters = list(groups.values())
    draws = []
    for _ in range(bootstrap):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        draws.append(sum(map(sum, sampled)) / sum(map(len, sampled)))
    draws.sort()
    latencies = sorted(float(r["latency_s"]) for r in rows)
    result = {
        "n": len(rows), "images": len(groups), "correct": correct,
        "accuracy": correct / len(rows),
        "invalid_rate": sum(r["prediction"] is None for r in rows) / len(rows),
        "random_expected_accuracy": sum(1 / r["n_candidates"] for r in rows) / len(rows),
        "wilson_95": wilson(correct, len(rows)),
        "image_bootstrap_95": [draws[int(bootstrap * .025)], draws[min(bootstrap - 1, int(bootstrap * .975))]],
        "latency_mean_s": sum(latencies) / len(rows),
        "latency_median_s": latencies[len(latencies) // 2],
        "latency_p95_s": latencies[min(len(latencies)-1, int(.95 * len(latencies)))],
    }
    if all("restricted_correct" in row for row in rows):
        result["restricted_next_token_accuracy"] = sum(row["restricted_correct"] for row in rows) / len(rows)
    return result


def stratify(rows: list[dict]) -> dict:
    result = {}
    for field in ("scene", "n_candidates", "spatial", "similar_objects"):
        groups = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        result[field] = {key: summarize(value, 200) for key, value in sorted(groups.items())}
    return result
