"""RoboRefIt provenance, conservative candidate construction, and split audit."""
from __future__ import annotations

import hashlib
import fnmatch
import json
import math
import random
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image
from .render import stable_seed

DRIVE_ID = "1pdGF1HaU_UiKfh5Z618hy3nRjVbq_VuW"
ARCHIVE_SHA256 = "26cbcb6b5a1b80965d68a59b20cd1fcbd0726f909199119245cdeb1b2f67394e"
SOURCE_URL = "https://github.com/luyh20/VL-Grasp"
PARTITIONS = ("train", "testA", "testB")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(root: str | Path) -> Path:
    import gdown
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "final_dataset.rar"
    if not archive.exists():
        temporary = root / "final_dataset.rar.partial"
        print("Downloading the author archive (~6.1 GB)...", flush=True)
        gdown.download(id=DRIVE_ID, output=str(temporary), quiet=True)
        temporary.replace(archive)
    archive_hash = sha256(archive)
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError("Archive checksum differs from the verified author release.")
    listed = subprocess.run(["unrar", "lb", str(archive)], check=True, capture_output=True, text=True)
    for name in listed.stdout.splitlines():
        path = PurePosixPath(name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or ":" in name:
            raise ValueError("Unsafe archive member")
    expected = [name.replace("\\", "/") for name in listed.stdout.splitlines()
                if not name.endswith("/") and
                (fnmatch.fnmatch(name.replace("\\", "/"), "final_dataset/*/image/*") or
                 fnmatch.fnmatch(name.replace("\\", "/"), "final_dataset/*/*.json"))]
    extraction = subprocess.run(["unrar", "x", "-o-", "-idq", str(archive),
                                 "final_dataset/*/image/*", "final_dataset/*/*.json", str(root) + "/"])
    # unrar returns 10 when -o- skips every existing file. Accept this only
    # after checking the complete expected RGB/annotation member inventory.
    if extraction.returncode not in (0, 10):
        extraction.check_returncode()
    missing = [name for name in expected if not (root / name).is_file()]
    if not expected or missing:
        raise RuntimeError(f"Incomplete extraction: {len(missing)} missing files; {missing[:5]}")
    provenance = {"source": SOURCE_URL, "drive_id": DRIVE_ID,
                  "archive_bytes": archive.stat().st_size, "archive_sha256": archive_hash,
                  "annotation_sha256": {part: sha256(root / "final_dataset" / part / f"roborefit_{part}.json")
                                        for part in PARTITIONS}}
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return root


def box_iou(a, b) -> float:
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return area / max(1e-8, (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area)


def normalize_box(box, width: int, height: int) -> tuple[tuple, bool]:
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("Invalid box")
    x1, y1, x2, y2 = map(float, box)
    result = (max(0, min(width, min(x1, x2))), max(0, min(height, min(y1, y2))),
              max(0, min(width, max(x1, x2))), max(0, min(height, max(y1, y2))))
    if result[2] - result[0] < 3 or result[3] - result[1] < 3:
        raise ValueError("Degenerate box")
    return result, result != tuple(box)


def build_partition(root: Path, part: str, maximum: int) -> tuple[list[dict], dict]:
    annotations = json.loads((root / "final_dataset" / part / f"roborefit_{part}.json").read_text())
    by_image = defaultdict(list)
    for row in annotations:
        by_image[row["rgb_path"].replace("\\", "/")].append(row)
    counts = Counter()
    rows, excluded, corrections = [], [], []
    for image_number, (path, records) in enumerate(sorted(by_image.items()), 1):
        if image_number % 2000 == 0:
            print(f"Auditing {part}: {image_number}/{len(by_image)} images", flush=True)
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise ValueError("Unsafe image path")
        image_id = part + ":" + Path(path).stem
        try:
            with Image.open(root / path) as image:
                width, height = image.size
                pixel_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            normalized = [normalize_box(r["bbox"], width, height) for r in records]
            boxes = sorted(set(box for box, _ in normalized))
            if not 2 <= len(boxes) <= maximum:
                raise ValueError("candidate_count")
            if any(box_iou(a, b) >= .9 for i, a in enumerate(boxes) for b in boxes[i+1:]):
                raise ValueError("overlapping_candidates")
            scenes = {r["scene"] for r in records}
            if len(scenes) != 1:
                raise ValueError("inconsistent_scene")
        except (ValueError, OSError) as error:
            reason = str(error)
            counts[reason] += len(records)
            excluded.append({"image_id": image_id, "reason": reason, "references": len(records)})
            continue
        classes = defaultdict(set)
        query_targets = defaultdict(set)
        for record, (box, corrected) in zip(records, normalized, strict=True):
            classes[box].add(str(record.get("class", "unknown")).lower())
            query_targets[record["text"].strip().lower()].add(box)
            if corrected:
                corrections.append({"id": f'{part}:{record["num"]}', "old": record["bbox"], "new": box})
        class_list = [sorted(classes[b]) for b in boxes]
        repeated_class = any(set(a) & set(b) - {"unknown"} for i, a in enumerate(class_list) for b in class_list[i+1:])
        for record, (box, _) in zip(records, normalized, strict=True):
            text = record["text"].strip()
            if not text or len(query_targets[text.lower()]) != 1:
                counts["ambiguous_or_empty_expression"] += 1
                excluded.append({"id": f'{part}:{record["num"]}', "reason": "ambiguous_or_empty_expression"})
                continue
            rows.append({"id": f'{part}:{record["num"]}', "image_id": image_id,
                         "official_split": part, "image_path": path, "pixel_sha256": pixel_hash,
                         "scene": record["scene"], "text": text, "boxes": boxes,
                         "target": boxes.index(box), "classes": class_list,
                         "spatial": bool(re.search(r"\b(left|right|top|bottom|above|below|behind|front|between|closest|nearest|middle)\b", text.lower())),
                         "similar_objects": bool(repeated_class)})
    return rows, {"original_references": len(annotations), "original_images": len(by_image),
                  "retained_references": len(rows), "excluded_counts": dict(counts),
                  "excluded": excluded, "box_corrections": corrections}


def sample_images(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """One deterministic random reference per image; no target-conditioned sampling."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_id"]].append(row)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    result = []
    for key in keys[:limit]:
        candidates = sorted(groups[key], key=lambda r: r["id"])
        result.append(random.Random(stable_seed(key, seed)).choice(candidates))
    return result


def prepare(root: str | Path, output: str | Path, config: dict) -> dict:
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    partitions, audit = {}, {}
    for part in PARTITIONS:
        partitions[part], audit[part] = build_partition(root, part, config["max_candidates"])
    test_hashes = {r["pixel_sha256"] for p in ("testA", "testB") for r in partitions[p]}
    overlap = [r["id"] for r in partitions["train"] if r["pixel_sha256"] in test_hashes]
    original_train = [r for r in partitions["train"] if r["pixel_sha256"] not in test_hashes]
    validation_scenes = set(config.get("validation_scenes", ["shelf"]))
    train = [r for r in original_train if r["scene"] not in validation_scenes]
    val = [r for r in original_train if r["scene"] in validation_scenes]
    val_hashes = {r["pixel_sha256"] for r in val}
    train = [r for r in train if r["pixel_sha256"] not in val_hashes]
    assert train and val
    assert not ({r["scene"] for r in train} & {r["scene"] for r in val})
    data_seed = config.get("data_seed", 42)
    manifests = {"train": sample_images(train, config["train_limit"], data_seed),
                 "val": sample_images(val, config["val_limit"], data_seed)}
    for part in ("testA", "testB"):
        manifests[part] = sample_images(partitions[part], config["test_limit"], data_seed)
    audit["protocol"] = {"validation_scenes": sorted(validation_scenes), "train_references_removed_for_test_pixel_overlap": overlap,
                         "sampling": "one random reference per sampled image", "selection_seed": data_seed,
                         "scene_limitation": "scene is a support-surface category, not a physical capture/session identifier. Train/validation categories are disjoint; physical train/test independence cannot be proven from released metadata.",
                         "selected_counts": {p: len(r) for p, r in manifests.items()}}
    for part, rows in manifests.items():
        (output / f"{part}.json").write_text(json.dumps(rows, indent=2))
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2))
    return audit["protocol"]
