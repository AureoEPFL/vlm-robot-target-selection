"""Target-independent rendering and deterministic candidate permutations."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont


def stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def candidate_order(image_id: str, count: int, seed: int, epoch: int = 0) -> list[int]:
    order = list(range(count))
    random.Random(stable_seed(image_id, seed, epoch)).shuffle(order)
    return order


def annotate(image: Image.Image, boxes: list, order: list[int], size: int = 512,
             perturb: bool = False) -> Image.Image:
    """No target or instruction is accepted by this function."""
    image = image.convert("RGB")
    if perturb:
        image = ImageEnhance.Brightness(image).enhance(.8)
        image = ImageEnhance.Contrast(image).enhance(.9)
    scale = size / max(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default(size=20)
    for number, index in enumerate(order, 1):
        x1, y1, x2, y2 = [round(v * scale) for v in boxes[index]]
        draw.rectangle((x1, y1, x2, y2), outline="#ffb000", width=3)
        label = str(number)
        right = min(image.width, x1 + 27)
        bottom = min(image.height, y1 + 27)
        draw.rectangle((max(0, right-27), max(0, bottom-27), right, bottom), fill="#ffb000")
        draw.text((max(0, right-24), max(0, bottom-26)), label, font=font, fill="black")
    return image


def make_input(row: dict, data_root: str | Path, seed: int, epoch: int = 0,
               size: int = 512, variant: str = "original") -> tuple[Image.Image, str, int, list[int]]:
    permutation_seed = seed + (100003 if variant == "permutation" else 0)
    order = candidate_order(row["image_id"], len(row["boxes"]), permutation_seed, epoch)
    with Image.open(Path(data_root) / row["image_path"]) as original:
        image = annotate(original, row["boxes"], order, size, variant == "visual")
    if variant == "paraphrase":
        question = f'Which numbered object matches this description: "{row["text"]}"?'
    else:
        question = f'Select the object described by: "{row["text"]}".'
    prompt = (f'The image contains {len(order)} candidate objects marked with numbered boxes. '
              f'{question} Reply with only one candidate number from 1 to {len(order)}.')
    answer = order.index(row["target"]) + 1
    return image, prompt, answer, order
