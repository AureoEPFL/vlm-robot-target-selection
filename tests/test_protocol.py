import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from robot_target.data import build_partition, normalize_box, sample_images, prepare
from robot_target.metrics import parse_answer, summarize
from robot_target.render import candidate_order, make_input


class ProtocolTests(unittest.TestCase):
    def test_strict_output_parser(self):
        for text in ("1 because", "object 2", "0", "9", "01", "1.0", "1\n2", ""):
            self.assertIsNone(parse_answer(text, 4))
        self.assertEqual(parse_answer(" 3\n", 4), 3)
        self.assertEqual(parse_answer("2.", 4), 2)

    def test_numbering_is_bijective_reproducible_and_variable(self):
        original = candidate_order("train:1", 8, 42)
        self.assertEqual(sorted(original), list(range(8)))
        self.assertEqual(original, candidate_order("train:1", 8, 42))
        self.assertNotEqual(original, candidate_order("train:1", 8, 43))

    def test_box_repair_is_geometric(self):
        box, changed = normalize_box([70, 80, -10, 20], 60, 60)
        self.assertEqual(box, (0, 20, 60, 60))
        self.assertTrue(changed)
        with self.assertRaises(ValueError):
            normalize_box([0, 0, float("nan"), 20], 60, 60)

    def test_candidates_and_rendering_do_not_depend_on_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "final_dataset/train"
            (folder / "image").mkdir(parents=True)
            Image.new("RGB", (100, 100), "white").save(folder / "image/000.png")
            common = {"rgb_path": "final_dataset\\train\\image\\000.png", "scene": "table", "class": "cup"}
            records = [dict(common, num=0, text="left cup", bbox=[5, 10, 30, 45]),
                       dict(common, num=1, text="right cup", bbox=[65, 10, 95, 45])]
            (folder / "roborefit_train.json").write_text(json.dumps(records))
            rows, _ = build_partition(root, "train", 8)
            self.assertEqual(rows[0]["boxes"], rows[1]["boxes"])
            a = make_input(rows[0], root, 42)
            b = make_input(rows[1], root, 42)
            self.assertEqual(a[0].tobytes(), b[0].tobytes())
            self.assertEqual(a[3], b[3])
            self.assertNotEqual(a[2], b[2])
            self.assertNotIn("000.png", a[1])
            self.assertEqual(len(sample_images(rows, 10, 42)), 1)

    def test_random_baseline_accounts_for_candidate_count(self):
        rows = [{"image_id": str(i), "correct": i == 0, "prediction": None if i else 1,
                 "n_candidates": n, "latency_s": .1} for i, n in enumerate([2, 4])]
        metrics = summarize(rows, 100)
        self.assertAlmostEqual(metrics["random_expected_accuracy"], .375)
        self.assertEqual(metrics["accuracy"], .5)
        self.assertEqual(metrics["invalid_rate"], .5)

    def test_split_holdout_and_duplicate_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for part, images in {
                "train": [("table", "red"), ("shelf", "blue"), ("table", "green")],
                "testA": [("table", "green")],
                "testB": [("drawer", "yellow")],
            }.items():
                folder = root / "final_dataset" / part
                (folder / "image").mkdir(parents=True)
                records = []
                for index, (scene, color) in enumerate(images):
                    image_path = folder / "image" / f"{index}.png"
                    Image.new("RGB", (100, 100), color).save(image_path)
                    for number, box in enumerate([[5, 10, 30, 45], [65, 10, 95, 45]]):
                        records.append({"num": 2*index+number, "scene": scene,
                                        "rgb_path": image_path.relative_to(root).as_posix(),
                                        "text": ["left object", "right object"][number], "bbox": box})
                (folder / f"roborefit_{part}.json").write_text(json.dumps(records))
            output = root / "manifests"
            protocol = prepare(root, output, {"max_candidates": 8, "seed": 42, "train_limit": 10, "val_limit": 10, "test_limit": 10})
            self.assertEqual(protocol["selected_counts"], {"train": 1, "val": 1, "testA": 1, "testB": 1})
            self.assertEqual(set(protocol["train_references_removed_for_test_pixel_overlap"]), {"train:4", "train:5"})
            train, val = [json.loads((output / f"{split}.json").read_text()) for split in ["train", "val"]]
            self.assertFalse({r["pixel_sha256"] for r in train} & {r["pixel_sha256"] for r in val})
            self.assertFalse({r["scene"] for r in train} & {r["scene"] for r in val})


if __name__ == "__main__":
    unittest.main()
