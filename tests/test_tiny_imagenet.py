import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from src.data.tiny_imagenet import TinyImageNet, resolve_tiny_imagenet_root
from src.data.prepare_tiny_imagenet import safe_extract


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)


class TinyImageNetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_root = Path(self.temp.name) / "data"
        self.root = self.data_root / "tiny-imagenet-200"
        self.root.mkdir(parents=True)
        (self.root / "wnids.txt").write_text("n0002\nn0001\n", encoding="utf-8")
        make_image(self.root / "train/n0002/images/b.JPEG", (255, 0, 0))
        make_image(self.root / "train/n0002/images/a.JPEG", (200, 0, 0))
        make_image(self.root / "train/n0001/images/c.JPEG", (0, 255, 0))
        make_image(self.root / "val/images/v2.JPEG", (0, 0, 255))
        make_image(self.root / "val/images/v1.JPEG", (0, 0, 200))
        (self.root / "val/val_annotations.txt").write_text(
            "v2.JPEG\tn0001\t0\t0\t64\t64\n"
            "v1.JPEG\tn0002\t0\t0\t64\t64\n",
            encoding="utf-8",
        )

    def test_resolves_parent_or_dataset_root(self):
        self.assertEqual(resolve_tiny_imagenet_root(self.data_root), self.root)
        self.assertEqual(resolve_tiny_imagenet_root(self.root), self.root)

    def test_train_uses_wnids_mapping_and_deterministic_order(self):
        dataset = TinyImageNet(self.data_root, split="train")

        self.assertEqual(dataset.classes, ["n0002", "n0001"])
        self.assertEqual(dataset.class_to_idx, {"n0002": 0, "n0001": 1})
        self.assertEqual([path.name for path, _ in dataset.samples], ["a.JPEG", "b.JPEG", "c.JPEG"])
        self.assertEqual(dataset.targets, [0, 0, 1])

    def test_official_validation_uses_same_mapping(self):
        dataset = TinyImageNet(self.data_root, split="val")

        self.assertEqual([path.name for path, _ in dataset.samples], ["v1.JPEG", "v2.JPEG"])
        self.assertEqual(dataset.targets, [0, 1])
        image, label = dataset[0]
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(label, 0)

    def test_rejects_unknown_validation_class(self):
        (self.root / "val/val_annotations.txt").write_text(
            "v1.JPEG\tn9999\t0\t0\t64\t64\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "unknown class"):
            TinyImageNet(self.root, split="val")

    def test_safe_extract_rejects_path_traversal(self):
        archive = Path(self.temp.name) / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("../outside.txt", "unsafe")
        with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
            safe_extract(archive, Path(self.temp.name) / "extract")


if __name__ == "__main__":
    unittest.main()
