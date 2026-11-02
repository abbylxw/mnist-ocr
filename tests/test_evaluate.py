"""Evaluation helpers on small synthetic images; no MNIST download."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

from src import evaluate
from src.inference import MnistInference
from tests.checkpoint_fixtures import make_checkpoint


def _row(label, png_pred, jpg_pred):
    """A per-image row with only the fields `summarize` reads."""
    png_correct, jpg_correct = png_pred == label, jpg_pred == label
    disagreement = png_pred != jpg_pred
    return {
        "label": label,
        "png_pred": png_pred,
        "jpg_pred": jpg_pred,
        "png_correct": png_correct,
        "jpg_correct": jpg_correct,
        "png_ce_loss": 0.1,
        "jpg_ce_loss": 0.3,
        "png_brier_loss": 0.2,
        "jpg_brier_loss": 0.4,
        "disagreement": disagreement,
        "correct_to_wrong": png_correct and not jpg_correct,
        "wrong_to_correct": not png_correct and jpg_correct,
        "wrong_to_different_wrong": not png_correct
        and not jpg_correct
        and disagreement,
    }


def _write_samples(directory, count=3):
    """Write paired PNG/JPEG files; return (samples, source images)."""
    generator = torch.Generator().manual_seed(0)
    images = torch.randint(
        0, 256, (count, 28, 28), dtype=torch.uint8, generator=generator
    )
    samples = []
    for index in range(count):
        label = index % 10
        # frombytes avoids numpy, which CI does not install.
        image = Image.frombytes("L", (28, 28), bytes(images[index].flatten().tolist()))
        png = Path(directory) / f"{index}_label_{label}.png"
        jpg = Path(directory) / f"{index}_label_{label}.jpg"
        image.save(png, format="PNG")
        image.save(jpg, format="JPEG", quality=evaluate.JPEG_QUALITY)
        samples.append((index, label, png, jpg))
    return samples, images


class ScoreTests(unittest.TestCase):
    def test_confident_correct_prediction_scores_near_zero(self):
        logits = torch.full((1, 10), -20.0)
        logits[0, 3] = 20.0
        prediction, cross_entropy, brier = evaluate.score(logits, 3)
        self.assertEqual(prediction, 3)
        self.assertLess(cross_entropy, 1e-6)
        self.assertLess(brier, 1e-6)

    def test_uniform_prediction(self):
        _, cross_entropy, brier = evaluate.score(torch.zeros((1, 10)), 3)
        self.assertTrue(math.isclose(cross_entropy, math.log(10), rel_tol=1e-6))
        # (1 - 0.1)^2 + 9 * 0.1^2
        self.assertTrue(math.isclose(brier, 0.9, rel_tol=1e-6))


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        rows = [
            _row(0, 0, 0),  # both correct
            _row(1, 1, 7),  # correct → wrong
            _row(1, 4, 1),  # wrong → correct
            _row(2, 5, 6),  # wrong → different wrong
            _row(2, 5, 5),  # same wrong class
        ]
        self.metrics, self.per_class, self.confusion = evaluate.summarize(rows)

    def test_accuracy_and_losses(self):
        self.assertEqual(self.metrics["source_pairs"], 5)
        self.assertEqual(self.metrics["png_accuracy"], 2 / 5)
        self.assertEqual(self.metrics["jpg_accuracy"], 2 / 5)
        self.assertTrue(math.isclose(self.metrics["png_cross_entropy_loss"], 0.1))
        self.assertTrue(math.isclose(self.metrics["jpg_brier_loss"], 0.4))

    def test_transitions_add_up_to_disagreements(self):
        self.assertEqual(self.metrics["correct_to_wrong"], 1)
        self.assertEqual(self.metrics["wrong_to_correct"], 1)
        self.assertEqual(self.metrics["wrong_to_different_wrong"], 1)
        self.assertEqual(self.metrics["disagreement_count"], 3)
        self.assertEqual(self.metrics["disagreement_rate"], 3 / 5)

    def test_per_class_accuracy_covers_present_digits(self):
        self.assertEqual(self.per_class["png"], {"0": 1.0, "1": 0.5, "2": 0.0})
        self.assertEqual(self.per_class["jpg"], {"0": 1.0, "1": 0.5, "2": 0.0})

    def test_confusion_rows_are_true_digits(self):
        png = self.confusion["png"]
        self.assertEqual(png[1][1], 1)
        self.assertEqual(png[1][4], 1)
        self.assertEqual(png[2][5], 2)
        self.assertEqual(sum(map(sum, png)), 5)


class EvaluatePairsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        checkpoint_path = self.directory / "checkpoint.pt"
        torch.save(make_checkpoint(), checkpoint_path)
        self.predictor = MnistInference(checkpoint_path, torch.device("cpu"))
        self.samples, self.images = _write_samples(self.directory)

    def test_rows_match_direct_prediction(self):
        rows = evaluate.evaluate_pairs(self.predictor, self.samples, self.images)
        self.assertEqual([row["index"] for row in rows], [0, 1, 2])
        for row, (_, label, png, _) in zip(rows, self.samples):
            expected = self.predictor.predict(self.images[row["index"]].unsqueeze(0))
            self.assertEqual(row["label"], label)
            self.assertEqual(row["png_pred"], expected["input_pred"])
            self.assertEqual(row["png_correct"], row["png_pred"] == label)
            self.assertEqual(row["png_path"], str(png.resolve()))

    def test_refuses_png_that_differs_from_source(self):
        altered = self.images.clone()
        altered[1, 0, 0] ^= 1
        with self.assertRaisesRegex(evaluate.EvaluationError, "source image 1"):
            evaluate.evaluate_pairs(self.predictor, self.samples, altered)


class JpegManifestTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.samples, _ = _write_samples(self.directory)
        self.manifest = self.directory / "test_jpeg.sha256"

    def _write_manifest(self, samples):
        self.manifest.write_text(
            "".join(
                f"{evaluate.sha256_of(jpg)}  data/test/{jpg.name}\n"
                for _, _, _, jpg in samples
            ),
            encoding="utf-8",
        )

    def test_matching_bytes_pass(self):
        self._write_manifest(self.samples)
        evaluate.check_jpeg_manifest(self.samples, self.manifest)

    def test_changed_bytes_are_refused(self):
        self._write_manifest(self.samples)
        jpg = self.samples[0][3]
        jpg.write_bytes(jpg.read_bytes() + b"\0")
        with self.assertRaisesRegex(evaluate.EvaluationError, "encodes differently"):
            evaluate.check_jpeg_manifest(self.samples, self.manifest)

    def test_unlisted_jpeg_is_refused(self):
        self._write_manifest(self.samples[:-1])
        with self.assertRaisesRegex(evaluate.EvaluationError, "exactly"):
            evaluate.check_jpeg_manifest(self.samples, self.manifest)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.checkpoint = self.directory / "checkpoint.pt"

    def _main(self, git, *extra):
        with (
            mock.patch.object(evaluate, "git_state", return_value=git),
            mock.patch.object(evaluate, "export_test_images") as export,
            mock.patch("sys.stderr"),
        ):
            code = evaluate.main(["--checkpoint", str(self.checkpoint), *extra])
        # Every refusal happens before any image is exported.
        export.assert_not_called()
        return code

    def test_refuses_uncommitted_changes(self):
        torch.save(make_checkpoint(), self.checkpoint)
        self.assertEqual(self._main(("abc123", True)), 1)

    def test_refuses_checkpoint_without_clean_provenance(self):
        checkpoint = make_checkpoint()
        checkpoint["provenance"] = {"git_dirty": True}
        torch.save(checkpoint, self.checkpoint)
        self.assertEqual(self._main(("abc123", False)), 1)
        # A checkpoint that records no provenance is treated the same way.
        torch.save(make_checkpoint(), self.checkpoint)
        self.assertEqual(self._main(("abc123", False)), 1)

    def test_refuses_unreadable_checkpoint(self):
        self.checkpoint.write_text("not a checkpoint", encoding="utf-8")
        self.assertEqual(self._main(("abc123", False)), 1)

    def test_refuses_to_overwrite_a_report(self):
        torch.save(make_checkpoint(), self.checkpoint)
        existing = self.directory / "report"
        existing.mkdir()
        self.assertEqual(self._main(("abc123", False), "--out", str(existing)), 1)


if __name__ == "__main__":
    unittest.main()
