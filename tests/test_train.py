"""Training pipeline on tiny synthetic data; no MNIST download."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src import train
from src.inference import MnistInference


def _config(**search):
    """A small config shaped like configs/train.toml."""
    grid = {"batch_sizes": [8], "learning_rates": [1e-3, 1e-2], "weight_decays": [0]}
    grid.update(search)
    return {
        "data": {"dir": "unused"},
        "split": {"train_ratio": 0.75, "seed": 11},
        "training": {"seed": 11, "epochs": 2},
        "search": grid,
        "selection": {"rule": train.SELECTION_RULE, "min_validation_accuracy": 0.0},
    }


def _synthetic_mnist(count=48):
    generator = torch.Generator().manual_seed(0)
    images = torch.randint(
        0, 256, (count, 1, 28, 28), dtype=torch.uint8, generator=generator
    )
    return images, torch.arange(count) % 10


def _candidate(
    loss, accuracy=0.99, *, learning_rate=1e-3, batch_size=64, weight_decay=0
):
    return {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "validation_loss": loss,
        "validation_accuracy": accuracy,
    }


class ConfigTests(unittest.TestCase):
    def test_project_config_loads(self):
        config, digest = train.load_config(train.DEFAULT_CONFIG)
        self.assertEqual(config["selection"]["rule"], train.SELECTION_RULE)
        self.assertEqual(len(digest), 64)

    def _write(self, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "train.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_rejects_missing_keys(self):
        text = train.DEFAULT_CONFIG.read_text(encoding="utf-8")
        path = self._write(text.replace("epochs = 5", ""))
        with self.assertRaisesRegex(train.TrainingError, r"\[training\] epochs"):
            train.load_config(path)

    def test_rejects_unknown_selection_rule(self):
        text = train.DEFAULT_CONFIG.read_text(encoding="utf-8")
        path = self._write(text.replace(train.SELECTION_RULE, "max_accuracy"))
        with self.assertRaisesRegex(train.TrainingError, "selection rule"):
            train.load_config(path)


class SelectWinnerTests(unittest.TestCase):
    def test_lowest_validation_loss_wins(self):
        winner = train.select_winner([_candidate(0.05), _candidate(0.03)], 0.9)
        self.assertEqual(winner["validation_loss"], 0.03)

    def test_ties_go_to_smaller_learning_rate_then_batch_size(self):
        candidates = [
            _candidate(0.03, learning_rate=1e-2, batch_size=32),
            _candidate(0.03, learning_rate=1e-3, batch_size=128),
            _candidate(0.03, learning_rate=1e-3, batch_size=64),
        ]
        winner = train.select_winner(candidates, 0.9)
        self.assertEqual((winner["learning_rate"], winner["batch_size"]), (1e-3, 64))

    def test_skips_diverged_candidates(self):
        winner = train.select_winner([_candidate(None), _candidate(0.2)], 0.9)
        self.assertEqual(winner["validation_loss"], 0.2)

    def test_refuses_winner_below_accuracy_floor(self):
        with self.assertRaisesRegex(train.TrainingError, "below the configured floor"):
            train.select_winner([_candidate(0.03, accuracy=0.95)], 0.98)

    def test_refuses_when_no_candidate_is_finite(self):
        with self.assertRaisesRegex(train.TrainingError, "finite"):
            train.select_winner([_candidate(None)], 0.0)


class ProvenanceTests(unittest.TestCase):
    def test_refuses_uncommitted_changes_by_default(self):
        with mock.patch.object(train, "git_state", return_value=("abc123", True)):
            with self.assertRaisesRegex(train.TrainingError, "Uncommitted"):
                train.provenance(torch.device("cpu"), "f" * 64, allow_dirty=False)

    def test_allow_dirty_runs_and_records_it(self):
        with mock.patch.object(train, "git_state", return_value=("abc123", True)):
            record = train.provenance(torch.device("cpu"), "f" * 64, allow_dirty=True)
        self.assertEqual(record["git_commit"], "abc123")
        self.assertTrue(record["git_dirty"])
        self.assertEqual(record["config_sha256"], "f" * 64)
        self.assertEqual(record["device"], "cpu")

    def test_clean_checkout_runs(self):
        with mock.patch.object(train, "git_state", return_value=("abc123", False)):
            record = train.provenance(torch.device("cpu"), "f" * 64, allow_dirty=False)
        self.assertFalse(record["git_dirty"])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.images, self.labels = _synthetic_mnist()
        self.config = _config()
        self.device = torch.device("cpu")

    def _search(self):
        return train.run_search(
            self.images, self.labels, self.config, self.device, log=lambda _: None
        )

    def test_search_records_every_candidate_and_epoch(self):
        result = self._search()
        self.assertEqual(result["split"]["train_samples"], 36)
        self.assertEqual(result["split"]["validation_samples"], 12)
        self.assertEqual(len(result["candidates"]), 2)
        for candidate in result["candidates"]:
            self.assertEqual([r["epoch"] for r in candidate["epochs"]], [1, 2])
            self.assertEqual(
                candidate["validation_loss"], candidate["epochs"][-1]["validation_loss"]
            )
            self.assertTrue(0.0 <= candidate["validation_accuracy"] <= 1.0)

    def test_search_is_repeatable(self):
        self.assertEqual(self._search(), self._search())

    def test_candidates_do_not_depend_on_grid_order(self):
        # Per-candidate seeding: a candidate's result must not depend on what ran before it.
        forward = self._search()["candidates"]
        self.config["search"]["learning_rates"] = [1e-2, 1e-3]
        reverse = self._search()["candidates"]
        self.assertEqual(forward[0], reverse[1])
        self.assertEqual(forward[1], reverse[0])

    def test_final_checkpoint_loads_through_inference(self):
        result = self._search()
        winner = train.select_winner(result["candidates"], 0.0)
        model, mean, std, history = train.run_final(
            self.images, self.labels, self.config, winner, self.device
        )
        search_record = {"provenance": {"git_commit": "abc123"}, **result}
        checkpoint = train.build_checkpoint(
            model,
            mean=mean,
            std=std,
            history=history,
            train_samples=len(self.labels),
            config=self.config,
            winner=winner,
            search_record=search_record,
            search_run="runs/test/search.json",
            search_run_sha256="0" * 64,
            run_provenance={"git_commit": "abc123", "git_dirty": False},
        )

        # Statistics come from all images, not the search's training part.
        self.assertNotEqual(mean, result["normalization"]["train_mean"])
        self.assertEqual(checkpoint["training"]["train_samples"], 48)
        self.assertEqual(len(checkpoint["training"]["epoch_train_losses"]), 2)
        self.assertEqual(
            checkpoint["selection"]["validation_loss"], winner["validation_loss"]
        )
        for tensor in checkpoint["model_state_dict"].values():
            self.assertEqual(tensor.device.type, "cpu")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(checkpoint, path)
            predictor = MnistInference(path, torch.device("cpu"))
        prediction = predictor.predict(self.images[0])
        self.assertIn(prediction["input_pred"], range(10))
        self.assertTrue(math.isclose(predictor.saved_preprocessing["train_mean"], mean))


class CommandLineTests(unittest.TestCase):
    def test_final_refuses_search_run_from_a_different_config(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "search.json"
            run.write_text(
                '{"provenance": {"config_sha256": "stale"}}', encoding="utf-8"
            )
            with mock.patch("sys.stderr"):
                code = train.main(
                    [
                        "final",
                        "--search-run",
                        str(run),
                        "--out",
                        str(Path(directory) / "c.pt"),
                    ]
                )
        self.assertEqual(code, 1)

    def test_final_refuses_to_overwrite_a_checkpoint(self):
        _, digest = train.load_config(train.DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "search.json"
            run.write_text(
                '{"provenance": {"config_sha256": "%s"}}' % digest, encoding="utf-8"
            )
            existing = Path(directory) / "c.pt"
            existing.write_bytes(b"keep me")
            with mock.patch("sys.stderr"):
                code = train.main(
                    ["final", "--search-run", str(run), "--out", str(existing)]
                )
            self.assertEqual(code, 1)
            self.assertEqual(existing.read_bytes(), b"keep me")


if __name__ == "__main__":
    unittest.main()
