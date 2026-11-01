"""Checkpoint metadata must match the architecture and preprocessing in code."""

import io
import tempfile
import unittest
from pathlib import Path

import torch

from src.inference import MnistInference
from tests.checkpoint_fixtures import ARCHITECTURE, PADDING_PER_SIDE, make_checkpoint

BUNDLED_CHECKPOINT = Path(__file__).resolve().parents[1] / "models" / "checkpoint.pt"


class CheckpointContractTests(unittest.TestCase):
    def _load(self, checkpoint):
        """Save checkpoint contents and construct a predictor from them."""
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            return MnistInference(checkpoint_path, torch.device("cpu"))

    def test_accepts_matching_checkpoint(self):
        predictor = self._load(make_checkpoint())
        self.assertFalse(predictor.loaded_model.training)

    def test_rejects_architecture_mismatch(self):
        # Weights still fit the network; only the recorded identifier differs.
        for architecture in ("lenet5_relu_maxpool_dense_v2", "", "resnet18"):
            with self.subTest(architecture=architecture):
                with self.assertRaisesRegex(ValueError, "architecture"):
                    self._load(make_checkpoint(architecture=architecture))

    def test_rejects_missing_architecture(self):
        checkpoint = make_checkpoint()
        del checkpoint["architecture"]
        with self.assertRaisesRegex(ValueError, "architecture"):
            self._load(checkpoint)

    def test_rejects_padding_per_side_mismatch(self):
        for padding in (0, 1, PADDING_PER_SIDE + 1):
            with self.subTest(padding_per_side=padding):
                with self.assertRaisesRegex(ValueError, "padding_per_side"):
                    self._load(make_checkpoint(padding_per_side=padding))

    def test_rejects_missing_padding_per_side(self):
        checkpoint = make_checkpoint()
        del checkpoint["preprocessing"]["padding_per_side"]
        with self.assertRaisesRegex(ValueError, "padding_per_side"):
            self._load(checkpoint)

    def test_rejects_file_that_is_not_a_checkpoint(self):
        # Each fails differently inside torch.load (UnpicklingError, KeyError,
        # EOFError, RuntimeError for a truncated archive).
        truncated = io.BytesIO()
        torch.save(make_checkpoint(), truncated)
        for content in (
            b"not a checkpoint",
            b"junk\n",
            b"",
            truncated.getvalue()[:200],
        ):
            with self.subTest(content=content[:20]):
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint_path = Path(directory) / "checkpoint.pt"
                    checkpoint_path.write_bytes(content)
                    with self.assertRaisesRegex(ValueError, "checkpoint"):
                        MnistInference(checkpoint_path, torch.device("cpu"))

    def test_rejects_checkpoint_that_is_not_a_dictionary(self):
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self._load(torch.zeros(3))

    def test_rejects_missing_model_state_dict(self):
        checkpoint = make_checkpoint()
        del checkpoint["model_state_dict"]
        with self.assertRaisesRegex(ValueError, "model_state_dict"):
            self._load(checkpoint)

    def test_rejects_weights_that_do_not_fit_the_network(self):
        # Metadata matches, but the tensors belong to a different network shape.
        resized = make_checkpoint()
        resized["model_state_dict"]["conv2d_relu_stack.0.weight"] = torch.zeros(
            3, 1, 5, 5
        )
        incomplete = make_checkpoint()
        del incomplete["model_state_dict"]["conv2d_relu_stack.0.bias"]
        for name, checkpoint in (("resized", resized), ("incomplete", incomplete)):
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, "model_state_dict"):
                    self._load(checkpoint)

    @unittest.skipUnless(BUNDLED_CHECKPOINT.is_file(), "trained checkpoint absent")
    def test_bundled_checkpoint_satisfies_contract(self):
        # The checkpoint was saved from MPS; map to CPU so it loads on any host.
        checkpoint = torch.load(
            BUNDLED_CHECKPOINT, map_location="cpu", weights_only=True
        )
        self.assertEqual(checkpoint["architecture"], ARCHITECTURE)
        self.assertEqual(
            checkpoint["preprocessing"]["padding_per_side"], PADDING_PER_SIDE
        )
        MnistInference(BUNDLED_CHECKPOINT, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
