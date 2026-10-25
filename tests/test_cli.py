import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.nn_model import NeuralNetwork
from tests.checkpoint_fixtures import make_checkpoint


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]

    def _run_cli(self, *arguments):
        """Run the CLI with consistent process settings and capture its output."""
        return subprocess.run(
            [sys.executable, "-m", "src.cli", *[str(arg) for arg in arguments]],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def _write_checkpoint(self, checkpoint_path, *, mean=25.0, std=70.0):
        """Save a valid CPU checkpoint and return its independent reference model."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            reference_model = NeuralNetwork().eval()
        torch.save(
            make_checkpoint(reference_model, mean=mean, std=std), checkpoint_path
        )
        return reference_model

    def test_cli_success(self):
        mean, std = 25.0, 70.0
        image = (torch.arange(28 * 28).reshape(1, 28, 28) % 256).to(torch.uint8)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            image_path = Path(directory) / "image.png"
            reference_model = self._write_checkpoint(
                checkpoint_path, mean=mean, std=std
            )

            # Compute the expected result independently of the inference class.
            padded = torch.nn.functional.pad(image.float(), (2, 2, 2, 2), value=0)
            reference_input = ((padded - mean) / std).unsqueeze(0)
            with torch.no_grad():
                expected_digit = reference_model(reference_input).argmax(dim=1).item()
            Image.fromarray(image.squeeze(0).numpy()).save(image_path, format="PNG")

            # Act: exercise the CLI as a separate process while fixtures exist.
            result = self._run_cli(image_path, "--checkpoint", checkpoint_path)

        # Assert: successful prediction through the complete CLI path.
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Predicted digit: {expected_digit}", result.stdout.splitlines())
        self.assertEqual(result.stderr, "")

    def test_cli_reject_missing_image(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"

            result = self._run_cli(image_path)

        # Assert: Failure through the nonexistent CLI path.
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn(str(image_path), result.stderr)
        self.assertIn("is not a valid file", result.stderr)
        self.assertIn("Accepted input:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_image_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            txt_path = Path(directory) / "image.txt"
            with open(txt_path, "w", encoding="utf-8") as file:
                file.write("This is a new text file created using Python.")
                file.write("It is a txt file named as image.")

            result = self._run_cli(txt_path)

        # Assert: Failure through the wrong image format.
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn(str(txt_path), result.stderr)
        self.assertIn("Could not decode", result.stderr)
        self.assertIn("Accepted input:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_rgb_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            image = (torch.arange(3 * 28 * 28).reshape(3, 28, 28) % 256).to(torch.uint8)
            image_path = Path(directory) / "image.png"
            checkpoint_path = Path(directory) / "checkpoint.pt"
            Image.fromarray(image.permute(1, 2, 0).numpy()).save(
                image_path, format="PNG"
            )
            self._write_checkpoint(checkpoint_path)

            result = self._run_cli(image_path, "--checkpoint", checkpoint_path)

        # Assert: Failure through a wrong input shape.
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("Expected input shape (1, 28, 28)", result.stderr)
        self.assertIn("Accepted input:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_size_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            image = (torch.arange(32 * 32).reshape(1, 32, 32) % 256).to(torch.uint8)
            image_path = Path(directory) / "image.png"
            checkpoint_path = Path(directory) / "checkpoint.pt"
            Image.fromarray(image.squeeze(0).numpy()).save(image_path, format="PNG")
            self._write_checkpoint(checkpoint_path)

            result = self._run_cli(image_path, "--checkpoint", checkpoint_path)

        # Assert: Failure through a wrong input shape.
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("Expected input shape (1, 28, 28)", result.stderr)
        self.assertIn("Accepted input:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_reject_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            image = (torch.arange(28 * 28).reshape(1, 28, 28) % 256).to(torch.uint8)
            image_path = Path(directory) / "image.png"
            checkpoint_path = Path(directory) / "missing.pt"
            Image.fromarray(image.squeeze(0).numpy()).save(image_path, format="PNG")

            result = self._run_cli(image_path, "--checkpoint", checkpoint_path)

        # Assert: Failure through a wrong checkpoint path.
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn("Could not load checkpoint", result.stderr)
        self.assertIn("Use --checkpoint PATH", result.stderr)
        self.assertNotIn("Accepted input:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_reject_checkpoint_contract_mismatch(self):
        cases = [
            ({"architecture": "lenet5_relu_maxpool_dense_v2"}, "architecture"),
            ({"padding_per_side": 0}, "padding_per_side"),
        ]
        for override, field in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as directory:
                    image = torch.zeros((1, 28, 28), dtype=torch.uint8)
                    image_path = Path(directory) / "image.png"
                    checkpoint_path = Path(directory) / "checkpoint.pt"
                    Image.fromarray(image.squeeze(0).numpy()).save(
                        image_path, format="PNG"
                    )
                    torch.save(make_checkpoint(**override), checkpoint_path)

                    result = self._run_cli(image_path, "--checkpoint", checkpoint_path)

                # Assert: an incompatible checkpoint is a setup error, not a crash.
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn("Could not load checkpoint", result.stderr)
                self.assertIn(field, result.stderr)
                self.assertNotIn("Accepted input:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
