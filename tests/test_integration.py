import tempfile
import unittest
from pathlib import Path

import torch

from src.inference import MnistInference
from src.nn_model import NeuralNetwork
from tests.checkpoint_fixtures import make_checkpoint


class CheckpointLoadingTests(unittest.TestCase):
    def test_restores_weights_and_preprocessing(self):
        # Arrange: create known checkpoint contents.
        source_model = NeuralNetwork()
        expected_weights = {
            name: tensor.clone() for name, tensor in source_model.state_dict().items()
        }
        checkpoint = make_checkpoint(source_model, mean=25.0, std=70.0)
        expected_preprocessing = dict(checkpoint["preprocessing"])

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)

            # Act: construction should load the checkpoint.
            predictor = MnistInference(checkpoint_path, torch.device("cpu"))

            # Assert: every saved parameter/buffer was restored.
            actual_weights = predictor.loaded_model.state_dict()
            self.assertEqual(set(actual_weights), set(expected_weights))
            for name, expected in expected_weights.items():
                with self.subTest(parameter=name):
                    torch.testing.assert_close(
                        actual_weights[name],
                        expected,
                        rtol=0,
                        atol=0,
                    )

            self.assertEqual(
                predictor.saved_preprocessing,
                expected_preprocessing,
            )

    def test_model_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            torch.save(make_checkpoint(), checkpoint_path)

            predictor = MnistInference(checkpoint_path, torch.device("cpu"))
            self.assertFalse(predictor.loaded_model.training)

    def test_image_wrong_shape_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            torch.save(make_checkpoint(mean=25.0, std=70.0), checkpoint_path)

            predictor = MnistInference(checkpoint_path, torch.device("cpu"))
            image1 = torch.zeros((3, 28, 28), dtype=torch.uint8)
            image2 = torch.zeros((3, 1, 28, 28), dtype=torch.uint8)
            image3 = torch.zeros((1, 28, 28))
            with self.assertRaisesRegex(ValueError, "shape"):
                predictor.predict(image1)
            with self.assertRaisesRegex(ValueError, "shape"):
                predictor.predict(image2)
            with self.assertRaisesRegex(ValueError, "dtype"):
                predictor.predict(image3)

    def test_model_prediction(self):
        # Arrange: preserve a reference model independent of the predictor.
        source_model = NeuralNetwork()
        source_model.eval()
        mean, std = 25.0, 70.0

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pt"
            torch.save(
                make_checkpoint(source_model, mean=mean, std=std), checkpoint_path
            )
            predictor = MnistInference(checkpoint_path, torch.device("cpu"))
            image = (torch.arange(28 * 28).reshape(1, 28, 28) % 256).to(torch.uint8)

            # Build the expected input directly from the contract.
            padded = torch.nn.functional.pad(image.float(), (2, 2, 2, 2), value=0)
            expected_input = ((padded - mean) / std).unsqueeze(0)
            with torch.no_grad():
                expected_logits = source_model(expected_input)
                expected_probabilities = expected_logits.softmax(dim=1)
                expected_digit = expected_logits.argmax(dim=1).item()

            # Act.
            result = predictor.predict(image)

            # Assert all three output values.
            torch.testing.assert_close(
                result["input_logits"],
                expected_logits,
                rtol=1e-5,
                atol=1e-6,
            )
            self.assertEqual(result["input_pred"], expected_digit)
            torch.testing.assert_close(
                result["input_prob"],
                expected_probabilities,
                rtol=1e-5,
                atol=1e-6,
            )
            self.assertFalse(result["input_logits"].requires_grad)
            self.assertFalse(result["input_prob"].requires_grad)
