"""Contract and preprocessing regression tests; no MNIST download required."""

import tempfile
import unittest
from pathlib import Path

import torch
from torchvision.io import read_image, write_png

from src import preprocessing


class InputContractTests(unittest.TestCase):
    def test_accepts_decoded_grayscale_image(self):
        preprocessing.check_data_input(torch.zeros(1, 28, 28, dtype=torch.uint8))

    def test_rejects_invalid_decoded_images(self):
        cases = [
            (torch.zeros(3, 28, 28, dtype=torch.uint8), "shape"),
            (torch.zeros(1, 32, 32, dtype=torch.uint8), "shape"),
            (torch.zeros(1, 28, 28), "dtype"),
        ]
        for image, message in cases:
            with self.subTest(shape=image.shape, dtype=image.dtype):
                with self.assertRaisesRegex(ValueError, message):
                    preprocessing.check_data_input(image)

    def test_accepts_model_batches(self):
        for size in (1, 3):
            preprocessing.check_model_input(torch.zeros(size, 1, 32, 32))

    def test_rejects_invalid_model_batches(self):
        cases = [
            (torch.zeros(1, 32, 32), "shape"),
            (torch.zeros(0, 1, 32, 32), "shape"),
            (torch.zeros(2, 3, 32, 32), "shape"),
            (torch.zeros(2, 1, 28, 28), "shape"),
            (torch.zeros(1, 1, 32, 32, dtype=torch.float64), "dtype"),
        ]
        for batch, message in cases:
            with self.subTest(shape=batch.shape, dtype=batch.dtype):
                with self.assertRaisesRegex(ValueError, message):
                    preprocessing.check_model_input(batch)


class PreprocessingTests(unittest.TestCase):
    def test_padding_preserves_pixels_and_does_not_scale_or_mutate_input(self):
        image = torch.full((1, 28, 28), 255, dtype=torch.uint8)
        original = image.clone()
        padded = preprocessing.padding_image(image)
        self.assertEqual(padded.shape, (1, 32, 32))
        self.assertEqual(padded.dtype, torch.float32)
        self.assertTrue(torch.equal(padded[:, 2:-2, 2:-2], image.float()))
        border = padded.clone()
        border[:, 2:-2, 2:-2] = 0
        self.assertEqual(torch.count_nonzero(border).item(), 0)
        self.assertTrue(torch.equal(image, original))

    def test_standardization_uses_supplied_statistics_including_border(self):
        padded = preprocessing.padding_image(
            torch.full((1, 28, 28), 100, dtype=torch.uint8)
        )
        original = padded.clone()
        normalized = preprocessing.normalize_image(padded, 20.0, 10.0)
        self.assertEqual(normalized.dtype, torch.float32)
        self.assertEqual(normalized.shape, (1, 32, 32))
        torch.testing.assert_close(
            normalized[:, 2:-2, 2:-2], torch.full((1, 28, 28), 8.0)
        )
        self.assertEqual(normalized[0, 0, 0].item(), -2.0)
        torch.testing.assert_close(padded, original, rtol=0, atol=0)

    def test_batch_processing_matches_individual_processing(self):
        images = torch.stack(
            [
                torch.zeros(1, 28, 28, dtype=torch.uint8),
                torch.full((1, 28, 28), 255, dtype=torch.uint8),
            ]
        )

        def prepare(image):
            return preprocessing.normalize_image(
                preprocessing.padding_image(image), 25.0, 70.0
            )

        actual = prepare(images)
        expected = torch.stack([prepare(image) for image in images])
        preprocessing.check_model_input(actual)
        torch.testing.assert_close(actual, expected)

    def test_png_round_trip_preserves_pixels_and_preprocessed_input(self):
        image = (torch.arange(28 * 28).reshape(1, 28, 28) % 256).to(torch.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "image.png")
            write_png(image, path)
            decoded = read_image(path)
        preprocessing.check_data_input(decoded)
        self.assertTrue(torch.equal(decoded, image))
        original_input = preprocessing.normalize_image(
            preprocessing.padding_image(image), 25.0, 70.0
        )
        decoded_input = preprocessing.normalize_image(
            preprocessing.padding_image(decoded), 25.0, 70.0
        )
        torch.testing.assert_close(decoded_input, original_input, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
