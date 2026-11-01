"""Training data helpers on small synthetic tensors; no MNIST download."""

import unittest

import torch
from torch.utils.data import TensorDataset, random_split

from src import data


def _seeded(seed=11):
    return torch.Generator().manual_seed(seed)


class SplitIndicesTests(unittest.TestCase):
    def test_disjoint_and_covering_with_expected_sizes(self):
        train, validation = data.split_indices(100, 0.8, generator=_seeded())
        self.assertEqual((len(train), len(validation)), (80, 20))
        self.assertEqual(train.dtype, torch.long)
        combined = torch.cat([train, validation]).sort().values
        self.assertTrue(torch.equal(combined, torch.arange(100)))

    def test_same_seed_repeats_and_matches_random_split(self):
        # The original checkpoint's split came from random_split on the dataset.
        train, validation = data.split_indices(100, 0.8, generator=_seeded())
        again, _ = data.split_indices(100, 0.8, generator=_seeded())
        reference_train, reference_validation = random_split(
            TensorDataset(torch.arange(100)), [80, 20], generator=_seeded()
        )
        self.assertTrue(torch.equal(train, again))
        self.assertEqual(train.tolist(), reference_train.indices)
        self.assertEqual(validation.tolist(), reference_validation.indices)

    def test_rejects_invalid_ratios_and_empty_subsets(self):
        for num_samples, ratio in [(100, 0), (100, 1), (100, 1.5), (1, 0.8)]:
            with self.subTest(num_samples=num_samples, ratio=ratio):
                with self.assertRaises(ValueError):
                    data.split_indices(num_samples, ratio, generator=_seeded())


class SelectTests(unittest.TestCase):
    def test_images_stay_aligned_with_labels(self):
        # Image i is filled with value i and labelled i % 10, so any
        # misalignment shows up as a label that no longer matches its pixels.
        count = 50
        images = torch.arange(count, dtype=torch.uint8).view(count, 1, 1, 1)
        images = images.expand(count, 1, 28, 28).clone()
        labels = torch.arange(count) % 10
        indices, _ = data.split_indices(count, 0.8, generator=_seeded())

        selected_images, selected_labels = data.select(images, labels, indices)

        self.assertEqual(len(selected_images), len(indices))
        for image, label, index in zip(selected_images, selected_labels, indices):
            self.assertTrue(torch.all(image == index))
            self.assertEqual(label.item(), index.item() % 10)


class StatisticsTests(unittest.TestCase):
    def _images(self):
        return (torch.arange(3 * 28 * 28) % 256).to(torch.uint8).view(3, 1, 28, 28)

    def test_statistics_include_the_padded_border(self):
        images = self._images()
        padded = torch.nn.functional.pad(images.float(), (2, 2, 2, 2), value=0)
        expected_std, expected_mean = torch.std_mean(padded)

        mean, std = data.fit_statistics(images)

        self.assertIsInstance(mean, float)
        self.assertIsInstance(std, float)
        self.assertAlmostEqual(mean, expected_mean.item(), places=4)
        self.assertAlmostEqual(std, expected_std.item(), places=4)

    def test_dataset_applies_given_statistics_and_keeps_labels(self):
        images = self._images()
        labels = torch.tensor([3, 1, 4])
        mean, std = 25.0, 70.0
        padded = torch.nn.functional.pad(images.float(), (2, 2, 2, 2), value=0)

        dataset = data.to_dataset(images, labels, mean, std)

        prepared, prepared_labels = dataset.tensors
        self.assertEqual(prepared.shape, (3, 1, 32, 32))
        self.assertEqual(prepared.dtype, torch.float32)
        torch.testing.assert_close(prepared, (padded - mean) / std)
        self.assertTrue(torch.equal(prepared_labels, labels))


if __name__ == "__main__":
    unittest.main()
