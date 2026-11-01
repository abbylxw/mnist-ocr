"""Training-only data loading and splitting; inference does not import this."""

import torch
from torch.utils.data import TensorDataset
from torchvision import datasets

from src import preprocessing


def load_mnist_train(data_dir):
    """Return raw training images (N, 1, 28, 28) uint8 and labels (N,) int64.

    Reads the dataset's stored tensors directly, bypassing any dataset
    transforms; downloads into `data_dir` if the files are missing.
    """
    mnist = datasets.MNIST(root=data_dir, train=True, download=True)
    return mnist.data.unsqueeze(1), mnist.targets


def split_indices(num_samples, train_ratio, *, generator):
    """Return disjoint (train, validation) index tensors covering all samples.

    Matches `random_split`, so a given generator seed reproduces the split used
    for the original checkpoint.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    train_size = int(train_ratio * num_samples)
    if min(train_size, num_samples - train_size) == 0:
        raise ValueError("Both subsets must contain at least one sample.")
    permutation = torch.randperm(num_samples, generator=generator)
    return permutation[:train_size], permutation[train_size:]


def select(images, labels, indices):
    """Select the same samples from images and labels, keeping them aligned."""
    return images.index_select(0, indices), labels.index_select(0, indices)


def fit_statistics(images):
    """Return (mean, std) over all pixels of the padded images, border included."""
    std, mean = torch.std_mean(preprocessing.padding_image(images))
    return mean.item(), std.item()


def to_dataset(images, labels, mean, std):
    """Pad and standardize images with the given statistics; pair with labels."""
    padded = preprocessing.padding_image(images)
    return TensorDataset(preprocessing.normalize_image(padded, mean, std), labels)
