import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.transforms.functional import normalize

PADDING_PER_SIDE = 2

_padding_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=False),
        nn.ConstantPad2d(PADDING_PER_SIDE, 0),
    ]
)


def padding_image(image):
    """Pad a PIL image or (N, C, H, W) tensor, preserving the 0–255 scale."""
    return _padding_transform(image)


def normalize_image(padded_image, train_u, train_std):
    """Standardize one image or a batch using training-only statistics."""
    return normalize(padded_image, mean=[train_u], std=[train_std])


def check_data_input(raw_data):
    """Validate one decoded grayscale MNIST image."""
    if raw_data.shape != (1, 28, 28):
        raise ValueError("Expected input shape (1, 28, 28).")
    if raw_data.dtype != torch.uint8:
        raise ValueError("Expected input dtype torch.uint8.")


def check_model_input(batch):
    """Validate a nonempty batch of preprocessed images."""
    if batch.ndim != 4 or batch.shape[0] == 0 or batch.shape[1:] != (1, 32, 32):
        raise ValueError("Expected model input shape (N, 1, 32, 32), N > 0.")
    if batch.dtype != torch.float32:
        raise ValueError("Expected model input dtype torch.float32.")
