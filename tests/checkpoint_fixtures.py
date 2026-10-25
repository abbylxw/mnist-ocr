"""Build checkpoints that satisfy the inference contract, independent of src/."""

from src.nn_model import NeuralNetwork

# Contract values written literally so tests do not trust the code under test.
ARCHITECTURE = "lenet5_relu_maxpool_dense_v1"
PADDING_PER_SIDE = 2


def make_checkpoint(
    model=None,
    *,
    mean=25.0,
    std=70.0,
    architecture=ARCHITECTURE,
    padding_per_side=PADDING_PER_SIDE,
):
    """Return checkpoint contents shaped like the training notebook's output."""
    if model is None:
        model = NeuralNetwork()
    return {
        "architecture": architecture,
        "model_state_dict": model.state_dict(),
        "preprocessing": {
            "input_shape": [1, 28, 28],
            "input_dtype": "uint8",
            "pixel_range": [0, 255],
            "padding_per_side": padding_per_side,
            "padding_value": 0,
            "output_dtype": "float32",
            "scale_to_unit_interval": False,
            "train_mean": mean,
            "train_std": std,
        },
    }
