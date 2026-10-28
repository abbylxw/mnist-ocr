import torch

from src import nn_model, preprocessing
from src.preprocessing import check_data_input, check_model_input


class MnistInference:
    def __init__(self, checkpoint_path, device):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.loaded_checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=True
        )
        architecture = self.loaded_checkpoint.get("architecture")
        if architecture != nn_model.ARCHITECTURE:
            raise ValueError(
                f"Expected architecture {nn_model.ARCHITECTURE!r}, "
                f"received {architecture!r}."
            )

        saved_preprocessing = self.loaded_checkpoint.get("preprocessing")
        if not isinstance(saved_preprocessing, dict):
            raise ValueError(
                "Expected preprocessing metadata containing padding_per_side."
            )
        padding = saved_preprocessing.get("padding_per_side")
        if padding != preprocessing.PADDING_PER_SIDE:
            raise ValueError(
                f"Expected padding_per_side {preprocessing.PADDING_PER_SIDE!r}, "
                f"received {padding!r}."
            )

        self.saved_preprocessing = saved_preprocessing
        self.loaded_model = nn_model.NeuralNetwork().to(self.device)
        self.loaded_model.load_state_dict(self.loaded_checkpoint["model_state_dict"])
        self.loaded_model.eval()

    def predict(self, image):
        # Load and preprocess the image
        check_data_input(image)
        padded_image = preprocessing.padding_image(image)
        normalized_image = preprocessing.normalize_image(
            padded_image,
            self.saved_preprocessing["train_mean"],
            self.saved_preprocessing["train_std"],
        )
        normalized_input = normalized_image.to(self.device).unsqueeze(0)
        check_model_input(normalized_input)

        with torch.no_grad():
            input_logits = self.loaded_model(normalized_input)
            input_pred = input_logits.argmax(dim=1).item()
            input_prob = torch.softmax(input_logits, dim=-1)

        return {
            "input_logits": input_logits,
            "input_pred": input_pred,
            "input_prob": input_prob,
        }
