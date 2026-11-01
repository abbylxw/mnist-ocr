import argparse
import sys
from pathlib import Path

from torchvision.io import read_image

from src import inference


def validate_existing_file(filepath):
    path = Path(filepath)
    if not path.is_file():
        # This message will be cleanly formatted into argparse's default error output
        print(
            "Accepted input: PNG or JPEG decoding to a 28×28 grayscale uint8 image.",
            file=sys.stderr,
        )
        print(
            "Content assumption: one centered light digit on a dark background.",
            file=sys.stderr,
        )
        raise argparse.ArgumentTypeError(f"'{filepath}' is not a valid file.")
    return path


def report_input_error(message):
    print(f"Error: {message}", file=sys.stderr)
    print(
        "Accepted input: PNG or JPEG decoding to a 28×28 grayscale uint8 image.",
        file=sys.stderr,
    )
    print(
        "Content assumption: one centered light digit on a dark background.",
        file=sys.stderr,
    )


def report_checkpoint_error(message):
    print(f"Error: {message}", file=sys.stderr)
    print(
        "Use --checkpoint PATH to select an existing checkpoint.",
        file=sys.stderr,
    )


def parse_args_and_predict():
    parser = argparse.ArgumentParser(
        prog="digit_recognizer", description="Gray Scale Digit Recognizer Inference"
    )
    parser.add_argument(
        "img_path",
        type=validate_existing_file,
        help="input image path, produce digit prediction, accepted input: PNG or JPEG decoding to a 28×28 grayscale uint8 image, content assumption: one centered light digit on a dark background.",
    )
    parser.add_argument(
        "--label",
        type=int,
        choices=range(10),
        help="image digit label if known, for evaluation purpose, must be an integer between 0 and 9 inclusive.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/checkpoint.pt"),
        help="path to the checkpoint file",
    )
    args = parser.parse_args()
    image_path = args.img_path
    label = args.label
    checkpoint_path = args.checkpoint

    try:
        img = read_image(image_path)
    except (OSError, RuntimeError, ValueError) as error:
        report_input_error(f"Could not decode '{image_path}': {error}")
        sys.exit(1)

    # Checkpoint failures.
    try:
        predictor = inference.MnistInference(
            checkpoint_path=checkpoint_path,
            device="cpu",
        )
    except (OSError, ValueError) as error:
        report_checkpoint_error(f"Could not load checkpoint {checkpoint_path}: {error}")
        sys.exit(1)

    try:
        image_result = predictor.predict(img)
    except ValueError as error:
        report_input_error(str(error))
        sys.exit(1)

    print(f"Image path: {image_path}")
    print(f"Predicted digit: {image_result['input_pred']}")
    if label is not None:
        print(f"True digit: {label}")
    sys.exit(0)


if __name__ == "__main__":
    parse_args_and_predict()
