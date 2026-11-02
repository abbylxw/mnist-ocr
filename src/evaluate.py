"""Evaluate a checkpoint on the MNIST test split through the shipped code path.

From the project root:

    uv run --locked python -m src.evaluate [--checkpoint models/checkpoint.pt]

Each test image is exported as a PNG and a quality-75 JPEG, decoded with the
CLI's reader, and predicted by `MnistInference` on CPU. The JPEG bytes are
checked against `manifests/test_jpeg.sha256` so every report describes the
same inputs. Writes `results/<id>/summary.json` and `per_image_results.csv`.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import PIL
import torch
import torch.nn.functional as F
from PIL import features
from torchvision import datasets
from torchvision.io import read_image

from src import inference
from src.train import PROJECT_ROOT, display_path, git_state, sha256_of

DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "checkpoint.pt"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
JPEG_MANIFEST = PROJECT_ROOT / "manifests" / "test_jpeg.sha256"
JPEG_QUALITY = 75
NUM_CLASSES = 10
ENCODINGS = ("png", "jpg")


class EvaluationError(Exception):
    """An evaluation cannot proceed; the message explains why."""


def export_test_images(data_dir):
    """Write each test image as PNG and JPEG; return [(index, label, png, jpg)]."""
    test_data = datasets.MNIST(root=data_dir, train=False, download=True)
    out = Path(data_dir) / "test"
    out.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, (image, label) in enumerate(test_data):
        png = out / f"{index}_label_{label}.png"
        jpg = out / f"{index}_label_{label}.jpg"
        image.save(png, format="PNG")
        image.save(jpg, format="JPEG", quality=JPEG_QUALITY)
        samples.append((index, label, png, jpg))
    return samples, test_data.data


def check_jpeg_manifest(samples, manifest):
    """Refuse JPEGs whose bytes differ from the recorded ones."""
    expected = {}
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        digest, path = line.split(maxsplit=1)
        expected[Path(path).name] = digest
    names = {jpg.name for _, _, _, jpg in samples}
    if names != set(expected):
        raise EvaluationError(f"{manifest} does not list exactly the exported JPEGs.")
    mismatched = [
        jpg.name for _, _, _, jpg in samples if sha256_of(jpg) != expected[jpg.name]
    ]
    if mismatched:
        raise EvaluationError(
            f"{len(mismatched)} JPEGs differ from {display_path(manifest)} "
            f"(first: {mismatched[0]}); this Pillow/libjpeg encodes differently, "
            "so results would not describe the recorded inputs."
        )


def score(logits, label):
    """Return (prediction, cross-entropy, Brier score) for one image's logits."""
    target = torch.tensor([label])
    probabilities = torch.softmax(logits, dim=-1)
    one_hot = F.one_hot(target, NUM_CLASSES).float()
    return (
        int(logits.argmax(dim=1).item()),
        F.cross_entropy(logits, target).item(),
        torch.sum((probabilities - one_hot) ** 2).item(),
    )


def evaluate_pairs(predictor, samples, source_images):
    """Predict both encodings of every sample; return per-image rows."""
    rows = []
    for index, label, png, jpg in samples:
        row = {"index": index, "label": label}
        for encoding, path in (("png", png), ("jpg", jpg)):
            image = read_image(path)
            if encoding == "png" and not torch.equal(
                image, source_images[index].unsqueeze(0)
            ):
                raise EvaluationError(f"PNG pixels differ from source image {index}.")
            logits = predictor.predict(image)["input_logits"].cpu()
            prediction, cross_entropy, brier = score(logits, label)
            row[f"{encoding}_path"] = display_path(path)
            row[f"{encoding}_pred"] = prediction
            row[f"{encoding}_ce_loss"] = cross_entropy
            row[f"{encoding}_brier_loss"] = brier
            row[f"{encoding}_correct"] = prediction == label
        png_correct, jpg_correct = row["png_correct"], row["jpg_correct"]
        row["disagreement"] = row["png_pred"] != row["jpg_pred"]
        # Direction: PNG → JPEG.
        row["correct_to_wrong"] = png_correct and not jpg_correct
        row["wrong_to_correct"] = not png_correct and jpg_correct
        row["wrong_to_different_wrong"] = (
            not png_correct and not jpg_correct and row["disagreement"]
        )
        rows.append(row)
    return rows


def summarize(rows):
    """Aggregate per-image rows into metrics for each encoding and the pairs."""
    count = len(rows)
    metrics: dict[str, int | float] = {"source_pairs": count}
    for encoding in ENCODINGS:
        metrics[f"{encoding}_accuracy"] = (
            sum(row[f"{encoding}_correct"] for row in rows) / count
        )
        metrics[f"{encoding}_cross_entropy_loss"] = (
            sum(row[f"{encoding}_ce_loss"] for row in rows) / count
        )
        metrics[f"{encoding}_brier_loss"] = (
            sum(row[f"{encoding}_brier_loss"] for row in rows) / count
        )
    disagreements = sum(row["disagreement"] for row in rows)
    metrics["disagreement_count"] = disagreements
    metrics["disagreement_rate"] = disagreements / count
    for transition in (
        "correct_to_wrong",
        "wrong_to_correct",
        "wrong_to_different_wrong",
    ):
        metrics[transition] = sum(row[transition] for row in rows)

    per_class = {}
    confusion = {}
    for encoding in ENCODINGS:
        # Rows are true digits, columns predicted digits.
        matrix = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
        for row in rows:
            matrix[row["label"]][row[f"{encoding}_pred"]] += 1
        confusion[encoding] = matrix
        per_class[encoding] = {
            str(digit): matrix[digit][digit] / sum(matrix[digit])
            for digit in range(NUM_CLASSES)
            if sum(matrix[digit])
        }
    return metrics, per_class, confusion


def checkpoint_summary(path, checkpoint):
    """Identify the evaluated checkpoint and where it came from."""
    return {
        "path": display_path(path),
        "sha256": sha256_of(path),
        "architecture": checkpoint["architecture"],
        "training": checkpoint.get("training"),
        "selection": checkpoint.get("selection"),
        "provenance": checkpoint.get("provenance"),
    }


def _write_report(out, report, rows):
    out.mkdir(parents=True)
    with (out / "per_image_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _evaluate_command(args):
    commit, dirty = git_state()
    if dirty and not args.allow_dirty:
        raise EvaluationError(
            "Uncommitted changes (or no git checkout): the recorded commit would "
            "not identify the code that ran. Commit first, or pass --allow-dirty "
            "for a report that must not be cited."
        )
    created_at = datetime.now(timezone.utc)
    out = args.out or PROJECT_ROOT / "results" / created_at.strftime("%Y%m%dT%H%M%SZ")
    if out.exists():
        raise EvaluationError(f"{out} already exists; reports are never overwritten.")

    try:
        predictor = inference.MnistInference(args.checkpoint, torch.device("cpu"))
    except (OSError, ValueError) as error:
        raise EvaluationError(f"Could not load checkpoint {args.checkpoint}: {error}")
    checkpoint = predictor.loaded_checkpoint
    checkpoint_dirty = (checkpoint.get("provenance") or {}).get("git_dirty", True)
    if checkpoint_dirty and not args.allow_dirty:
        raise EvaluationError(
            f"{args.checkpoint} was trained from uncommitted code (or records no "
            "provenance); it must not be released. Pass --allow-dirty to evaluate "
            "it anyway."
        )

    samples, source_images = export_test_images(args.data_dir)
    check_jpeg_manifest(samples, JPEG_MANIFEST)
    rows = evaluate_pairs(predictor, samples, source_images)
    metrics, per_class, confusion = summarize(rows)

    report = {
        "created_at_utc": created_at.isoformat(),
        "checkpoint": checkpoint_summary(args.checkpoint, checkpoint),
        "dataset": {
            "name": "MNIST",
            "split": "test",
            "sample_count": len(rows),
            "source_id": "original index in MNIST test split",
        },
        "encoding": {
            "pillow_version": PIL.__version__,
            "jpeg_codec_version": features.version_codec("jpg"),
            "image_mode": "L",
            "jpeg_quality": JPEG_QUALITY,
            "png_save_options": "Pillow defaults",
            "jpeg_manifest_sha256": sha256_of(JPEG_MANIFEST),
        },
        "preprocessing": predictor.saved_preprocessing,
        "provenance": {
            "git_commit": commit,
            "git_dirty": dirty,
            "torch_version": str(torch.__version__),
            "device": "cpu",
        },
        "metrics": metrics,
        "per_class_accuracy": per_class,
        "confusion_matrix": confusion,
    }
    _write_report(out, report, rows)
    print(
        f"PNG accuracy {metrics['png_accuracy']:.2%}, "
        f"JPEG accuracy {metrics['jpg_accuracy']:.2%}, "
        f"{metrics['disagreement_count']} disagreements"
    )
    print(f"Saved evaluation to {display_path(out)}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="MNIST download and exported test images",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="report directory to create (default: results/<UTC time>)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="evaluate with uncommitted changes or a dirty checkpoint; recorded",
    )
    args = parser.parse_args(argv)
    try:
        _evaluate_command(args)
    except EvaluationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
