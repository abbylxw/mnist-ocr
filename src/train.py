"""Train the digit classifier: hyperparameter search, then a refit on all data.

From the project root:

    uv run --locked python -m src.train search
    uv run --locked python -m src.train final --search-run runs/<id> --out PATH

`search` compares candidates on a train/validation split and writes
`runs/<id>/search.json`. `final` selects the winner from that run, retrains it
on all training images, and writes a CPU checkpoint. Both refuse to run with
uncommitted changes unless `--allow-dirty` is given; that flag is recorded.
"""

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src import data, inference, nn_model, preprocessing

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train.toml"
SELECTION_RULE = "min_final_validation_loss"

_REQUIRED_CONFIG = {
    "data": ["dir"],
    "split": ["train_ratio", "seed"],
    "training": ["seed", "epochs"],
    "search": ["batch_sizes", "learning_rates", "weight_decays"],
    "selection": ["rule", "min_validation_accuracy"],
}


class TrainingError(Exception):
    """A training run cannot proceed; the message explains why."""


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_path(path):
    """Return the path relative to the project root when it lies inside it."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_config(path):
    """Return (config, sha256) for a training TOML file."""
    raw = Path(path).read_bytes()
    config = tomllib.loads(raw.decode("utf-8"))
    for section, keys in _REQUIRED_CONFIG.items():
        for key in keys:
            if key not in config.get(section, {}):
                raise TrainingError(f"Config {path} is missing [{section}] {key}.")
    rule = config["selection"]["rule"]
    if rule != SELECTION_RULE:
        raise TrainingError(
            f"Unsupported selection rule {rule!r}; expected {SELECTION_RULE!r}."
        )
    return config, hashlib.sha256(raw).hexdigest()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_state():
    """Return (commit, dirty); a missing git checkout counts as dirty."""
    try:
        commit = _git("rev-parse", "HEAD").strip()
        dirty = bool(_git("status", "--porcelain").strip())
    except (OSError, subprocess.CalledProcessError):
        return None, True
    return commit, dirty


def _git(*arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def provenance(device, config_sha256, *, allow_dirty):
    """Record what produced a run; refuse uncommitted code unless allowed."""
    commit, dirty = git_state()
    if dirty and not allow_dirty:
        raise TrainingError(
            "Uncommitted changes (or no git checkout): the recorded commit would "
            "not identify the code that ran. Commit first, or pass --allow-dirty "
            "for a run that must not be released."
        )
    lockfile = PROJECT_ROOT / "uv.lock"
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "config_sha256": config_sha256,
        "uv_lock_sha256": sha256_of(lockfile) if lockfile.is_file() else None,
        "torch_version": str(torch.__version__),
        "device": str(device),
    }


def _finite_or_none(value):
    """Keep JSON valid: a diverged run's NaN or infinite loss becomes null."""
    return value if math.isfinite(value) else None


def train_model(
    train_set,
    *,
    batch_size,
    learning_rate,
    weight_decay,
    epochs,
    seed,
    device,
    validation_set=None,
):
    """Train from a fixed seed; return the model and per-epoch losses.

    Seeding immediately before building the model gives every run the same
    initial weights and shuffle order, whatever its position in a search.
    """
    torch.manual_seed(seed)
    shuffle_generator = torch.Generator().manual_seed(seed)
    model = nn_model.NeuralNetwork().to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, generator=shuffle_generator, shuffle=True
    )
    validation_loader = None
    if validation_set is not None:
        validation_loader = DataLoader(
            validation_set, batch_size=batch_size, shuffle=False
        )

    history = []
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = nn_model.train_one_epoch(
            model, optimizer, loss_function, train_loader
        )
        record = {"epoch": epoch, "train_loss": _finite_or_none(train_loss)}
        if validation_loader is not None:
            validation_loss, _, _ = nn_model.model_eval(
                model, validation_loader, loss_function
            )
            record["validation_loss"] = _finite_or_none(validation_loss)
        history.append(record)
    return model, history


def accuracy(model, dataset, *, batch_size, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in DataLoader(dataset, batch_size=batch_size):
            predictions = model(images.to(device)).argmax(dim=1)
            correct += (predictions == labels.to(device)).sum().item()
    return correct / len(dataset)


def run_search(images, labels, config, device, log=print):
    """Train every grid candidate on the split; return the search results."""
    split = config["split"]
    train_indices, validation_indices = data.split_indices(
        len(labels),
        split["train_ratio"],
        generator=torch.Generator().manual_seed(split["seed"]),
    )
    train_images, train_labels = data.select(images, labels, train_indices)
    validation_images, validation_labels = data.select(
        images, labels, validation_indices
    )
    # Statistics come from the training part only and are reused for validation.
    mean, std = data.fit_statistics(train_images)
    train_set = data.to_dataset(train_images, train_labels, mean, std)
    validation_set = data.to_dataset(validation_images, validation_labels, mean, std)

    grid = config["search"]
    candidates = []
    for batch_size, learning_rate, weight_decay in itertools.product(
        grid["batch_sizes"], grid["learning_rates"], grid["weight_decays"]
    ):
        log(
            f"batch_size={batch_size} learning_rate={learning_rate} "
            f"weight_decay={weight_decay}"
        )
        model, history = train_model(
            train_set,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=config["training"]["epochs"],
            seed=config["training"]["seed"],
            device=device,
            validation_set=validation_set,
        )
        candidates.append(
            {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "validation_loss": history[-1]["validation_loss"],
                "validation_accuracy": accuracy(
                    model, validation_set, batch_size=batch_size, device=device
                ),
                "epochs": history,
            }
        )
    return {
        "split": {
            "seed": split["seed"],
            "train_ratio": split["train_ratio"],
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
        },
        "normalization": {"train_mean": mean, "train_std": std},
        "candidates": candidates,
    }


def select_winner(candidates, min_validation_accuracy):
    """Apply the selection rule; refuse a winner below the accuracy floor.

    Lowest final-epoch validation loss wins; ties go to the smaller learning
    rate, then batch size, then weight decay. Diverged candidates are skipped.
    """
    finite = [c for c in candidates if c["validation_loss"] is not None]
    if not finite:
        raise TrainingError("Search run has no candidate with a finite loss.")
    winner = min(
        finite,
        key=lambda c: (
            c["validation_loss"],
            c["learning_rate"],
            c["batch_size"],
            c["weight_decay"],
        ),
    )
    if winner["validation_accuracy"] < min_validation_accuracy:
        raise TrainingError(
            f"Winner's validation accuracy {winner['validation_accuracy']:.4%} is "
            f"below the configured floor {min_validation_accuracy:.4%}."
        )
    return winner


def run_final(images, labels, config, winner, device):
    """Retrain the winner on all images; return (model, mean, std, history)."""
    mean, std = data.fit_statistics(images)
    model, history = train_model(
        data.to_dataset(images, labels, mean, std),
        batch_size=winner["batch_size"],
        learning_rate=winner["learning_rate"],
        weight_decay=winner["weight_decay"],
        epochs=config["training"]["epochs"],
        seed=config["training"]["seed"],
        device=device,
    )
    return model, mean, std, history


def build_checkpoint(
    model,
    *,
    mean,
    std,
    history,
    train_samples,
    config,
    winner,
    search_record,
    search_run,
    search_run_sha256,
    run_provenance,
):
    """Assemble a device-neutral checkpoint for the inference contract.

    `selection` holds the search evidence for the chosen hyperparameters; it
    describes the candidate trained on the split, not this model.
    """
    return {
        "architecture": nn_model.ARCHITECTURE,
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
        "preprocessing": {
            "input_shape": [1, 28, 28],
            "input_dtype": "uint8",
            "pixel_range": [0, 255],
            "padding_per_side": preprocessing.PADDING_PER_SIDE,
            "padding_value": 0,
            "output_dtype": "float32",
            "scale_to_unit_interval": False,
            "train_mean": mean,
            "train_std": std,
        },
        "training": {
            "optimizer": "Adam",
            "batch_size": winner["batch_size"],
            "learning_rate": winner["learning_rate"],
            "weight_decay": winner["weight_decay"],
            "epochs": len(history),
            "seed": config["training"]["seed"],
            "train_samples": train_samples,
            "epoch_train_losses": [record["train_loss"] for record in history],
        },
        "selection": {
            "rule": SELECTION_RULE,
            "search_run": search_run,
            "search_run_sha256": search_run_sha256,
            "search_git_commit": search_record["provenance"]["git_commit"],
            "split": search_record["split"],
            "validation_loss": winner["validation_loss"],
            "validation_accuracy": winner["validation_accuracy"],
        },
        "provenance": run_provenance,
    }


def _search_command(args):
    config, config_sha256 = load_config(args.config)
    device = resolve_device(args.device)
    run_provenance = provenance(device, config_sha256, allow_dirty=args.allow_dirty)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or PROJECT_ROOT / "runs" / run_id
    if out.exists():
        raise TrainingError(f"{out} already exists; search runs are never overwritten.")

    images, labels = data.load_mnist_train(_data_dir(config))
    result = run_search(images, labels, config, device)

    out.mkdir(parents=True)
    record = {
        "provenance": run_provenance,
        "config": config,
        "selection_rule": SELECTION_RULE,
        **result,
    }
    (out / "search.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved search run to {display_path(out)}")


def _final_command(args):
    config, config_sha256 = load_config(args.config)
    search_path = args.search_run
    if search_path.is_dir():
        search_path = search_path / "search.json"
    record = json.loads(search_path.read_text(encoding="utf-8"))
    if record["provenance"]["config_sha256"] != config_sha256:
        raise TrainingError(
            f"{args.config} changed since {display_path(search_path)} was made; "
            "run a new search with the current config."
        )
    if args.out.exists():
        raise TrainingError(
            f"{args.out} already exists; checkpoints are never overwritten."
        )

    device = resolve_device(args.device)
    run_provenance = provenance(device, config_sha256, allow_dirty=args.allow_dirty)
    winner = select_winner(
        record["candidates"], config["selection"]["min_validation_accuracy"]
    )
    print(
        f"Winner: batch_size={winner['batch_size']} "
        f"learning_rate={winner['learning_rate']} "
        f"weight_decay={winner['weight_decay']} "
        f"(validation loss {winner['validation_loss']:.6f}, "
        f"accuracy {winner['validation_accuracy']:.2%})"
    )

    images, labels = data.load_mnist_train(_data_dir(config))
    model, mean, std, history = run_final(images, labels, config, winner, device)
    checkpoint = build_checkpoint(
        model,
        mean=mean,
        std=std,
        history=history,
        train_samples=len(labels),
        config=config,
        winner=winner,
        search_record=record,
        search_run=display_path(search_path),
        search_run_sha256=sha256_of(search_path),
        run_provenance=run_provenance,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)

    # The checkpoint must load through the shipped inference path.
    inference.MnistInference(args.out, torch.device("cpu"))
    print(
        f"Saved checkpoint to {display_path(args.out)} (sha256 {sha256_of(args.out)})"
    )


def _data_dir(config):
    path = Path(config["data"]["dir"])
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m src.train", description=__doc__.split("\n\n")[0]
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    common.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    common.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run with uncommitted changes; recorded as git_dirty=true",
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    search = modes.add_parser(
        "search", parents=[common], help="compare grid candidates on a split"
    )
    search.add_argument(
        "--out", type=Path, help="run directory to create (default: runs/<UTC time>)"
    )
    final = modes.add_parser(
        "final", parents=[common], help="retrain the search winner on all data"
    )
    final.add_argument(
        "--search-run",
        type=Path,
        required=True,
        help="search run directory or its search.json",
    )
    final.add_argument(
        "--out", type=Path, required=True, help="checkpoint file to create"
    )
    args = parser.parse_args(argv)

    try:
        if args.mode == "search":
            _search_command(args)
        else:
            _final_command(args)
    except TrainingError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
