# Development environment

Python 3.13 is used locally and in the development container. `pyproject.toml`
declares dependencies; `uv.lock` records their resolved versions. This project
runs from the repository root and is not installed as a Python package.

## Local setup

Install uv using its official installation instructions, then run from the
project root:

```bash
uv sync --locked --group notebooks
```

This creates `.venv` with runtime, development, and notebook dependencies.
For CLI-only development, use `uv sync --locked`. Select `.venv/bin/python`
in VS Code and for notebooks; restart notebook kernels after dependency updates.
Use `--group notebooks` with subsequent syncs when you need notebook dependencies.

Commands from the project root:

```bash
uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
uv run --locked pyrefly check
uv run --locked python -m src.cli tests/test_images/1_label_2.png --label 2
```

The default trained artifact is `models/checkpoint.pt`, produced by
`python -m src.train` (see Training). Run notebooks with `notebooks/` as their
working directory. They are for exploration: the training notebook saves its
own checkpoint to `runs/notebook/` (ignored), never over the shipped one.
Tests create temporary checkpoints and do not require the trained artifact;
one contract test also checks `models/checkpoint.pt` when it is present.

## Training

Training is configured by `configs/train.toml` and runs in two steps from the
project root:

```bash
uv run --locked python -m src.train search
uv run --locked python -m src.train final --search-run runs/<id> --out PATH
```

`search` trains every grid candidate on the 80/20 split and writes
`runs/<id>/search.json` (per-epoch losses, validation accuracy, provenance).
`final` selects the winner by the configured rule, refuses it below the
configured validation-accuracy floor, retrains it on all 60,000 training
images, and writes a CPU checkpoint that is checked by loading it through
`MnistInference`.

Both refuse to run with uncommitted changes, so the recorded git commit
identifies the code. `--allow-dirty` overrides this for experiments and is
recorded as `git_dirty: true`; such checkpoints must not be released. `final`
also refuses a search run made with a different config, and never overwrites
an existing checkpoint. `--device` defaults to CUDA, then MPS, then CPU.

## Evaluation

```bash
uv run --locked python -m src.evaluate                # models/checkpoint.pt
uv run --locked python -m src.evaluate --checkpoint PATH
```

This exports the 10,000 MNIST test images to `data/test/` as PNG and
quality-75 JPEG, decodes each file as the CLI does, and predicts on CPU through
`MnistInference`, the code path the Docker image ships. It writes
`results/<id>/summary.json` (accuracy, cross-entropy, Brier score, PNG→JPEG
transitions, per-class accuracy, confusion matrices, checkpoint hash and
provenance) and `per_image_results.csv`. It takes about 10 seconds.

PNG pixels are checked against MNIST. JPEG bytes depend on the Pillow/libjpeg
version, so the script refuses to run if they differ from
`manifests/test_jpeg.sha256`: the results would no longer describe the
recorded inputs. To check exported JPEGs by hand:

```bash
shasum -a 256 -c --quiet manifests/test_jpeg.sha256
```

Like training, evaluation refuses uncommitted changes, and also refuses a
checkpoint whose provenance is dirty or missing; `--allow-dirty` overrides
both for experiments. Evaluate a model once its training is final: the test
set is not used to choose between models.

## Releasing a model

1. Train: `search`, then `final --out` to a path outside the repository.
2. Copy the checkpoint to `models/checkpoint.pt`, commit, and evaluate it
   from that clean commit.
3. Commit the report, un-ignore its `results/<id>/` and the cited `runs/<id>/`
   in `.gitignore` (dropping the previous ones), and update the README model
   card and checkpoint hash.
4. Merge the PR only with green CI.
5. To publish, bump `version` in `pyproject.toml` (then `uv lock`) in that PR,
   and after merging tag the merge commit:

   ```bash
   git checkout main && git pull
   git tag -a v1.0.0 -m "v1.0.0"
   git push origin v1.0.0
   ```

`.github/workflows/release.yml` runs on the tag. It refuses a tag that is not
on `main`, does not match the project version, or already has an image; a
bundled checkpoint that does not match the committed report or was trained
from uncommitted code; and an image that fails the example prediction. It then
pushes `ghcr.io/abbylxw/mnist-ocr:<version>` labelled with the git commit and
checkpoint SHA-256. One git tag = one image tag = one checkpoint hash: to ship
a different checkpoint, release a new version.

## Artifacts kept in the repository

| Kept | Generated, not committed |
|---|---|
| `models/checkpoint.pt` (bundled by the Dockerfile) | Docker image (build it from `Dockerfile`) |
| `tests/test_images/1_label_2.png` (documentation example) | `data/` (MNIST download and exported test images) |
| the `results/<id>/` report cited in the README | other `results/` reports, `*.pth` scratch weights |
| the `runs/<id>/` search run the checkpoint cites | other `runs/` |
| `manifests/test_jpeg.sha256` (exported JPEG checksums) | |

To select a checkpoint explicitly:

```bash
uv run --locked python -m src.cli tests/test_images/1_label_2.png \
  --checkpoint models/checkpoint.pt
```

Ruff checks imports and common Python errors and provides formatting. Pyrefly
checks types in `src/` and `tests/`, including unannotated function bodies.
Notebooks are excluded from Ruff for now. Formatting and fixes are explicit:
`uv run ruff format src tests` and `uv run ruff check --fix src tests` modify
files; review their changes. Existing lint findings are not automatically fixed
by this setup.

To change dependencies, use `uv add NAME`, `uv add --dev NAME`, or
`uv add --group notebooks NAME`. Commit both `pyproject.toml` and `uv.lock`.
These are the dependency sources used by local setup and Docker builds;
no separate `requirements.txt` is maintained.

PyTorch and torchvision retain the versions used by this project. Linux resolves
CPU builds from the official PyTorch CPU index; macOS uses PyPI builds with MPS
support. Other dependencies were resolved during migration, so the new environment
may differ from the one that generated historical evaluation reports. Those
reports retain their recorded encoding versions.

## Dev Container CLI

Docker Desktop must be running. Node.js and npm are required on the host. The
Dev Container CLI is project-local and pinned in `package-lock.json`:

```bash
npm ci
npm run devcontainer -- up --workspace-folder .
npm run devcontainer -- exec --workspace-folder . uv run --locked python -m unittest discover -s tests -v
npm run devcontainer -- exec --workspace-folder . uv run --locked pyrefly check
```

You can also use VS Code's **Dev Containers: Reopen in Container** command.
The container installs runtime, development, and notebook dependencies with
`uv sync --locked --group notebooks`. Its Linux `.venv`
lives in a named Docker volume mounted over the workspace's `.venv`, keeping it
separate from the host's macOS environment. Source files, data, and checkpoints
are shared with the host. Linux containers on this Mac run on CPU; use the local
environment for MPS training.

The recommended VS Code extensions are recorded in `.vscode/extensions.json`;
they are installed automatically when opening the dev container. Ruff is the
Python formatter, and Pyrefly provides type checking. Formatting on save is not
enabled by this setup.

References: [uv installation](https://docs.astral.sh/uv/getting-started/installation/),
[uv projects](https://docs.astral.sh/uv/guides/projects/),
[Ruff configuration](https://docs.astral.sh/ruff/configuration/),
[Pyrefly configuration](https://pyrefly.org/en/docs/configuration/),
[Dev Container CLI](https://code.visualstudio.com/docs/devcontainers/devcontainer-cli).
