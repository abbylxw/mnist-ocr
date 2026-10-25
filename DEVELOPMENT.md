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

The default trained artifact is `models/checkpoint.pt`. Run notebooks with
`notebooks/` as their working directory; their checkpoint path is
`../models/checkpoint.pt`. The training notebook creates `models/` before saving.
Tests create temporary checkpoints and do not require the trained artifact;
one contract test also checks `models/checkpoint.pt` when it is present.

## Artifacts kept in the repository

| Kept | Generated, not committed |
|---|---|
| `models/checkpoint.pt` (bundled by the Dockerfile) | Docker image (build it from `Dockerfile`) |
| `tests/test_images/1_label_2.png` (documentation example) | `data/` (MNIST download and exported test images) |
| `results/20261019T042353529283Z/` (report cited in the README) | other `results/` runs, `*.pth` scratch weights |
| `manifests/test_jpeg.sha256` (exported JPEG checksums) | |

`notebooks/inference.ipynb` exports `data/test/`. PNG pixels are checked
against MNIST during evaluation. JPEG bytes depend on the Pillow/libjpeg
version, so after exporting, confirm they match the evaluated inputs:

```bash
shasum -a 256 -c --quiet manifests/test_jpeg.sha256
```

No output means every JPEG matches. A mismatch means the recorded results no
longer describe these inputs.

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
