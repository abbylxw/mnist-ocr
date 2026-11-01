# MNIST project: progress and remaining work

Updated: 2026-10-27. Goal: take the project through the full cycle (model
development, evaluation, inference/deployment, version control) to a
production-level standard. Only items a production model requires are listed:
a versioned, reproducible, traceable artifact; checks that block bad changes;
and honest documentation of what the model can and cannot do.

## Decisions recorded

- **Training pipeline:** the hyperparameter search and the final training run
  both move into a script. The notebook stays for exploration and display.
- **Final model:** after selecting hyperparameters on the 48k/12k split, retrain
  on all 60,000 training images. Consequences, to handle in N3–N5:
  - The final model has no validation set, so the epoch count and
    hyperparameters come from the search unchanged.
  - Normalization statistics are recomputed from all 60,000 images.
  - Search-time validation metrics are recorded as selection evidence, not as
    metrics of the final model.
  - The new checkpoint replaces `models/checkpoint.pt`; README hashes and
    results must be regenerated.
- **Dropped:** training resumption (training takes ~20 s); a separate
  stratified-split task (MNIST is near-balanced).
- **Dropped as not required for production (2026-10-27):** loss-curve plots
  (per-epoch numbers are recorded instead), the misclassified-image grid,
  seed-variance study, a separate regeneration doc (the training and
  evaluation commands are the instructions), and branch protection (needs a
  paid plan for private repos; merge only on green CI by habit instead).
- **Interface stays a one-shot CLI in Docker.** An HTTP service is out of scope.

## Remaining (in order)

- [x] **P1. Commit the pending CI fix** via branch → PR (PR #1, merged
  2026-10-28). CI failed only on the 10 contract tests, as expected.
  From here on, every change goes branch → PR → green CI → merge.
- [x] **P2. Checkpoint contract check** (PR #2, merged 2026-10-28; first fully
  green CI). `ARCHITECTURE` and `PADDING_PER_SIDE` are constants in
  `nn_model.py` and `preprocessing.py`; `MnistInference` validates both before
  loading weights. Follow-up in PR (a): unreadable files, non-dictionary
  checkpoints, missing or ill-fitting `model_state_dict`, and directory paths
  also exit 1 without a traceback.
- [ ] **P3. Training script.** Design agreed 2026-10-31:
  - **Config:** TOML file (stdlib `tomllib`), e.g. `configs/train.toml`,
    holding the search grid, epochs, seeds, split ratio, selection rule, and
    sanity floor. Its contents or hash are stored in every output.
  - **Modes (one required):** `--search` writes all rows, per-epoch losses,
    and metadata (git commit, config hash, torch version) to `runs/<id>/`.
    `--final --search-run runs/<id>/` **auto-selects the winner** from that
    run and retrains it on all 60k images.
  - **Auto-select guardrails:** explicit rule (lowest final-epoch validation
    loss; fixed tie-break) recorded by name; `--final` refuses if the winner's
    validation accuracy is below the config's floor; the cited search run is
    committed so model changes show their evidence in the PR.
  - **60k refit:** normalization statistics recomputed on 60k; **same epoch
    count** as the search (≈25% more steps per epoch); the reason is recorded
    in the model card.
  - **Checkpoint:** tensors on CPU; search evidence (run path and hash, rule,
    winner's validation loss and accuracy) stored separately from this model's
    own fields; git commit, `git_dirty`, config hash, `uv.lock` hash, torch
    version, device.
  - **Git cleanliness:** refuse if `git status --porcelain` is non-empty
    unless `--allow-dirty`, which records `git_dirty: true`. Releases (P4/P6)
    reject dirty checkpoints.
  - **`runs/`:** ignored except the one run the model card cites.
  - **PR (a):** move `prepare_mnist_subset` into `src/` with an image-label
    alignment test; make the CLI exit 1 without a traceback for a non-checkpoint
    file, missing `model_state_dict`, or mismatched weights (P2 review); tick
    P1 and P2.
  - **PR (b):** the training script. Write the checkpoint outside `models/`;
    swapping it in happens in P4.
  - Done when: both modes run from a clean checkout; the batch 64 / lr 1e-3 /
    decay 0 search row reproduces `eval_loss 0.042647618…`; the new checkpoint
    loads through `MnistInference` and passes the contract tests; CI is green
    on both PRs.
- [ ] **P4. Evaluation script.** Evaluate the new checkpoint on CPU through
  `MnistInference` (the shipped code path), writing to `results/`. Replace the
  cited report, README hash, and metrics; confirm
  `manifests/test_jpeg.sha256` still verifies.
- [ ] **P5. Model card** (README section): intended use, accepted inputs,
  training data and the 60k refit, selection rationale, test metrics with the
  report link, per-class accuracy, known limitations (inverted, off-centre,
  multi-digit inputs), and an honest statement of test-set use. Include the
  two reproduce commands (train, evaluate).
- [ ] **P6. Release v1.0.0.** A CD workflow triggered by a `v*` tag builds the
  image in CI and pushes it to GHCR as `ghcr.io/abbylxw/mnist-ocr:1.0.0`, with
  the git commit and checkpoint SHA-256 as image labels. Document the rule:
  one git tag = one image tag = one checkpoint hash.

Resume here: **P3, PR (b)** (the training script), once PR (a) on branch
`train-data-helpers` is merged. PR (a) adds `src/data.py` (load, split,
select, fit statistics, build dataset); the refactored notebook reproduces
`models/checkpoint.pt` byte for byte.

## Done

- **Validation strategy:** fixed 80/20 split (48,000/12,000, split seed 11);
  normalization statistics from the training subset only; validation
  cross-entropy for candidate comparison.
- **Training:** reusable batch-update, training, and evaluation functions in
  `src/nn_model.py`; five-epoch training; 36-configuration search over batch
  size, learning rate, and weight decay; per-configuration re-seeding
  (2026-10-24); saved model (batch 64, lr 0.001, decay 0; validation accuracy
  98.725%).
- **Reproducibility:** checkpoint records seeds, hyperparameters, sample
  counts, preprocessing statistics, optimizer state, and epochs; dependencies
  locked in `pyproject.toml`/`uv.lock`. A fresh-process run of the final-model
  cells on MPS reproduced `models/checkpoint.pt` byte for byte (SHA-256
  `103a1679…`, 2026-10-24); search and other devices not re-run.
- **Validation:** evaluation mode, no gradients, sample-weighted mean
  cross-entropy, preprocessing shared with training.
- **Evaluation:** paired PNG/JPEG (quality 75) test evaluation with accuracy,
  cross-entropy, Brier score, and correctness transitions; report in
  `results/20261019T042353529283Z/` matches the checkpoint hash; README
  updated (PNG 98.74%, JPEG 98.72%, four disagreements).
- **Inference separation:** PyTorch checkpoint and preprocessing contract
  documented (README section 8); `src/inference.py` loads weights and
  preprocessing without training code; prediction verified in a fresh CLI
  process and in the container.
- **Application interface:** one-shot CLI (image path, optional label and
  checkpoint); input validation with explicit failure behavior; decoding,
  preprocessing, and inference in separate modules.
- **Tests:** preprocessing, checkpoint restoration, prediction outputs,
  subprocess CLI behavior, and error cases; shared `make_checkpoint` fixture;
  contract tests written for N2.
- **Packaging:** `pyproject.toml`/`uv.lock` only; dev container; application
  Dockerfile (locked dependencies, bundled checkpoint, non-root user);
  container success and error exit codes verified, including offline from an
  isolated directory; `.dockerignore`; Docker usage in README.
- **Lean repository:** `.gitignore` excludes `data/`, extra `results/` runs,
  and `*.pth`; JPEG checksums in `manifests/test_jpeg.sha256`; artifact policy
  in `DEVELOPMENT.md`; unused test files removed (2026-10-24).
- **Version control:** git repo on `main`, initial commit `4b4a6bc`; private
  remote `https://github.com/abbylxw/mnist-ocr`, pushed 2026-10-25.
- **CI:** `.github/workflows/ci.yml` on pushes to `main` and PRs. Job
  `checks` runs uv at the `uv.lock` version, then tests, ruff, format check,
  and pyrefly. Job `docker` builds the image and runs an offline prediction.
  Actions are pinned by SHA. The first run (40224825126) caught an
  MPS-tagged checkpoint load in a test; fixed in P1.
