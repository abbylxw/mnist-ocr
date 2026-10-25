# MNIST project: progress and remaining work

Updated: 2026-10-24. Goal: experience the full cycle once (model development,
evaluation, inference/deployment, version control). Only necessary items are
listed; nice-to-haves were deliberately left out.

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
  stratified-split task (MNIST is near-balanced; one sentence in the decision
  log covers it).

## Remaining (in order)

- [ ] **N1. Version control.** `git init`, review `.gitignore`, first commit,
  small meaningful commits from then on, push to a remote.
- [ ] **N2. Checkpoint contract check.** Validate `architecture` and
  `preprocessing.padding_per_side` against code at load time; make
  `tests/test_checkpoint_contract.py` and the CLI contract test pass
  (10 failing tests today, by design).
- [ ] **N3. Training script.** One entry point (e.g. `python -m src.train`)
  that runs the search on the 48k/12k split, then retrains the selected
  configuration on all 60k images and writes the checkpoint. Move
  `prepare_mnist_subset` and related helpers into `src/`. Decide the
  checkpoint fields for a model trained without a validation split.
- [ ] **N4. Run records.** Save per-epoch train/validation loss for each search
  candidate and per-epoch train loss for the final run, plus environment
  metadata (torch version, device, git commit, lockfile hash) in the checkpoint
  and run outputs. Plot the curves.
- [ ] **N5. Re-run and decision log.** Run the search with per-configuration
  seeding; confirm the batch 64 / lr 1e-3 / decay 0 row reproduces
  `eval_loss 0.042647618…`. Write one decision log covering the split, search
  scope, chosen configuration, epoch count (from the curves), and the 60k
  refit.
- [ ] **N6. Evaluate through the shipped path.** Evaluate the new checkpoint on
  CPU through `MnistInference`, not a notebook copy of the pipeline. Update the
  README hash, metrics, and cited report; check `manifests/test_jpeg.sha256`
  still verifies.
- [ ] **N7. Error analysis and test-set note.** Confusion matrix, a grid of
  misclassified images, limitations. State plainly what was and was not frozen
  before test results were seen, for both the original and the 60k model.
- [ ] **N8. Regeneration instructions.** Document how to reproduce every
  artifact: checkpoint, exported test images and checksums, results, Docker
  image.
- [ ] **N9. CI.** GitHub Actions: tests, ruff, pyrefly, and Docker build on
  every push.
- [ ] **N10. Release v1.** Git tag, matching Docker image tag, and a written
  rule relating code version, model version (checkpoint hash), and image tag.

Next small task: N1, then N2 as the first real commit.

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
