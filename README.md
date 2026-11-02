[Development setup and tool commands](DEVELOPMENT.md)

The trained checkpoint is stored at `models/checkpoint.pt`. It contains model
weights, preprocessing statistics, training settings, selection evidence, and
provenance. The CLI uses this path by default when run from the project root;
use `--checkpoint PATH` to override it. See the [model card](#model-card) for
what the model is for and how well it works.

## Run with Docker

Install and start Docker (Docker Desktop on macOS). From the project root:

```bash
docker build -t mnist-ocr .
docker run --rm \
  --mount "type=bind,source=$(pwd)/tests/test_images,target=/inputs,readonly" \
  mnist-ocr /inputs/1_label_2.png --label 2
```

Expected output:

```text
Image path: /inputs/1_label_2.png
Predicted digit: 2
True digit: 2
```

Released images are published to GitHub Container Registry. The repository
is private, so log in first with a personal access token that has the
`read:packages` scope:

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u abbylxw --password-stdin
docker run --rm \
  --mount "type=bind,source=$(pwd)/tests/test_images,target=/inputs,readonly" \
  ghcr.io/abbylxw/mnist-ocr:1.0.0 /inputs/1_label_2.png --label 2
```

Each image records its git commit and checkpoint SHA-256 as labels:
`docker image inspect ghcr.io/abbylxw/mnist-ocr:1.0.0 --format '{{json .Config.Labels}}'`.

For your own image, change `source` to its directory's absolute host path and
pass `/inputs/<filename>` to the CLI. CLI paths refer to files inside the
container. `--label` is optional; it displays a known label for comparison.

Inputs must be PNG or JPEG images decoding to 28×28 grayscale uint8 pixels,
with one centered light digit on a dark background (MNIST-style). The checkpoint
is bundled; no local Python setup or checkpoint mount is required. The container
prints the prediction and exits; `--rm` removes it afterward.

## Model card

| | |
|---|---|
| Version | 1.0.0 (`ghcr.io/abbylxw/mnist-ocr:1.0.0`) |
| Model | LeNet-5-style CNN, `lenet5_relu_maxpool_dense_v1` (61,706 parameters; section 5) |
| Checkpoint | `models/checkpoint.pt`, SHA-256 `eedb4358ce3940d96fa6e5843b0c3d95544b086571d8e12f5ffcc6a0a3209aaf` |
| Trained from | commit `d8b292a`, clean tree, PyTorch 2.14.0 on MPS |
| Evaluated by | [results/20261102T040050Z](results/20261102T040050Z/summary.json), commit `ccf458a`, CPU |

**Intended use.** Classifying one handwritten digit (0–9) in an MNIST-style
image, as a hands-on learning project covering training, evaluation, packaging,
and release. It is not intended for real-world documents, photographs, or any
decision with consequences.

**Inputs and output.** A PNG or JPEG decoding to a 28×28 grayscale uint8 image
containing one centered light digit on a dark background. Shape, channel count,
and dtype are validated; image content is not. The CLI prints the most likely
digit; the model always outputs one of the ten digits, with no rejection of
non-digit inputs.

**Training data.** All 60,000 images of the MNIST training split. Images are
zero-padded to 32×32 and standardized with statistics from all 60,000 padded
images (mean 25.5094, standard deviation 70.1804).

**Training procedure.** Adam, batch size 64, learning rate 0.001, no weight
decay, 5 epochs, seed 11 (`configs/train.toml`). Final-epoch training loss
0.0299.

**How the settings were chosen.** A 36-candidate grid (batch size 32/64/128 ×
learning rate 1e-4/1e-3/1e-2 × weight decay 0/1e-4/1e-3/1e-2) was trained on a
fixed 48,000/12,000 split (seed 11). The rule, chosen before the search, takes
the lowest final-epoch validation cross-entropy, with a validation accuracy floor
of 98%. The winner reached validation loss 0.0426 and accuracy 98.73%
([search run](runs/20261101T074423Z/search.json)). It was then retrained on all
60,000 images for the same 5 epochs (938 instead of 750 steps per epoch), so the
shipped model uses all labelled training data. The trade-off: the shipped model
has no held-out validation set of its own, and its validation numbers above
describe the 48,000-image candidate, not this model.

**Test results** (10,000 MNIST test images, each evaluated as PNG and as
quality-75 JPEG; section 7):

| Metric | PNG | JPEG |
|---|---:|---:|
| Accuracy | 99.05% | 99.11% |
| Mean cross-entropy | 0.0304 | 0.0299 |
| Mean Brier score | 0.0149 | 0.0147 |

Per-class accuracy (PNG): 0: 99.6%, 1: 99.5%, 2: 99.5%, 3: 99.2%, 4: 99.3%,
5: 98.9%, 6: 98.4%, 7: 99.0%, 8: 98.5%, 9: 98.5%. The most frequent errors are
6 read as 5 (7 images), 4 as 9 (6), 5 as 3 (5), and 9 as 7 (5). Full confusion
matrices are in the report.

**Limitations.**

- Measured only on MNIST, which comes from the same distribution as the
  training data. Performance on other handwriting, scans, or photos is unknown.
- Dark digits on a light background, off-centre or rescaled digits, several
  digits, and non-digit images pass input validation but are likely to be
  misclassified, and the model gives no warning.
- Inputs that are not 28×28 grayscale are rejected rather than resized.
- One training seed was run; the variation between seeds is not measured, so
  small differences from other models (such as the previous 48,000-image
  checkpoint's 98.74%) are not established as significant.

**Use of the test set.** Hyperparameters were selected on the validation split
only. The test set was evaluated once for the previous 48,000-image checkpoint
and once for this model. No training or selection setting was changed after
seeing this model's test results.

**Reproduce** (from a clean checkout; `data/` is downloaded on first use):

```bash
uv run --locked python -m src.train search     # prints "Saved search run to runs/RUN_ID"
uv run --locked python -m src.train final --search-run runs/RUN_ID --out /tmp/checkpoint.pt
uv run --locked python -m src.evaluate --checkpoint /tmp/checkpoint.pt
```

On the same hardware and library versions, `search` reproduces the cited run's
losses. GPU results can differ slightly between devices and library versions.



1. What exactly is the input to my application?
   JPG/JPEG/PNG common image files, but for training I'll directly use torchvision.dataset and leave the image -> model input format (fixed-size numeric representation) to the inference layer.
   Training: fixed-size numeric representation → preprocessing → model → loss → update
   Inference: Raw JPG/PNG files → fixed-size numeric representation -> preprocessing → model → prediction
   Fixed-size numeric representation before preprocessing:
   a. tensor (1*28*28) with data type torch.uint8 for each image;
   b. pixel-value range 0 ~ 255 (PILToTensor does not scale data)
   c. label type: int
   Preprocessing: add a 2-pixel zero-valued border on all sides, convert to torch.float32, then standardize. Compute one mean and standard deviation across all pixels of the padded training images, including the border. Reuse these fixed statistics at inference time.
   Keep the original training data and create a separate preprocessed tensor; labels are unchanged.
   Model input after preprocessing: torch.float32 with shape (1, 32, 32) per image or (N, 1, 32, 32) per batch.

   V1 input format decision: based on 0_image_5 training data sample.
   | Format | Pixels changed | Mean absolute error | Maximum absolute error |
   |---|---:|---:|---:|
   | PNG | 0% | 0 | 0 |
   | Default JPEG | 46.94% | 3.00 | 30 |
   | Quality-95 JPEG | 40.43% | 0.77 | 6 |

Reasoning:

1. mismatch percentage alone only reflect the fraction of pixel mismatch, but not necessarily large change to each pixel
2. For my testing case, change to inputs are fine because those files are still the same label just with variation of the image encoding itself. The trained models should be able to recognize these images even if they loss some information during JPEG encoding.
3. For each source digit image, I will create a JPEG with quality explicitly set to 75 and a PNG. I will compare paired predictions and report accuracy, mean cross-entropy, and mean Brier score separately for each encoding, as defined in section 7.

   What's not acceptable:
   1. RGB color digit image
   2. multiple digits in one image
   3. real world image where digit is not centralized, and background is varied.

2) What exactly is the output?
   Digit prediction 0 - 9

3) What does MNIST give me versus what a real user would give me?
   Real user photos may have blurred background/multiple characters/different colors including digits in a photo; while mnist gives only 1 white digit in the center in a dark background.
   Version 1:
   In my case, given i don't have labeled real-world pictures where multiple digits appear in different backgrounds, i'd stick to MNIST dataset.
   Training: MNIST training dataset tensor as feature and label as target. Prediction: export each MNIST test sample as paired JPEG and PNG files, decode each file and convert it to a tensor at inference time, and compare each predicted digit with the source test label.
   Guardrail: V1 accept MNIST dataset samples only.
   Input check:
   1. whether the image is gray scale 28\*28 PIL image with pixel-value 0 - 255.
   2. Assumptions lacking validation: whether the image contains a single digit in the center, whether the digit is white with dark background. These are assumed because i feed MNIST images.

   Version 2: multiple digit MNIST
   Version 3: multiple digit in less-controlled image (not centered, colored)
   Version 4: multiple digits embeeded in real-world photographs

4) What preprocessing is therefore required?
V1: convert the (1, 28, 28) uint8 input in the 0–255 range to a (1, 32, 32) float32 model input using zero padding, floating-point conversion, and standardization as defined in section 1.
Future input scopes are listed in section 3; their preprocessing is not implemented.
<!--
a. identify digit/digits location
b. center it
c. rotation?
d. grayscale / polarity normalization
e. resize to 28×28 -->

## 5. What does "OCR backbone" mean?

model architecture to identify optical character.

### Simpler models for baseline/comparison:

**Multiclass logistic regression:** each pixel treated as one feature, find their linear relationship with target 0~9 10 classes.

**MLP:** nonlinear combinations/interactions of pixels.
Hidden layers can learn combinations/interactions of pixels, while nonlinear activation functions allow the network to represent nonlinear decision boundaries. Multiclass logistic regression has only a linear mapping from pixels to class logits.

**CNN:**

nonlinear relationships + explicit local spatial structure + shared weights across locations

### Final model selection

LeNet-5-style CNN variant:

**Decision:**

Pad MNIST 28×28 → 32×32 before standardization.

**Primary reason:**

Use LeNet-5-style input size and layer dimensions for V1, with ReLU, max pooling, and dense convolution connectivity. This is a variant, not an exact reproduction of the original architecture.

**Secondary benefit:**

The zero-valued border represents dark background before standardization. Zero after standardization would instead represent the training mean used in that transformation.

**Trade-off:**

Adds artificial zero-valued border pixels and slightly more computation.
MNIST digits are generally centered, so the practical accuracy benefit
may be small.

Other OCR backbones are overkill, given MNIST images are so simple. When expanding the project to more complex versions, would visit other OCR backbones.

**Architecture choices:**

- Apply ReLU after Conv1, Conv2, Conv3, and FC1. FC2 produces 10 logits.
- Use max pooling for both pooling layers.
- Use dense convolution connectivity: every output channel connects to all input channels. This simplification fits my compute budget and permits more feature combinations; better performance remains a hypothesis to evaluate.
- Conv2 has 6 × 16 × 5 × 5 = 2,400 kernel weights plus 16 biases, totaling 2,416 trainable parameters.

### Model architecture

```text
60000: total training dataset size
N: batch size
1: channels (gray scale)
raw MNIST sample
      ↓
N, 1, 28, 28
      ↓ zero padding (2 pixels per side), float32 conversion, standardization
[N, 1, 32, 32]
      ↓ 6 5 * 5 kernel to extract features
Conv1 → N, 6, 28, 28
      ↓ ReLU activation followed by max pooling 2 * 2
Pool1 → N, 6, 14, 14
      ↓ 16 5 _ 5 kernel to extract features
Conv2 → N, 16, 10, 10
      ↓ ReLU activation followed by max pooling 2 * 2
Pool2 → N, 16, 5, 5
      ↓ 120 5 * 5 kernal
Conv3 -> N, 120, 1, 1
      ↓ ReLU activation followed by flatten
   N, 120
      ↓ FC1
   N, 84
      ↓ ReLU activation followed by FC2
   N, 10 logits
      ↓ argmax
   predicted digit
```

### Training

```text
image
↓
model
↓
10 logits ──────────┐
                    ↓
target class ─→ CrossEntropyLoss
                    ↓
                  loss
                    ↓
               backpropagation
```

### Resources Expectation

```text
Model: LeNet-style CNN
Dataset: MNIST, 60k training examples
Training workload: ~938 optimizer steps/epoch at batch size 64
Compute: ~0.42M forward MACs/image by rough calculation
Hardware candidates: M4 CPU vs MPS
Training-time estimate: to be established empirically by timing representative training batches.
```

## 6. What candidate backbones should I investigate?

OCR backbone is overkill for V1. V1 model selection process in section 5.

### CNN Backbones (Traditional & Lightweight)

Examples: ResNet (ResNet34, ResNet50), MobileNet, PPLCNetV4, and PPHGNetV2.

### Vision Transformers & MetaFormer/VLM Backbones (Modern)

Examples: CLIP ViT, Phi-3.5 vision encoders, or document-native encoders like MonkeyOCRv2 Vision Encoder and Qwen/MetaFormer blocks.

### Hybrid / State Space Models (SSMs)

Examples: CNN + Mamba or Transformer sequence encoders.

7. How will I know whether the system works?
   given a bunch of labeled images, it can give correct predictions with accuracy > 90%.

### Training configuration

Training batches are shuffled each epoch by explicitly setting `shuffle=True` in the training DataLoader. Evaluation uses `shuffle=False` to keep sample ordering predictable; source IDs identify matching JPEG/PNG pairs.

For each reported training run, record the random seed(s) used for initialization and shuffling, the training configuration, and the checkpoint identifier. The exploratory shuffle comparison remains in the notebook.

### V1 evaluation: paired PNG/JPEG comparison

#### Experiment setup

- Source test samples and sample count: 10000 test samples from MNIST test dataset.
- Pairing: create one PNG and one JPEG (explicit quality = 75) from each source image. This produces 20,000 encoded files representing 10,000 source pairs.
- Source ID used to match each pair: idx for each test sample.
- Model: use the same fixed trained model for both encodings.
- Shared preprocessing: zero padding to 32×32, float32 conversion, and standardization using the fixed mean/std computed from padded training images, including border pixels.
- Encoding library/version and export settings: recorded below and in the linked evaluation report.
- Comparison reference: PNG. Correctness is determined by the source dataset label.

Encoding details:
Pillow: 12.3.0
JPEG codec: 6.2
Image mode: L
JPEG setting: quality = 75
PNG: no explicit save options; Pillow defaults

#### Results recorded for each source image

| Field                                              | Definition / representation                                                                                                  |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Source ID and true label                           | Source index in the MNIST test split and its dataset label                                                                   |
| PNG and JPEG predicted classes                     | arg.max(p)                                                                                                                   |
| Model outputs needed for probability-based metrics | predicted probability for each class                                                                                         |
| Cross-entropy for each encoding                    | For one image: `-ln(p_true)`, where `p_true` is the probability assigned to its true class                                   |
| Brier score for each encoding                      | For one image: sum `(p_k - y_k)^2` over all 10 classes, where `y` is the one-hot true label                                  |
| Prediction disagreement indicator                  | True when the PNG and JPEG predicted class IDs differ; false otherwise                                                       |
| Correctness transition from PNG to JPEG            | Compare each prediction with the true label and assign the applicable outcome below; otherwise mark the prediction unchanged |

#### Paired prediction changes

| Outcome (PNG → JPEG)          | Interpretation                                 | Count                                                                                                                   | Rate denominator |
| ----------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------- |
| Correct → wrong               | Degradation in correctness                     | # of test samples correctly predicted by PNG encoding but wrongly by JPEG encoding                                      | 10k              |
| Wrong → correct               | Improvement in correctness                     | # of test samples wrongly predicted by PNG encoding but correctly by JPEG encoding                                      | 10k              |
| Wrong → different wrong class | Disagreement without a correctness improvement | # of test samples wrongly predicted by both PNG encoding and by JPEG encoding and their predicted classes are different | 10k              |

- Total disagreement count: number of source pairs with different predicted classes, equal to the sum of the three transition counts above. Disagreement rate: this count divided by 10,000 source pairs.
- Pairs with unchanged predicted classes contribute zero to disagreement and remain in all accuracy and loss calculations. Both predictions are either correct or the same wrong class. Their losses may still differ because the predicted probabilities can change; a lower loss indicates improvement under that metric, not necessarily a change in correctness.

#### Aggregate metrics by encoding

| Metric             | Definition and aggregation across images                                                                           | PNG result   | JPEG result  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------ | ------------ | ------------ |
| Accuracy           | Correct predictions divided by 10,000 images for that encoding                                                     | 99.05% | 99.11% |
| Mean cross-entropy | Sum the per-image cross-entropy values and divide by 10,000 images for that encoding                               | 0.03039010 | 0.02991649 |
| Mean Brier score   | Sum the per-image Brier scores (each already summed over 10 classes) and divide by 10,000 images for that encoding | 0.01490250 | 0.01469340 |

Recorded evaluation: [summary JSON](results/20261102T040050Z/summary.json)
and [per-image results](results/20261102T040050Z/per_image_results.csv),
created November 2, 2026 (UTC) by `python -m src.evaluate` on CPU from commit
`ccf458a`, using JPEG quality 75. The report's checkpoint SHA-256 matches the
bundled `models/checkpoint.pt`:
`eedb4358ce3940d96fa6e5843b0c3d95544b086571d8e12f5ffcc6a0a3209aaf`.

Across 10,000 source pairs, eleven predictions changed (0.11%): two correct →
wrong, eight wrong → correct, and one wrong → different wrong class. Both
encodings exceeded the V1 accuracy target of 90%. The model card describes how
the test set was used.

#### Success criteria and interpretation

- Accuracy target (> 90%): both PNG and JPEG encoding(s) must meet it.
- What the paired results establish: how PNG versus JPEG at quality 75 affects this fixed model’s predictions and evaluation metrics on these MNIST test samples.
- What this experiment does not establish: model performance on real user images outside the controlled MNIST input assumptions.

## 8. What artifact must training produce so inference does not depend on training code?

**V1 decision:** `python -m src.train final` saves a PyTorch checkpoint, which
is committed as `models/checkpoint.pt`. The application Docker image bundles
this checkpoint, the Python source, and runtime dependencies. Users do not need
to run training or supply a separate checkpoint.

The checkpoint is a dictionary with CPU tensors:

- `architecture`: identifier checked against `src/nn_model.py` before loading.
- `model_state_dict`: trained weights.
- `preprocessing`: input contract and the fixed training mean and standard
  deviation; `padding_per_side` is checked against `src/preprocessing.py`.
- `training`: optimizer, hyperparameters, epochs, seed, sample count, and
  per-epoch training loss of this model.
- `selection`: the search run (path and SHA-256) that chose the
  hyperparameters, the rule, and the winning candidate's validation metrics.
  These describe the candidate trained on the 48,000-image split, not this model.
- `provenance`: git commit and dirty flag, config and `uv.lock` hashes, PyTorch
  version, and training device.

Optimizer state is not saved: training takes under a minute, so a model is
retrained rather than resumed.

The model architecture is defined in `src/nn_model.py`; the checkpoint stores
its identifier and weights, not an executable architecture. Image validation
and preprocessing are implemented in `src/preprocessing.py`. At inference,
`src/inference.py` constructs the model, loads its weights and preprocessing
configuration, and predicts without running training. The CLI in `src/cli.py`
decodes the input image and prints the prediction.

```text
Docker image
├── Python + PyTorch + runtime dependencies
├── src/ (CLI, inference, preprocessing, model definition)
└── models/checkpoint.pt (weights, preprocessing, training metadata)

Mounted image → CLI → preprocessing → PyTorch model → prediction → exit
```

**Why a checkpoint for V1:** it fits the existing Python/PyTorch training and
inference code without an additional export step, and `torch.load` with
`weights_only=True` loads only tensors and plain data, not arbitrary code.

**Future option:** ONNX can provide model interoperability across frameworks
and inference runtimes. It requires an export step and verification that the
exported model preserves predictions. Performance and deployment-size benefits
depend on the runtime, hardware, and optimizations and must be measured. That
additional workflow is unnecessary for the current V1 scope.
