[Development setup and tool commands](DEVELOPMENT.md)

The trained checkpoint is stored at `models/checkpoint.pt`. It contains model
weights, preprocessing statistics, and training metadata. The CLI uses this path
by default when run from the project root; use `--checkpoint PATH` to override it.

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

For your own image, change `source` to its directory's absolute host path and
pass `/inputs/<filename>` to the CLI. CLI paths refer to files inside the
container. `--label` is optional; it displays a known label for comparison.

Inputs must be PNG or JPEG images decoding to 28×28 grayscale uint8 pixels,
with one centered light digit on a dark background (MNIST-style). The checkpoint
is bundled; no local Python setup or checkpoint mount is required. The container
prints the prediction and exits; `--rm` removes it afterward.



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
| Accuracy           | Correct predictions divided by 10,000 images for that encoding                                                     | 98.74% | 98.72% |
| Mean cross-entropy | Sum the per-image cross-entropy values and divide by 10,000 images for that encoding                               | 0.03556320 | 0.03586963 |
| Mean Brier score   | Sum the per-image Brier scores (each already summed over 10 classes) and divide by 10,000 images for that encoding | 0.01830397 | 0.01854628 |

Recorded evaluation: [summary JSON](results/20261019T042353529283Z/summary.json)
and [per-image results](results/20261019T042353529283Z/per_image_results.csv),
created October 19, 2026 (UTC), using MPS and JPEG quality 75. The report's
checkpoint SHA-256 matches the bundled `models/checkpoint.pt`:
`103a167951fa73b17ef2696fe57d8f4af50615e9ba8877d7d8549aa5e12245b9`.

Across 10,000 source pairs, four predictions changed (0.04%): three correct →
wrong, one wrong → correct, and zero wrong → different wrong class. Both
encodings exceeded the V1 accuracy target of 90%. These are recorded test
results; they do not establish that model-selection decisions were frozen
before evaluation.

#### Success criteria and interpretation

- Accuracy target (> 90%): both PNG and JPEG encoding(s) must meet it.
- What the paired results establish: how PNG versus JPEG at quality 75 affects this fixed model’s predictions and evaluation metrics on these MNIST test samples.
- What this experiment does not establish: model performance on real user images outside the controlled MNIST input assumptions.

## 8. What artifact must training produce so inference does not depend on training code?

**V1 decision:** training saves a PyTorch checkpoint at `models/checkpoint.pt`.
The application Docker image bundles this checkpoint, the Python source, and
runtime dependencies. Users do not need to run the training notebook or supply
a separate checkpoint.

The checkpoint contains:

- Trained model weights (`model_state_dict`).
- Preprocessing configuration, including the fixed training mean and standard deviation.
- Architecture identifier, training configuration, seeds, and validation metrics.
- Optimizer state and completed epoch count to support resuming training.

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
local inference workflow, and can support training resumption without an
additional model export step. The saved optimizer state and epoch count help
with resumption; exact continuation also requires restoring random-number and
data-order state, which the current checkpoint does not fully capture.

**Future option:** ONNX can provide model interoperability across frameworks
and inference runtimes. It requires an export step and verification that the
exported model preserves predictions. Performance and deployment-size benefits
depend on the runtime, hardware, and optimizations and must be measured. That
additional workflow is unnecessary for the current V1 scope.
