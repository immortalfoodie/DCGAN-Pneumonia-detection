# Pneumonia Detection Project - Complete Architecture Documentation

This document explains the project from basics to detailed internals, including:

- End-to-end architecture flow
- Model components and how they interact
- Exact hyperparameters (optimizer, epochs, batch size, image size, etc.)
- Which models are used (DenseNet, ResNet, VGG16, Inception)
- Where each component is implemented and used
- Training, inference, explainability, and app behavior

---

## 1) Project Overview

This repository currently contains **two tracks**:

1. **Enhanced PyTorch production pipeline** (active system):
   - GAN branch for minority class balancing
   - DenseNet-121 + FPN backbone
   - Classification head + anchor-free detection head
   - Grad-CAM explainability
   - Streamlit application for diagnosis and visualization

2. **Legacy TensorFlow notebooks** (baseline experiments):
   - Custom CNN
   - DenseNet
   - VGG16
   - ResNet50
   - InceptionV3

---

## 2) High-Level System Architecture (Enhanced PyTorch)

### 2.1 Pipeline Stages

1. **Data Ingestion**
   - Reads chest X-ray images from class folders (`NORMAL`, `PNEUMONIA`).
   - Converts grayscale image to 3-channel RGB (required by ImageNet-pretrained backbones).

2. **Data Preprocessing & Augmentation**
   - Train: resize + geometric + photometric augmentation + normalization.
   - Validation/Test: resize + normalization only.

3. **Class Imbalance Handling**
   - Option A: `WeightedRandomSampler` in dataloader.
   - Option B: include synthetic GAN images in training dataset.
   - Option C: on-the-fly GAN batch balancing in model forward pass (if generator loaded).

4. **Feature Extraction**
   - DenseNet-121 backbone extracts multi-level features.
   - FPN builds `P3`, `P4`, `P5` feature maps.

5. **Task Heads**
   - **Classification head** predicts pneumonia probability.
   - **Detection head** predicts dense anchor-free boxes + confidence scores.

6. **Post-processing**
   - Detection output merged across pyramid levels.
   - NMS applied.
   - Label decided by classification threshold.

7. **Explainability**
   - Grad-CAM and Grad-CAM++ heatmaps generated from target convolution layer.

8. **Serving**
   - Streamlit app allows upload (PNG/JPG/DICOM), diagnosis, overlay display, PDF report, synthetic gallery, and metrics dashboard.

---

## 3) Repository Structure and Purpose

- `config.py`:
  - Global paths and hyperparameters.
  - Device selection (`cuda` if available).
- `models/`:
  - `dcgan.py`: Generator, Discriminator, self-attention, gradient penalty.
  - `densenet_fpn.py`: DenseNet-121 + FPN backbone.
  - `classifier.py`: classifier head.
  - `detector.py`: FCOS-style detection head + NMS.
  - `full_model.py`: unified model wiring all components.
- `training/`:
  - `train_gan.py`: GAN training loop.
  - `train_classifier.py`: classifier/detector training loop.
  - `losses.py`: focal, CIoU, hinge, combined losses.
- `utils/`:
  - `dataset.py`: dataset scanning, dataloader, balancing sampler.
  - `augmentation.py`: Albumentations transforms.
  - `grad_cam.py`: Grad-CAM utilities.
  - `metrics.py`: training/evaluation metrics and curve plotting.
  - `visualization.py`: box drawing and image helpers.
- `scripts/`:
  - `predict.py`: command-line single image inference.
  - `download_data.py`: dataset setup instructions.
- `app/streamlit_app.py`:
  - Full UI application.
- `checkpoints/metrics.json`:
  - persisted metrics and threshold after training.
- `*.ipynb`:
  - TensorFlow baseline experiments.

---

## 4) Exact Global Hyperparameters (Enhanced Pipeline)

Defined in `config.py`:

- `IMAGE_SIZE = 512`
- `GAN_IMAGE_SIZE = 128`
- `IN_CHANNELS = 1`
- `MODEL_CHANNELS = 3`
- `LATENT_DIM = 100`
- `GAN_LR = 0.0002`
- `GAN_BETAS = (0.5, 0.999)`
- `GAN_EPOCHS = 100`
- `NUM_DISCRIMINATORS = 2`
- `BATCH_SIZE = 16`
- `LR = 1e-4`
- `WEIGHT_DECAY = 1e-5`
- `EPOCHS = 50`
- `PATIENCE = 10`
- `NUM_CLASSES = 1`
- `FPN_CHANNELS = 256`
- `DROPOUT = 0.5`
- `FOCAL_ALPHA = 0.25`
- `FOCAL_GAMMA = 2.0`
- `LAMBDA_DET = 1.0`
- `DEVICE = "cuda" if available else "cpu"`

---

## 5) Optimizers, Epochs, Batch Size, Model Types (What + Where Used)

## 5.1 Enhanced PyTorch Training

- **Classifier training**
  - Optimizer: `Adam`
  - LR: `1e-4` (default from `Config.LR`)
  - Weight decay: `1e-5`
  - Epochs: `50` (default, with early stopping)
  - Batch size: `16` (default)
  - Scheduler: `ReduceLROnPlateau(mode="max", patience=5, factor=0.5)` on validation AUC
  - Mixed precision: enabled if CUDA is available
  - File: `training/train_classifier.py`

- **GAN training**
  - Optimizer: `Adam` (separate for generator, discriminator-1, discriminator-2)
  - LR: `0.0002`
  - Betas: `(0.5, 0.999)`
  - Epochs: `100`
  - Batch size: `16` (from same global config)
  - Loss: Hinge + Gradient Penalty (`lambda_gp = 10.0` hardcoded in loop)
  - File: `training/train_gan.py`

## 5.2 Legacy TensorFlow Notebooks

From notebook content:

- **Custom CNN notebook**
  - Optimizer: `rmsprop`
  - Loss: `binary_crossentropy`
  - Epochs: `12`
  - Batch size: `32`
  - Source: `pneumonia-detection-using-tensorflow.ipynb`

- **Transfer-learning notebook models**
  - **DenseNet121**: optimizer `adam`, epochs `10`
  - **VGG16**: optimizer `Adam(learning_rate=0.001)`, epochs `10`
  - **ResNet50**: optimizer `Adam(learning_rate=0.001)`, epochs `10`
  - **InceptionV3**: optimizer `Adam(learning_rate=0.001)`, epochs `10`
  - Typical generators include train batch size `8`, validation/test batch size `1` in parts of the notebook
  - Source: `Transfer-learning-pneumonia-detection.ipynb`

---

## 6) Core Model Details (Enhanced)

## 6.1 DenseNet-FPN Backbone

File: `models/densenet_fpn.py`

- Uses `torchvision` `densenet121(weights=IMAGENET1K_V1)` when pretrained.
- Extracts backbone stages and constructs FPN:
  - `C3`, `C4`, `C5` converted to FPN channels via 1x1 lateral convs.
  - Top-down upsampling and addition.
  - 3x3 smoothing convs create `P3`, `P4`, `P5`.
- Output:
  - Dictionary with `P3`, `P4`, `P5` tensors.

## 6.2 Classification Head

File: `models/classifier.py`

- Input: `P5` feature map.
- Pipeline:
  - Global average pooling
  - FC(256 -> 512)
  - ReLU
  - Dropout (`0.5`)
  - FC(512 -> 1)
  - Sigmoid
- Output:
  - `prob`: pneumonia probability
  - `features`: latent vector
  - `cam_target`: for explainability

## 6.3 Detection Head (Anchor-Free)

File: `models/detector.py`

- FCOS-like head for levels `P3`, `P4`, `P5`.
- For each level:
  - Shared conv + ReLU
  - `box_head` predicts 4 channels (`dx, dy, dw, dh`)
  - `score_head` predicts confidence (sigmoid)
- Box decoding:
  - Converts relative offsets to absolute center-format boxes (`cx, cy, w, h`) using level stride.
- NMS:
  - Converts to `xyxy` internally and applies torchvision NMS.

## 6.4 Unified Full Model

File: `models/full_model.py`

- Components:
  - `DenseNetFPN` backbone
  - `ClassificationHead`
  - `DetectionHead`
  - `Generator` (for optional balancing)
- Forward behavior:
  - Optional GAN batch balancing in training mode.
  - Runs backbone -> classification + detection heads.
  - Merges multi-scale detections, applies NMS per sample.
  - Returns classification probability, selected boxes/scores, and label.

---

## 7) GAN Architecture Details

File: `models/dcgan.py`

- **Generator**
  - Input latent vector `z` dimension: `100`
  - Linear projection to 4x4x512
  - Multiple transposed-conv upsampling blocks
  - Self-attention block
  - Spectral normalization on conv layers
  - Output: 1-channel 128x128 image with `tanh` range `[-1, 1]`

- **Discriminator**
  - Spectral-normalized conv blocks + LeakyReLU
  - Self-attention in intermediate layer
  - No sigmoid output (hinge-loss style)

- **Double-SGAN pattern**
  - One generator + two discriminators (`D1`, `D2`)
  - Generator loss averaged over both discriminator signals.

- **Regularization**
  - Gradient penalty function implemented and added to discriminator loss.

---

## 8) Data Pipeline in Detail

## 8.1 Dataset Loading

File: `utils/dataset.py`

- Reads split folders:
  - `train/NORMAL`, `train/PNEUMONIA`
  - `val/NORMAL`, `val/PNEUMONIA`
  - `test/NORMAL`, `test/PNEUMONIA`
- Supports image extensions: `.jpeg`, `.jpg`, `.png`
- Converts loaded grayscale images to RGB for model input.
- Optional RSNA annotations support:
  - Reads `data/rsna/stage_2_train_labels.csv` if present.
  - Stores boxes per patient ID.

## 8.2 Dataloader Balancing

- Uses `WeightedRandomSampler` based on inverse class frequency.
- Optionally appends synthetic GAN images to dataset during training.

## 8.3 Augmentations

File: `utils/augmentation.py`

- Train:
  - Resize
  - Horizontal flip
  - Brightness/contrast jitter
  - Gaussian noise
  - Shift/scale/rotate
  - Coarse dropout
  - ImageNet normalization
  - Tensor conversion
- Validation/Test:
  - Resize + normalize + tensor conversion

---

## 9) Training Mechanics

## 9.1 Classifier/Detector Training Loop

File: `training/train_classifier.py`

- Builds train/val datasets.
- Uses balanced loader for train.
- Loads generator checkpoint if available (unless `--no_gan`).
- Loss:
  - `CombinedLoss = FocalLoss + lambda_det * CIoULoss`
- Current implementation detail:
  - In epoch loop, detection loss path receives empty tensors (`pred_boxes` and `tgt_boxes` set to zero-sized), so effectively optimization is classification-dominated in this script.
- Metrics:
  - Accuracy, precision, recall, F1, AUC, confusion matrix, ROC arrays.
- Threshold calibration:
  - Chooses best threshold from precision-recall curve maximizing F1.
- Checkpointing:
  - `best_model.pth` when val AUC improves.
  - `classifier_last.pth` every epoch.
- Early stopping:
  - Stops after `PATIENCE` epochs without AUC improvement.

## 9.2 GAN Training Loop

File: `training/train_gan.py`

- Finds minority class automatically from training split.
- Trains generator + 2 discriminators.
- Discriminator objective:
  - Hinge discriminator loss + `10 * gradient_penalty`.
- Generator objective:
  - Mean of adversarial hinge losses against both discriminators.
- Artifacts:
  - Saves sample image grids every 5 epochs.
  - Computes FID every 10 epochs (if torchmetrics FID available).
  - Saves best generator by FID.
  - Always saves `gan_last.pth`.

---

## 10) Loss Functions

File: `training/losses.py`

- `FocalLoss(alpha=0.25, gamma=2.0)` for imbalanced binary classification.
- `CIoULoss` for box regression in center-format.
- `GANHingeLoss` static methods for generator/discriminator.
- `CombinedLoss` blends classification and detection via `lambda_det`.

---

## 11) Inference Flow

## 11.1 CLI Inference

File: `scripts/predict.py`

1. Loads `best_model.pth`.
2. Loads decision threshold from `metrics.json` (or fallback from checkpoint).
3. Preprocesses input image (currently with transform size `224` in script).
4. Runs model.
5. Applies threshold to classify `NORMAL` vs `PNEUMONIA`.
6. Optionally draws predicted boxes and saves output image.

## 11.2 Streamlit Inference

File: `app/streamlit_app.py`

- Supports upload: PNG/JPG/JPEG/DICOM.
- Preprocesses to input size `224`.
- If model exists:
  - Runs inference
  - Generates Grad-CAM heatmap
  - Draws boxes and overlay
  - Displays confidence gauge and decision threshold
  - Allows PDF report download
- If checkpoints are missing:
  - Runs demo placeholder outputs.

---

## 12) Explainability Architecture

File: `utils/grad_cam.py`

- Registers forward and backward hooks on target layer.
- Computes:
  - Grad-CAM (mean-gradient weighting)
  - Grad-CAM++ (second/third-order weighted formulation)
- Normalizes heatmap to [0, 1].
- Overlay function blends heatmap with original image.

Used in:

- `app/streamlit_app.py` (`diagnose_page`) to generate overlay visual explanation for user-facing diagnosis.

---

## 13) Metrics and Reporting

File: `utils/metrics.py`

- Classification metrics:
  - accuracy, precision, recall, f1, auc, confusion matrix, ROC.
- Detection metric utility:
  - simple mAP proxy (`precision * recall`) at IoU threshold.
- GAN metrics:
  - FID support.
- Visualization exports:
  - ROC and confusion matrix to HTML/PNG.
  - Training curves PNG.

Persisted metrics example:

- `checkpoints/metrics.json` includes:
  - `accuracy`, `auc`, `f1`, `precision`, `recall`
  - confusion matrix
  - ROC arrays
  - calibrated threshold
  - epoch-wise training history

---

## 14) Streamlit App Architecture

File: `app/streamlit_app.py`

### Pages

1. **Diagnose**
   - Upload image -> run model -> show prediction + overlay + report.
2. **GAN Gallery**
   - Generate synthetic images from GAN (or show existing samples).
3. **Performance Dashboard**
   - Reads `metrics.json` and visualizes key metrics and curves.
4. **Paper Comparison**
   - Compares architecture/performance with literature baselines.

### Runtime Caching

- Model and generator are loaded with `st.cache_resource`.

### Compatibility Loader

- Includes checkpoint compatibility logic for legacy spectral norm key formats and DataParallel key prefix stripping.

---

## 15) Legacy Notebook Model Summary (from repository notebooks)

- **Custom CNN**
  - `optimizer="rmsprop"`
  - `loss="binary_crossentropy"`
  - `epochs=12`
  - train batch size `32` in fit flow.

- **DenseNet121 transfer learning**
  - Keras DenseNet121 pretrained on ImageNet.
  - `optimizer='adam'`
  - `loss='binary_crossentropy'`
  - `epochs=10`

- **VGG16 transfer learning**
  - Keras VGG16 pretrained.
  - `Adam(learning_rate=0.001)`
  - `epochs=10`

- **ResNet50 transfer learning**
  - Keras ResNet50 pretrained.
  - `Adam(learning_rate=0.001)`
  - `epochs=10`

- **InceptionV3 transfer learning**
  - Keras InceptionV3 pretrained.
  - `Adam(learning_rate=0.001)`
  - `epochs=10`

Notebook-reported accuracy values in repo README:

- Custom CNN: 91.98%
- DenseNet: 87.18%
- VGG16: 66.19%
- ResNet: 73.40%
- InceptionNet: 76.76%

---

## 16) Practical End-to-End Execution Order

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Prepare dataset under `chest_xray` (or `data/chest_xray` fallback).
3. Train GAN:
   - `python training/train_gan.py`
4. Train unified model:
   - `python training/train_classifier.py`
5. Launch app:
   - `streamlit run app/streamlit_app.py`
6. Optional CLI prediction:
   - `python scripts/predict.py --image <path> --output <path>`

---

## 17) Important Implementation Notes

- The enhanced architecture uses **DenseNet** in active pipeline, not ResNet/VGG/Inception.
- ResNet, VGG16, Inception are present in **legacy notebook experiments**.
- Detection head exists and runs in forward path, but current `train_classifier.py` primarily optimizes classification due to empty detection targets in the loss call.
- Inference preprocessing size in app/CLI is currently 224, while training default image size is 512 (this is an implementation choice to be aware of when comparing behavior).

---

## 18) Quick "Where Is It Used?" Index

- Global hyperparameters: `config.py`
- DenseNet backbone/FPN: `models/densenet_fpn.py`
- Classification head: `models/classifier.py`
- Detection head and NMS: `models/detector.py`
- Unified model wiring: `models/full_model.py`
- GAN components: `models/dcgan.py`
- GAN training loop: `training/train_gan.py`
- Classifier training loop: `training/train_classifier.py`
- Loss functions: `training/losses.py`
- Dataset and balancing: `utils/dataset.py`
- Augmentation pipelines: `utils/augmentation.py`
- Grad-CAM: `utils/grad_cam.py`
- Metrics utilities: `utils/metrics.py`
- Visualization helpers: `utils/visualization.py`
- CLI prediction: `scripts/predict.py`
- Web UI: `app/streamlit_app.py`
- Legacy TensorFlow baselines: `pneumonia-detection-using-tensorflow.ipynb`, `Transfer-learning-pneumonia-detection.ipynb`

---

This document is generated from the current codebase and notebook content in this repository.
