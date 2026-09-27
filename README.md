# Pneumonia Detection from Chest X-ray Images Using Deep Learning

---

## 1. Project Overview

Pneumonia is an acute respiratory infection affecting the lungs, characterized by fluid or inflammatory exudate accumulation in the pulmonary alveoli. In clinical environments, chest radiography (X-ray) is the standard initial imaging modality utilized for diagnostic evaluation. 

This project implements an end-to-end, academic deep learning pipeline for binary classification of chest radiographs into two categories:
- **NORMAL**: Radiographs exhibiting clear lung fields without focal consolidation or infiltrates.
- **PNEUMONIA**: Radiographs exhibiting pulmonary opacities, consolidations, or infiltrates associated with bacterial or viral pneumonia.

The project evaluates both a custom 4-block Convolutional Neural Network (CNN) baseline and a transfer-learning architecture based on **MobileNetV2** (utilizing ImageNet pretrained weights). To provide interpretability for model predictions, **Grad-CAM (Gradient-weighted Class Activation Mapping)** is incorporated at convolutional layer `Conv_1` to visualize image regions contributing to network activations. A standalone inference pipeline and a local **Streamlit web application** provide an interactive interface with explicit input modality confirmation and safety warnings.

> **Academic & Research Scope**: This system was developed solely for academic research, education, and viva demonstration. It is **not** a certified medical device and has **not** been clinically validated. It must **not** be utilized for medical diagnosis, clinical screening, treatment planning, or clinical decision support.

---

## 2. Problem Statement

Classifying chest radiographs presents multiple technical challenges in computational image analysis:
1. **Visual Ambiguity**: Subtle or diffuse interstitial infiltrates and bacterial consolidations can exhibit visual overlap with normal anatomical variations, vascular markings, and imaging artifacts.
2. **Class Imbalance & Sample Scarcity**: Clinical radiograph datasets frequently exhibit significant class imbalance and limited sample sizes, increasing the risk of overfitting in deep networks trained from scratch.
3. **Black-Box Nature of Deep Learning**: High classification accuracy alone is insufficient for high-stakes medical imaging applications; stakeholders require visual evidence indicating whether predictions stem from pulmonary regions rather than extraneous artifacts, image borders, or radiopaque markers.
4. **Out-of-Distribution Inputs**: Standard convolutional classifiers are trained on domain-specific radiographs and lack intrinsic mechanisms to recognize or reject non-medical images (e.g., charts, documents, photographs), necessitating rigorous application-level input handling.

This project addresses these challenges by developing a reproducible transfer-learning pipeline, conducting an audit of dataset integrity and patient overlap, integrating Grad-CAM explainability, and enforcing explicit user confirmation gates in the deployment interface.

---

## 3. Project Objectives

The project executed the following verified technical objectives:
1. **Dataset Audit & Analysis**: Comprehensively audit the Kaggle Chest X-Ray Images dataset for corrupted files, format distributions, class balance, exact duplicates, and cross-split patient ID leakage.
2. **Preprocessing Pipeline**: Implement standardization pipelines including format conversion, 3-channel RGB alignment, bilinear resizing to 224×224 pixels, and MobileNetV2 normalization to `[-1.0, 1.0]`.
3. **Baseline CNN Development**: Design, train, and evaluate a custom 4-block CNN architecture (250,561 parameters) on a patient-aware development split.
4. **MobileNetV2 Transfer Learning**: Implement a 2-phase transfer-learning strategy (Phase 1 feature extraction and Phase 2 fine-tuning with frozen Batch Normalization layers) yielding a 2,259,265 parameter model.
5. **Comprehensive Evaluation**: Evaluate models using Accuracy, Precision, Recall/Sensitivity, Specificity, F1 Score, ROC-AUC, and PR-AUC.
6. **Grad-CAM Explainability**: Implement Grad-CAM from the verified target layer `Conv_1` and generate multi-panel visual explanations.
7. **Standalone Prediction API**: Build a modular, cached prediction pipeline (`src/predict.py`) for single-image inference with fixed 0.50 thresholding.
8. **Streamlit Demonstration Interface**: Develop a web application (`app/streamlit_app.py`) featuring radiograph previews, prediction metrics, explainability visualizations, and a mandatory X-ray confirmation gate.
9. **Regression Testing**: Validate system integrity through an automated 61-test regression suite.

---

## 4. Dataset Overview & Analysis

The project utilizes the canonical Kaggle **Chest X-Ray Images (Pneumonia)** dataset (Kermany et al., 2018).

### 4.1 Canonical Dataset Partitioning

The raw dataset contains 5,856 images organized across three canonical partitions:

| Partition | NORMAL Images | PNEUMONIA Images | Total Images | Proportion | Class Distribution |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`train/`** | 1,341 | 3,875 | 5,216 | 89.07% | 25.71% NORMAL / 74.29% PNEUMONIA |
| **`val/`** | 8 | 8 | 16 | 0.27% | 50.00% NORMAL / 50.00% PNEUMONIA |
| **`test/`** | 234 | 390 | 624 | 10.66% | 37.50% NORMAL / 62.50% PNEUMONIA |
| **Total** | **1,583** | **4,273** | **5,856** | **100.00%** | **27.03% NORMAL / 72.97% PNEUMONIA** |

### 4.2 Verified Dataset Characteristics
An exploratory data analysis (`src/dataset_analysis.py`, output in `outputs/metrics/dataset_analysis.json`) established the following verified properties:
- **File Format**: 100.0% of images (5,856/5,856) are valid JPEG files (`.jpeg`).
- **File Integrity**: Exactly 0 corrupted or unreadable images were detected.
- **Color Mode Distribution**:
  - Grayscale ('L' mode): 5,573 images (**95.17%**)
  - RGB ('RGB' mode): 283 images (**4.83%**)
- **Image Resolution**: Native dimensions range from $384 \times 127$ to $2916 \times 2713$ pixels (mean width: 1,327.9 px, mean height: 970.7 px, mean aspect ratio: 1.44).
- **Exact File Duplicates**:
  - Total identical MD5 duplicate groups: 30 groups (all intra-split duplicates within `train/`).
  - Cross-split exact file leakage: **0 images** (no identical image files cross split boundaries).

---

## 5. Dataset Partitioning & Patient-ID Limitation

### 5.1 Patient-ID Overlap in Benchmark Dataset
A patient-level identifier audit identified a critical structural limitation inherent to the Kaggle benchmark release:
- **NORMAL Class**: 0 patient IDs overlap between the training and official test partitions.
- **PNEUMONIA Class**: Exactly **170 patient IDs** appear in both the canonical `train/` split and the canonical `test/` split.

> **Evaluation Nomenclature**: Because 170 pneumonia patients have radiographs in both the training set and the official test set, the final Stage 7 test evaluation is strictly characterized as an **"official held-out test split evaluation"**, and must **not** be characterized as a "fully patient-independent test evaluation".

### 5.2 Patient-Aware Development Split Strategy
Because the official `val/` partition contains only 16 images (insufficient for reliable statistical validation, early stopping, or checkpoint selection), the 5,216-image training set was partitioned into a dedicated, patient-aware development split:
- **Development Training**: 4,697 images (1,201 NORMAL, 3,496 PNEUMONIA) across 2,569 unique patients.
- **Development Validation**: 519 images (140 NORMAL, 379 PNEUMONIA) across 277 unique patients.
- **Patient Leakage**: Exactly **0 patient overlap** between development training and development validation.
- **Official Splits**: The official 16-image validation set and the 624-image test set were completely isolated from model selection and hyperparameter tuning.

---

## 6. Preprocessing & Data Augmentation

### 6.1 Inference & Preprocessing Pipeline
All radiographs entering the MobileNetV2 network undergo deterministic standardization via `src/preprocessing.py` and `src/predict.py`:
1. **Format Validation**: Validates image existence, non-empty dimensions, and supported extension (`.jpeg`, `.jpg`, `.png`).
2. **Color Mode Alignment**: Grayscale ('L') and RGBA images are converted to 3-channel RGB.
3. **Spatial Resizing**: Resized to the target dimension of $224 \times 224 \times 3$ pixels using bilinear interpolation.
4. **MobileNetV2 Normalization**: Pixel values in $[0, 255]$ are normalized via $\frac{\text{pixel}}{127.5} - 1.0$, producing floating-point tensors in $[-1.0, 1.0]$.

### 6.2 Training Augmentation
To mitigate overfitting on training data, stochastic data augmentation was applied **exclusively** during model training:
- Random rotation: $\pm 15^\circ$
- Random translation: $\pm 5\%$ horizontal and vertical shifts
- Random zoom: $\pm 10\%$ scale adjustment
- Random horizontal flip: Enabled
- Vertical flip: **Disabled** (inappropriate for anatomical chest radiograph orientation)

Validation and official test images were **never** augmented, preserving raw radiograph fidelity for deterministic evaluation.

---

## 7. Baseline CNN Architecture & Validation Results

### 7.1 Architecture Design
A custom 4-block convolutional network (`src/model_cnn.py`) was implemented as an experimental baseline:
- **Input Layer**: $224 \times 224 \times 3$ (normalized to $[0.0, 1.0]$ via $\frac{\text{pixel}}{255.0}$)
- **Block 1**: Conv2D (32 filters, $3 \times 3$, padding='same') $\rightarrow$ BatchNormalization $\rightarrow$ ReLU $\rightarrow$ MaxPooling2D ($2 \times 2$)
- **Block 2**: Conv2D (64 filters, $3 \times 3$, padding='same') $\rightarrow$ BatchNormalization $\rightarrow$ ReLU $\rightarrow$ MaxPooling2D ($2 \times 2$)
- **Block 3**: Conv2D (128 filters, $3 \times 3$, padding='same') $\rightarrow$ BatchNormalization $\rightarrow$ ReLU $\rightarrow$ MaxPooling2D ($2 \times 2$)
- **Block 4**: Conv2D (128 filters, $3 \times 3$, padding='same') $\rightarrow$ BatchNormalization $\rightarrow$ ReLU $\rightarrow$ MaxPooling2D ($2 \times 2$)
- **Classifier Head**: GlobalAveragePooling2D $\rightarrow$ Dense (64 units, ReLU) $\rightarrow$ Dropout (0.40) $\rightarrow$ Dense (1 unit, Sigmoid)
- **Total Parameters**: **250,561** (249,857 trainable, 704 non-trainable)

### 7.2 Baseline Performance on Patient-Aware Validation Set
The baseline CNN was trained with Adam ($\text{lr} = 10^{-3}$), balanced class weights, EarlyStopping, and ModelCheckpoint saving to `models/baseline_cnn_best.keras`. Performance on the 519-image patient-aware validation split (threshold = 0.50):

| Metric | Measured Validation Value |
| :--- | :---: |
| **Accuracy** | 83.82% (0.8382) |
| **Precision** | 84.71% (0.8471) |
| **Recall / Sensitivity** | 94.99% (0.9499) |
| **Specificity** | 53.57% (0.5357) |
| **F1 Score** | 89.55% (0.8955) |
| **ROC-AUC** | 0.9188 |
| **PR-AUC** | 0.9707 |

- **Confusion Matrix**: True Negatives = 75, False Positives = 65, False Negatives = 19, True Positives = 360 (Total = 519).

---

## 8. MobileNetV2 Transfer Learning Architecture & Strategy

### 8.1 Model Architecture
The primary model (`src/model_transfer.py`) leverages the **MobileNetV2** backbone pretrained on ImageNet:
- **Input Tensor**: $224 \times 224 \times 3$
- **Feature Extractor**: MobileNetV2 (`include_top=False`, ImageNet weights)
- **Pooling**: GlobalAveragePooling2D
- **Regularization**: Dropout (0.40 rate)
- **Output Layer**: Dense (1 unit, Sigmoid activation)
- **Total Parameters**: **2,259,265**

### 8.2 Two-Phase Training Strategy
Training (`src/train_transfer.py`) executed across two structured phases:
1. **Phase 1: Feature Extraction**:
   - MobileNetV2 backbone completely frozen (154 layers non-trainable).
   - Only the classification head was trained.
   - Trainable parameters: **1,281** (2,257,984 frozen).
   - Optimizer: Adam ($\text{lr} = 10^{-4}$).
   - Epochs: 10 epochs with early stopping monitoring validation loss.
2. **Phase 2: Limited Fine-Tuning**:
   - Layers from `block_16_expand` onward (blocks 16, 17, and final point-wise convolution) unfrozen.
   - **Batch Normalization Freezing**: All 52 BatchNormalization layers remained explicitly frozen (`layer.trainable = False`) to prevent distortion of pretrained population statistics.
   - Trainable parameters: **880,321** (1,378,944 frozen).
   - Optimizer: Adam ($\text{lr} = 10^{-5}$).
   - Best checkpoint saved to `models/mobilenetv2_transfer_best.keras`.

---

## 9. MobileNetV2 Development Validation Results

Evaluated on the identical 519-image patient-aware development validation partition at the fixed classification threshold of 0.50 (`outputs/metrics/mobilenetv2_validation_metrics.json`):

| Evaluation Metric | Patient-Aware Validation Value |
| :--- | :---: |
| **Accuracy** | **93.64%** (0.9364) |
| **Precision** | **92.40%** (0.9240) |
| **Recall / Sensitivity** | **99.47%** (0.9947) |
| **Specificity** | **77.86%** (0.7786) |
| **F1 Score** | **95.81%** (0.9581) |
| **ROC-AUC** | **0.9908** |
| **PR-AUC** | **0.9966** |

### Confusion Matrix (Development Validation)
$$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix} = \begin{pmatrix} 109 & 31 \\ 2 & 377 \end{pmatrix}$$
- **True Negatives (TN)**: 109 normal radiographs correctly classified.
- **False Positives (FP)**: 31 normal radiographs classified as pneumonia.
- **False Negatives (FN)**: 2 pneumonia radiographs classified as normal.
- **True Positives (TP)**: 377 pneumonia radiographs correctly classified.

---

## 10. Official Held-Out Test Split Evaluation

The finalized MobileNetV2 model (`models/mobilenetv2_transfer_best.keras`) was evaluated on the untouched canonical 624-image test set at the fixed 0.50 threshold (`outputs/metrics/final_test_metrics.json`):

| Metric | Official Held-Out Test Value | Formula / Basis |
| :--- | :---: | :--- |
| **Accuracy** | **74.36%** (0.7436) | $(76 + 388) / 624$ |
| **Precision** | **71.06%** (0.7106) | $388 / (388 + 158)$ |
| **Recall / Sensitivity** | **99.49%** (0.9949) | $388 / (388 + 2)$ |
| **Specificity** | **32.48%** (0.3248) | $76 / (76 + 158)$ |
| **F1 Score** | **82.91%** (0.8291) | $2 \cdot (\text{Prec} \cdot \text{Rec}) / (\text{Prec} + \text{Rec})$ |
| **ROC-AUC** | **0.9530** | Area under Receiver Operating Characteristic curve |
| **PR-AUC** | **0.9711** | Area under Precision-Recall curve |

### Confusion Matrix (Official Test Set)
$$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix} = \begin{pmatrix} 76 & 158 \\ 2 & 388 \end{pmatrix}$$

### Performance Analysis
- **High Sensitivity (99.49%)**: Only 2 out of 390 pneumonia cases in the official test partition were misclassified, producing a false negative rate of 0.51%.
- **Low Specificity (32.48%)**: 158 of 234 normal radiographs in the test partition were classified as pneumonia. This specificity drop relative to the development validation set (77.86%) reflects known domain shifts in pediatric age distribution and imaging equipment between the Kaggle test set and training set.
- **Threshold Integrity**: The 0.50 threshold was pre-specified and strictly maintained without post-hoc tuning on the test set.

---

## 11. Baseline CNN vs. MobileNetV2 Comparison

Direct quantitative comparison evaluated on the identical 519-image patient-aware development validation partition:

| Metric | Baseline CNN (Custom 4-Block) | MobileNetV2 (Transfer Learning) | Absolute Difference |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | 250,561 | 2,259,265 | +2,008,704 |
| **Validation Accuracy** | 83.82% | **93.64%** | +9.82% |
| **Precision** | 84.71% | **92.40%** | +7.69% |
| **Recall / Sensitivity** | 94.99% | **99.47%** | +4.48% |
| **Specificity** | 53.57% | **77.86%** | +24.29% |
| **F1 Score** | 89.55% | **95.81%** | +6.26% |
| **ROC-AUC** | 0.9188 | **0.9908** | +0.0720 |
| **PR-AUC** | 0.9707 | **0.9966** | +0.0259 |
| **False Negatives (out of 379)** | 19 | **2** | -17 FN |
| **False Positives (out of 140)** | 65 | **31** | -34 FP |

---

## 12. Grad-CAM Explainability

Gradient-Weighted Class Activation Mapping (`src/gradcam.py`) was implemented to inspect spatial activation patterns in the finalized MobileNetV2 checkpoint.

### 12.1 Target Layer Verification
- **Layer Name**: `Conv_1`
- **Output Tensor Shape**: `(None, 7, 7, 1280)`
- **Architectural Function**: `Conv_1` is the final point-wise convolutional layer of the MobileNetV2 backbone prior to global pooling, retaining rich semantic features with $7 \times 7$ spatial resolution.

### 12.2 Computation Methodology
1. **Forward Pass**: Forward pass through a gradient model capturing `Conv_1` activations $A^k$ and output probability $y$.
2. **Gradients**: Calculate gradients $\frac{\partial y}{\partial A^k}$ using `tf.GradientTape`.
3. **Global Average Pooling**: Compute channel importance weights $\alpha_k = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y}{\partial A_{i,j}^k}$.
4. **Weighted Combination & ReLU**: Calculate $L_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$, isolating features with positive contribution to the predicted class.
5. **Normalization**: Min-max normalize heatmap values into $[0.0, 1.0]$.
6. **Bilinear Upsampling**: Interpolate heatmap from $7 \times 7$ to native radiograph dimensions.
7. **Overlay Blending**: Map heatmap through the `jet` colormap and blend onto the original radiograph ($\alpha = 0.45$).

> **Interpretability Disclaimer**: Grad-CAM highlights regions receiving greater activation for the model's prediction. These visualizations describe computational model behavior and do not establish clinical evidence of pathology or diagnostic validity.

### 12.3 Representative Test Cases
Visualizations were audited across four test outcomes (`outputs/metrics/gradcam_examples.json`):
- **Correct NORMAL** (`NORMAL2-IM-0307-0001.jpeg`): Probability = 0.0290, True Negative (`outputs/gradcam/gradcam_correct_normal.png`).
- **Correct PNEUMONIA** (`person124_bacteria_589.jpeg`): Probability = 0.9999, True Positive (`outputs/gradcam/gradcam_correct_pneumonia.png`).
- **False Positive** (`NORMAL2-IM-0256-0001.jpeg`): Probability = 0.9999, False Positive (`outputs/gradcam/gradcam_false_positive.png`).
- **False Negative** (`person117_bacteria_557.jpeg`): Probability = 0.4793, False Negative (`outputs/gradcam/gradcam_false_negative.png`).

---

## 13. Standalone Prediction Pipeline

A standalone, reusable inference interface is provided in `src/predict.py`.

### 13.1 Key Classes & Functions
- **`validate_and_load_image(image_path)`**: Validates file presence, verifies extensions (`.jpeg`, `.jpg`, `.png`), standardizes grayscale/RGBA to 3-channel RGB, resizes to $224 \times 224$, and applies MobileNetV2 normalization $[-1.0, 1.0]$.
- **`PneumoniaPredictor`**: Encapsulates checkpoint loading, warm-up, and single-image inference without reloading overhead.
- **`get_predictor()`**: Factory function managing cached singleton instances.
- **`predict_image()`**: Functional API accepting an image path and returning structured prediction metadata.

### 13.2 Command-Line Interface (CLI)
```powershell
# Standard single-image prediction
.\venv\Scripts\python.exe -m src.predict --image "dataset\chest_xray\test\NORMAL\IM-0001-0001.jpeg"

# Single-image prediction with Grad-CAM visualization
.\venv\Scripts\python.exe -m src.predict --image "dataset\chest_xray\test\PNEUMONIA\person1_virus_6.jpeg" --gradcam
```

Example CLI Output:
```text
=======================================================
PNEUMONIA DETECTION - INFERENCE RESULT
=======================================================
Image:                 person1_virus_6.jpeg
Original size:         1317 x 1857
Pneumonia probability: 99.96%
Prediction:            PNEUMONIA
Threshold:             50.00%
Grad-CAM overlay:      outputs\gradcam\predictions\person1_virus_6_gradcam.png
-------------------------------------------------------
Disclaimer: For educational/research use only. Does not constitute clinical diagnosis.
=======================================================
```

---

## 14. Streamlit Web Application & Stage 10A Input Confirmation Gate

An interactive web application is implemented in `app/streamlit_app.py`.

### 14.1 Key Features
- **Header & Non-Diagnostic Warning**: Upfront disclaimer stating educational and research scope.
- **File Uploader**: Accepts `.jpeg`, `.jpg`, and `.png` chest radiographs with file validation.
- **Radiograph Preview**: Displays uploaded image with original pixel dimensions ($W \times H$) and file size.
- **Prediction Card**: Displays Predicted Class (**NORMAL** or **PNEUMONIA**), **Pneumonia Probability** (percentage), and fixed decision threshold (50.00%).
- **Grad-CAM Visualization**: Optional 3-panel visualization showing original image, feature heatmap, and overlay.
- **Technical Expanders**: Detailed specifications for model architecture, official evaluation metrics, and inference flow.

### 14.2 Stage 10A Input Confirmation Gate
Because the model was trained on chest radiographs and produces uninterpretable predictions when given out-of-distribution images (such as charts, screenshots, or documents), Stage 10A introduced an explicit user confirmation gate:
- **Input Requirement Notice**: Prominently displayed near the uploader:
  > *"Input requirement: Upload a chest X-ray image. This application does not automatically verify whether an uploaded image is a chest X-ray. Predictions for unrelated image types are not meaningful."*
- **Confirmation Checkbox**: Mandatory checkbox (default unchecked):
  > `[ ] I confirm that the uploaded image is a chest X-ray.`
- **Inference Gate**: If the user clicks **Analyze X-ray** without checking the confirmation box, inference is **blocked**, no model call occurs, no Grad-CAM is generated, and a warning is displayed:
  > *"Please confirm that the uploaded image is a chest X-ray before analysis."*
- **Post-Prediction Assumption Notice**:
  > *"Input assumption: This result assumes that the uploaded image is a chest X-ray. The application does not automatically verify image modality."*

> **Safety Notice**: The application does **not** automatically determine whether an arbitrary image is a chest X-ray. It enforces a user confirmation safety gate to prevent accidental analysis of unrelated images.

---

## 15. System Inference Workflow

The end-to-end data and decision pipeline follows a sequential, strictly non-training workflow:

```text
User uploads image (.jpeg / .jpg / .png)
       ↓
Input validation (validate_and_load_image)
       ↓
User confirms: "I confirm that the uploaded image is a chest X-ray"
       ↓
[Analyze X-ray clicked]
       ↓
Channel conversion (Grayscale/RGBA → 3-channel RGB)
       ↓
Bilinear resizing to 224 × 224 pixels
       ↓
MobileNetV2 normalization: (pixel / 127.5) - 1.0 → [-1.0, 1.0]
       ↓
MobileNetV2 model forward pass (models/mobilenetv2_transfer_best.keras)
       ↓
Sigmoid output: Pneumonia Probability p ∈ [0.0, 1.0]
       ↓
Threshold comparison at fixed 0.50 (p ≥ 0.50 ? PNEUMONIA : NORMAL)
       ↓
Result display (Model Prediction, Pneumonia Probability, Threshold 50.00%)
       ↓
Post-prediction input assumption notice
       ↓
Optional Grad-CAM generation (Conv_1 target layer, overlay blend)
       ↓
Model-behavior explanation display
```

---

## 16. Project Directory Structure

```text
pneumonia_detection/
├── dataset/
│   └── chest_xray/
│       ├── train/                  # Canonical training partition (5,216 images)
│       │   ├── NORMAL/             # 1,341 images
│       │   └── PNEUMONIA/          # 3,875 images
│       ├── val/                    # Canonical validation partition (16 images)
│       │   ├── NORMAL/             # 8 images
│       │   └── PNEUMONIA/          # 8 images
│       └── test/                   # Canonical held-out test partition (624 images)
│           ├── NORMAL/             # 234 images
│           └── PNEUMONIA/          # 390 images
├── models/
│   ├── baseline_cnn_best.keras                    # Saved baseline CNN checkpoint (250,561 params)
│   ├── mobilenetv2_feature_extractor_best.keras  # Saved Phase 1 transfer model checkpoint
│   └── mobilenetv2_transfer_best.keras           # Finalized Phase 2 model checkpoint (2,259,265 params)
├── src/
│   ├── __init__.py
│   ├── config.py                   # Centralized project configuration & paths
│   ├── utils.py                    # Logging, seed management, and JSON serialization
│   ├── data_loader.py              # Patient-aware dataset splitting & streaming generators
│   ├── dataset_analysis.py         # Dataset audit, MD5 duplicates, and patient overlap
│   ├── preprocessing.py           # Preprocessing & augmentation pipelines
│   ├── model_cnn.py                # 4-block custom CNN architecture definition
│   ├── model_transfer.py           # MobileNetV2 transfer learning architecture definition
│   ├── train.py                    # Baseline CNN training script
│   ├── train_transfer.py           # Two-phase transfer learning & fine-tuning script
│   ├── evaluate.py                 # Development validation evaluation script
│   ├── evaluate_test.py            # Official held-out test set evaluation script
│   ├── gradcam.py                  # Grad-CAM calculation & visualization generation
│   └── predict.py                  # Standalone prediction engine & CLI
├── app/
│   ├── streamlit_app.py            # Primary Streamlit web application with confirmation gate
│   └── app.py                      # Application entry-point wrapper
├── outputs/
│   ├── metrics/                    # Verified evaluation JSON metrics & test predictions CSV
│   │   ├── dataset_analysis.json
│   │   ├── baseline_cnn_validation_metrics.json
│   │   ├── mobilenetv2_validation_metrics.json
│   │   ├── final_test_metrics.json
│   │   ├── final_test_predictions.csv
│   │   └── gradcam_examples.json
│   ├── plots/                      # Confusion matrices, ROC curves, and PR curves
│   └── gradcam/                    # Saved multi-panel Grad-CAM visualization figures
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py             # Dataset structure, counts, and generator tests
│   ├── test_preprocessing.py      # Resizing, channel handling, and augmentation tests
│   ├── test_models.py              # Baseline CNN shapes, layers, and forward-pass tests
│   ├── test_transfer.py            # MobileNetV2 architecture, freezing, and fine-tuning tests
│   ├── test_evaluation_test.py     # Official test set pipeline and artifact tests
│   ├── test_gradcam.py             # Grad-CAM heatmap, overlay, and Conv_1 target tests
│   ├── test_predict.py             # Standalone predictor API, formats, and threshold tests
│   └── test_streamlit_app.py       # Streamlit UI structure, confirmation gate, and safety tests
├── notebooks/                      # Exploratory Data Analysis & viva notes
├── requirements.txt                # Pinned dependency requirements
└── README.md                       # Comprehensive academic project documentation
```

---

## 17. Regression Testing & Quality Assurance

The codebase includes an automated regression test suite covering all modules:

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```

### Verified Test Suite Breakdown (61/61 Passed)
- `tests/test_dataset.py` (7 tests): Verifies dataset paths, exact counts (5,856 total, 624 test), class weights, generator streaming, and test set isolation.
- `tests/test_preprocessing.py` (10 tests): Verifies $224 \times 224 \times 3$ sizing, $[-1.0, 1.0]$ normalization, augmentation behavior (training only), and label mapping.
- `tests/test_models.py` (8 tests): Verifies baseline CNN shapes, binary sigmoid output, callbacks, and patient-aware zero-leakage split.
- `tests/test_transfer.py` (8 tests): Verifies MobileNetV2 architecture, Phase 1 layer freezing, Phase 2 fine-tuning with frozen BatchNormalization, forward pass, and checkpoint reloading.
- `tests/test_evaluation_test.py` (1 test): Verifies official test set pipeline and evaluation metrics artifacts.
- `tests/test_gradcam.py` (5 tests): Verifies checkpoint availability, target layer `Conv_1`, input preprocessing, heatmap generation in $[0.0, 1.0]$, and overlay generation.
- `tests/test_predict.py` (11 tests): Verifies checkpoint presence, binary output, RGB/Grayscale/RGBA format support, error handling on corrupted/unsupported inputs, 0.50 threshold behavior, and Grad-CAM integration.
- `tests/test_streamlit_app.py` (11 tests): Verifies app module files, checkpoint integrity, metrics JSON verification, backend imports, absence of retraining triggers, official test isolation, confirmation checkbox (default unchecked), non-X-ray warnings, confirmation gate logic, and confirmed inference.

**Final Suite Summary**: **61 tests collected, 61 passed, 0 failed** (duration: 214.68 seconds).

---

## 18. Model Checkpoint & Artifact Integrity

All experiments utilize pre-saved checkpoints and metric artifacts. The integrity of the final model checkpoint was strictly verified:

- **Checkpoint File**: [`models/mobilenetv2_transfer_best.keras`](file:///e:/pneumonia_detection/models/mobilenetv2_transfer_best.keras)
- **File Size**: Exactly **16,666,293 bytes** (16.67 MB)
- **SHA-256 Checksum**:
  ```
  8104988636be7dd15786d58c88663ec747da5986aa20f7d569da622e39b7aae2
  ```
- **Total Parameters**: 2,259,265
- **Input Dimensions**: `(None, 224, 224, 3)`
- **Output Dimensions**: `(None, 1)`
- **Integrity Status**: The SHA-256 checksum was recorded in Stage 10A and confirmed 100% identical in Stages 11 and 12 (zero retraining, zero fine-tuning, zero re-saving).

---

## 19. Project Limitations

The project explicitly acknowledges the following verified scientific and engineering limitations:
1. **Academic/Research Scope**: The model is an academic prototype developed for educational demonstration and viva evaluation.
2. **Lack of Clinical Validation**: The system has not undergone clinical trials, multi-center prospective validation, or regulatory clearance (e.g., FDA/CE).
3. **Non-Diagnostic Nature**: Outputs represent statistical model behavior and must not be used for medical diagnosis, clinical screening, or treatment selection.
4. **No Automatic Modality Detection**: The convolutional neural network does not intrinsically determine whether an uploaded image is a chest radiograph.
5. **Reliance on User Confirmation**: The application relies on an explicit user confirmation gate to prevent the submission of out-of-distribution inputs (e.g., charts, photos).
6. **Meaningless Predictions on Non-X-rays**: Model predictions for non-radiograph images are arbitrary artifacts of out-of-distribution feature mapping.
7. **Patient-ID Overlap in Benchmark**: The original Kaggle benchmark release contains 170 PNEUMONIA patient IDs appearing in both training and official test partitions.
8. **Test Evaluation Characterization**: The official test set evaluation is characterized as an "official held-out test split evaluation", not a "patient-independent evaluation".
9. **Small Canonical Validation Partition**: The canonical validation split contains only 16 images, requiring a custom patient-aware development split for model selection.
10. **Hardware & Environment**: All training, inference, and testing were conducted in a CPU-only environment, which affects computational latency.

---

## 20. How to Run Locally

### 20.1 Local Streamlit Application
To launch the interactive Streamlit demonstration application:
```powershell
# From the project root in PowerShell:
.\venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```
The server will start locally and provide an accessible interface:
```text
Local URL: http://localhost:8501
Network URL: http://<local-ip>:8501
```

### 20.2 Standalone Inference CLI
To run single-radiograph inference from the command line:
```powershell
.\venv\Scripts\python.exe -m src.predict --image "dataset\chest_xray\test\NORMAL\IM-0001-0001.jpeg" --gradcam
```

### 20.3 Automated Test Suite
To execute the full regression test suite:
```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```

---

## 21. Environment & Reproducibility

The project environment is fully specified with pinned dependencies:
- **Operating System**: Windows 11 (native execution, CPU-optimized)
- **Python**: 3.12.0
- **TensorFlow**: 2.21.0
- **Keras**: 3.15.1
- **Streamlit**: 1.64.0
- **NumPy**: 2.0.2
- **Pillow**: 11.1.0
- **Matplotlib**: 3.10.0
- **Scikit-Learn**: 1.6.1
- **pytest**: 9.1.1
- **Random Seeds**: Configured globally (`seed = 42`) across Python `random`, NumPy, and TensorFlow for deterministic data splitting and batch generation.

---

## 22. Project Stage Roadmap

- [x] **Stage 1: Project Setup & Environment Baseline**
- [x] **Stage 2: Dataset Setup, Integrity Check & Exploratory Data Analysis**
- [x] **Stage 3: Preprocessing & Data Augmentation Pipelines**
- [x] **Stage 4: Baseline CNN Implementation**
- [x] **Stage 5: Baseline CNN Training & Evaluation**
- [x] **Stage 6: Transfer Learning (MobileNetV2 Phase 1 & Phase 2 Fine-Tuning)**
- [x] **Stage 7: Final Model Evaluation on Official Held-Out Test Set**
- [x] **Stage 8: Grad-CAM Explainability Implementation**
- [x] **Stage 9: Standalone Prediction & Inference Pipeline**
- [x] **Stage 10: Streamlit Web Application**
- [x] **Stage 10A: Input Modality Handling & Confirmation Gate**
- [x] **Stage 11: Final Integration & Full Regression Testing**
- [x] **Stage 12: Academic Project Documentation**

---

*Academic project documentation completed. All values, parameters, and findings reflect verified implementation artifacts.*
