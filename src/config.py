"""
Configuration module for Pneumonia Detection project.
Centralizes all paths, hyperparameters, model choices, and pipeline constants.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Dict


# Base workspace directory
BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_dataset_dir(base: Path) -> Path:
    """
    Resolve the active chest_xray dataset directory.
    Supports either direct structure (dataset/train) or standard Kaggle archive (dataset/chest_xray/train).
    """
    nested = base / "dataset" / "chest_xray"
    if (nested / "train").exists():
        return nested
    direct = base / "dataset"
    if (direct / "train").exists():
        return direct
    return nested


RESOLVED_DATASET_DIR = _resolve_dataset_dir(BASE_DIR)


@dataclass(frozen=True)
class Config:
    """Project-wide immutable configuration."""

    # -----------------------------
    # Directories
    # -----------------------------
    PROJECT_ROOT: Path = BASE_DIR
    RAW_DATASET_DIR: Path = BASE_DIR / "dataset"
    DATASET_DIR: Path = RESOLVED_DATASET_DIR
    MODELS_DIR: Path = BASE_DIR / "models"
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"
    PLOTS_DIR: Path = BASE_DIR / "outputs" / "plots"
    METRICS_DIR: Path = BASE_DIR / "outputs" / "metrics"
    GRADCAM_DIR: Path = BASE_DIR / "outputs" / "gradcam"
    NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"
    APP_DIR: Path = BASE_DIR / "app"

    # -----------------------------
    # Dataset Splits & Structure
    # -----------------------------
    TRAIN_DIR: Path = RESOLVED_DATASET_DIR / "train"
    VAL_DIR: Path = RESOLVED_DATASET_DIR / "val"
    TEST_DIR: Path = RESOLVED_DATASET_DIR / "test"

    CLASS_NAMES: Tuple[str, ...] = ("NORMAL", "PNEUMONIA")
    LABEL_MAPPING: Dict[str, int] = field(default_factory=lambda: {"NORMAL": 0, "PNEUMONIA": 1})
    INV_LABEL_MAPPING: Dict[int, str] = field(default_factory=lambda: {0: "NORMAL", 1: "PNEUMONIA"})

    # -----------------------------
    # Validation Strategy
    # -----------------------------
    # "patient_aware_split": Derives robust patient-grouped validation split (10% ~519 images) from train
    # "official": Uses the original official 16-image validation split
    VALIDATION_STRATEGY: str = "patient_aware_split"
    VAL_SPLIT_RATIO: float = 0.10

    # -----------------------------
    # Image & Preprocessing Params
    # -----------------------------
    IMAGE_HEIGHT: int = 224
    IMAGE_WIDTH: int = 224
    IMAGE_CHANNELS: int = 3
    IMAGE_SIZE: Tuple[int, int] = (224, 224)
    INPUT_SHAPE: Tuple[int, int, int] = (224, 224, 3)

    # Pixel normalization
    NORMALIZATION_MODE: str = "rescaling_0_1"
    RESCALE_FACTOR: float = 1.0 / 255.0

    # Data augmentation (training only, conservative for radiographs)
    AUGMENTATION_ENABLED: bool = True
    AUG_ROTATION_FACTOR: float = 0.05        # ~±18 degrees rotation
    AUG_TRANSLATION_HEIGHT: float = 0.05     # ±5% vertical shift
    AUG_TRANSLATION_WIDTH: float = 0.05      # ±5% horizontal shift
    AUG_ZOOM_HEIGHT: float = 0.08            # ±8% zoom
    AUG_ZOOM_WIDTH: float = 0.08
    AUG_HORIZONTAL_FLIP: bool = True         # horizontal flip
    AUG_FILL_MODE: str = "nearest"

    # -----------------------------
    # Reproducibility & Hardware
    # -----------------------------
    RANDOM_SEED: int = 42

    # -----------------------------
    # Training Hyperparameters
    # -----------------------------
    BATCH_SIZE: int = 32
    EPOCHS_BASELINE: int = 15
    EPOCHS_TRANSFER_PHASE1: int = 10
    EPOCHS_TRANSFER_PHASE2: int = 10
    LEARNING_RATE: float = 1e-4
    FINE_TUNE_LEARNING_RATE: float = 1e-5
    DECISION_THRESHOLD: float = 0.5

    # Early stopping & Reduce LR callbacks
    EARLY_STOPPING_PATIENCE: int = 5
    REDUCE_LR_PATIENCE: int = 2
    REDUCE_LR_FACTOR: float = 0.5
    MIN_LEARNING_RATE: float = 1e-7

    # -----------------------------
    # Model Choices
    # -----------------------------
    # DenseNet121 is the primary transfer learning candidate due to dense feature connectivity.
    # MobileNetV2 is available as an optional lightweight alternative for constrained environments.
    # Selection will be validated empirically against test-set metrics.
    PRIMARY_TRANSFER_MODEL: str = "densenet121"
    SUPPORTED_TRANSFER_MODELS: Tuple[str, ...] = (
        "densenet121",
        "mobilenetv2",
        "efficientnetb0",
        "resnet50"
    )

    # -----------------------------
    # Explainability (Grad-CAM)
    # -----------------------------
    # Target convolutional layer names for Grad-CAM
    GRADCAM_TARGET_LAYERS: Dict[str, str] = field(
        default_factory=lambda: {
            "baseline_cnn": "conv2d_block4_conv",
            "densenet121": "conv5_block16_2_conv",
            "mobilenetv2": "Conv_1",
            "resnet50": "conv5_block3_out",
        }
    )

    # -----------------------------
    # Mandatory Disclaimer
    # -----------------------------
    DISCLAIMER: str = (
        "This application is intended for educational and research purposes only "
        "and is not a substitute for professional medical diagnosis."
    )


# Instantiate default configuration instance
config = Config()
