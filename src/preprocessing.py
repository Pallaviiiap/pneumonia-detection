"""
Preprocessing and Data Augmentation Module for Pneumonia Detection.
Provides configurable, CPU-friendly image preprocessing, normalization, and
conservative data augmentation tailored specifically for chest radiographs.
"""

from pathlib import Path
from typing import Tuple, Optional, Any, List, Dict
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import config
from src.utils import get_logger

logger = get_logger("preprocessing")


def build_augmentation_layers(config_obj=config) -> keras.Sequential:
    """
    Builds the training-time augmentation sequential model.
    Applies conservative spatial transformations suitable for medical chest X-rays:
    - Small rotation (e.g. ±5% / ~18°)
    - Small translation (±5% height and width)
    - Small zoom (±8% zoom in/out)
    - Optional horizontal flip (subtle lateral variation)
    """
    layers = []
    
    # 1. Subtle rotation
    if config_obj.AUG_ROTATION_FACTOR > 0:
        layers.append(
            keras.layers.RandomRotation(
                factor=config_obj.AUG_ROTATION_FACTOR,
                fill_mode=config_obj.AUG_FILL_MODE,
                seed=config_obj.RANDOM_SEED,
                name="random_rotation"
            )
        )

    # 2. Subtle translation (horizontal & vertical)
    if config_obj.AUG_TRANSLATION_HEIGHT > 0 or config_obj.AUG_TRANSLATION_WIDTH > 0:
        layers.append(
            keras.layers.RandomTranslation(
                height_factor=config_obj.AUG_TRANSLATION_HEIGHT,
                width_factor=config_obj.AUG_TRANSLATION_WIDTH,
                fill_mode=config_obj.AUG_FILL_MODE,
                seed=config_obj.RANDOM_SEED,
                name="random_translation"
            )
        )

    # 3. Subtle zoom
    if config_obj.AUG_ZOOM_HEIGHT > 0 or config_obj.AUG_ZOOM_WIDTH > 0:
        layers.append(
            keras.layers.RandomZoom(
                height_factor=config_obj.AUG_ZOOM_HEIGHT,
                width_factor=config_obj.AUG_ZOOM_WIDTH,
                fill_mode=config_obj.AUG_FILL_MODE,
                seed=config_obj.RANDOM_SEED,
                name="random_zoom"
            )
        )

    # 4. Horizontal flip
    if config_obj.AUG_HORIZONTAL_FLIP:
        layers.append(
            keras.layers.RandomFlip(
                mode="horizontal",
                seed=config_obj.RANDOM_SEED,
                name="random_flip"
            )
        )

    aug_model = keras.Sequential(layers, name="data_augmentation")
    return aug_model


def build_normalization_layer(mode: Optional[str] = None, config_obj=config) -> keras.layers.Layer:
    """
    Builds the pixel normalization layer based on configuration or explicit mode.
    - 'rescaling_0_1': Maps pixel values from [0, 255] to [0.0, 1.0] (for baseline CNN)
    - 'mobilenetv2' / 'rescaling_neg1_1': Maps pixel values from [0, 255] to [-1.0, 1.0] (for MobileNetV2 ImageNet weights)
    """
    norm_mode = mode or config_obj.NORMALIZATION_MODE
    if norm_mode == "rescaling_0_1":
        return keras.layers.Rescaling(
            scale=config_obj.RESCALE_FACTOR,
            name="rescaling_0_1"
        )
    elif norm_mode in ("mobilenetv2", "rescaling_neg1_1"):
        return keras.layers.Rescaling(
            scale=1.0 / 127.5,
            offset=-1.0,
            name="rescaling_mobilenetv2"
        )
    raise ValueError(f"Unsupported normalization mode: {norm_mode}")


def preprocess_input_mobilenetv2(x: tf.Tensor) -> tf.Tensor:
    """
    Normalizes raw uint8 or float32 image tensors [0, 255] to [-1.0, 1.0]
    for pretrained MobileNetV2 ImageNet weights.
    Prevents double normalization when fed raw image tensors.
    """
    return (tf.cast(x, tf.float32) / 127.5) - 1.0



def build_preprocessing_pipeline(is_training: bool = False, config_obj=config) -> keras.Sequential:
    """
    Constructs the end-to-end preprocessing pipeline.
    - If is_training=True: Augmentation + Normalization
    - If is_training=False: Deterministic Normalization ONLY (No augmentation!)
    """
    layers = []
    if is_training and config_obj.AUGMENTATION_ENABLED:
        layers.append(build_augmentation_layers(config_obj=config_obj))
    
    layers.append(build_normalization_layer(config_obj=config_obj))
    pipeline = keras.Sequential(layers, name="preprocessing_pipeline")
    return pipeline


def load_raw_and_preprocessed_image(
    file_path: Path,
    target_size: Tuple[int, int] = config.IMAGE_SIZE,
    normalization_mode: str = "rescaling_0_1",
    config_obj=config
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Loads an image from disk in its native resolution (expanded to 3 channels),
    and creates its deterministically preprocessed counterpart.
    Returns:
    - raw_img: np.ndarray [H, W, 3] (uint8)
    - preprocessed_img: np.ndarray [target_H, target_W, 3] (float32, [0.0, 1.0] or [-1.0, 1.0])
    - metadata: Dict with original dimensions, aspect ratio, filename, and pixel stats
    """
    file_path = Path(file_path)
    with Image.open(file_path) as pil_img:
        # Convert to RGB if grayscale or RGBA
        if pil_img.mode != "RGB":
            pil_img_rgb = pil_img.convert("RGB")
        else:
            pil_img_rgb = pil_img.copy()

    raw_img = np.array(pil_img_rgb)
    orig_h, orig_w = raw_img.shape[:2]

    # Preprocessing: resize with bilinear interpolation
    resized_pil = pil_img_rgb.resize(target_size, resample=Image.Resampling.BILINEAR)
    resized_arr = np.array(resized_pil, dtype=np.float32)

    if normalization_mode == "rescaling_0_1":
        preprocessed_img = resized_arr * (1.0 / 255.0)
    elif normalization_mode in ("mobilenetv2", "rescaling_neg1_1"):
        preprocessed_img = (resized_arr / 127.5) - 1.0
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization_mode}")

    metadata = {
        "filename": file_path.name,
        "original_width": orig_w,
        "original_height": orig_h,
        "aspect_ratio": round(orig_w / orig_h, 4),
        "target_size": target_size,
        "normalization_mode": normalization_mode,
        "pixel_min": float(np.min(preprocessed_img)),
        "pixel_max": float(np.max(preprocessed_img)),
        "pixel_mean": float(np.mean(preprocessed_img)),
    }

    return raw_img, preprocessed_img, metadata


def generate_preprocessing_visualization(
    save_path: Optional[Path] = None,
    config_obj=config,
    num_augmented_samples: int = 4
) -> Path:
    """
    Generates a publication-quality comparison figure illustrating:
    - Column 1: Original Raw Chest Radiograph (Native dimensions & dynamic range)
    - Column 2: Preprocessed Image (224x224, Normalized to [0.0, 1.0])
    - Columns 3-6: Diverse training-time augmented variations of the same radiograph
    
    Evaluates 3 distinct clinical cases:
    1. NORMAL healthy radiograph
    2. PNEUMONIA (Bacterial consolidation)
    3. PNEUMONIA (Viral interstitial infiltration)
    """
    plots_dir = config_obj.PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    if save_path is None:
        save_path = plots_dir / "preprocessing_and_augmentation.png"

    aug_model = build_augmentation_layers(config_obj=config_obj)

    # Find sample images from train set
    train_normal = config_obj.TRAIN_DIR / "NORMAL"
    train_pneumonia = config_obj.TRAIN_DIR / "PNEUMONIA"

    sample_normal = next(train_normal.glob("*.jpeg"))
    
    # Locate bacterial and viral samples specifically
    bacterial_samples = list(train_pneumonia.glob("*bacteria*.jpeg"))
    viral_samples = list(train_pneumonia.glob("*virus*.jpeg"))

    sample_bacterial = bacterial_samples[0] if bacterial_samples else next(train_pneumonia.glob("*.jpeg"))
    sample_viral = viral_samples[0] if viral_samples else next(train_pneumonia.glob("*.jpeg"))

    cases = [
        ("NORMAL (Healthy)", sample_normal, "#2b5c8f"),
        ("PNEUMONIA (Bacterial)", sample_bacterial, "#d95f02"),
        ("PNEUMONIA (Viral)", sample_viral, "#7570b3")
    ]

    total_rows = len(cases)
    total_cols = 2 + num_augmented_samples  # Raw, Preprocessed, + 4 Augmentations

    fig, axes = plt.subplots(total_rows, total_cols, figsize=(18, 10.5), dpi=300)
    fig.suptitle(
        "Chest X-Ray Preprocessing & Data Augmentation Pipeline\n"
        "(Deterministic Resizing 224x224 & [0, 1] Rescaling vs Training-Time Conservative Augmentation)",
        fontsize=15, fontweight="bold", y=0.98
    )

    column_headers = [
        "1. Original Radiograph\n(Native Resolution)",
        "2. Preprocessed\n(224x224, [0, 1])",
        "3. Augmented Var. A\n(Training Mode)",
        "4. Augmented Var. B\n(Training Mode)",
        "5. Augmented Var. C\n(Training Mode)",
        "6. Augmented Var. D\n(Training Mode)"
    ]

    for row_idx, (case_title, img_path, badge_color) in enumerate(cases):
        raw_img, prep_img, meta = load_raw_and_preprocessed_image(img_path, config_obj=config_obj)

        # Generate batch of 4 augmentations from the preprocessed tensor
        prep_tensor = tf.expand_dims(tf.constant(prep_img, dtype=tf.float32), axis=0)
        # Augmentations applied repeatedly with training=True
        aug_tensors = []
        for seed_offset in range(num_augmented_samples):
            # Pass through augmentation model
            # Multiply back to [0, 255] if aug expects raw or pass directly (Keras random layers work on both)
            aug_out = aug_model(prep_tensor * 255.0, training=True) / 255.0
            aug_tensors.append(tf.squeeze(aug_out, axis=0).numpy())

        # Col 0: Original
        ax_raw = axes[row_idx, 0]
        ax_raw.imshow(raw_img)
        ax_raw.set_title(
            f"{case_title}\n{meta['filename'][:18]}...\n"
            f"{meta['original_width']}x{meta['original_height']} px (Raw)",
            fontsize=9, fontweight="bold", color=badge_color
        )
        ax_raw.axis("off")

        # Col 1: Preprocessed
        ax_prep = axes[row_idx, 1]
        ax_prep.imshow(prep_img)
        ax_prep.set_title(
            f"224x224 Normalized\nRange: [{meta['pixel_min']:.2f}, {meta['pixel_max']:.2f}]\nMean: {meta['pixel_mean']:.2f}",
            fontsize=9, fontweight="semibold"
        )
        ax_prep.axis("off")

        # Cols 2 to 5: Augmentations
        for aug_idx in range(num_augmented_samples):
            ax_aug = axes[row_idx, 2 + aug_idx]
            aug_img = aug_tensors[aug_idx]
            # Clip display values to [0, 1]
            display_img = np.clip(aug_img, 0.0, 1.0)
            ax_aug.imshow(display_img)
            a_min, a_max = float(np.min(aug_img)), float(np.max(aug_img))
            ax_aug.set_title(
                f"Augmented #{aug_idx + 1}\nRange: [{a_min:.2f}, {a_max:.2f}]\n"
                f"Rot/Zoom/Shift/Flip",
                fontsize=8.5
            )
            ax_aug.axis("off")

    # Set column titles on top row
    for col_idx in range(total_cols):
        axes[0, col_idx].annotate(
            column_headers[col_idx],
            xy=(0.5, 1.28), xycoords="axes fraction",
            ha="center", va="bottom",
            fontsize=10.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f4f8", edgecolor="#b0bec5")
        )

    plt.figtext(0.5, 0.01, config_obj.DISCLAIMER, ha="center", fontsize=8, color="#555555", style="italic")
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved preprocessing and augmentation visualization to: {save_path}")
    return save_path


def main():
    logger.info("Running standalone preprocessing and augmentation pipeline test...")
    logger.info(f"Target size: {config.IMAGE_SIZE}")
    logger.info(f"Normalization mode: {config.NORMALIZATION_MODE} (scale: {config.RESCALE_FACTOR})")
    logger.info(f"Augmentation enabled: {config.AUGMENTATION_ENABLED}")
    logger.info(f"Augmentation params: rotation={config.AUG_ROTATION_FACTOR}, translation=({config.AUG_TRANSLATION_HEIGHT}, {config.AUG_TRANSLATION_WIDTH}), zoom=({config.AUG_ZOOM_HEIGHT}, {config.AUG_ZOOM_WIDTH}), flip={config.AUG_HORIZONTAL_FLIP}")

    # Generate plot
    plot_path = generate_preprocessing_visualization()
    print(f"\nVisualization generated successfully at: {plot_path}")


if __name__ == "__main__":
    main()
