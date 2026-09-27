"""
Stage 8: Grad-CAM (Gradient-Weighted Class Activation Mapping) Explainability Module.
Provides visual explanation of convolutional feature activations contributing to the finalized
MobileNetV2 pneumonia classification predictions.

Visualizations are strictly model-explanation artifacts indicating regions receiving greater
activation for a given prediction, not clinical diagnostic proof or medical evidence.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.utils import get_logger, save_json
from src.preprocessing import preprocess_input_mobilenetv2

logger = get_logger("gradcam")


def load_image_for_gradcam(
    image_path: Path,
    target_size: Tuple[int, int] = config.IMAGE_SIZE
) -> Tuple[np.ndarray, tf.Tensor, Dict[str, Any]]:
    """
    Loads an image from disk, preserves the original native-resolution RGB array,
    and constructs the preprocessed MobileNetV2 input tensor in range [-1.0, 1.0].
    
    Returns:
    - raw_img_rgb: np.ndarray [orig_H, orig_W, 3] (uint8)
    - input_tensor: tf.Tensor [1, 224, 224, 3] (float32, [-1.0, 1.0])
    - meta: metadata dictionary with dimensions and filename
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    with Image.open(image_path) as pil_img:
        if pil_img.mode != "RGB":
            pil_img_rgb = pil_img.convert("RGB")
        else:
            pil_img_rgb = pil_img.copy()

    raw_img_rgb = np.array(pil_img_rgb, dtype=np.uint8)
    orig_h, orig_w = raw_img_rgb.shape[:2]

    # Preprocessing: resize to target (224, 224) and scale to [-1.0, 1.0]
    resized_pil = pil_img_rgb.resize(target_size, resample=Image.Resampling.BILINEAR)
    resized_arr = np.array(resized_pil, dtype=np.float32)
    # Apply MobileNetV2 normalization [-1.0, 1.0]
    normalized_tensor = preprocess_input_mobilenetv2(tf.constant(resized_arr))
    input_tensor = tf.expand_dims(normalized_tensor, axis=0)  # Shape: (1, 224, 224, 3)

    meta = {
        "filename": image_path.name,
        "image_path": str(image_path.resolve()),
        "original_width": orig_w,
        "original_height": orig_h,
        "aspect_ratio": round(orig_w / orig_h, 4),
        "target_size": list(target_size),
        "input_tensor_shape": list(input_tensor.shape),
        "input_tensor_min": float(tf.reduce_min(input_tensor)),
        "input_tensor_max": float(tf.reduce_max(input_tensor))
    }
    return raw_img_rgb, input_tensor, meta


def compute_gradcam_heatmap(
    model: keras.Model,
    input_tensor: tf.Tensor,
    target_layer_name: str = "Conv_1",
    target_class_idx: Optional[int] = None
) -> Tuple[np.ndarray, float, int]:
    """
    Computes 2D Grad-CAM class activation map for a target convolutional layer.
    
    Method:
    1. Evaluates forward pass capturing target layer feature activations and output prediction.
    2. Identifies the prediction probability p and target class (0 = NORMAL, 1 = PNEUMONIA).
    3. Calculates gradients of the target class score w.r.t. the target feature maps.
    4. Computes spatial global average pooling of gradients -> channel importance weights.
    5. Computes weighted linear combination of feature channels.
    6. Applies ReLU to retain features with a positive influence on the target class.
    7. Normalizes the resulting heatmap to [0.0, 1.0].
    
    Returns:
    - heatmap: np.ndarray [H_feat, W_feat] float32 in [0.0, 1.0]
    - pred_prob: float (raw model output probability for PNEUMONIA)
    - pred_class: int (0 for NORMAL, 1 for PNEUMONIA at threshold 0.50)
    """
    # Verify target layer exists
    try:
        target_layer = model.get_layer(target_layer_name)
    except Exception as e:
        raise ValueError(f"Target layer '{target_layer_name}' not found in model: {e}")

    # Build dual-output gradient model
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor, training=False)
        pred_prob = float(predictions[0, 0].numpy())
        pred_class = 1 if pred_prob >= config.DECISION_THRESHOLD else 0

        # Class to explain
        class_to_explain = target_class_idx if target_class_idx is not None else pred_class
        if class_to_explain == 1:
            target_score = predictions[:, 0]
        else:
            target_score = 1.0 - predictions[:, 0]

    # Compute gradients of class score w.r.t. convolutional feature maps
    grads = tape.gradient(target_score, conv_outputs)
    if grads is None:
        raise RuntimeError(f"Gradients w.r.t. layer '{target_layer_name}' evaluated to None.")

    # Global average pooling over spatial dimensions (H, W) -> shape: (C,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by importance weights
    conv_outputs = conv_outputs[0]  # shape: (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep only features that positively correlate with the target score
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalize heatmap to [0.0, 1.0]
    max_val = tf.reduce_max(heatmap)
    if max_val > 0.0:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)

    heatmap_np = heatmap.numpy().astype(np.float32)
    return heatmap_np, pred_prob, pred_class


def create_gradcam_overlay(
    raw_img_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resizes the low-resolution convolutional heatmap to the native image resolution,
    converts it to an RGB colormap, and blends it over the original radiograph.
    
    Returns:
    - resized_heatmap: np.ndarray [orig_H, orig_W] float32 in [0.0, 1.0]
    - overlay_img: np.ndarray [orig_H, orig_W, 3] uint8 blended RGB image
    """
    orig_h, orig_w = raw_img_rgb.shape[:2]

    # Resize heatmap to match original image dimensions using PIL bilinear resampling
    heatmap_pil = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0))
    resized_pil = heatmap_pil.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)
    resized_heatmap = np.array(resized_pil, dtype=np.float32) / 255.0

    # Apply color map
    cmap = matplotlib.colormaps.get_cmap(colormap_name)
    colored_heatmap = cmap(resized_heatmap)[:, :, :3]  # Drop alpha channel -> [0.0, 1.0]
    colored_heatmap = np.uint8(colored_heatmap * 255.0)

    # Blend original grayscale X-ray and colored heatmap
    # Keep original radiograph structure crisp and visible
    raw_float = raw_img_rgb.astype(np.float32)
    heat_float = colored_heatmap.astype(np.float32)
    overlay = (1.0 - alpha) * raw_float + alpha * heat_float
    overlay_img = np.clip(overlay, 0, 255).astype(np.uint8)

    return resized_heatmap, overlay_img


def render_gradcam_comparison_panel(
    raw_img_rgb: np.ndarray,
    resized_heatmap: np.ndarray,
    overlay_img: np.ndarray,
    example_info: Dict[str, Any],
    save_path: Path
) -> Path:
    """
    Renders publication-ready 3-panel figure:
    1. Original Radiograph (Native resolution)
    2. Grad-CAM Activation Heatmap
    3. Grad-CAM Heatmap Overlay
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=300)

    category_title = example_info.get("category_title", "Grad-CAM Activation")
    filename = example_info.get("filename", "")
    true_cls = example_info.get("true_class", "")
    pred_cls = example_info.get("predicted_class", "")
    prob = example_info.get("predicted_probability", 0.0)
    is_correct = example_info.get("is_correct", True)
    target_layer = example_info.get("target_layer", "Conv_1")

    status_str = "CORRECT" if is_correct else "INCORRECT"
    status_color = "#1b5e20" if is_correct else "#b71c1c"

    fig.suptitle(
        f"MobileNetV2 Grad-CAM Explainability: {category_title}\n"
        f"Filename: {filename} | True: {true_cls} | Predicted: {pred_cls} (P(Pneumonia) = {prob:.4f}) | {status_str}",
        fontsize=13, fontweight="bold", y=0.98, color=status_color
    )

    # 1. Original X-Ray
    axes[0].imshow(raw_img_rgb)
    axes[0].set_title(
        f"1. Original Radiograph\n{raw_img_rgb.shape[1]}x{raw_img_rgb.shape[0]} px | Ground Truth: {true_cls}",
        fontsize=10.5, fontweight="semibold", pad=8
    )
    axes[0].axis("off")

    # 2. Grad-CAM Heatmap
    im1 = axes[1].imshow(resized_heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title(
        f"2. Grad-CAM Heatmap\nTarget Layer: '{target_layer}' (7x7 -> {raw_img_rgb.shape[1]}x{raw_img_rgb.shape[0]})",
        fontsize=10.5, fontweight="semibold", pad=8
    )
    axes[1].axis("off")
    cbar = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Activation Intensity", rotation=270, labelpad=12, fontsize=9)

    # 3. Heatmap Overlay
    axes[2].imshow(overlay_img)
    axes[2].set_title(
        f"3. Heatmap Overlay\nPredicted: {pred_cls} ({prob:.2%})",
        fontsize=10.5, fontweight="semibold", pad=8
    )
    axes[2].axis("off")

    disclaimer_text = (
        "Explanation Note: Highlights regions receiving greater activation for the model's prediction. "
        "Does not constitute clinical evidence of pathology or diagnostic reasoning. Educational/research use only."
    )
    plt.figtext(0.5, 0.02, disclaimer_text, ha="center", fontsize=7.5, color="#555555", style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Grad-CAM panel visualization to: {save_path}")
    return save_path


def generate_representative_gradcam_artifacts(
    model_checkpoint_path: Optional[Path] = None,
    predictions_csv_path: Optional[Path] = None,
    target_layer_name: str = "Conv_1",
    config_obj=config
) -> Dict[str, Any]:
    """
    Selects 4 representative examples from the official held-out test predictions:
    1. Correct NORMAL prediction
    2. Correct PNEUMONIA prediction
    3. False-positive NORMAL -> PNEUMONIA
    4. False-negative PNEUMONIA -> NORMAL
    
    Generates high-resolution 3-panel Grad-CAM visualizations, individual assets,
    and metadata JSON.
    """
    ckpt_path = Path(model_checkpoint_path) if model_checkpoint_path else config_obj.MODELS_DIR / "mobilenetv2_transfer_best.keras"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    csv_path = Path(predictions_csv_path) if predictions_csv_path else config_obj.METRICS_DIR / "final_test_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Test predictions CSV not found: {csv_path}")

    logger.info(f"Loading final model from {ckpt_path}...")
    model = keras.models.load_model(ckpt_path)

    # Verify target layer
    target_layer = model.get_layer(target_layer_name)
    logger.info(f"Verified target Grad-CAM layer '{target_layer_name}': output shape = {target_layer.output.shape}")

    df = pd.read_csv(csv_path)

    # Query representative cases
    # 1. Correct Normal (high confidence normal)
    cn_candidates = df[(df["true_label"] == 0) & (df["predicted_label"] == 0)].sort_values("predicted_probability")
    sample_cn = cn_candidates.iloc[0]

    # 2. Correct Pneumonia (high confidence pneumonia)
    cp_candidates = df[(df["true_label"] == 1) & (df["predicted_label"] == 1)].sort_values("predicted_probability", ascending=False)
    sample_cp = cp_candidates.iloc[0]

    # 3. False Positive (True Normal, predicted Pneumonia)
    fp_candidates = df[(df["true_label"] == 0) & (df["predicted_label"] == 1)].sort_values("predicted_probability", ascending=False)
    sample_fp = fp_candidates.iloc[0]

    # 4. False Negative (True Pneumonia, predicted Normal)
    fn_candidates = df[(df["true_label"] == 1) & (df["predicted_label"] == 0)].sort_values("predicted_probability")
    sample_fn = fn_candidates.iloc[0]

    examples = [
        {
            "category_key": "correct_normal",
            "category_title": "Correct NORMAL Diagnosis (True Normal, Predicted Normal)",
            "output_filename": "gradcam_correct_normal.png",
            "row": sample_cn
        },
        {
            "category_key": "correct_pneumonia",
            "category_title": "Correct PNEUMONIA Diagnosis (True Pneumonia, Predicted Pneumonia)",
            "output_filename": "gradcam_correct_pneumonia.png",
            "row": sample_cp
        },
        {
            "category_key": "false_positive",
            "category_title": "False Positive Error (True Normal, Predicted Pneumonia)",
            "output_filename": "gradcam_false_positive.png",
            "row": sample_fp
        },
        {
            "category_key": "false_negative",
            "category_title": "False Negative Error (True Pneumonia, Predicted Normal)",
            "output_filename": "gradcam_false_negative.png",
            "row": sample_fn
        }
    ]

    gradcam_dir = config_obj.OUTPUTS_DIR / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = gradcam_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    results_metadata: Dict[str, Any] = {
        "model_name": "mobilenetv2_transfer",
        "checkpoint_path": str(ckpt_path),
        "target_layer_name": target_layer_name,
        "target_layer_output_shape": list(target_layer.output.shape),
        "decision_threshold": config_obj.DECISION_THRESHOLD,
        "disclaimer": "Regions receiving greater Grad-CAM activation for the model's prediction. Not clinical evidence.",
        "examples": {}
    }

    for ex in examples:
        row = ex["row"]
        img_path = Path(row["image_path"])
        cat_key = ex["category_key"]

        logger.info(f"Processing Grad-CAM for {cat_key} ({row['filename']})...")
        raw_img_rgb, input_tensor, meta = load_image_for_gradcam(img_path)

        heatmap_raw, pred_prob, pred_cls = compute_gradcam_heatmap(
            model,
            input_tensor,
            target_layer_name=target_layer_name
        )

        resized_heatmap, overlay_img = create_gradcam_overlay(
            raw_img_rgb,
            heatmap_raw,
            alpha=0.45,
            colormap_name="jet"
        )

        example_info = {
            "category_key": cat_key,
            "category_title": ex["category_title"],
            "filename": row["filename"],
            "patient_id": str(row["patient_id"]),
            "image_path": str(img_path),
            "true_label": int(row["true_label"]),
            "true_class": str(row["class_name"]),
            "predicted_label": int(row["predicted_label"]),
            "predicted_class": str(row["predicted_class"]),
            "predicted_probability": float(row["predicted_probability"]),
            "is_correct": bool(row["is_correct"]),
            "target_layer": target_layer_name,
            "original_dimensions": [meta["original_width"], meta["original_height"]],
            "heatmap_resolution": list(heatmap_raw.shape),
            "panel_plot_path": str(gradcam_dir / ex["output_filename"]),
            "individual_assets": {
                "original_image": str(examples_dir / f"{cat_key}_original.png"),
                "heatmap_image": str(examples_dir / f"{cat_key}_heatmap.png"),
                "overlay_image": str(examples_dir / f"{cat_key}_overlay.png")
            }
        }

        # 1. Render 3-panel comparison figure
        panel_save_path = gradcam_dir / ex["output_filename"]
        render_gradcam_comparison_panel(
            raw_img_rgb,
            resized_heatmap,
            overlay_img,
            example_info,
            panel_save_path
        )

        # 2. Save individual assets for downstream usage (e.g. Streamlit app)
        Image.fromarray(raw_img_rgb).save(examples_dir / f"{cat_key}_original.png")
        Image.fromarray(np.uint8(resized_heatmap * 255.0)).save(examples_dir / f"{cat_key}_heatmap.png")
        Image.fromarray(overlay_img).save(examples_dir / f"{cat_key}_overlay.png")

        results_metadata["examples"][cat_key] = example_info

    # Save metadata JSON
    metrics_json_path = config_obj.METRICS_DIR / "gradcam_examples.json"
    save_json(results_metadata, metrics_json_path)
    logger.info(f"Grad-CAM metadata saved to: {metrics_json_path}")

    return results_metadata


def main():
    print("\n" + "=" * 65)
    print("STAGE 8: GRAD-CAM EXPLAINABILITY PIPELINE GENERATION")
    print("=" * 65)
    summary = generate_representative_gradcam_artifacts()
    print(f"Target Layer:            {summary['target_layer_name']} ({summary['target_layer_output_shape']})")
    print(f"Generated Examples:")
    for k, ex in summary["examples"].items():
        print(f"  - [{k}] {ex['filename']} | True: {ex['true_class']} -> Pred: {ex['predicted_class']} (Prob: {ex['predicted_probability']:.4f})")
    print(f"Metadata JSON:           outputs/metrics/gradcam_examples.json")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
