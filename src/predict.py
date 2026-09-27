"""
Stage 9: Standalone Prediction / Inference Pipeline for Pneumonia Detection.
Provides a clean, reusable standalone inference API and CLI for predicting pneumonia
from single chest X-ray radiographs using the finalized MobileNetV2 transfer learning model,
with optional Grad-CAM explainability integration.

Intended strictly for educational and research purposes.
Does not constitute a clinical diagnostic system or substitute for professional medical advice.
"""

import os
# Suppress non-error TensorFlow warnings before importing framework
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
from PIL import Image, UnidentifiedImageError

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
import keras

from src.config import config
from src.utils import get_logger
from src.preprocessing import preprocess_input_mobilenetv2
from src.gradcam import (
    compute_gradcam_heatmap,
    create_gradcam_overlay,
    render_gradcam_comparison_panel
)

logger = get_logger("predict")

# Supported image file extensions
SUPPORTED_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def validate_and_load_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = config.IMAGE_SIZE
) -> Tuple[np.ndarray, tf.Tensor, int, int]:
    """
    Validates input image existence, format, and integrity.
    Loads image into memory, converts grayscale/RGBA to 3-channel RGB,
    preserves native dimensions, and resizes to target size with MobileNetV2 normalization [-1.0, 1.0].
    
    Returns:
    - raw_img_rgb: np.ndarray [orig_H, orig_W, 3] (uint8)
    - input_tensor: tf.Tensor [1, target_H, target_W, 3] (float32 in [-1.0, 1.0])
    - orig_w: original width in pixels
    - orig_h: original height in pixels
    
    Raises:
    - FileNotFoundError: If image file does not exist.
    - ValueError: If file extension is unsupported or image file is corrupt / unreadable.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Specified path is not a file: {path}")

    # Validate file extension
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image file format '{ext}' for file '{path.name}'. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # Safely load and convert image
    try:
        with Image.open(path) as pil_img:
            orig_w, orig_h = pil_img.size
            if orig_w <= 0 or orig_h <= 0:
                raise ValueError(f"Image has invalid dimensions: {orig_w}x{orig_h}")
            
            # Convert to 3-channel RGB (handles grayscale 'L', bilevel '1', palette 'P', 'RGBA', etc.)
            if pil_img.mode != "RGB":
                pil_img_rgb = pil_img.convert("RGB")
            else:
                pil_img_rgb = pil_img.copy()
            
            # Force load image data to catch any mid-stream corruption
            pil_img_rgb.load()
    except (UnidentifiedImageError, OSError, ValueError) as err:
        raise ValueError(f"Unable to read or parse image file '{path.name}': {err}") from err

    raw_img_rgb = np.array(pil_img_rgb, dtype=np.uint8)

    # Preprocessing: resize with bilinear interpolation to target size (224, 224)
    resized_pil = pil_img_rgb.resize(target_size, resample=Image.Resampling.BILINEAR)
    resized_arr = np.array(resized_pil, dtype=np.float32)

    # MobileNetV2 normalization: pixel / 127.5 - 1.0 -> [-1.0, 1.0]
    normalized_tensor = preprocess_input_mobilenetv2(tf.constant(resized_arr))
    input_tensor = tf.expand_dims(normalized_tensor, axis=0)  # Shape: (1, 224, 224, 3)

    return raw_img_rgb, input_tensor, orig_w, orig_h


class PneumoniaPredictor:
    """
    Reusable predictor instance that loads and caches the finalized MobileNetV2 model
    for single-image inference without reload overhead.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config_obj=config
    ):
        self.config = config_obj
        self.model_path = (
            Path(model_path) if model_path
            else self.config.MODELS_DIR / "mobilenetv2_transfer_best.keras"
        )
        self._model: Optional[keras.Model] = None
        self._load_model()

    def _load_model(self) -> keras.Model:
        """Loads and verifies the saved Keras checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {self.model_path}. "
                "Ensure Stage 6 transfer learning has completed."
            )
        logger.info(f"Loading MobileNetV2 checkpoint from: {self.model_path}")
        self._model = keras.models.load_model(self.model_path)
        return self._model

    @property
    def model(self) -> keras.Model:
        if self._model is None:
            self._load_model()
        return self._model

    def predict(
        self,
        image_path: Union[str, Path],
        threshold: float = config.DECISION_THRESHOLD,
        generate_gradcam: bool = False,
        gradcam_output_dir: Optional[Union[str, Path]] = None,
        gradcam_target_layer: str = "Conv_1"
    ) -> Dict[str, Any]:
        """
        Runs inference on a single chest X-ray image.

        Args:
            image_path: Path to the radiograph (.jpeg, .jpg, .png).
            threshold: Classification threshold (fixed at 0.50 by project specification).
            generate_gradcam: If True, generates a 3-panel Grad-CAM visualization.
            gradcam_output_dir: Destination directory for Grad-CAM output. Defaults to outputs/gradcam/predictions.
            gradcam_target_layer: Convolutional target layer name (defaults to 'Conv_1').

        Returns:
            Dict containing:
                - image_path: Path to image as string
                - original_width: Original image width
                - original_height: Original image height
                - pneumonia_probability: Model scalar probability in [0.0, 1.0]
                - predicted_class: "NORMAL" or "PNEUMONIA"
                - predicted_label: 0 or 1
                - threshold: Applied threshold (0.50)
                - gradcam_path: (optional) Path to saved Grad-CAM figure if requested
        """
        path = Path(image_path)
        raw_img_rgb, input_tensor, orig_w, orig_h = validate_and_load_image(
            path, target_size=self.config.IMAGE_SIZE
        )

        # Run model inference
        prediction_tensor = self.model(input_tensor, training=False)
        pneumonia_prob = float(prediction_tensor[0, 0].numpy())

        # Determine predicted class based on threshold (0 = NORMAL, 1 = PNEUMONIA)
        predicted_label = 1 if pneumonia_prob >= threshold else 0
        predicted_class = self.config.INV_LABEL_MAPPING[predicted_label]

        result: Dict[str, Any] = {
            "image_path": str(path.resolve()),
            "original_width": orig_w,
            "original_height": orig_h,
            "pneumonia_probability": round(pneumonia_prob, 6),
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "threshold": float(threshold)
        }

        # Optional Grad-CAM explainability generation
        if generate_gradcam:
            out_dir = (
                Path(gradcam_output_dir) if gradcam_output_dir
                else self.config.OUTPUTS_DIR / "gradcam" / "predictions"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            gradcam_save_path = out_dir / f"{path.stem}_gradcam.png"

            # Compute Grad-CAM on target convolutional layer
            heatmap_raw, _, _ = compute_gradcam_heatmap(
                self.model,
                input_tensor,
                target_layer_name=gradcam_target_layer
            )

            resized_heatmap, overlay_img = create_gradcam_overlay(
                raw_img_rgb,
                heatmap_raw,
                alpha=0.45,
                colormap_name="jet"
            )

            example_info = {
                "category_title": f"Single Prediction: {predicted_class}",
                "filename": path.name,
                "true_class": "N/A (Inference Input)",
                "predicted_class": predicted_class,
                "predicted_probability": pneumonia_prob,
                "is_correct": True,  # Neutral green styling for single inference
                "target_layer": gradcam_target_layer
            }

            render_gradcam_comparison_panel(
                raw_img_rgb,
                resized_heatmap,
                overlay_img,
                example_info,
                gradcam_save_path
            )

            result["gradcam_path"] = str(gradcam_save_path.resolve())

        return result


# Global default predictor instance for repeated function calls
_GLOBAL_PREDICTOR: Optional[PneumoniaPredictor] = None


def get_predictor(
    model_path: Optional[Union[str, Path]] = None,
    config_obj=config
) -> PneumoniaPredictor:
    """Returns a cached or new PneumoniaPredictor instance."""
    global _GLOBAL_PREDICTOR
    if model_path is not None:
        return PneumoniaPredictor(model_path=model_path, config_obj=config_obj)
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = PneumoniaPredictor(config_obj=config_obj)
    return _GLOBAL_PREDICTOR


def predict_image(
    image_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None,
    threshold: float = config.DECISION_THRESHOLD,
    generate_gradcam: bool = False,
    gradcam_output_dir: Optional[Union[str, Path]] = None,
    config_obj=config
) -> Dict[str, Any]:
    """
    High-level functional API for single chest radiograph inference.

    Args:
        image_path: Path to the image file.
        model_path: Optional path to custom model checkpoint.
        threshold: Classification threshold (default 0.50).
        generate_gradcam: Whether to generate Grad-CAM visualization.
        gradcam_output_dir: Output folder for Grad-CAM figure.

    Returns:
        Structured prediction dictionary.
    """
    predictor = get_predictor(model_path=model_path, config_obj=config_obj)
    return predictor.predict(
        image_path=image_path,
        threshold=threshold,
        generate_gradcam=generate_gradcam,
        gradcam_output_dir=gradcam_output_dir
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser for standalone inference."""
    parser = argparse.ArgumentParser(
        description="Pneumonia Detection: Standalone Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to the chest X-ray image file (.jpeg, .jpg, .png)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to Keras model checkpoint (defaults to models/mobilenetv2_transfer_best.keras)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=config.DECISION_THRESHOLD,
        help="Decision threshold for pneumonia classification (fixed project default: 0.50)"
    )
    parser.add_argument(
        "--gradcam", "-g",
        action="store_true",
        help="Generate and save Grad-CAM explainability visualization"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save Grad-CAM visualization (defaults to outputs/gradcam/predictions)"
    )
    return parser


def main():
    """CLI entry point for standalone prediction."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        result = predict_image(
            image_path=args.image,
            model_path=args.model,
            threshold=args.threshold,
            generate_gradcam=args.gradcam,
            gradcam_output_dir=args.output_dir
        )
    except Exception as err:
        print(f"\n[ERROR] Inference failed: {err}\n", file=sys.stderr)
        sys.exit(1)

    # Format clean output
    img_name = Path(args.image).name
    prob_pct = result["pneumonia_probability"] * 100.0
    thresh_pct = result["threshold"] * 100.0

    print("\n" + "=" * 55)
    print("PNEUMONIA DETECTION - INFERENCE RESULT")
    print("=" * 55)
    print(f"Image:                 {img_name}")
    print(f"Original size:         {result['original_width']} x {result['original_height']}")
    print(f"Pneumonia probability: {prob_pct:.2f}%")
    print(f"Prediction:            {result['predicted_class']}")
    print(f"Threshold:             {thresh_pct:.2f}%")
    if "gradcam_path" in result:
        print(f"Grad-CAM overlay:      {result['gradcam_path']}")
    print("-" * 55)
    print("Disclaimer: For educational/research use only. Does not constitute clinical diagnosis.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
