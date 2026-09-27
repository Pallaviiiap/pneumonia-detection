"""
Stage 9: Automated Tests for Standalone Prediction / Inference Pipeline.
Validates:
1. Model loading: checkpoint exists, model loads, output shape compatible with binary classification.
2. Preprocessing: tensor shape (1, 224, 224, 3), pixel range in [-1.0, 1.0], grayscale -> 3 channels, RGBA -> 3 channels.
3. Prediction: scalar probability in [0, 1], binary label (0 or 1), class ("NORMAL" or "PNEUMONIA"), threshold defaults to 0.50.
4. Input validation: nonexistent image, unsupported format, and corrupt image handling.
5. Grad-CAM integration: single-image Grad-CAM panel generation, file existence, and image validity.
"""

from pathlib import Path
import pytest
import numpy as np
from PIL import Image
import tensorflow as tf
import keras

from src.config import config
from src.predict import (
    validate_and_load_image,
    PneumoniaPredictor,
    get_predictor,
    predict_image,
    SUPPORTED_EXTENSIONS
)


@pytest.fixture(scope="module")
def predictor():
    """Module-scoped predictor to avoid reloading checkpoint multiple times."""
    return get_predictor()


@pytest.fixture
def sample_rgb_image(tmp_path):
    """Creates a temporary synthetic RGB JPEG image."""
    img_path = tmp_path / "synthetic_rgb.jpeg"
    arr = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(img_path)
    return img_path


@pytest.fixture
def sample_grayscale_image(tmp_path):
    """Creates a temporary synthetic grayscale PNG image."""
    img_path = tmp_path / "synthetic_gray.png"
    arr = np.random.randint(0, 255, (150, 180), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(img_path)
    return img_path


@pytest.fixture
def sample_rgba_image(tmp_path):
    """Creates a temporary synthetic RGBA PNG image."""
    img_path = tmp_path / "synthetic_rgba.png"
    arr = np.random.randint(0, 255, (140, 140, 4), dtype=np.uint8)
    Image.fromarray(arr, mode="RGBA").save(img_path)
    return img_path


@pytest.fixture
def corrupt_image(tmp_path):
    """Creates a file with corrupt non-image bytes but a valid extension."""
    img_path = tmp_path / "corrupted_xray.jpeg"
    img_path.write_bytes(b"CORRUPT_NOT_A_VALID_IMAGE_FILE_DATA_HEADER_123456789")
    return img_path


@pytest.fixture
def unsupported_format_image(tmp_path):
    """Creates an image with an unsupported extension."""
    img_path = tmp_path / "unsupported_xray.bmp"
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(img_path)
    return img_path


# =========================================================================
# 1. Model Loading Tests
# =========================================================================

def test_final_checkpoint_exists():
    """Verify that the finalized MobileNetV2 checkpoint exists."""
    ckpt_path = config.MODELS_DIR / "mobilenetv2_transfer_best.keras"
    assert ckpt_path.exists(), f"Model checkpoint not found at: {ckpt_path}"
    assert ckpt_path.stat().st_size > 1_000_000, "Checkpoint file is unexpectedly small"


def test_model_loads_and_has_binary_output(predictor):
    """Verify that the predictor's model loads and has compatible binary classification output."""
    model = predictor.model
    assert isinstance(model, keras.Model)
    assert model.output_shape == (None, 1), f"Expected output shape (None, 1), got {model.output_shape}"


# =========================================================================
# 2. Preprocessing & Input Validation Tests
# =========================================================================

def test_preprocessing_rgb_image(sample_rgb_image):
    """Verify RGB image preprocessing produces (1, 224, 224, 3) tensor in [-1.0, 1.0]."""
    raw_img, tensor, w, h = validate_and_load_image(sample_rgb_image)
    
    assert w == 160 and h == 120
    assert raw_img.shape == (120, 160, 3)
    assert raw_img.dtype == np.uint8
    
    assert tensor.shape == (1, 224, 224, 3)
    t_min = float(tf.reduce_min(tensor))
    t_max = float(tf.reduce_max(tensor))
    assert t_min >= -1.05 and t_max <= 1.05


def test_preprocessing_grayscale_image(sample_grayscale_image):
    """Verify grayscale single-channel image is converted to 3-channel RGB."""
    raw_img, tensor, w, h = validate_and_load_image(sample_grayscale_image)
    
    assert w == 180 and h == 150
    assert raw_img.shape == (150, 180, 3), "Grayscale image should be converted to 3-channel RGB"
    assert tensor.shape == (1, 224, 224, 3)


def test_preprocessing_rgba_image(sample_rgba_image):
    """Verify RGBA 4-channel image is converted to 3-channel RGB."""
    raw_img, tensor, w, h = validate_and_load_image(sample_rgba_image)
    
    assert w == 140 and h == 140
    assert raw_img.shape == (140, 140, 3), "RGBA image should be converted to 3-channel RGB"
    assert tensor.shape == (1, 224, 224, 3)


def test_input_validation_nonexistent_file():
    """Verify FileNotFoundError is raised for nonexistent path."""
    nonexistent = Path("dataset/chest_xray/nonexistent_image_12345.jpeg")
    with pytest.raises(FileNotFoundError, match="Image file does not exist"):
        validate_and_load_image(nonexistent)


def test_input_validation_unsupported_extension(unsupported_format_image):
    """Verify ValueError is raised for unsupported file format."""
    with pytest.raises(ValueError, match="Unsupported image file format"):
        validate_and_load_image(unsupported_format_image)


def test_input_validation_corrupted_file(corrupt_image):
    """Verify ValueError is raised for corrupt or unreadable image file."""
    with pytest.raises(ValueError, match="Unable to read or parse image file"):
        validate_and_load_image(corrupt_image)


# =========================================================================
# 3. Prediction Output & Thresholding Tests
# =========================================================================

def test_predict_image_structure_and_types(sample_rgb_image):
    """Verify inference output structure, field names, and value types."""
    result = predict_image(sample_rgb_image)
    
    # Required keys according to specification
    expected_keys = {
        "image_path",
        "original_width",
        "original_height",
        "pneumonia_probability",
        "predicted_class",
        "predicted_label",
        "threshold"
    }
    assert expected_keys.issubset(result.keys())
    
    assert isinstance(result["original_width"], int) and result["original_width"] == 160
    assert isinstance(result["original_height"], int) and result["original_height"] == 120
    assert isinstance(result["pneumonia_probability"], float)
    assert 0.0 <= result["pneumonia_probability"] <= 1.0
    
    assert result["predicted_label"] in (0, 1)
    assert result["predicted_class"] in ("NORMAL", "PNEUMONIA")
    assert result["threshold"] == 0.50


def test_predict_threshold_behavior(predictor, sample_rgb_image):
    """Verify that predictions accurately reflect the configured threshold."""
    # Obtain raw probability
    res_default = predictor.predict(sample_rgb_image, threshold=0.50)
    prob = res_default["pneumonia_probability"]
    
    # Setting threshold = 0.0 should guarantee label 1 (PNEUMONIA)
    res_low = predictor.predict(sample_rgb_image, threshold=0.0)
    assert res_low["predicted_label"] == 1
    assert res_low["predicted_class"] == "PNEUMONIA"
    
    # Setting threshold = 1.01 should guarantee label 0 (NORMAL)
    res_high = predictor.predict(sample_rgb_image, threshold=1.01)
    assert res_high["predicted_label"] == 0
    assert res_high["predicted_class"] == "NORMAL"


# =========================================================================
# 4. Grad-CAM Integration Tests
# =========================================================================

def test_predict_with_gradcam_integration(predictor, sample_rgb_image, tmp_path):
    """Verify that Grad-CAM visualization can be optionally generated and saved."""
    gradcam_out = tmp_path / "gradcam_test"
    result = predictor.predict(
        sample_rgb_image,
        generate_gradcam=True,
        gradcam_output_dir=gradcam_out
    )
    
    assert "gradcam_path" in result
    gradcam_file = Path(result["gradcam_path"])
    assert gradcam_file.exists(), f"Grad-CAM file was not created: {gradcam_file}"
    assert gradcam_file.stat().st_size > 1000, "Grad-CAM file size is unexpectedly small"
    
    # Verify saved image is valid and readable
    with Image.open(gradcam_file) as img:
        assert img.format == "PNG"
        w, h = img.size
        assert w > 0 and h > 0
