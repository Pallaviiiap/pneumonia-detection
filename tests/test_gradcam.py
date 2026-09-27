"""
Stage 8: Automated Tests for Grad-CAM Explainability Pipeline.
Validates:
1. Target layer existence and output shape in the finalized MobileNetV2 checkpoint.
2. Input preprocessing produces (1, 224, 224, 3) tensor in [-1.0, 1.0].
3. Model prediction output is a scalar probability in [0, 1].
4. Grad-CAM heatmap calculation, dimensions (7x7), finiteness, and normalization in [0, 1].
5. Overlay blending generation matches original image shape.
6. Representative Stage 8 output files and metrics JSON exist and contain valid fields.
"""

from pathlib import Path
import json
import pytest
import numpy as np
import tensorflow as tf
import keras

from src.config import config
from src.gradcam import (
    load_image_for_gradcam,
    compute_gradcam_heatmap,
    create_gradcam_overlay,
    generate_representative_gradcam_artifacts
)


@pytest.fixture(scope="module")
def loaded_model():
    """Loads the finalized MobileNetV2 checkpoint once for test efficiency."""
    ckpt_path = config.MODELS_DIR / "mobilenetv2_transfer_best.keras"
    assert ckpt_path.exists(), f"Model checkpoint missing at {ckpt_path}"
    model = keras.models.load_model(ckpt_path)
    return model


@pytest.fixture(scope="module")
def sample_test_image_path():
    """Provides a sample image path from the test set for unit testing."""
    test_normal_dir = config.TEST_DIR / "NORMAL"
    assert test_normal_dir.exists(), f"Directory not found: {test_normal_dir}"
    images = list(test_normal_dir.glob("*.jpeg")) + list(test_normal_dir.glob("*.jpg")) + list(test_normal_dir.glob("*.png"))
    assert len(images) > 0, "No test images found"
    return images[0]


def test_gradcam_checkpoint_and_target_layer_exists(loaded_model):
    """Verifies that the target convolutional layer 'Conv_1' exists and has expected shape."""
    target_layer_name = "Conv_1"
    layer_names = [l.name for l in loaded_model.layers]
    assert target_layer_name in layer_names, f"Target layer '{target_layer_name}' not found in model layers: {layer_names[:10]}..."
    
    target_layer = loaded_model.get_layer(target_layer_name)
    output_shape = target_layer.output.shape
    # Shape is (None, 7, 7, 1280)
    assert output_shape[1] == 7 and output_shape[2] == 7, f"Expected 7x7 spatial feature map, got {output_shape}"
    assert output_shape[3] == 1280, f"Expected 1280 channels for Conv_1, got {output_shape}"


def test_gradcam_image_preprocessing(sample_test_image_path):
    """Verifies image loading and preprocessing produces (1, 224, 224, 3) tensor in [-1.0, 1.0]."""
    raw_img, input_tensor, meta = load_image_for_gradcam(sample_test_image_path)
    
    assert isinstance(raw_img, np.ndarray)
    assert raw_img.ndim == 3 and raw_img.shape[2] == 3
    assert raw_img.dtype == np.uint8
    
    assert isinstance(input_tensor, tf.Tensor)
    assert input_tensor.shape == (1, 224, 224, 3)
    
    tensor_min = float(tf.reduce_min(input_tensor))
    tensor_max = float(tf.reduce_max(input_tensor))
    assert tensor_min >= -1.05 and tensor_max <= 1.05, f"Tensor range outside [-1, 1]: min={tensor_min}, max={tensor_max}"
    
    assert "original_width" in meta and "original_height" in meta
    assert meta["original_width"] == raw_img.shape[1]
    assert meta["original_height"] == raw_img.shape[0]


def test_gradcam_prediction_and_heatmap_computation(loaded_model, sample_test_image_path):
    """Verifies model forward pass and Grad-CAM computation on target layer."""
    _, input_tensor, _ = load_image_for_gradcam(sample_test_image_path)
    
    heatmap, pred_prob, pred_class = compute_gradcam_heatmap(
        loaded_model,
        input_tensor,
        target_layer_name="Conv_1"
    )
    
    # Check prediction scalar
    assert isinstance(pred_prob, float)
    assert 0.0 <= pred_prob <= 1.0
    assert pred_class in (0, 1)
    
    # Check heatmap properties
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (7, 7), f"Expected 7x7 heatmap, got {heatmap.shape}"
    assert np.all(np.isfinite(heatmap)), "Heatmap contains NaN or Inf values"
    assert np.min(heatmap) >= 0.0, f"Heatmap min < 0: {np.min(heatmap)}"
    assert np.max(heatmap) <= 1.0, f"Heatmap max > 1: {np.max(heatmap)}"


def test_gradcam_overlay_generation(sample_test_image_path):
    """Verifies overlay creation resizes heatmap and blends with original image dimensions."""
    raw_img, _, _ = load_image_for_gradcam(sample_test_image_path)
    mock_heatmap = np.random.uniform(0.0, 1.0, size=(7, 7)).astype(np.float32)
    
    resized_heatmap, overlay_img = create_gradcam_overlay(
        raw_img,
        mock_heatmap,
        alpha=0.45,
        colormap_name="jet"
    )
    
    orig_h, orig_w = raw_img.shape[:2]
    assert resized_heatmap.shape == (orig_h, orig_w)
    assert overlay_img.shape == (orig_h, orig_w, 3)
    assert overlay_img.dtype == np.uint8
    assert np.all(np.isfinite(resized_heatmap))
    assert 0.0 <= np.min(resized_heatmap) and np.max(resized_heatmap) <= 1.0


def test_representative_output_artifacts_exist():
    """Verifies that all 4 required representative visualization panels and metadata JSON exist."""
    gradcam_dir = config.OUTPUTS_DIR / "gradcam"
    metrics_json_path = config.METRICS_DIR / "gradcam_examples.json"
    
    expected_panels = [
        "gradcam_correct_normal.png",
        "gradcam_correct_pneumonia.png",
        "gradcam_false_positive.png",
        "gradcam_false_negative.png"
    ]
    
    for panel in expected_panels:
        panel_path = gradcam_dir / panel
        assert panel_path.exists(), f"Expected panel file {panel_path} does not exist"
        assert panel_path.stat().st_size > 1000, f"File {panel_path} is unexpectedly small ({panel_path.stat().st_size} bytes)"
        
    assert metrics_json_path.exists(), f"Metadata JSON {metrics_json_path} does not exist"
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "model_name" in data
    assert "target_layer_name" in data
    assert data["target_layer_name"] == "Conv_1"
    assert "examples" in data
    assert set(data["examples"].keys()) == {"correct_normal", "correct_pneumonia", "false_positive", "false_negative"}
    
    for key, ex in data["examples"].items():
        assert "filename" in ex
        assert "patient_id" in ex
        assert "true_class" in ex
        assert "predicted_class" in ex
        assert "predicted_probability" in ex
        assert "is_correct" in ex
        assert "target_layer" in ex
        assert "original_dimensions" in ex
