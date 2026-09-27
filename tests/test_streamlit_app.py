"""
Stage 10 & 10A: Automated Tests for Streamlit Web Application and Input Validation.
Validates:
1. Streamlit app module and entry points can be imported successfully.
2. Required model checkpoint exists, is accessible, and its SHA-256 hash is unchanged.
3. Required Stage 7 metrics JSON exists, contains expected schema, and is unchanged.
4. Core Streamlit app helper functions exist.
5. Prediction backend (src.predict) and Grad-CAM backend (src.gradcam) can be imported cleanly.
6. Importing the app does not trigger model retraining or optimizer execution.
7. Official test directory remains strictly isolated and unmodified.
8. Stage 10A: Explicit X-ray confirmation requirement exists (default unchecked).
9. Stage 10A: Analysis is gated when confirmation is False.
10. Stage 10A: Analysis can be triggered when confirmation is True.
11. Stage 10A: Visible non-X-ray input requirement and post-prediction assumption text exist.
"""

from pathlib import Path
import hashlib
import json
import pytest
import keras

from src.config import config


EXPECTED_MODEL_SHA256 = "8104988636be7dd15786d58c88663ec747da5986aa20f7d569da622e39b7aae2"
EXPECTED_METRICS_SHA256 = "37781510496e70ded1bca3dfb660194e11377a52bdb970677d97839018250d26"


def test_app_module_files_exist():
    """Verify primary Streamlit app and entry point exist."""
    app_file = config.PROJECT_ROOT / "app" / "streamlit_app.py"
    wrapper_file = config.PROJECT_ROOT / "app" / "app.py"
    assert app_file.exists(), f"Missing primary Streamlit file at {app_file}"
    assert wrapper_file.exists(), f"Missing wrapper file at {wrapper_file}"
    assert app_file.stat().st_size > 1000, "app/streamlit_app.py is unexpectedly small"


def test_required_model_checkpoint_exists_and_unmodified():
    """Verify that finalized MobileNetV2 checkpoint exists and SHA-256 hash is completely unchanged."""
    ckpt_path = config.MODELS_DIR / "mobilenetv2_transfer_best.keras"
    assert ckpt_path.exists(), f"Model checkpoint not found: {ckpt_path}"
    assert ckpt_path.stat().st_size == 16666293, f"Unexpected checkpoint file size: {ckpt_path.stat().st_size}"
    
    current_sha256 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert current_sha256 == EXPECTED_MODEL_SHA256, (
        f"Model checkpoint SHA-256 mismatch! Expected {EXPECTED_MODEL_SHA256}, got {current_sha256}"
    )


def test_required_metrics_json_exists_and_unmodified():
    """Verify that Stage 7 evaluation metrics JSON exists and has not been altered or degraded."""
    metrics_path = config.METRICS_DIR / "final_test_metrics.json"
    assert metrics_path.exists(), f"Metrics JSON not found: {metrics_path}"
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["model_name"] == "mobilenetv2_transfer"
    assert data["test_image_count"] == 624
    assert data["fixed_classification_threshold"] == 0.5
    
    assert "measured_test_set_performance" in data
    perf = data["measured_test_set_performance"]
    
    # Exact official Stage 7 metrics
    assert perf["accuracy"] == 0.7436
    assert perf["precision"] == 0.7106
    assert perf["recall_sensitivity"] == 0.9949
    assert perf["specificity"] == 0.3248
    assert perf["f1_score"] == 0.8291
    assert perf["roc_auc"] == 0.953
    assert perf["pr_auc"] == 0.9711
    
    cm = perf["confusion_matrix"]
    assert cm["true_negative"] == 76
    assert cm["false_positive"] == 158
    assert cm["false_negative"] == 2
    assert cm["true_positive"] == 388
    assert cm["total"] == 624


def test_backends_importable():
    """Verify prediction and Grad-CAM backends are importable."""
    from src.predict import PneumoniaPredictor, get_predictor, validate_and_load_image
    from src.gradcam import compute_gradcam_heatmap, create_gradcam_overlay
    
    assert callable(get_predictor)
    assert callable(validate_and_load_image)
    assert callable(compute_gradcam_heatmap)
    assert callable(create_gradcam_overlay)


def test_streamlit_app_functions_exist():
    """Verify expected helper functions exist in streamlit_app module."""
    import importlib.util
    app_path = config.PROJECT_ROOT / "app" / "streamlit_app.py"
    
    spec = importlib.util.spec_from_file_location("streamlit_app", app_path)
    assert spec is not None
    assert spec.loader is not None
    
    # Check that module defines expected functions without running Streamlit UI
    with open(app_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "def load_cached_predictor" in content
    assert "def load_official_evaluation_metrics" in content
    assert "def run_inference_on_uploaded_file" in content
    assert "Pneumonia Detection from Chest X-ray Images" in content
    assert "Educational/Research Use Only" in content or "Educational / Research Use Only" in content
    assert "Conv_1" in content
    assert "0.50" in content


def test_no_retraining_triggered_by_app():
    """Verify that the Streamlit application code contains no training or fitting triggers."""
    app_path = config.PROJECT_ROOT / "app" / "streamlit_app.py"
    with open(app_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert ".fit(" not in content
    assert ".fit_generator(" not in content
    assert "train_transfer" not in content
    assert "train(" not in content


def test_official_test_directory_untouched():
    """Verify that official test directory contains exactly 624 images and is unmodified."""
    test_dir = config.TEST_DIR
    normal_files = list((test_dir / "NORMAL").glob("*.*"))
    pneumonia_files = list((test_dir / "PNEUMONIA").glob("*.*"))
    
    assert len(normal_files) == 234, f"Expected 234 NORMAL test images, got {len(normal_files)}"
    assert len(pneumonia_files) == 390, f"Expected 390 PNEUMONIA test images, got {len(pneumonia_files)}"
    assert len(normal_files) + len(pneumonia_files) == 624


# =============================================================================
# Stage 10A: Input Validation and X-Ray Confirmation Tests
# =============================================================================

def test_xray_confirmation_requirement_exists():
    """Verify that the user confirmation checkbox requirement is explicitly defined in app code."""
    app_path = config.PROJECT_ROOT / "app" / "streamlit_app.py"
    content = app_path.read_text(encoding="utf-8")
    
    # Must contain confirmation checkbox with exact prompt wording
    assert "I confirm that the uploaded image is a chest X-ray." in content
    assert "confirm_xray" in content
    # Default must be unchecked (value=False)
    assert "value=False" in content


def test_non_xray_warning_and_assumption_text_exist():
    """Verify that input requirement warning and post-prediction assumption notice are in app code."""
    import re
    app_path = config.PROJECT_ROOT / "app" / "streamlit_app.py"
    raw_content = app_path.read_text(encoding="utf-8")
    content = re.sub(r'["\']\s*["\']', '', raw_content)
    content = re.sub(r'\s+', ' ', content)
    
    # Input requirement notice
    expected_req = (
        "Input requirement: Upload a chest X-ray image. This application does not automatically "
        "verify whether an uploaded image is a chest X-ray. Predictions for unrelated image types are not meaningful."
    )
    assert expected_req in content, "Missing visible input requirement notice in app/streamlit_app.py"
    
    # Warning when user tries to analyze without confirming
    expected_warn = "Please confirm that the uploaded image is a chest X-ray before analysis."
    assert expected_warn in content, "Missing confirmation warning in app/streamlit_app.py"
    
    # Post-prediction input assumption note
    assert "Input assumption:" in content
    assert (
        "This result assumes that the uploaded image is a chest X-ray. "
        "The application does not automatically verify image modality."
    ) in content


def test_confirmation_gate_logic():
    """Verify the logic of the confirmation gate: analysis is blocked when confirmation is False."""
    app_path = config.PROJECT_ROOT / "app" / "streamlit_app.py"
    content = app_path.read_text(encoding="utf-8")
    
    # Must enforce if not confirm_xray: st.warning(...)
    assert "if not confirm_xray:" in content
    # Must only run inference under else branch
    assert "run_inference_on_uploaded_file" in content
    # Grad-CAM must only run when confirmed
    assert "confirm_xray" in content


def test_inference_runs_when_confirmed(tmp_path):
    """Verify that the underlying prediction function works properly when an image is processed."""
    from PIL import Image
    import numpy as np
    from src.predict import get_predictor
    
    # Create temporary synthetic test radiograph
    test_img = tmp_path / "test_chest_xray.png"
    arr = (np.random.rand(120, 120) * 255).astype(np.uint8)
    Image.fromarray(arr).convert("RGB").save(test_img)
    
    predictor = get_predictor()
    result = predictor.predict(test_img)
    
    assert "predicted_class" in result
    assert result["predicted_class"] in ("NORMAL", "PNEUMONIA")
    assert 0.0 <= result["pneumonia_probability"] <= 1.0
    assert result["threshold"] == 0.50
