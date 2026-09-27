"""
Stage 10: Streamlit Web Application for Pneumonia Detection.
Provides a clean, professional academic and research demonstration of the finalized
MobileNetV2 transfer learning model for chest radiograph classification.

Intended strictly for educational and research demonstration purposes.
Not a clinical diagnostic tool or medical device.
"""

import os
# Suppress framework logs before imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.utils import get_logger
from src.predict import (
    PneumoniaPredictor,
    get_predictor,
    SUPPORTED_EXTENSIONS
)

logger = get_logger("streamlit_app")

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Pneumonia Detection from Chest X-ray Images",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #475569;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .project-desc {
        font-size: 0.95rem;
        color: #64748b;
        font-style: italic;
        margin-bottom: 1.2rem;
    }
    .disclaimer-card {
        background-color: #fef3c7;
        border-left: 5px solid #d97706;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        color: #92400e;
    }
    .disclaimer-title {
        font-weight: 700;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .disclaimer-body {
        font-size: 0.88rem;
        line-height: 1.4;
        margin: 0;
    }
    .prediction-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 0.5rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-value-normal {
        font-size: 2rem;
        font-weight: 800;
        color: #0d9488;
    }
    .metric-value-pneumonia {
        font-size: 2rem;
        font-weight: 800;
        color: #e11d48;
    }
    .metric-prob {
        font-size: 1.7rem;
        font-weight: 700;
        color: #334155;
    }
    .explanation-box {
        background-color: #f1f5f9;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 0.9rem;
        color: #334155;
        margin-top: 1rem;
    }
    .gradcam-note {
        font-size: 0.85rem;
        color: #475569;
        font-style: italic;
        background: #f8fafc;
        padding: 0.6rem 0.8rem;
        border-radius: 4px;
        border-left: 3px solid #64748b;
        margin-top: 0.5rem;
    }
    .flow-diagram {
        font-family: monospace;
        font-size: 0.95rem;
        background-color: #0f172a;
        color: #38bdf8;
        padding: 1.2rem;
        border-radius: 8px;
        line-height: 1.6;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Resources and Data
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading MobileNetV2 checkpoint...")
def load_cached_predictor() -> PneumoniaPredictor:
    """Loads and caches the finalized MobileNetV2 predictor once across sessions."""
    logger.info("Initializing cached MobileNetV2 predictor for Streamlit session.")
    return get_predictor()


@st.cache_data
def load_official_evaluation_metrics() -> Dict[str, Any]:
    """Loads the precomputed Stage 7 evaluation metrics from disk."""
    metrics_path = config.METRICS_DIR / "final_test_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as err:
            logger.error(f"Error reading metrics JSON: {err}")
    return {}


# -----------------------------------------------------------------------------
# Temporary File Inference Helper
# -----------------------------------------------------------------------------
def run_inference_on_uploaded_file(
    uploaded_file,
    show_gradcam: bool = False
) -> Dict[str, Any]:
    """
    Safely writes an uploaded image to a temporary file, executes the existing
    Stage 9 prediction pipeline, and cleans up the temporary file immediately.
    """
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{ext}'. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # Create isolated temporary file with matching extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp_file.name)

    try:
        predictor = load_cached_predictor()
        result = predictor.predict(
            image_path=tmp_path,
            threshold=config.DECISION_THRESHOLD,
            generate_gradcam=show_gradcam
        )
        # Replace temp filename with original upload filename
        result["filename"] = uploaded_file.name
        return result
    finally:
        if tmp_path.exists():
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {tmp_path}: {e}")


# -----------------------------------------------------------------------------
# UI Layout: Header and Disclaimer
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">Pneumonia Detection from Chest X-ray Images</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Learning-Based Image Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="project-desc">An educational deep-learning demonstration using transfer learning with MobileNetV2.</div>',
    unsafe_allow_html=True
)

# Prominent Main Disclaimer
st.markdown(
    """
    <div class="disclaimer-card">
        <div class="disclaimer-title">⚠️ Educational / Research Use Only</div>
        <div class="disclaimer-body">
            This application demonstrates an image-classification model trained for an academic project.
            It is <strong>not</strong> a clinically validated diagnostic system and should not be used for medical diagnosis,
            treatment recommendations, or clinical decision-making.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Sidebar: System Specifications & Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("✅ MobileNetV2 Model Ready")
    
    st.markdown("### Specifications")
    st.markdown(f"**Architecture:** MobileNetV2 (Transfer Learning)")
    st.markdown(f"**Checkpoint:** `mobilenetv2_transfer_best.keras`")
    st.markdown(f"**Input Size:** {config.IMAGE_WIDTH} × {config.IMAGE_HEIGHT} × {config.IMAGE_CHANNELS}")
    st.markdown(f"**Normalization:** `[-1.0, 1.0]`")
    st.markdown(f"**Decision Threshold:** `{config.DECISION_THRESHOLD:.2f}` (Fixed)")
    st.markdown(f"**Grad-CAM Layer:** `Conv_1`")
    
    st.markdown("---")
    st.markdown("### Class Mapping")
    st.markdown("- `0` : **NORMAL** (Healthy lung)")
    st.markdown("- `1` : **PNEUMONIA** (Opacities)")
    
    st.markdown("---")
    st.markdown("### Reference")
    st.markdown("Academic Deep Learning Pipeline (Stages 1–10)")


# -----------------------------------------------------------------------------
# Main Section: Upload and Analysis
# -----------------------------------------------------------------------------
st.subheader("1. Upload Chest Radiograph")

st.info(
    "Input requirement: Upload a chest X-ray image. This application does not automatically "
    "verify whether an uploaded image is a chest X-ray. Predictions for unrelated image types are not meaningful."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image (.jpeg, .jpg, .png)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    help="Upload a chest X-ray image in JPG, JPEG, or PNG format."
)

# Reset state if uploaded file changed
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
    st.session_state.prediction_result = None

if uploaded_file is not None and uploaded_file.name != st.session_state.current_file_name:
    st.session_state.current_file_name = uploaded_file.name
    st.session_state.prediction_result = None

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("#### Radiograph Preview")
    if uploaded_file is not None:
        try:
            # Preview uploaded image
            pil_image = Image.open(uploaded_file)
            orig_w, orig_h = pil_image.size
            st.image(
                pil_image,
                caption=f"Uploaded Image: {uploaded_file.name} ({orig_w} × {orig_h} px)",
                use_container_width=True
            )
            
            st.markdown(f"**File size:** `{uploaded_file.size / 1024:.1f} KB` | **Mode:** `{pil_image.mode}`")
            
            # Mandatory X-ray Confirmation Checkbox (default unchecked)
            confirm_xray = st.checkbox(
                "I confirm that the uploaded image is a chest X-ray.",
                value=False,
                help="Confirmation that this image is an actual chest radiograph."
            )

            # If user unchecks confirmation after a previous analysis, clear existing prediction
            if not confirm_xray and st.session_state.prediction_result is not None:
                st.session_state.prediction_result = None
            
            # Options
            show_gradcam = st.checkbox(
                "Show Grad-CAM explanation",
                value=False,
                help="Generates a 3-panel activation map showing regions contributing to the prediction."
            )
            
            analyze_clicked = st.button("Analyze X-ray", type="primary", use_container_width=True)
            
            if analyze_clicked:
                if not confirm_xray:
                    st.warning("Please confirm that the uploaded image is a chest X-ray before analysis.")
                else:
                    with st.spinner("Analyzing radiograph through MobileNetV2 pipeline..."):
                        try:
                            res = run_inference_on_uploaded_file(uploaded_file, show_gradcam=show_gradcam)
                            st.session_state.prediction_result = res
                        except Exception as err:
                            st.error(f"Inference failed: {err}")
                            logger.error(f"Inference error on {uploaded_file.name}: {err}", exc_info=True)
                        
        except (UnidentifiedImageError, OSError) as e:
            st.error(f"Unable to read or preview image file: {e}")
            uploaded_file = None
    else:
        confirm_xray = False
        st.info("Please upload a chest X-ray image to continue.")

with col_right:
    st.markdown("#### Model Prediction")
    result = st.session_state.prediction_result
    
    if result is not None:
        pred_class = result["predicted_class"]
        prob = result["pneumonia_probability"]
        prob_pct = prob * 100.0
        threshold_pct = result["threshold"] * 100.0
        
        # Display Result Card
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="metric-label">Model Prediction</div>', unsafe_allow_html=True)
            if pred_class == "NORMAL":
                st.markdown(f'<div class="metric-value-normal">{pred_class}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-value-pneumonia">{pred_class}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-label">Pneumonia Probability</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-prob">{prob_pct:.2f}%</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-label">Decision Threshold</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-prob">{threshold_pct:.2f}%</div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Progress bar showing probability relative to 50% threshold
        st.progress(float(np.clip(prob, 0.0, 1.0)), text=f"Pneumonia Probability: {prob_pct:.2f}% (Threshold: {threshold_pct:.1f}%)")
        
        # Result Interpretation
        if pred_class == "NORMAL":
            interp_text = "At the fixed 0.50 classification threshold, the model classified this image as NORMAL."
        else:
            interp_text = "At the fixed 0.50 classification threshold, the model classified this image as PNEUMONIA."
            
        st.markdown(f'<div class="explanation-box">ℹ️ {interp_text}</div>', unsafe_allow_html=True)

        # Post-Prediction Input Assumption Notice
        st.markdown(
            """
            <div class="assumption-note" style="background-color: #f1f5f9; border-left: 3px solid #64748b; padding: 0.6rem 0.8rem; border-radius: 4px; font-size: 0.84rem; color: #475569; margin-top: 0.75rem;">
                <strong>Input assumption:</strong> This result assumes that the uploaded image is a chest X-ray. The application does not automatically verify image modality.
            </div>
            """,
            unsafe_allow_html=True
        )
        
    elif uploaded_file is not None:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 2rem; text-align: center; color: #64748b;">
                Click <strong>Analyze X-ray</strong> on the left to run inference on the uploaded radiograph.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px; padding: 2rem; text-align: center; color: #94a3b8;">
                No radiograph loaded.
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# Grad-CAM Visualization Section (if generated)
# -----------------------------------------------------------------------------
if st.session_state.prediction_result is not None:
    result = st.session_state.prediction_result
    
    # Check if Grad-CAM was requested or needs generation
    if show_gradcam and "gradcam_path" in result:
        st.markdown("---")
        st.subheader("2. Grad-CAM Model Explanation")
        gradcam_path = Path(result["gradcam_path"])
        
        if gradcam_path.exists():
            st.image(
                str(gradcam_path),
                caption="Grad-CAM Activation: Original Radiograph | Feature Heatmap | Heatmap Overlay",
                use_container_width=True
            )
            st.markdown(
                """
                <div class="gradcam-note">
                    <strong>Interpretation Note:</strong> Grad-CAM highlights regions receiving greater activation for the model's prediction.
                    These visualizations describe model behavior and do not establish clinical evidence of pathology.
                </div>
                """,
                unsafe_allow_html=True
            )
    elif show_gradcam and "gradcam_path" not in result and uploaded_file is not None and confirm_xray:
        # User checked Grad-CAM after running initial prediction; generate it only with confirmation
        with st.spinner("Generating Grad-CAM visualization for target layer Conv_1..."):
            try:
                res = run_inference_on_uploaded_file(uploaded_file, show_gradcam=True)
                st.session_state.prediction_result = res
                st.rerun()
            except Exception as e:
                st.warning(f"Grad-CAM generation encountered an issue: {e}")


# -----------------------------------------------------------------------------
# Technical Sections (Expanders)
# -----------------------------------------------------------------------------
st.markdown("---")

col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        **Architecture & Backbone:**
        - Model: MobileNetV2 (Transfer Learning)
        - Base Weights: Pretrained on ImageNet
        - Trainable Strategy: 2-Phase Training (Frozen Head -> Fine-tuning top 40 layers)
        
        **Pipeline Specifications:**
        - Input Dimensions: 224 × 224 × 3
        - Normalization: `[-1.0, 1.0]` (`pixel / 127.5 - 1.0`)
        - Fixed Threshold: `0.50`
        - Classes: `NORMAL` (0), `PNEUMONIA` (1)
        - Target Explainability Layer: `Conv_1` (7 × 7 × 1280)
        """)

with col_exp2:
    with st.expander("📊 Model Evaluation"):
        metrics = load_official_evaluation_metrics()
        perf = metrics.get("measured_test_set_performance", {})
        
        st.markdown("**Official Held-Out Test Split Results** (624 images)")
        st.caption("These results are from the project's evaluation dataset and do not represent clinical validation.")
        
        if perf:
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{perf.get('accuracy', 0.7436) * 100:.2f}%")
            m1.metric("Precision", f"{perf.get('precision', 0.7106) * 100:.2f}%")
            m1.metric("Recall", f"{perf.get('recall_sensitivity', 0.9949) * 100:.2f}%")
            m1.metric("Specificity", f"{perf.get('specificity', 0.3248) * 100:.2f}%")
            
            m2.metric("F1-Score", f"{perf.get('f1_score', 0.8291) * 100:.2f}%")
            m2.metric("ROC-AUC", f"{perf.get('roc_auc', 0.9530):.4f}")
            m2.metric("PR-AUC", f"{perf.get('pr_auc', 0.9711):.4f}")
            
            cm = perf.get("confusion_matrix", {})
            st.markdown(
                f"**Confusion Matrix:** TN={cm.get('true_negative', 76)}, "
                f"FP={cm.get('false_positive', 158)}, "
                f"FN={cm.get('false_negative', 2)}, "
                f"TP={cm.get('true_positive', 388)}"
            )
        else:
            st.markdown("""
            - Accuracy: 74.36%
            - Precision: 71.06%
            - Recall: 99.49%
            - Specificity: 32.48%
            - F1-Score: 82.91%
            - ROC-AUC: 0.9530
            - PR-AUC: 0.9711
            """)
            
        st.warning(
            "Dataset Limitation Note: The original public dataset release contains 170 PNEUMONIA "
            "patient IDs appearing in both the training and official test partitions. Therefore, the "
            "official test evaluation should not be interpreted as a fully patient-independent evaluation."
        )

with col_exp3:
    with st.expander("🔄 How the Prediction Works"):
        st.markdown(
            """
            <div class="flow-diagram">
Uploaded X-ray (.jpg, .png)
      ↓
Image validation & RGB conversion
      ↓
224 × 224 × 3 preprocessing ([-1, 1])
      ↓
MobileNetV2 Forward Pass
      ↓
Pneumonia probability P(Pneumonia)
      ↓
Fixed threshold 0.50 comparison
      ↓
NORMAL / PNEUMONIA classification
      ↓
Optional Grad-CAM (Target Layer: Conv_1)
            </div>
            """,
            unsafe_allow_html=True
        )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.82rem;'>"
    "Pneumonia Detection Academic Project — Stage 10 Demonstration Interface<br>"
    "Educational/Research Use Only. Not intended for clinical or diagnostic use."
    "</div>",
    unsafe_allow_html=True
)
