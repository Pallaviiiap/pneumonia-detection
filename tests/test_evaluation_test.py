"""
Unit and Integration Tests for Stage 7: Final Official Held-Out Test Evaluation.
Verifies test set evaluation pipeline, predictions CSV schema and integrity,
metric computations, confusion matrix row counts, and output artifacts.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.config import config
from src.evaluate_test import run_official_test_evaluation


def test_official_test_evaluation_pipeline_and_artifacts():
    """Verify that Stage 7 test evaluation runs on all 624 images and generates all expected artifacts."""
    summary = run_official_test_evaluation()

    assert summary["test_image_count"] == 624
    assert summary["fixed_classification_threshold"] == 0.50
    assert summary["test_isolation_verification"]["test_set_used_in_stage5_training"] is False
    assert summary["test_isolation_verification"]["test_set_used_in_stage6_training"] is False

    m = summary["measured_test_set_performance"]
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall_sensitivity"] <= 1.0
    assert 0.0 <= m["specificity"] <= 1.0
    assert 0.0 <= m["f1_score"] <= 1.0
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["pr_auc"] <= 1.0

    cm = m["confusion_matrix"]
    assert cm["total"] == 624
    assert cm["true_negative"] + cm["false_positive"] == 234  # NORMAL count
    assert cm["false_negative"] + cm["true_positive"] == 390  # PNEUMONIA count

    # Verify CSV predictions
    csv_path = config.METRICS_DIR / "final_test_predictions.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    df = pd.read_csv(csv_path)
    assert len(df) == 624
    assert len(df["image_path"].unique()) == 624  # Zero duplicates
    assert set(["image_path", "true_label", "predicted_label", "predicted_probability"]).issubset(df.columns)
    assert df["predicted_probability"].between(0.0, 1.0).all()
    assert not df["predicted_probability"].isna().any()

    # Verify plots
    assert (config.PLOTS_DIR / "final_test_confusion_matrix.png").exists()
    assert (config.PLOTS_DIR / "final_test_roc_curve.png").exists()
    assert (config.PLOTS_DIR / "final_test_precision_recall_curve.png").exists()

    # Verify patient overlap documentation
    assert "170 PNEUMONIA patient IDs" in summary["patient_overlap_warning"]["warning"]
