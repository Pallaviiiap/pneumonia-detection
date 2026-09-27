"""
Unit and Integration Tests for Stage 4: Baseline CNN Model and Training Infrastructure.
Verifies model architecture, input/output tensor shapes, binary sigmoid outputs,
forward pass, training step, data loader integration, and patient-aware validation strategy.
"""

from pathlib import Path
import pytest
import numpy as np
import tensorflow as tf
import keras

from src.config import config
from src.model_cnn import build_baseline_cnn
from src.train import get_callbacks, run_training_sanity_check
from src.evaluate import compute_classification_metrics
from src.data_loader import (
    scan_dataset,
    create_patient_aware_validation_split,
    get_data_loaders
)


def test_baseline_cnn_creation_and_shapes():
    """Verify that baseline CNN instantiates with correct input, output, and layer shapes."""
    model = build_baseline_cnn(input_shape=(224, 224, 3))
    
    assert model.name == "baseline_cnn"
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 1)

    # Check total parameters is compact (~250k)
    total_params = model.count_params()
    assert 200_000 < total_params < 300_000, f"Unexpected parameter count: {total_params}"


def test_baseline_cnn_target_gradcam_layer():
    """Verify that the target Grad-CAM convolutional layer exists and is named correctly."""
    model = build_baseline_cnn()
    expected_layer_name = config.GRADCAM_TARGET_LAYERS["baseline_cnn"]
    
    layer_names = [l.name for l in model.layers]
    assert expected_layer_name in layer_names, f"Target layer '{expected_layer_name}' not found in model layers"
    
    target_layer = model.get_layer(expected_layer_name)
    assert isinstance(target_layer, keras.layers.Conv2D), f"Expected Conv2D, got {type(target_layer)}"


def test_forward_pass_and_binary_sigmoid_output():
    """Verify forward pass on synthetic batch produces valid probabilities [0.0, 1.0]."""
    model = build_baseline_cnn()
    dummy_input = tf.random.uniform((4, 224, 224, 3), minval=0.0, maxval=1.0)
    
    preds = model(dummy_input, training=False)
    preds_np = preds.numpy()

    assert preds.shape == (4, 1)
    assert preds.dtype == tf.float32
    assert np.all(preds_np >= 0.0), f"Probabilities below 0.0 detected: {np.min(preds_np)}"
    assert np.all(preds_np <= 1.0), f"Probabilities above 1.0 detected: {np.max(preds_np)}"


def test_data_loader_compatibility_with_baseline_cnn():
    """Verify that actual batches from data_loader seamlessly feed into the baseline CNN."""
    train_ds, val_ds, test_ds, metadata = get_data_loaders(
        batch_size=4,
        augment_train=True,
        normalize=True,
        validation_strategy="patient_aware_split"
    )
    model = build_baseline_cnn()

    # Feed one batch from train loader
    batch_x, batch_y = next(iter(train_ds))
    preds = model(batch_x, training=False)
    
    assert preds.shape == (4, 1)
    assert batch_y.shape == (4,)


def test_training_callbacks_construction():
    """Verify that EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau are properly configured."""
    callbacks = get_callbacks(model_name="baseline_cnn")
    callback_types = [type(cb) for cb in callbacks]

    assert keras.callbacks.ModelCheckpoint in callback_types
    assert keras.callbacks.EarlyStopping in callback_types
    assert keras.callbacks.ReduceLROnPlateau in callback_types


def test_patient_aware_validation_split_zero_leakage():
    """Verify that derived validation split has zero patient overlap with training split."""
    splits = scan_dataset()
    train_df = splits["train"]
    
    dev_train, dev_val = create_patient_aware_validation_split(train_df, val_ratio=0.10, seed=42)
    
    pts_train = set(dev_train["patient_id"])
    pts_val = set(dev_val["patient_id"])
    
    assert len(pts_train & pts_val) == 0, f"Patient overlap found: {pts_train & pts_val}"
    assert len(dev_train) + len(dev_val) == len(train_df)
    assert len(dev_val) > 400, f"Derived validation set too small: {len(dev_val)}"


def test_training_sanity_check_execution():
    """Verify that lightweight training sanity check passes and generates valid metric records."""
    results = run_training_sanity_check(batch_size=4)
    
    assert results["sanity_check_passed"] is True
    assert results["total_parameters"] > 0
    assert not np.isnan(results["initial_forward_loss"])
    assert not np.isnan(results["single_step_train_loss"])
    assert results["output_shape"] == [None, 1]


def test_evaluation_metric_computations():
    """Verify classification metrics calculation (precision, recall, specificity, ROC-AUC)."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred_probs = np.array([0.1, 0.4, 0.8, 0.9, 0.2])
    
    metrics = compute_classification_metrics(y_true, y_pred_probs, threshold=0.5)
    
    assert "accuracy" in metrics
    assert "recall_sensitivity" in metrics
    assert "specificity" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["recall_sensitivity"] <= 1.0
    assert 0.0 <= metrics["specificity"] <= 1.0
