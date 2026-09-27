"""
Unit and Integration Tests for Stage 6: Transfer Learning with MobileNetV2.
Verifies model architecture, pretrained ImageNet weights, Phase 1 frozen base,
Phase 2 fine-tuning configuration, BatchNormalization freeze integrity,
MobileNetV2-specific preprocessing range, forward pass, training step,
and checkpoint reloadability.
"""

from pathlib import Path
import tempfile
import pytest
import numpy as np
import tensorflow as tf
import keras

from src.config import config
from src.model_transfer import build_mobilenetv2_transfer_model, setup_phase2_fine_tuning
from src.preprocessing import build_normalization_layer, preprocess_input_mobilenetv2
from src.data_loader import get_data_loaders
from src.train_transfer import run_mobilenetv2_sanity_check


def test_mobilenetv2_transfer_model_architecture():
    """Verify MobileNetV2 model architecture, shapes, and layer naming."""
    model = build_mobilenetv2_transfer_model(input_shape=(224, 224, 3))

    assert model.name == "mobilenetv2_transfer"
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 1)

    layer_names = [l.name for l in model.layers]
    assert "global_average_pooling2d" in layer_names
    assert "dropout_head" in layer_names
    assert "dense_head" in layer_names
    assert "Conv_1" in layer_names


def test_mobilenetv2_pretrained_weights_and_phase1_freezing():
    """Verify ImageNet weights are loaded and base layers are frozen in Phase 1."""
    model = build_mobilenetv2_transfer_model()

    total_params = int(model.count_params())
    trainable_params = int(sum(tf.size(w).numpy() for w in model.trainable_weights))

    assert total_params == 2_259_265, f"Unexpected total params: {total_params}"
    assert trainable_params == 1_281, f"Expected 1,281 trainable params in Phase 1, got {trainable_params}"

    # Verify that base layers are frozen
    conv1_layer = model.get_layer("Conv_1")
    assert conv1_layer.trainable is False


def test_phase2_fine_tuning_setup_and_bn_freezing():
    """Verify Phase 2 unfreezes only block_16 onwards while keeping all BatchNormalization layers frozen."""
    model = build_mobilenetv2_transfer_model()
    model = setup_phase2_fine_tuning(model, unfreeze_from_layer="block_16_expand", learning_rate=1e-5)

    trainable_params = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
    assert trainable_params == 880_321, f"Expected 880,321 trainable params in Phase 2, got {trainable_params}"

    # Verify that block_16_expand is trainable, but all BatchNormalizations are NOT
    block16_conv = model.get_layer("block_16_expand")
    assert block16_conv.trainable is True

    bn_layers = [
        l for l in model.layers
        if isinstance(l, (keras.layers.BatchNormalization, tf.keras.layers.BatchNormalization))
        or "BatchNormalization" in l.__class__.__name__
    ]
    assert len(bn_layers) > 0, "No BatchNormalization layers found"
    for bn in bn_layers:
        assert bn.trainable is False, f"BatchNormalization layer '{bn.name}' must remain frozen in Phase 2!"


def test_mobilenetv2_preprocessing_range_and_loader():
    """Verify MobileNetV2 preprocessing maps [0, 255] to [-1.0, 1.0] without double normalization."""
    # 1. Direct helper test
    test_tensor = tf.constant([0.0, 127.5, 255.0], dtype=tf.float32)
    normalized = preprocess_input_mobilenetv2(test_tensor).numpy()
    assert np.allclose(normalized, [-1.0, 0.0, 1.0], atol=1e-4)

    # 2. Normalization layer test
    norm_layer = build_normalization_layer(mode="mobilenetv2")
    layer_out = norm_layer(test_tensor).numpy()
    assert np.allclose(layer_out, [-1.0, 0.0, 1.0], atol=1e-4)

    # 3. Data loader test
    train_ds, val_ds, test_ds, metadata = get_data_loaders(
        batch_size=2,
        augment_train=False,
        normalize=True,
        normalization_mode="mobilenetv2",
        validation_strategy="patient_aware_split"
    )
    batch_x, batch_y = next(iter(train_ds))
    x_min, x_max = float(np.min(batch_x)), float(np.max(batch_x))
    assert x_min >= -1.05 and x_max <= 1.05, f"Unexpected range: [{x_min}, {x_max}]"
    assert x_min < -0.5, "Values do not exhibit negative range expected by MobileNetV2"


def test_baseline_vs_mobilenetv2_loader_preprocessing_pipeline():
    """
    Prove:
    1. Baseline CNN still receives its original [0, 1] input with default loader.
    2. Raw loader output [0, 255] -> MobileNetV2 preprocessing -> correct [-1, 1] range.
    3. MobileNetV2 data loader directly provides [-1, 1] range without double normalization.
    """
    # 1. Baseline CNN loader check (default parameters untouched)
    train_ds_baseline, _, _, _ = get_data_loaders(batch_size=4, normalize=True)
    b_batch, _ = next(iter(train_ds_baseline))
    b_min, b_max = float(np.min(b_batch)), float(np.max(b_batch))
    assert b_min >= 0.0 and b_max <= 1.0, f"Baseline CNN expected [0, 1], got [{b_min}, {b_max}]"

    # 2. Raw loader output -> MobileNetV2 preprocessing
    train_ds_raw, _, _, _ = get_data_loaders(batch_size=4, normalize=False)
    raw_batch, _ = next(iter(train_ds_raw))
    raw_min, raw_max = float(np.min(raw_batch)), float(np.max(raw_batch))
    assert raw_min >= 0.0 and raw_max > 1.0, f"Expected raw [0, 255], got [{raw_min}, {raw_max}]"

    m_processed = preprocess_input_mobilenetv2(raw_batch)
    m_min, m_max = float(np.min(m_processed)), float(np.max(m_processed))
    assert m_min >= -1.05 and m_max <= 1.05, f"Expected MobileNetV2 range [-1, 1], got [{m_min}, {m_max}]"
    assert m_min < -0.5, "Negative dynamic range missing in MobileNetV2 preprocessing"

    # 3. Direct MobileNetV2 loader
    train_ds_m, _, _, _ = get_data_loaders(batch_size=4, normalization_mode="mobilenetv2")
    m_loader_batch, _ = next(iter(train_ds_m))
    ml_min, ml_max = float(np.min(m_loader_batch)), float(np.max(m_loader_batch))
    assert ml_min >= -1.05 and ml_max <= 1.05, f"Expected [-1, 1], got [{ml_min}, {ml_max}]"



def test_forward_pass_and_train_step_mobilenetv2():
    """Verify forward pass and single-step optimization on synthetic batch."""
    model = build_mobilenetv2_transfer_model()
    dummy_x = tf.random.uniform((4, 224, 224, 3), minval=-1.0, maxval=1.0)
    dummy_y = tf.constant([0, 1, 0, 1], dtype=tf.float32)

    preds = model(dummy_x, training=False)
    preds_np = preds.numpy()

    assert preds.shape == (4, 1)
    assert np.all(preds_np >= 0.0) and np.all(preds_np <= 1.0)

    loss_metrics = model.train_on_batch(dummy_x, dummy_y, return_dict=True)
    assert not np.isnan(loss_metrics["loss"])


def test_mobilenetv2_sanity_check_function():
    """Verify run_mobilenetv2_sanity_check executes and validates all constraints."""
    sanity = run_mobilenetv2_sanity_check(batch_size=2)
    assert sanity["sanity_check_passed"] is True
    assert sanity["phase1_trainable_parameters"] == 1281
    assert sanity["output_shape"] == [None, 1]


def test_checkpoint_save_and_reload():
    """Verify that a MobileNetV2 transfer checkpoint can be saved and accurately reloaded."""
    model = build_mobilenetv2_transfer_model()
    dummy_x = tf.random.uniform((2, 224, 224, 3), minval=-1.0, maxval=1.0)
    original_preds = model(dummy_x, training=False).numpy()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_mobilenet_checkpoint.keras"
        model.save(ckpt_path)
        assert ckpt_path.exists()

        reloaded_model = keras.models.load_model(ckpt_path)
        reloaded_preds = reloaded_model(dummy_x, training=False).numpy()

        assert np.allclose(original_preds, reloaded_preds, atol=1e-5), "Reloaded model predictions mismatch"
