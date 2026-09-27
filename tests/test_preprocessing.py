"""
Unit and Integration Tests for Stage 3: Image Preprocessing & Data Pipeline.
Verifies target image sizing (224x224x3), normalization [0, 1], training-only augmentation,
deterministic validation/test loaders, label mapping, and batch tensor shapes.
"""

from pathlib import Path
import pytest
import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import config
from src.preprocessing import (
    build_augmentation_layers,
    build_normalization_layer,
    build_preprocessing_pipeline,
    load_raw_and_preprocessed_image
)
from src.data_loader import (
    scan_dataset,
    compute_class_weights_dict,
    get_data_loaders,
    create_tf_dataset,
    load_raw_image_tensor
)


def test_target_image_sizing_and_3channels():
    """Verify that images are consistently resized to 224x224 and converted to 3 channels."""
    train_normal = config.TRAIN_DIR / "NORMAL"
    sample_path = next(train_normal.glob("*.jpeg"))
    
    raw_img, prep_img, meta = load_raw_and_preprocessed_image(sample_path)
    
    assert prep_img.shape == (224, 224, 3), f"Expected shape (224, 224, 3), got {prep_img.shape}"
    assert raw_img.shape[2] == 3, f"Expected 3 channels in raw image, got {raw_img.shape[2]}"
    assert meta["target_size"] == (224, 224)


def test_normalization_pixel_range():
    """Verify that pixel values are strictly normalized into [0.0, 1.0]."""
    train_pneumonia = config.TRAIN_DIR / "PNEUMONIA"
    sample_path = next(train_pneumonia.glob("*.jpeg"))

    _, prep_img, meta = load_raw_and_preprocessed_image(sample_path)
    
    min_val = np.min(prep_img)
    max_val = np.max(prep_img)
    
    assert min_val >= 0.0, f"Pixel min value {min_val} is below 0.0"
    assert max_val <= 1.0, f"Pixel max value {max_val} exceeds 1.0"
    assert meta["pixel_min"] >= 0.0
    assert meta["pixel_max"] <= 1.0


def test_augmentation_layers_output():
    """Verify that augmentation layers preserve tensor dimensions and return valid floats."""
    aug_model = build_augmentation_layers()
    dummy_input = tf.random.uniform((4, 224, 224, 3), minval=0.0, maxval=255.0, dtype=tf.float32)
    
    augmented = aug_model(dummy_input, training=True)
    
    assert augmented.shape == (4, 224, 224, 3), f"Augmentation changed shape to {augmented.shape}"
    assert augmented.dtype == tf.float32


def test_augmentation_applied_only_in_training_mode():
    """
    Verify that when training=False, augmentation acts deterministically as an identity
    transform (so validation and test sets remain strictly deterministic and uncorrupted).
    """
    aug_model = build_augmentation_layers()
    dummy_input = tf.random.uniform((2, 224, 224, 3), minval=0.0, maxval=255.0, dtype=tf.float32)

    # In evaluation / inference mode (training=False)
    eval_out1 = aug_model(dummy_input, training=False)
    eval_out2 = aug_model(dummy_input, training=False)

    np.testing.assert_allclose(
        eval_out1.numpy(),
        eval_out2.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="Evaluation mode must be strictly deterministic across calls!"
    )
    np.testing.assert_allclose(
        eval_out1.numpy(),
        dummy_input.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="When training=False, random augmentation must not alter the input!"
    )


def test_training_loader_tensor_shapes_and_types():
    """Verify actual batch shapes and types produced by the training data loader."""
    train_ds, _, _, metadata = get_data_loaders(batch_size=8, augment_train=True, normalize=True)
    
    batch_images, batch_labels = next(iter(train_ds))
    
    assert batch_images.shape == (8, 224, 224, 3), f"Expected (8, 224, 224, 3), got {batch_images.shape}"
    assert batch_labels.shape == (8,), f"Expected (8,), got {batch_labels.shape}"
    assert batch_images.dtype == tf.float32
    assert batch_labels.dtype == tf.int64
    assert float(tf.reduce_min(batch_images)) >= 0.0
    assert float(tf.reduce_max(batch_images)) <= 1.0


def test_validation_loader_deterministic_and_unaugmented():
    """Verify validation loader tensor shapes and ensure deterministic evaluation."""
    _, val_ds, _, _ = get_data_loaders(batch_size=8, augment_train=True, normalize=True)
    
    # Val set has 16 images total
    batch_images1, batch_labels1 = next(iter(val_ds))
    batch_images2, batch_labels2 = next(iter(val_ds))

    assert batch_images1.shape == (8, 224, 224, 3)
    assert batch_labels1.shape == (8,)
    # Verify bit-for-bit identical batches (deterministic, no random augmentations)
    np.testing.assert_array_equal(batch_images1.numpy(), batch_images2.numpy())
    np.testing.assert_array_equal(batch_labels1.numpy(), batch_labels2.numpy())


def test_test_loader_shapes_and_isolation():
    """Verify test loader tensor shapes and strict isolation."""
    _, _, test_ds, metadata = get_data_loaders(batch_size=16, augment_train=True, normalize=True)
    
    batch_images, batch_labels = next(iter(test_ds))
    
    assert batch_images.shape == (16, 224, 224, 3)
    assert batch_labels.shape == (16,)
    assert metadata["augmentation_applied_to_val"] is False
    assert metadata["augmentation_applied_to_test"] is False


def test_label_mapping_integrity():
    """Verify that NORMAL maps to 0 and PNEUMONIA maps to 1."""
    splits = scan_dataset()
    train_df = splits["train"]
    
    normal_subset = train_df[train_df["class_name"] == "NORMAL"]
    pneumonia_subset = train_df[train_df["class_name"] == "PNEUMONIA"]

    assert set(normal_subset["label"].unique()) == {0}
    assert set(pneumonia_subset["label"].unique()) == {1}
    assert config.LABEL_MAPPING["NORMAL"] == 0
    assert config.LABEL_MAPPING["PNEUMONIA"] == 1


def test_class_weights_address_imbalance():
    """Verify that class weights are correctly computed and favor minority NORMAL class."""
    splits = scan_dataset()
    weights = compute_class_weights_dict(splits["train"])
    
    assert weights[0] > weights[1]
    assert np.isclose(weights[0], 1.9448, atol=0.01)
    assert np.isclose(weights[1], 0.6730, atol=0.01)


def test_visualization_artifact_present():
    """Verify that preprocessing and augmentation plot artifact exists."""
    expected_plot = config.PLOTS_DIR / "preprocessing_and_augmentation.png"
    assert expected_plot.exists(), f"Plot artifact missing: {expected_plot}"
    assert expected_plot.stat().st_size > 50000, f"Plot file is too small: {expected_plot.stat().st_size} bytes"
