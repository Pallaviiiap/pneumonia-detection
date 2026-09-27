"""
Unit and Integration Tests for Stage 2: Dataset Setup, Loader & Analysis.
Verifies directory structure, image counts, data loader streaming, batch shapes,
and output artifact generation.
"""

from pathlib import Path
import pytest
import tensorflow as tf

from src.config import config
from src.data_loader import (
    scan_dataset,
    get_dataset_summary,
    compute_class_weights_dict,
    create_tf_dataset,
    get_data_loaders
)


def test_dataset_paths_exist():
    """Verify that resolved dataset directories exist and have proper structure."""
    assert config.DATASET_DIR.exists(), f"DATASET_DIR not found: {config.DATASET_DIR}"
    assert config.TRAIN_DIR.exists(), f"TRAIN_DIR not found: {config.TRAIN_DIR}"
    assert config.VAL_DIR.exists(), f"VAL_DIR not found: {config.VAL_DIR}"
    assert config.TEST_DIR.exists(), f"TEST_DIR not found: {config.TEST_DIR}"

    for split_dir in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
        for cls in config.CLASS_NAMES:
            cls_dir = split_dir / cls
            assert cls_dir.exists(), f"Class dir missing: {cls_dir}"
            assert cls_dir.is_dir(), f"Class path is not directory: {cls_dir}"


def test_dataset_exact_counts():
    """Verify exact counts of images per split and class from the physical dataset."""
    splits = scan_dataset()
    
    # Train
    train_normal = len(splits["train"][splits["train"]["class_name"] == "NORMAL"])
    train_pneumonia = len(splits["train"][splits["train"]["class_name"] == "PNEUMONIA"])
    assert train_normal == 1341
    assert train_pneumonia == 3875
    assert len(splits["train"]) == 5216

    # Val
    val_normal = len(splits["val"][splits["val"]["class_name"] == "NORMAL"])
    val_pneumonia = len(splits["val"][splits["val"]["class_name"] == "PNEUMONIA"])
    assert val_normal == 8
    assert val_pneumonia == 8
    assert len(splits["val"]) == 16

    # Test
    test_normal = len(splits["test"][splits["test"]["class_name"] == "NORMAL"])
    test_pneumonia = len(splits["test"][splits["test"]["class_name"] == "PNEUMONIA"])
    assert test_normal == 234
    assert test_pneumonia == 390
    assert len(splits["test"]) == 624

    total_images = sum(len(df) for df in splits.values())
    assert total_images == 5856


def test_dataset_summary_calculation():
    """Verify summary calculations and percentages."""
    splits = scan_dataset()
    summary = get_dataset_summary(splits)

    assert summary["total_images"] == 5856
    assert summary["overall_distribution"]["counts"]["NORMAL"] == 1583
    assert summary["overall_distribution"]["counts"]["PNEUMONIA"] == 4273
    assert summary["overall_distribution"]["percentages"]["NORMAL"] == 27.03
    assert summary["overall_distribution"]["percentages"]["PNEUMONIA"] == 72.97


def test_balanced_class_weights():
    """Verify balanced class weight computation handles class imbalance properly."""
    splits = scan_dataset()
    class_weights = compute_class_weights_dict(splits["train"])

    assert 0 in class_weights
    assert 1 in class_weights
    # NORMAL is minority class (~25.7%), so weight should be > 1.0 (~1.94)
    # PNEUMONIA is majority class (~74.3%), so weight should be < 1.0 (~0.67)
    assert class_weights[0] > class_weights[1]
    assert 1.8 < class_weights[0] < 2.1
    assert 0.6 < class_weights[1] < 0.8


def test_streaming_tf_dataset_generator():
    """
    Verify streaming tf.data.Dataset loads batch without holding full dataset in RAM,
    properly resizes to (224, 224), and expands channels to 3 (RGB).
    """
    splits = scan_dataset()
    sample_df = splits["val"]  # Small split with 16 samples
    batch_size = 4
    target_size = (224, 224)

    ds = create_tf_dataset(
        sample_df,
        batch_size=batch_size,
        image_size=target_size,
        shuffle=False,
        normalize=True
    )

    batch_x, batch_y = next(iter(ds))
    assert batch_x.shape == (batch_size, 224, 224, 3)
    assert batch_y.shape == (batch_size,)
    assert batch_x.dtype == tf.float32
    assert tf.reduce_max(batch_x) <= 1.0
    assert tf.reduce_min(batch_x) >= 0.0


def test_test_set_isolation():
    """Verify that test dataset is strictly separate from train and val."""
    splits = scan_dataset()
    train_paths = set(splits["train"]["filepath"].tolist())
    val_paths = set(splits["val"]["filepath"].tolist())
    test_paths = set(splits["test"]["filepath"].tolist())

    assert len(train_paths & test_paths) == 0, "Data leakage detected between train and test sets!"
    assert len(train_paths & val_paths) == 0, "Data leakage detected between train and val sets!"
    assert len(val_paths & test_paths) == 0, "Data leakage detected between val and test sets!"


def test_metrics_and_plots_artifacts():
    """Verify that Stage 2 metrics and plot files have been generated on disk."""
    expected_metrics = [
        config.METRICS_DIR / "dataset_analysis.json",
        config.METRICS_DIR / "dataset_summary.csv"
    ]
    for m in expected_metrics:
        assert m.exists(), f"Metric file missing: {m}"
        assert m.stat().st_size > 0, f"Metric file is empty: {m}"

    expected_plots = [
        config.PLOTS_DIR / "class_distribution.png",
        config.PLOTS_DIR / "sample_normal_xrays.png",
        config.PLOTS_DIR / "sample_pneumonia_xrays.png",
        config.PLOTS_DIR / "image_dimensions_distribution.png"
    ]
    for p in expected_plots:
        assert p.exists(), f"Plot file missing: {p}"
        assert p.stat().st_size > 1000, f"Plot file is suspiciously small or empty: {p}"
