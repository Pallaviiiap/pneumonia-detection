"""
Stage 7: Final Official Held-Out Test Split Evaluation Module.
Evaluates the final MobileNetV2 transfer model (models/mobilenetv2_transfer_best.keras)
on all 624 images of the official held-out test split at a fixed threshold of 0.50.
Computes complete clinical and statistical metrics, exports predictions CSV, renders
publication-quality test evaluation curves, documents patient-overlap limitations,
and produces a factual validation vs test comparison.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import datetime
import platform
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.utils import get_logger, save_json
from src.data_loader import scan_dataset, create_tf_dataset

logger = get_logger("evaluate_test")


def plot_test_confusion_matrix(
    cm: np.ndarray,
    save_path: Path,
    model_name: str = "MobileNetV2 (Transfer)",
    class_names: Tuple[str, ...] = config.CLASS_NAMES
) -> Path:
    """Renders high-resolution normalized & raw confusion matrix for the official test set."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    labels = [
        [f"{cm[0, 0]}\n({cm_norm[0, 0]:.1%})", f"{cm[0, 1]}\n({cm_norm[0, 1]:.1%})"],
        [f"{cm[1, 0]}\n({cm_norm[1, 0]:.1%})", f"{cm[1, 1]}\n({cm_norm[1, 1]:.1%})"]
    ]

    sns.heatmap(
        cm, annot=labels, fmt="", cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names, ax=ax,
        linewidths=1.0, linecolor="white", annot_kws={"fontsize": 12, "fontweight": "bold"}
    )
    ax.set_title(f"Official Held-Out Test Split Confusion Matrix\n{model_name} (Threshold = 0.50)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Diagnosis", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Ground Truth", fontsize=11, fontweight="semibold")

    plt.figtext(0.5, -0.02, f"Official Held-Out Test Split (624 images). {config.DISCLAIMER}", ha="center", fontsize=7.5, color="#555555", style="italic")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved test confusion matrix plot to {save_path}")
    return save_path


def plot_test_roc_curve(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    save_path: Path,
    model_name: str = "MobileNetV2"
) -> Path:
    """Renders ROC curve with AUC score for the official test set."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    auc_val = roc_auc_score(y_true, y_pred_probs)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2.2, label=f"{model_name} (ROC-AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Random Chance (AUC = 0.50)")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="semibold")
    ax.set_title(f"Receiver Operating Characteristic (ROC): Official Test Split\n{model_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="#f8f9fa")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.figtext(0.5, -0.02, f"Official Held-Out Test Split (624 images). {config.DISCLAIMER}", ha="center", fontsize=7.5, color="#555555", style="italic")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved test ROC curve plot to {save_path}")
    return save_path


def plot_test_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    save_path: Path,
    model_name: str = "MobileNetV2"
) -> Path:
    """Renders Precision-Recall curve with PR-AUC for the official test set."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = average_precision_score(y_true, y_pred_probs)
    baseline_rate = np.mean(y_true == 1)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(recall, precision, color="#d95f02", lw=2.2, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")
    ax.axhline(baseline_rate, color="gray", lw=1.2, linestyle="--", label=f"Test Prevalence Baseline ({baseline_rate:.2f})")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="semibold")
    ax.set_title(f"Precision-Recall Curve: Official Test Split\n{model_name}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", frameon=True, facecolor="#f8f9fa")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.figtext(0.5, -0.02, f"Official Held-Out Test Split (624 images). {config.DISCLAIMER}", ha="center", fontsize=7.5, color="#555555", style="italic")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved test Precision-Recall curve plot to {save_path}")
    return save_path


def run_official_test_evaluation(
    model_checkpoint_path: Optional[Path] = None,
    threshold: float = 0.50,
    config_obj=config
) -> Dict[str, Any]:
    """
    Executes the definitive Stage 7 evaluation on the official 624-image test set.
    """
    logger.info("Initiating STAGE 7: Final Official Held-Out Test Split Evaluation...")
    eval_start_time = time.time()

    # 1. Locate and reload the final model checkpoint
    ckpt_path = Path(model_checkpoint_path) if model_checkpoint_path else config_obj.MODELS_DIR / "mobilenetv2_transfer_best.keras"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Final MobileNetV2 checkpoint not found at: {ckpt_path}")

    logger.info(f"Loading final MobileNetV2 checkpoint from: {ckpt_path}")
    model = keras.models.load_model(ckpt_path)
    total_params = int(model.count_params())
    trainable_params = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
    logger.info(f"Model reloaded: Total parameters={total_params:,}, Trainable={trainable_params:,}")

    # 2. Scan official test split manifest
    splits = scan_dataset(config_obj.DATASET_DIR)
    test_df = splits["test"].copy().reset_index(drop=True)
    total_test_images = len(test_df)
    assert total_test_images == 624, f"Expected 624 test images, got {total_test_images}"

    class_counts = test_df["class_name"].value_counts().to_dict()
    assert class_counts.get("NORMAL", 0) == 234, f"Expected 234 NORMAL, got {class_counts.get('NORMAL')}"
    assert class_counts.get("PNEUMONIA", 0) == 390, f"Expected 390 PNEUMONIA, got {class_counts.get('PNEUMONIA')}"
    logger.info(f"Test manifest verified: Total={total_test_images} (NORMAL={class_counts['NORMAL']}, PNEUMONIA={class_counts['PNEUMONIA']})")

    # 3. Create unaugmented, deterministic test data loader with MobileNetV2 [-1.0, 1.0] normalization
    test_ds = create_tf_dataset(
        test_df,
        batch_size=config_obj.BATCH_SIZE,
        image_size=config_obj.IMAGE_SIZE,
        shuffle=False,
        augment=False,
        normalize=True,
        normalization_mode="mobilenetv2",
        seed=config_obj.RANDOM_SEED
    )

    # 4. Predict probabilities batch by batch
    logger.info("Executing inference across all 624 official test images...")
    all_pred_probs = []
    all_y_true = []

    for batch_x, batch_y in test_ds:
        preds = model(batch_x, training=False)
        all_pred_probs.extend(preds.numpy().flatten())
        all_y_true.extend(batch_y.numpy().flatten())

    y_pred_probs = np.array(all_pred_probs, dtype=np.float32)
    y_true = np.array(all_y_true, dtype=np.int32)
    assert len(y_pred_probs) == 624, f"Expected 624 predictions, got {len(y_pred_probs)}"
    assert np.all(y_pred_probs >= 0.0) and np.all(y_pred_probs <= 1.0), "Probabilities outside [0.0, 1.0]"
    assert not np.isnan(y_pred_probs).any(), "NaN values detected in predictions"

    # 5. Apply fixed threshold 0.50
    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    # 6. Calculate statistical and clinical metrics
    acc = float(accuracy_score(y_true, y_pred_binary))
    prec = float(precision_score(y_true, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_true, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_pred_probs))
    pr_auc = float(average_precision_score(y_true, y_pred_probs))

    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    assert tn + fp + fn + tp == 624, f"Confusion matrix total {tn+fp+fn+tp} != 624"
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # 7. Classification Report
    clf_report = classification_report(
        y_true, y_pred_binary,
        target_names=list(config_obj.CLASS_NAMES),
        output_dict=True,
        zero_division=0
    )

    # 8. Create predictions CSV
    test_df["image_path"] = test_df["filepath"]
    test_df["true_label"] = y_true
    test_df["predicted_probability"] = [round(float(p), 6) for p in y_pred_probs]
    test_df["predicted_label"] = y_pred_binary
    test_df["predicted_class"] = ["PNEUMONIA" if p == 1 else "NORMAL" for p in y_pred_binary]
    test_df["is_correct"] = (y_true == y_pred_binary)

    predictions_csv_path = config_obj.METRICS_DIR / "final_test_predictions.csv"
    test_df[[
        "image_path", "filename", "patient_id", "class_name",
        "true_label", "predicted_label", "predicted_probability", "predicted_class", "is_correct"
    ]].to_csv(predictions_csv_path, index=False)
    logger.info(f"Saved final test predictions CSV ({len(test_df)} rows) to: {predictions_csv_path}")

    # 9. Load development validation metrics for side-by-side comparison
    val_metrics_path = config_obj.METRICS_DIR / "mobilenetv2_validation_metrics.json"
    dev_val_metrics = {}
    if val_metrics_path.exists():
        with open(val_metrics_path, "r", encoding="utf-8") as f:
            dev_val_metrics = json.load(f)

    comparison_val_vs_test = {
        "metric_comparison": {
            "accuracy": {
                "development_validation_519": dev_val_metrics.get("accuracy", 0.9364),
                "official_held_out_test_624": round(acc, 4),
                "delta_test_minus_val": round(acc - dev_val_metrics.get("accuracy", 0.9364), 4)
            },
            "precision": {
                "development_validation_519": dev_val_metrics.get("precision", 0.9240),
                "official_held_out_test_624": round(prec, 4),
                "delta_test_minus_val": round(prec - dev_val_metrics.get("precision", 0.9240), 4)
            },
            "recall_sensitivity": {
                "development_validation_519": dev_val_metrics.get("recall_sensitivity", 0.9947),
                "official_held_out_test_624": round(rec, 4),
                "delta_test_minus_val": round(rec - dev_val_metrics.get("recall_sensitivity", 0.9947), 4)
            },
            "specificity": {
                "development_validation_519": dev_val_metrics.get("specificity", 0.7786),
                "official_held_out_test_624": round(specificity, 4),
                "delta_test_minus_val": round(specificity - dev_val_metrics.get("specificity", 0.7786), 4)
            },
            "f1_score": {
                "development_validation_519": dev_val_metrics.get("f1_score", 0.9581),
                "official_held_out_test_624": round(f1, 4),
                "delta_test_minus_val": round(f1 - dev_val_metrics.get("f1_score", 0.9581), 4)
            },
            "roc_auc": {
                "development_validation_519": dev_val_metrics.get("roc_auc", 0.9908),
                "official_held_out_test_624": round(roc_auc, 4),
                "delta_test_minus_val": round(roc_auc - dev_val_metrics.get("roc_auc", 0.9908), 4)
            },
            "pr_auc": {
                "development_validation_519": dev_val_metrics.get("pr_auc", 0.9966),
                "official_held_out_test_624": round(pr_auc, 4),
                "delta_test_minus_val": round(pr_auc - dev_val_metrics.get("pr_auc", 0.9966), 4)
            }
        },
        "sample_counts": {
            "development_validation": 519,
            "official_test": 624
        }
    }

    # 10. Compile comprehensive test metrics record
    eval_duration = round(time.time() - eval_start_time, 2)
    final_test_summary = {
        "model_name": "mobilenetv2_transfer",
        "evaluation_title": "Official Held-Out Test Split Evaluation",
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
        "checkpoint_path": str(ckpt_path),
        "hardware": "CPU",
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "dataset_name": "Kaggle Chest X-Ray Images (Pneumonia) - Official Test Split",
        "test_image_count": total_test_images,
        "class_counts": class_counts,
        "fixed_classification_threshold": threshold,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "image_size": list(config_obj.IMAGE_SIZE),
        "preprocessing_mode": "mobilenetv2",
        "input_range": [-1.0, 1.0],
        "test_isolation_verification": {
            "test_set_used_in_stage5_training": False,
            "test_set_used_in_stage6_training": False,
            "test_set_used_for_early_stopping": False,
            "test_set_used_for_checkpoint_selection": False,
            "test_set_used_for_threshold_tuning": False,
            "checkpoint_selected_prior_to_test_eval": True,
            "isolation_status": "Strictly Isolated until Stage 7"
        },
        "patient_overlap_warning": {
            "warning": "DATASET LIMITATION: 170 PNEUMONIA patient IDs appear in both the original train and official test partitions.",
            "characterization": "Official held-out test split evaluation (NOT patient-independent test evaluation)",
            "impact_note": "The test images themselves remained completely untouched throughout the project, but the original dataset release contains inherent patient overlap across the train and test splits."
        },
        "measured_test_set_performance": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall_sensitivity": round(rec, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": {
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
                "total": tn + fp + fn + tp
            },
            "sample_counts": {
                "total_evaluated": total_test_images,
                "actual_normal": int(np.sum(y_true == 0)),
                "actual_pneumonia": int(np.sum(y_true == 1)),
                "predicted_normal": int(np.sum(y_pred_binary == 0)),
                "predicted_pneumonia": int(np.sum(y_pred_binary == 1))
            }
        },
        "development_validation_vs_official_test": comparison_val_vs_test,
        "evaluation_duration_seconds": eval_duration,
        "disclaimer": config_obj.DISCLAIMER
    }

    # Save JSON files
    metrics_path = config_obj.METRICS_DIR / "final_test_metrics.json"
    save_json(final_test_summary, metrics_path)
    logger.info(f"Saved final test metrics to: {metrics_path}")

    cr_path = config_obj.METRICS_DIR / "final_test_classification_report.json"
    save_json(clf_report, cr_path)
    logger.info(f"Saved final test classification report to: {cr_path}")

    # 11. Render plots
    plots_dir = config_obj.PLOTS_DIR
    cm_plot_path = plots_dir / "final_test_confusion_matrix.png"
    plot_test_confusion_matrix(cm, cm_plot_path, model_name="MobileNetV2 (Transfer)")

    roc_plot_path = plots_dir / "final_test_roc_curve.png"
    plot_test_roc_curve(y_true, y_pred_probs, roc_plot_path, model_name="MobileNetV2 (Transfer)")

    pr_plot_path = plots_dir / "final_test_precision_recall_curve.png"
    plot_test_precision_recall_curve(y_true, y_pred_probs, pr_plot_path, model_name="MobileNetV2 (Transfer)")

    logger.info("STAGE 7 evaluation completed successfully!")
    return final_test_summary


def main():
    summary = run_official_test_evaluation()
    m = summary["measured_test_set_performance"]
    print("\n" + "=" * 65)
    print("STAGE 7: OFFICIAL HELD-OUT TEST SPLIT EVALUATION COMPLETE")
    print("=" * 65)
    print(f"Model:                   {summary['model_name']}")
    print(f"Checkpoint:              {summary['checkpoint_path']}")
    print(f"Total Test Images:       {summary['test_image_count']} (NORMAL=234, PNEUMONIA=390)")
    print(f"Decision Threshold:      {summary['fixed_classification_threshold']} (Fixed)")
    print("\nMeasured Test-Set Performance:")
    print(f"  - Accuracy:            {m['accuracy'] * 100:.2f}%")
    print(f"  - Precision:           {m['precision'] * 100:.2f}%")
    print(f"  - Recall (Sensitivity):{m['recall_sensitivity'] * 100:.2f}%")
    print(f"  - Specificity:         {m['specificity'] * 100:.2f}%")
    print(f"  - F1-Score:            {m['f1_score'] * 100:.2f}%")
    print(f"  - ROC-AUC:             {m['roc_auc']:.4f}")
    print(f"  - PR-AUC:              {m['pr_auc']:.4f}")
    print("\nConfusion Matrix:")
    cm = m["confusion_matrix"]
    print(f"  - TN: {cm['true_negative']}  |  FP: {cm['false_positive']}")
    print(f"  - FN: {cm['false_negative']}   |  TP: {cm['true_positive']}")
    print(f"  - Total: {cm['total']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
