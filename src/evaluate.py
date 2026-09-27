"""
Evaluation and Metrics Reporting Module for Pneumonia Detection Models.
Computes clinically critical metrics: Accuracy, Precision, Recall/Sensitivity,
Specificity, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix, and renders publication-ready plots.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
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

from src.config import config
from src.utils import get_logger, save_json

logger = get_logger("evaluate")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = config.DECISION_THRESHOLD
) -> Dict[str, Any]:
    """
    Computes standard and clinical binary classification metrics.
    - Sensitivity / Recall: True Positive Rate (identifying pneumonia cases)
    - Specificity: True Negative Rate (correctly ruling out healthy cases)
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred_probs = np.asarray(y_pred_probs).astype(float).ravel()
    y_pred_binary = (y_pred_probs >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred_binary))
    prec = float(precision_score(y_true, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_true, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))

    try:
        auc_score = float(roc_auc_score(y_true, y_pred_probs))
    except Exception:
        auc_score = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_pred_probs))
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    metrics = {
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall_sensitivity": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc_score, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "sample_counts": {
            "total_samples": len(y_true),
            "actual_normal": int(np.sum(y_true == 0)),
            "actual_pneumonia": int(np.sum(y_true == 1)),
            "predicted_normal": int(np.sum(y_pred_binary == 0)),
            "predicted_pneumonia": int(np.sum(y_pred_binary == 1))
        }
    }
    return metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Path,
    model_name: str = "baseline_cnn",
    class_names: Tuple[str, ...] = config.CLASS_NAMES
) -> Path:
    """Renders high-resolution normalized & raw confusion matrix."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    
    # Calculate percentages
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
    ax.set_title(f"Confusion Matrix: {model_name.upper()}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Diagnosis", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Ground Truth", fontsize=11, fontweight="semibold")

    plt.figtext(0.5, -0.02, config.DISCLAIMER, ha="center", fontsize=7.5, color="#555555", style="italic")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix plot to {save_path}")
    return save_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    save_path: Path,
    model_name: str = "baseline_cnn"
) -> Path:
    """Renders ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    auc_val = roc_auc_score(y_true, y_pred_probs)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(fpr, tpr, color="#2b5c8f", lw=2.2, label=f"{model_name} (ROC-AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.2, linestyle="--", label="Random Chance (AUC = 0.50)")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="semibold")
    ax.set_title(f"Receiver Operating Characteristic (ROC): {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="#f8f9fa")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.figtext(0.5, -0.02, config.DISCLAIMER, ha="center", fontsize=7.5, color="#555555", style="italic")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ROC curve plot to {save_path}")
    return save_path


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    save_path: Path,
    model_name: str = "baseline_cnn"
) -> Path:
    """Renders Precision-Recall curve with PR-AUC / Average Precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = average_precision_score(y_true, y_pred_probs)
    baseline_rate = np.mean(y_true == 1)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.plot(recall, precision, color="#d95f02", lw=2.2, label=f"{model_name} (PR-AUC = {pr_auc:.4f})")
    ax.axhline(baseline_rate, color="gray", lw=1.2, linestyle="--", label=f"Baseline Prevalence ({baseline_rate:.2f})")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="semibold")
    ax.set_title(f"Precision-Recall Curve: {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", frameon=True, facecolor="#f8f9fa")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.figtext(0.5, -0.02, config.DISCLAIMER, ha="center", fontsize=7.5, color="#555555", style="italic")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Precision-Recall curve plot to {save_path}")
    return save_path


def plot_training_history(
    history_dict: Dict[str, List[float]],
    save_path: Path,
    model_name: str = "baseline_cnn"
) -> Path:
    """
    Renders comprehensive 6-panel training history curves:
    Loss, Accuracy, Precision, Recall, AUC, and Learning Rate.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    epochs = range(1, len(history_dict["loss"]) + 1)
    fig.suptitle(f"Training & Validation History: {model_name.upper()}", fontsize=16, fontweight="bold", y=0.98)

    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, history_dict["loss"], 'o-', color="#2b5c8f", label="Train Loss", lw=1.8, ms=4)
    if "val_loss" in history_dict:
        ax.plot(epochs, history_dict["val_loss"], 's--', color="#d95f02", label="Val Loss", lw=1.8, ms=4)
    ax.set_title("Binary Cross-Entropy Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # 2. Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history_dict.get("accuracy", []), 'o-', color="#2b5c8f", label="Train Acc", lw=1.8, ms=4)
    if "val_accuracy" in history_dict:
        ax.plot(epochs, history_dict["val_accuracy"], 's--', color="#d95f02", label="Val Acc", lw=1.8, ms=4)
    ax.set_title("Classification Accuracy", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # 3. AUC
    ax = axes[0, 2]
    ax.plot(epochs, history_dict.get("auc", []), 'o-', color="#2b5c8f", label="Train AUC", lw=1.8, ms=4)
    if "val_auc" in history_dict:
        ax.plot(epochs, history_dict["val_auc"], 's--', color="#d95f02", label="Val AUC", lw=1.8, ms=4)
    ax.set_title("ROC-AUC Score", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # 4. Recall (Sensitivity)
    ax = axes[1, 0]
    ax.plot(epochs, history_dict.get("recall", []), 'o-', color="#2b5c8f", label="Train Recall", lw=1.8, ms=4)
    if "val_recall" in history_dict:
        ax.plot(epochs, history_dict["val_recall"], 's--', color="#d95f02", label="Val Recall", lw=1.8, ms=4)
    ax.set_title("Recall / Sensitivity (Pneumonia)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # 5. Precision
    ax = axes[1, 1]
    ax.plot(epochs, history_dict.get("precision", []), 'o-', color="#2b5c8f", label="Train Precision", lw=1.8, ms=4)
    if "val_precision" in history_dict:
        ax.plot(epochs, history_dict["val_precision"], 's--', color="#d95f02", label="Val Precision", lw=1.8, ms=4)
    ax.set_title("Precision (Positive Predictive Value)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    # 6. Learning Rate
    ax = axes[1, 2]
    lrs = history_dict.get("learning_rate", history_dict.get("lr", []))
    if lrs:
        ax.plot(epochs, lrs, 'd-', color="#7570b3", label="Learning Rate", lw=1.8, ms=4)
        ax.set_yscale("log")
        ax.set_title("Learning Rate Schedule", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate (log scale)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "Constant Learning Rate", ha="center", va="center", fontsize=12)
        ax.axis("off")

    plt.figtext(0.5, 0.01, config.DISCLAIMER, ha="center", fontsize=8, color="#555555", style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved training history plot to {save_path}")
    return save_path


def evaluate_model(
    model: keras.Model,
    data_loader: tf.data.Dataset,
    model_name: str = "baseline_cnn",
    split_name: str = "validation",
    save_plots: bool = True,
    config_obj=config
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluates model predictions batch by batch over the specified dataset.
    Generates quantitative report, classification report, and plots.
    Returns: (metrics_dict, classification_report_dict)
    """
    logger.info(f"Evaluating '{model_name}' on '{split_name}' dataset...")
    
    all_y_true = []
    all_y_pred_probs = []

    for batch_x, batch_y in data_loader:
        preds = model(batch_x, training=False)
        all_y_true.extend(batch_y.numpy().flatten())
        all_y_pred_probs.extend(preds.numpy().flatten())

    y_true = np.array(all_y_true)
    y_pred_probs = np.array(all_y_pred_probs)
    y_pred_binary = (y_pred_probs >= config_obj.DECISION_THRESHOLD).astype(int)

    metrics = compute_classification_metrics(y_true, y_pred_probs, threshold=config_obj.DECISION_THRESHOLD)
    metrics["model_name"] = model_name
    metrics["split_name"] = split_name

    clf_report = classification_report(
        y_true, y_pred_binary,
        target_names=list(config_obj.CLASS_NAMES),
        output_dict=True,
        zero_division=0
    )

    metrics_dir = config_obj.METRICS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = config_obj.PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save validation metrics JSON
    metrics_file_name = f"{model_name}_{split_name}_metrics.json" if split_name != "validation" else f"{model_name}_validation_metrics.json"
    metrics_path = metrics_dir / metrics_file_name
    save_json(metrics, metrics_path)
    logger.info(f"Saved evaluation metrics to {metrics_path}")

    # 2. Save classification report JSON
    cr_file_name = f"{model_name}_classification_report.json" if split_name == "validation" else f"{model_name}_{split_name}_classification_report.json"
    cr_path = metrics_dir / cr_file_name
    save_json(clf_report, cr_path)
    logger.info(f"Saved classification report to {cr_path}")

    # 3. Save plots if requested
    if save_plots:
        cm = confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
        cm_path = plots_dir / f"{model_name}_confusion_matrix.png"
        plot_confusion_matrix(cm, cm_path, model_name=model_name)

        roc_path = plots_dir / f"{model_name}_roc_curve.png"
        plot_roc_curve(y_true, y_pred_probs, roc_path, model_name=model_name)

        pr_path = plots_dir / f"{model_name}_precision_recall_curve.png"
        plot_precision_recall_curve(y_true, y_pred_probs, pr_path, model_name=model_name)

    return metrics, clf_report
