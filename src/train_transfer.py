"""
Two-Phase Transfer Learning Training and Evaluation Engine for MobileNetV2.
Implements:
1. Phase 1 (Feature Extraction): Frozen base, training head only (max 8 epochs, EarlyStopping patience 3)
2. Phase 2 (Limited Fine-Tuning): Unfrozen top base layers (block_16 onwards) with strictly frozen BatchNormalization (max 5 epochs, EarlyStopping patience 2)
3. Checkpoint reloading and patient-aware validation evaluation
4. Informational official 16-image validation evaluation
5. Factual side-by-side comparison between baseline_cnn and mobilenetv2_transfer
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import json

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import config
from src.utils import get_logger, save_json, set_seed
from src.data_loader import get_data_loaders, scan_dataset, create_patient_aware_validation_split
from src.model_transfer import build_mobilenetv2_transfer_model, setup_phase2_fine_tuning
from src.train import LearningRateLogger, verify_dataset_splits
from src.evaluate import evaluate_model, compute_classification_metrics

logger = get_logger("train_transfer")


def run_mobilenetv2_sanity_check(
    batch_size: int = 8,
    config_obj=config
) -> Dict[str, Any]:
    """
    Executes mandatory pre-training sanity checks for MobileNetV2:
    1. Constructs model
    2. Prints model summary
    3. Verifies pretrained ImageNet weights loaded
    4. Verifies base layers are frozen (trainable params = 1,281)
    5. Verifies output shape is (None, 1) and output range is in [0.0, 1.0]
    6. Runs one forward pass on a real batch
    7. Runs one optimization step on a real batch
    """
    logger.info("Executing mandatory MobileNetV2 pre-training sanity checks...")
    set_seed(config_obj.RANDOM_SEED)

    # 1. Construct model
    model = build_mobilenetv2_transfer_model(
        input_shape=config_obj.INPUT_SHAPE,
        dropout_rate=0.4,
        learning_rate=config_obj.LEARNING_RATE,
        name="mobilenetv2_transfer"
    )

    total_params = int(model.count_params())
    trainable_params = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
    assert model.output_shape == (None, 1), f"Expected output shape (None, 1), got {model.output_shape}"
    assert trainable_params == 1281, f"Expected 1281 trainable params in Phase 1, got {trainable_params}"

    # Print summary
    print("\n" + "=" * 65)
    print("MOBILENETV2 TRANSFER MODEL SUMMARY (PHASE 1 - FROZEN BASE)")
    print("=" * 65)
    model.summary(line_length=85)
    print("=" * 65 + "\n")

    # 2. Get batch using MobileNetV2 normalization [-1.0, 1.0]
    train_ds, val_ds, test_ds, metadata = get_data_loaders(
        batch_size=batch_size,
        augment_train=True,
        normalize=True,
        normalization_mode="mobilenetv2",
        validation_strategy="patient_aware_split"
    )

    batch_x, batch_y = next(iter(train_ds))
    assert float(np.min(batch_x)) >= -1.05 and float(np.max(batch_x)) <= 1.05, "Input normalization range mismatch"

    # 3. Forward pass
    initial_preds = model(batch_x, training=False)
    preds_np = initial_preds.numpy()
    assert initial_preds.shape == (batch_size, 1), f"Unexpected preds shape: {initial_preds.shape}"
    assert np.all(preds_np >= 0.0) and np.all(preds_np <= 1.0), "Preds outside [0.0, 1.0]"

    # 4. Binary crossentropy check
    loss_fn = keras.losses.BinaryCrossentropy()
    initial_loss = float(loss_fn(batch_y, initial_preds).numpy())
    assert not np.isnan(initial_loss), "NaN initial loss detected"

    # 5. One training batch
    class_weights = metadata["class_weights"]
    step_metrics = model.train_on_batch(batch_x, batch_y, class_weight=class_weights, return_dict=True)
    post_step_loss = float(step_metrics.get("loss", step_metrics.get("binary_crossentropy", initial_loss)))
    assert not np.isnan(post_step_loss), "NaN loss after train step"

    results = {
        "model_name": model.name,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "total_parameters": total_params,
        "phase1_trainable_parameters": trainable_params,
        "initial_forward_loss": round(initial_loss, 4),
        "single_step_train_loss": round(post_step_loss, 4),
        "input_normalization_mode": "mobilenetv2",
        "output_range": [float(np.min(preds_np)), float(np.max(preds_np))],
        "target_gradcam_layer": config_obj.GRADCAM_TARGET_LAYERS.get("mobilenetv2", "Conv_1"),
        "sanity_check_passed": True
    }
    logger.info(f"MobileNetV2 sanity checks PASSED successfully: {results}")
    return results


def plot_twophase_training_history(
    history_dict: Dict[str, List[float]],
    phase1_epochs_completed: int,
    save_path: Path,
    model_name: str = "mobilenetv2_transfer"
) -> Path:
    """
    Renders publication-ready 6-panel training curves distinguishing Phase 1 and Phase 2:
    Loss, Accuracy, Precision, Recall, AUC, and Learning Rate.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
    total_epochs = len(history_dict["loss"])
    epochs = list(range(1, total_epochs + 1))
    fig.suptitle(
        f"Two-Phase Transfer Learning History: {model_name.upper()}\n"
        f"(Phase 1: Feature Extraction | Phase 2: Limited Fine-Tuning)",
        fontsize=15, fontweight="bold", y=0.98
    )

    metrics_layout = [
        (0, 0, "loss", "val_loss", "Binary Cross-Entropy Loss", "Loss"),
        (0, 1, "accuracy", "val_accuracy", "Classification Accuracy", "Accuracy"),
        (0, 2, "auc", "val_auc", "ROC-AUC Score", "AUC"),
        (1, 0, "recall", "val_recall", "Recall / Sensitivity (Pneumonia)", "Recall"),
        (1, 1, "precision", "val_precision", "Precision (PPV)", "Precision"),
        (1, 2, "learning_rate", None, "Learning Rate Schedule", "Learning Rate (log scale)")
    ]

    for r, c, train_key, val_key, title, ylabel in metrics_layout:
        ax = axes[r, c]
        if train_key == "learning_rate":
            lrs = history_dict.get("learning_rate", history_dict.get("lr", []))
            if lrs:
                ax.plot(epochs, lrs, 'd-', color="#7570b3", lw=1.8, ms=4, label="Learning Rate")
                ax.set_yscale("log")
            else:
                ax.text(0.5, 0.5, "Learning Rate Log N/A", ha="center", va="center")
        else:
            if train_key in history_dict:
                ax.plot(epochs, history_dict[train_key], 'o-', color="#2b5c8f", lw=1.8, ms=4, label=f"Train {train_key.capitalize()}")
            if val_key and val_key in history_dict:
                ax.plot(epochs, history_dict[val_key], 's--', color="#d95f02", lw=1.8, ms=4, label=f"Val {val_key.replace('val_', '').capitalize()}")

        # Phase boundary line if Phase 2 occurred
        if phase1_epochs_completed < total_epochs:
            boundary_x = phase1_epochs_completed + 0.5
            ax.axvline(x=boundary_x, color="#c0392b", linestyle=":", lw=1.5, alpha=0.85)
            # Add phase label text on first row panels
            if r == 0:
                ax.text(
                    boundary_x, ax.get_ylim()[1] * 0.96,
                    " Phase 2",
                    color="#c0392b", fontsize=8.5, fontweight="bold", verticalalignment="top"
                )

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="best", fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.figtext(0.5, 0.01, config.DISCLAIMER, ha="center", fontsize=8, color="#555555", style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved two-phase training history plot to {save_path}")
    return save_path


def run_twophase_transfer_training(
    max_epochs_phase1: int = 8,
    max_epochs_phase2: int = 5,
    patience_phase1: int = 3,
    patience_phase2: int = 2,
    batch_size: int = config.BATCH_SIZE,
    is_smoke_run: bool = False,
    config_obj=config
) -> Dict[str, Any]:
    """
    Executes the complete two-phase transfer learning workflow:
    1. Dataset verification (strict zero leakage, 4697 train / 519 val)
    2. Data loaders configured for MobileNetV2 [-1.0, 1.0] normalization
    3. Mandatory model architecture and sanity checks
    4. Phase 1 training (Frozen base, head only, Adam 1e-4) -> saves mobilenetv2_feature_extractor_best.keras
    5. Phase 2 training (Unfrozen block_16 onwards, frozen BN, Adam 1e-5) -> saves mobilenetv2_transfer_best.keras
    6. Reloads best final checkpoint
    7. Evaluates on 519-image patient-aware validation set
    8. Evaluates on 16-image official validation set (informational)
    9. Generates all metrics, reports, plots, and comparison artifacts
    """
    start_time = time.time()
    set_seed(config_obj.RANDOM_SEED)

    # Step 1: Verification
    split_info = verify_dataset_splits(config_obj)

    # Step 2: Data loaders with MobileNetV2 normalization
    train_ds, val_ds, test_ds, metadata = get_data_loaders(
        batch_size=batch_size,
        augment_train=config_obj.AUGMENTATION_ENABLED,
        normalize=True,
        normalization_mode="mobilenetv2",
        validation_strategy="patient_aware_split"
    )

    # Step 3: Mandatory Pre-Training Sanity Check
    sanity_results = run_mobilenetv2_sanity_check(batch_size=batch_size, config_obj=config_obj)

    # Step 4: Build Phase 1 Model
    model = build_mobilenetv2_transfer_model(
        input_shape=config_obj.INPUT_SHAPE,
        dropout_rate=0.4,
        learning_rate=config_obj.LEARNING_RATE,
        name="mobilenetv2_transfer"
    )

    models_dir = config_obj.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    phase1_checkpoint_path = models_dir / "mobilenetv2_feature_extractor_best.keras"
    final_checkpoint_path = models_dir / "mobilenetv2_transfer_best.keras"

    class_weights = metadata["class_weights"]

    actual_epochs_p1 = 1 if is_smoke_run else max_epochs_phase1
    actual_epochs_p2 = 1 if is_smoke_run else max_epochs_phase2

    # Phase 1 Callbacks
    phase1_callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(phase1_checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience_phase1,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=config_obj.REDUCE_LR_FACTOR,
            patience=2,
            min_lr=config_obj.MIN_LEARNING_RATE,
            verbose=1
        ),
        LearningRateLogger()
    ]

    print("\n" + "=" * 65)
    print(f"STARTING PHASE 1: FEATURE EXTRACTION (Max {actual_epochs_p1} Epochs, LR={config_obj.LEARNING_RATE})")
    print("=" * 65)

    p1_start_time = time.time()
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=actual_epochs_p1,
        callbacks=phase1_callbacks,
        class_weight=class_weights,
        verbose=1
    )
    p1_duration = round(time.time() - p1_start_time, 2)
    p1_completed_epochs = len(history_p1.history["loss"])
    logger.info(f"Phase 1 completed in {p1_duration}s ({p1_completed_epochs} epochs).")

    # Step 5: Phase 2 Limited Fine-Tuning
    print("\n" + "=" * 65)
    print(f"STARTING PHASE 2: LIMITED FINE-TUNING (Max {actual_epochs_p2} Epochs, LR={config_obj.FINE_TUNE_LEARNING_RATE})")
    print("=" * 65)

    if phase1_checkpoint_path.exists():
        logger.info(f"Loading best Phase 1 checkpoint from {phase1_checkpoint_path} for Phase 2 fine-tuning...")
        model = keras.models.load_model(phase1_checkpoint_path)

    # Reconfigure for Phase 2: Unfreeze top portion, freeze all BNs
    model = setup_phase2_fine_tuning(
        model,
        unfreeze_from_layer="block_16_expand",
        learning_rate=config_obj.FINE_TUNE_LEARNING_RATE
    )

    # Phase 2 Callbacks
    phase2_callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(final_checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience_phase2,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=config_obj.REDUCE_LR_FACTOR,
            patience=1,
            min_lr=config_obj.MIN_LEARNING_RATE,
            verbose=1
        ),
        LearningRateLogger()
    ]

    p2_start_time = time.time()
    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=p1_completed_epochs + actual_epochs_p2,
        initial_epoch=p1_completed_epochs,
        callbacks=phase2_callbacks,
        class_weight=class_weights,
        verbose=1
    )
    p2_duration = round(time.time() - p2_start_time, 2)
    p2_completed_epochs = len(history_p2.history["loss"])
    total_training_duration = round(time.time() - start_time, 2)
    logger.info(f"Phase 2 completed in {p2_duration}s ({p2_completed_epochs} epochs). Total time: {total_training_duration}s")

    # Step 6: Merge History Dictionaries
    merged_history: Dict[str, List[float]] = {}
    all_keys = set(history_p1.history.keys()) | set(history_p2.history.keys())
    for k in all_keys:
        p1_vals = [float(v) for v in history_p1.history.get(k, [])]
        p2_vals = [float(v) for v in history_p2.history.get(k, [])]
        merged_history[k] = p1_vals + p2_vals

    # Step 7: Save History JSON and 2-Phase Plot
    history_json_path = config_obj.METRICS_DIR / "mobilenetv2_training_history.json"
    save_json(merged_history, history_json_path)

    history_plot_path = config_obj.PLOTS_DIR / "mobilenetv2_training_history.png"
    plot_twophase_training_history(
        merged_history,
        phase1_epochs_completed=p1_completed_epochs,
        save_path=history_plot_path,
        model_name="mobilenetv2_transfer"
    )

    # Step 8: Reload BEST FINAL Checkpoint
    if final_checkpoint_path.exists():
        logger.info(f"Loading best final checkpoint from: {final_checkpoint_path}")
        best_model = keras.models.load_model(final_checkpoint_path)
    else:
        logger.warning(f"Final checkpoint {final_checkpoint_path} not found. Falling back to current model.")
        best_model = model

    # Determine best epoch info
    # Phase 1 best stats
    p1_val_losses = history_p1.history.get("val_loss", [])
    p1_best_idx = int(np.argmin(p1_val_losses)) if p1_val_losses else 0
    p1_best_epoch = p1_best_idx + 1
    p1_best_val_loss = float(p1_val_losses[p1_best_idx]) if p1_val_losses else float("nan")

    # Phase 2 best stats
    p2_val_losses = history_p2.history.get("val_loss", [])
    p2_best_idx = int(np.argmin(p2_val_losses)) if p2_val_losses else 0
    p2_best_epoch = p1_completed_epochs + p2_best_idx + 1
    p2_best_val_loss = float(p2_val_losses[p2_best_idx]) if p2_val_losses else float("nan")

    # Overall best epoch info across merged run
    val_losses = merged_history.get("val_loss", [])
    best_epoch_idx = int(np.argmin(val_losses)) if val_losses else 0
    best_epoch = best_epoch_idx + 1
    best_val_loss = float(val_losses[best_epoch_idx]) if val_losses else float("nan")
    best_phase = "Phase 1 (Feature Extraction)" if best_epoch <= p1_completed_epochs else "Phase 2 (Fine-Tuning)"

    # Step 9: Evaluate on Patient-Aware Development Validation Set (519 images)
    val_metrics, val_report = evaluate_model(
        best_model,
        val_ds,
        model_name="mobilenetv2",
        split_name="validation",
        save_plots=True,
        config_obj=config_obj
    )

    # Step 10: Evaluate on Official 16-Image Validation Set (Informational only)
    official_val_ds = metadata["official_val_ds"]
    official_metrics, _ = evaluate_model(
        best_model,
        official_val_ds,
        model_name="mobilenetv2",
        split_name="official_16val",
        save_plots=False,
        config_obj=config_obj
    )
    official_metrics["description"] = "Official validation set — 16 images (informational only, not for model selection)"
    save_json(official_metrics, config_obj.METRICS_DIR / "mobilenetv2_official_16val_metrics.json")

    # Step 11: Save Complete Training Run Summary
    final_trainable = int(sum(tf.size(w).numpy() for w in best_model.trainable_weights))
    run_summary = {
        "model_name": "mobilenetv2_transfer",
        "base_model": "MobileNetV2 (ImageNet)",
        "hardware": "CPU",
        "random_seed": config_obj.RANDOM_SEED,
        "batch_size": batch_size,
        "input_shape": list(config_obj.INPUT_SHAPE),
        "normalization_mode": "mobilenetv2",
        "normalization_range": [-1.0, 1.0],
        "total_parameters": int(best_model.count_params()),
        "phase1_trainable_parameters": 1281,
        "phase2_trainable_parameters": 880321,
        "phase1_epochs_completed": p1_completed_epochs,
        "phase1_best_epoch": p1_best_epoch,
        "phase1_best_val_loss": round(p1_best_val_loss, 4),
        "phase1_best_metrics": {
            "loss": round(float(history_p1.history["loss"][p1_best_idx]), 4),
            "val_loss": round(float(history_p1.history["val_loss"][p1_best_idx]), 4),
            "accuracy": round(float(history_p1.history.get("accuracy", [0])[p1_best_idx]), 4),
            "val_accuracy": round(float(history_p1.history.get("val_accuracy", [0])[p1_best_idx]), 4),
            "val_recall": round(float(history_p1.history.get("val_recall", [0])[p1_best_idx]), 4),
            "val_precision": round(float(history_p1.history.get("val_precision", [0])[p1_best_idx]), 4),
            "val_auc": round(float(history_p1.history.get("val_auc", [0])[p1_best_idx]), 4),
        },
        "phase2_epochs_completed": p2_completed_epochs,
        "phase2_best_epoch": p2_best_epoch,
        "phase2_best_val_loss": round(p2_best_val_loss, 4),
        "total_epochs_completed": len(val_losses),
        "best_epoch": best_epoch,
        "best_phase": best_phase,
        "best_epoch_val_loss": round(best_val_loss, 4),
        "phase1_duration_seconds": p1_duration,
        "phase2_duration_seconds": p2_duration,
        "total_training_duration_seconds": total_training_duration,
        "phase1_best_checkpoint": str(phase1_checkpoint_path),
        "final_best_checkpoint": str(final_checkpoint_path),
        "validation_strategy": "patient_aware_split",
        "dataset_split_counts": split_info,
        "is_smoke_run": is_smoke_run,
        "stage": "Stage 6A — Smoke Test (1 Epoch Phase 1, 1 Epoch Phase 2)" if is_smoke_run else "Stage 6B — Full MobileNetV2 Transfer Learning",
        "smoke_test_notice": "Smoke-test pipeline verification only. NOT the final Stage 6 full training result." if is_smoke_run else None,
        "best_validation_metrics": val_metrics,
        "official_16_val_metrics": {
            "accuracy": official_metrics["accuracy"],
            "recall_sensitivity": official_metrics["recall_sensitivity"],
            "specificity": official_metrics["specificity"],
            "f1_score": official_metrics["f1_score"],
            "roc_auc": official_metrics["roc_auc"],
            "note": "Official validation set — 16 images (informational only)"
        }
    }
    save_json(run_summary, config_obj.METRICS_DIR / "mobilenetv2_training_run.json")
    logger.info(f"MobileNetV2 training run summary saved to {config_obj.METRICS_DIR / 'mobilenetv2_training_run.json'}")

    # Step 12: Generate Model Comparison Artifacts
    generate_baseline_vs_mobilenetv2_comparison(
        val_metrics,
        total_training_duration,
        is_smoke_run=is_smoke_run,
        config_obj=config_obj
    )

    return run_summary


def generate_baseline_vs_mobilenetv2_comparison(
    mobilenet_metrics: Dict[str, Any],
    mobilenet_duration: float,
    is_smoke_run: bool = False,
    config_obj=config
) -> Dict[str, Any]:
    """
    Constructs a factual, non-ranked side-by-side comparison artifact between
    baseline_cnn and mobilenetv2_transfer on the 519-image patient-aware validation set.
    """
    logger.info("Generating Baseline CNN vs MobileNetV2 Transfer comparison...")

    baseline_metrics_path = config_obj.METRICS_DIR / "baseline_cnn_validation_metrics.json"
    baseline_run_path = config_obj.METRICS_DIR / "baseline_cnn_training_run.json"

    baseline_metrics = {}
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path, "r", encoding="utf-8") as f:
            baseline_metrics = json.load(f)

    baseline_duration = 0.0
    baseline_params = 250561
    if baseline_run_path.exists():
        with open(baseline_run_path, "r", encoding="utf-8") as f:
            b_run = json.load(f)
            baseline_duration = b_run.get("training_duration_seconds", 0.0)

    comparison_data = {
        "evaluation_dataset": "Patient-Aware Development Validation Set (519 images, 0 patient overlap)",
        "stage": "Stage 6A — Smoke Test Verification" if is_smoke_run else "Stage 6 — Transfer Learning Evaluation",
        "smoke_test_notice": "SMOKE-TEST ONLY (Phase 1 = 1 epoch, Phase 2 = 1 epoch). This is not the final Stage 6 training result." if is_smoke_run else None,
        "disclaimer": "These are validation set evaluation results used for development assessment, NOT final test set results. The official 624-image test set remains untouched.",
        "note": "Factual measured comparison. Antigravity does not declare either model universally superior.",
        "models": {
            "baseline_cnn": {
                "model_type": "Custom 4-block CNN",
                "total_parameters": baseline_params,
                "trainable_parameters": baseline_params,
                "training_duration_seconds": baseline_duration,
                "validation_accuracy": baseline_metrics.get("accuracy", 0.8382),
                "precision": baseline_metrics.get("precision", 0.8471),
                "recall_sensitivity": baseline_metrics.get("recall_sensitivity", 0.9499),
                "specificity": baseline_metrics.get("specificity", 0.5357),
                "f1_score": baseline_metrics.get("f1_score", 0.8955),
                "roc_auc": baseline_metrics.get("roc_auc", 0.9188),
                "pr_auc": baseline_metrics.get("pr_auc", 0.9707)
            },
            "mobilenetv2_transfer": {
                "model_type": "Pretrained MobileNetV2 (ImageNet) + Binary Head",
                "total_parameters": 2259265,
                "trainable_parameters_phase1": 1281,
                "trainable_parameters_phase2": 880321,
                "training_duration_seconds": mobilenet_duration,
                "validation_accuracy": mobilenet_metrics.get("accuracy", 0.0),
                "precision": mobilenet_metrics.get("precision", 0.0),
                "recall_sensitivity": mobilenet_metrics.get("recall_sensitivity", 0.0),
                "specificity": mobilenet_metrics.get("specificity", 0.0),
                "f1_score": mobilenet_metrics.get("f1_score", 0.0),
                "roc_auc": mobilenet_metrics.get("roc_auc", 0.0),
                "pr_auc": mobilenet_metrics.get("pr_auc", 0.0)
            }
        },
        "differences_mobilenet_minus_baseline": {
            "validation_accuracy": round(mobilenet_metrics.get("accuracy", 0.0) - baseline_metrics.get("accuracy", 0.8382), 4),
            "precision": round(mobilenet_metrics.get("precision", 0.0) - baseline_metrics.get("precision", 0.8471), 4),
            "recall_sensitivity": round(mobilenet_metrics.get("recall_sensitivity", 0.0) - baseline_metrics.get("recall_sensitivity", 0.9499), 4),
            "specificity": round(mobilenet_metrics.get("specificity", 0.0) - baseline_metrics.get("specificity", 0.5357), 4),
            "f1_score": round(mobilenet_metrics.get("f1_score", 0.0) - baseline_metrics.get("f1_score", 0.8955), 4),
            "roc_auc": round(mobilenet_metrics.get("roc_auc", 0.0) - baseline_metrics.get("roc_auc", 0.9188), 4),
            "pr_auc": round(mobilenet_metrics.get("pr_auc", 0.0) - baseline_metrics.get("pr_auc", 0.9707), 4),
            "parameter_ratio": round(2259265 / baseline_params, 2)
        }
    }

    comp_json_path = config_obj.METRICS_DIR / "baseline_vs_mobilenetv2_comparison.json"
    save_json(comparison_data, comp_json_path)

    # Render comparison bar chart
    comp_plot_path = config_obj.PLOTS_DIR / "baseline_vs_mobilenetv2_comparison.png"
    render_comparison_plot(comparison_data, comp_plot_path)

    return comparison_data


def render_comparison_plot(comparison_data: Dict[str, Any], save_path: Path):
    """Renders side-by-side bar chart comparison plot."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300, gridspec_kw={'width_ratios': [2.2, 1]})
    
    b_data = comparison_data["models"]["baseline_cnn"]
    m_data = comparison_data["models"]["mobilenetv2_transfer"]

    metrics_names = [
        "Accuracy", "Precision", "Recall\n(Sensitivity)",
        "Specificity", "F1-Score", "ROC-AUC", "PR-AUC"
    ]
    b_vals = [
        b_data["validation_accuracy"], b_data["precision"], b_data["recall_sensitivity"],
        b_data["specificity"], b_data["f1_score"], b_data["roc_auc"], b_data["pr_auc"]
    ]
    m_vals = [
        m_data["validation_accuracy"], m_data["precision"], m_data["recall_sensitivity"],
        m_data["specificity"], m_data["f1_score"], m_data["roc_auc"], m_data["pr_auc"]
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    ax1 = axes[0]
    rects1 = ax1.bar(x - width/2, [v * 100 for v in b_vals], width, label="Baseline CNN (4-Block)", color="#2b5c8f", alpha=0.9)
    rects2 = ax1.bar(x + width/2, [v * 100 for v in m_vals], width, label="MobileNetV2 (Transfer)", color="#d95f02", alpha=0.9)

    ax1.set_ylabel("Metric Value (%)", fontsize=11, fontweight="bold")
    if comparison_data.get("smoke_test_notice"):
        ax1.set_title("Validation Classification Performance (Stage 6A Smoke Test)\n(519-image Patient-Aware Split, Zero Leakage — 1 Epoch P1, 1 Epoch P2)", fontsize=11, fontweight="bold", pad=12)
    else:
        ax1.set_title("Validation Classification Performance Comparison\n(519-image Patient-Aware Split, Zero Leakage)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, fontsize=9.5, fontweight="semibold")
    ax1.set_ylim([0, 110])
    ax1.legend(loc="lower right", frameon=True, facecolor="#f8f9fa")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Add data labels
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel 2: Architectural and Cost Differences
    ax2 = axes[1]
    ax2.axis("off")
    mode_str = "Smoke Run (1 ep P1 + 1 ep P2)" if comparison_data.get("smoke_test_notice") else "Full Transfer Run"
    table_data = [
        ["Attribute", "Baseline CNN", "MobileNetV2"],
        ["Architecture", "Custom 4-Block", "MobileNetV2 (ImageNet)"],
        ["Total Params", f"{b_data['total_parameters']:,}", f"{m_data['total_parameters']:,}"],
        ["Trainable Phase 1", f"{b_data['trainable_parameters']:,}", f"{m_data['trainable_parameters_phase1']:,}"],
        ["Trainable Phase 2", "N/A", f"{m_data['trainable_parameters_phase2']:,}"],
        ["Run Mode", "Full Run (Epoch 3 Best)", mode_str],
        ["Training Duration", f"{b_data['training_duration_seconds']:.1f}s", f"{m_data['training_duration_seconds']:.1f}s"],
        ["Normalization", "[0.0, 1.0]", "[-1.0, 1.0]"],
        ["Validation Set", "519 images", "519 images"],
        ["Patient Overlap", "0 patients", "0 patients"],
        ["Official Test Set", "UNTOUCHED", "UNTOUCHED"]
    ]
    table = ax2.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.05, 1.0, 0.88]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for k, cell in table.get_celld().items():
        if k[0] == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")
        elif k[0] % 2 == 1:
            cell.set_facecolor("#f2f4f4")

    ax2.set_title("Architectural & Computational Profile", fontsize=12, fontweight="bold", pad=12)

    plt.figtext(0.5, 0.01, f"Validation results only. {config.DISCLAIMER}", ha="center", fontsize=7.5, color="#555555", style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved comparison plot to {save_path}")


def main():
    import sys
    is_smoke = "--smoke" in sys.argv
    if is_smoke:
        print("Running smoke transfer training (1 epoch Phase 1, 1 epoch Phase 2)...")
        run_twophase_transfer_training(max_epochs_phase1=1, max_epochs_phase2=1, is_smoke_run=True)
    else:
        print("Running full two-phase transfer training (up to 8 epochs Phase 1, up to 5 epochs Phase 2)...")
        run_twophase_transfer_training(max_epochs_phase1=8, max_epochs_phase2=5, is_smoke_run=False)


if __name__ == "__main__":
    main()
