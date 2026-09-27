"""
Training Infrastructure and Execution Engine for Baseline CNN.
Includes validation split verification, LearningRateLogger callback, full multi-epoch
training with EarlyStopping and ReduceLROnPlateau, best checkpoint reloading,
and post-training evaluation on patient-aware validation sets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import time
import json
import numpy as np
import tensorflow as tf
import keras

from src.config import config
from src.utils import get_logger, save_json, set_seed
from src.data_loader import get_data_loaders, scan_dataset, create_patient_aware_validation_split
from src.model_cnn import build_baseline_cnn
from src.evaluate import evaluate_model, plot_training_history

logger = get_logger("train")


class LearningRateLogger(keras.callbacks.Callback):
    """Logs the active learning rate at the end of each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = self.model.optimizer.learning_rate
        if hasattr(lr, "numpy"):
            val = float(lr.numpy())
        else:
            val = float(lr)
        logs["learning_rate"] = val


def verify_dataset_splits(config_obj=config) -> Dict[str, Any]:
    """
    Verifies that the patient-aware development split satisfies all strict constraints:
    - Development training: 4,697 images
    - Development validation: 519 images
    - Patient overlap: 0
    - Official validation: 16 images preserved separately
    - Official test: 624 images completely untouched
    """
    logger.info("Verifying dataset splits and patient-aware partitioning...")
    splits = scan_dataset(config_obj.DATASET_DIR)
    
    dev_train, dev_val = create_patient_aware_validation_split(
        splits["train"],
        val_ratio=config_obj.VAL_SPLIT_RATIO,
        seed=config_obj.RANDOM_SEED
    )
    
    train_count = len(dev_train)
    val_count = len(dev_val)
    official_val_count = len(splits["val"])
    official_test_count = len(splits["test"])

    pts_train = set(dev_train["patient_id"])
    pts_val = set(dev_val["patient_id"])
    overlap_count = len(pts_train & pts_val)

    assert train_count == 4697, f"Expected 4697 train images, got {train_count}"
    assert val_count == 519, f"Expected 519 val images, got {val_count}"
    assert overlap_count == 0, f"Expected 0 patient overlap, got {overlap_count}"
    assert official_val_count == 16, f"Expected 16 official val images, got {official_val_count}"
    assert official_test_count == 624, f"Expected 624 official test images, got {official_test_count}"

    verification_info = {
        "dev_train_count": train_count,
        "dev_val_count": val_count,
        "patient_overlap": overlap_count,
        "official_val_count": official_val_count,
        "official_test_count": official_test_count,
        "verified": True
    }
    logger.info(f"Dataset split verification PASSED: {verification_info}")
    return verification_info


def get_callbacks(
    model_name: str = "baseline_cnn",
    checkpoint_dir: Optional[Path] = None,
    config_obj=config
) -> List[keras.callbacks.Callback]:
    """
    Constructs standardized training callbacks:
    - ModelCheckpoint: Saves best performing model weights (.keras format)
    - EarlyStopping: Halts training when validation loss stops improving (restore_best_weights=True)
    - ReduceLROnPlateau: Decays learning rate when validation loss plateaus
    - LearningRateLogger: Tracks active learning rate per epoch
    """
    models_dir = Path(checkpoint_dir) if checkpoint_dir else config_obj.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_dir / f"{model_name}_best.keras"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config_obj.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=config_obj.REDUCE_LR_FACTOR,
            patience=config_obj.REDUCE_LR_PATIENCE,
            min_lr=config_obj.MIN_LEARNING_RATE,
            verbose=1
        ),
        LearningRateLogger()
    ]
    logger.info(f"Configured 4 callbacks for '{model_name}'. Checkpoint: {checkpoint_path}")
    return callbacks


def run_training_sanity_check(
    model: Optional[keras.Model] = None,
    batch_size: int = 8,
    config_obj=config
) -> Dict[str, Any]:
    """
    Performs a lightweight sanity check on the model and data pipeline:
    1. Verifies model construction and parameter count
    2. Performs one forward pass on a real dataset batch
    3. Verifies output shapes (batch_size, 1) and sigmoid range [0, 1]
    4. Computes binary crossentropy loss
    5. Performs exactly one training optimization step via train_on_batch
    """
    logger.info("Initiating lightweight training sanity check...")
    set_seed(config_obj.RANDOM_SEED)

    if model is None:
        model = build_baseline_cnn(input_shape=config_obj.INPUT_SHAPE, learning_rate=config_obj.LEARNING_RATE)

    total_params = int(model.count_params())
    trainable_params = int(sum(tf.size(w).numpy() for w in model.trainable_weights))

    train_ds, val_ds, test_ds, metadata = get_data_loaders(
        batch_size=batch_size,
        augment_train=True,
        normalize=True,
        validation_strategy=config_obj.VALIDATION_STRATEGY
    )

    batch_x, batch_y = next(iter(train_ds))
    initial_preds = model(batch_x, training=False)
    preds_np = initial_preds.numpy()

    assert initial_preds.shape == (batch_size, 1)
    assert np.all(preds_np >= 0.0) and np.all(preds_np <= 1.0)

    loss_fn = keras.losses.BinaryCrossentropy()
    initial_loss = float(loss_fn(batch_y, initial_preds).numpy())
    assert not np.isnan(initial_loss)

    class_weights = metadata["class_weights"]
    step_metrics = model.train_on_batch(batch_x, batch_y, class_weight=class_weights, return_dict=True)
    post_step_loss = float(step_metrics.get("loss", step_metrics.get("binary_crossentropy", initial_loss)))
    assert not np.isnan(post_step_loss)

    results = {
        "model_name": model.name,
        "input_shape": list(model.input_shape),
        "output_shape": list(model.output_shape),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "loss_function": "binary_crossentropy",
        "optimizer": model.optimizer.__class__.__name__,
        "learning_rate": float(config_obj.LEARNING_RATE),
        "initial_forward_loss": round(initial_loss, 4),
        "single_step_train_loss": round(post_step_loss, 4),
        "output_range": [float(np.min(preds_np)), float(np.max(preds_np))],
        "target_gradcam_layer": config_obj.GRADCAM_TARGET_LAYERS.get("baseline_cnn", "conv2d_block4_conv"),
        "validation_strategy": config_obj.VALIDATION_STRATEGY,
        "sanity_check_passed": True
    }

    metrics_path = config_obj.METRICS_DIR / "baseline_cnn_sanity_check.json"
    save_json(results, metrics_path)
    logger.info(f"Sanity check completed successfully! Saved to {metrics_path}")
    return results


def run_training_pipeline(
    epochs: int = config.EPOCHS_BASELINE,
    batch_size: int = config.BATCH_SIZE,
    is_smoke_run: bool = False,
    config_obj=config
) -> Dict[str, Any]:
    """
    Executes the training and evaluation workflow:
    1. Verifies dataset splits and zero patient overlap
    2. Builds model and callbacks
    3. Runs training loop on CPU
    4. Saves training history JSON and plots
    5. Reloads best checkpoint from disk
    6. Evaluates best checkpoint on development validation split
    7. Evaluates on official 16-image validation split (for reporting)
    8. Records complete training run summary
    """
    start_time = time.time()
    set_seed(config_obj.RANDOM_SEED)

    # Step 1: Verification
    split_info = verify_dataset_splits(config_obj)

    # Step 2: Data loaders
    train_ds, val_ds, test_ds, metadata = get_data_loaders(
        batch_size=batch_size,
        augment_train=config_obj.AUGMENTATION_ENABLED,
        normalize=True,
        validation_strategy="patient_aware_split"
    )

    # Step 3: Build model
    model = build_baseline_cnn(
        input_shape=config_obj.INPUT_SHAPE,
        learning_rate=config_obj.LEARNING_RATE,
        name="baseline_cnn"
    )

    actual_epochs = 1 if is_smoke_run else epochs
    model_name = "baseline_cnn"
    callbacks = get_callbacks(model_name=model_name, config_obj=config_obj)
    class_weights = metadata["class_weights"]

    logger.info(f"Starting training run (is_smoke_run={is_smoke_run}, epochs={actual_epochs}, batch_size={batch_size})...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=actual_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    elapsed_time_seconds = round(time.time() - start_time, 2)
    logger.info(f"Training completed in {elapsed_time_seconds} seconds ({actual_epochs} epochs).")

    # Format history dictionary
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    # Step 4: Save history JSON and plot
    history_json_path = config_obj.METRICS_DIR / f"{model_name}_training_history.json"
    save_json(history_dict, history_json_path)

    history_plot_path = config_obj.PLOTS_DIR / f"{model_name}_training_history.png"
    plot_training_history(history_dict, history_plot_path, model_name=model_name)

    # Step 5: Reload BEST checkpoint from disk
    best_checkpoint_path = config_obj.MODELS_DIR / f"{model_name}_best.keras"
    if best_checkpoint_path.exists():
        logger.info(f"Loading best checkpoint from: {best_checkpoint_path}")
        best_model = keras.models.load_model(best_checkpoint_path)
    else:
        logger.warning(f"Best checkpoint not found at {best_checkpoint_path}. Using current model.")
        best_model = model

    # Determine best epoch details
    val_losses = history_dict.get("val_loss", [])
    best_epoch_idx = int(np.argmin(val_losses)) if val_losses else 0
    best_epoch = best_epoch_idx + 1
    best_val_loss = float(val_losses[best_epoch_idx]) if val_losses else float("nan")

    # Step 6: Evaluate best model on Patient-Aware Development Validation Set
    val_metrics, val_report = evaluate_model(
        best_model,
        val_ds,
        model_name=model_name,
        split_name="validation",
        save_plots=True,
        config_obj=config_obj
    )

    # Step 7: Evaluate on Official 16-Image Validation Set (Clearly labeled, not for selection)
    official_val_ds = metadata["official_val_ds"]
    official_metrics, _ = evaluate_model(
        best_model,
        official_val_ds,
        model_name=model_name,
        split_name="official_16val",
        save_plots=False,
        config_obj=config_obj
    )
    official_metrics["description"] = "Official validation set — 16 images (informational only, not for model selection)"
    save_json(official_metrics, config_obj.METRICS_DIR / f"{model_name}_official_16val_metrics.json")

    # Step 8: Save complete training run summary
    lrs = history_dict.get("learning_rate", [config_obj.LEARNING_RATE])
    training_run_summary = {
        "model_name": model_name,
        "hardware": "CPU",
        "random_seed": config_obj.RANDOM_SEED,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_completed": len(val_losses),
        "is_smoke_run": is_smoke_run,
        "initial_learning_rate": float(config_obj.LEARNING_RATE),
        "final_learning_rate": float(lrs[-1]) if lrs else float(config_obj.LEARNING_RATE),
        "best_epoch": best_epoch,
        "best_epoch_val_loss": round(best_val_loss, 4),
        "training_duration_seconds": elapsed_time_seconds,
        "saved_model_path": str(best_checkpoint_path),
        "validation_strategy": "patient_aware_split",
        "dataset_split_counts": split_info,
        "best_validation_metrics": val_metrics,
        "official_16_val_metrics": {
            "accuracy": official_metrics["accuracy"],
            "recall_sensitivity": official_metrics["recall_sensitivity"],
            "specificity": official_metrics["specificity"],
            "f1_score": official_metrics["f1_score"],
            "roc_auc": official_metrics["roc_auc"],
            "note": "Official validation set — 16 images"
        }
    }

    run_summary_path = config_obj.METRICS_DIR / f"{model_name}_training_run.json"
    save_json(training_run_summary, run_summary_path)
    logger.info(f"Complete training run summary saved to {run_summary_path}")

    return training_run_summary


def main():
    import sys
    if "--smoke" in sys.argv:
        print("Running smoke training (1 epoch)...")
        summary = run_training_pipeline(epochs=1, is_smoke_run=True)
    else:
        print("Running full training (up to 15 epochs)...")
        summary = run_training_pipeline(epochs=config.EPOCHS_BASELINE, is_smoke_run=False)

    print("\n" + "=" * 65)
    print("STAGE 5 TRAINING PIPELINE COMPLETED")
    print("=" * 65)
    print(f"Model Name:              {summary['model_name']}")
    print(f"Epochs Completed:        {summary['epochs_completed']}")
    print(f"Best Epoch:              {summary['best_epoch']}")
    print(f"Best Val Loss:           {summary['best_epoch_val_loss']}")
    print(f"Training Duration:       {summary['training_duration_seconds']}s")
    print("\nPatient-Aware Validation Metrics:")
    vm = summary["best_validation_metrics"]
    print(f"  - Accuracy:            {vm['accuracy']}")
    print(f"  - Precision:           {vm['precision']}")
    print(f"  - Recall (Sensitivity):{vm['recall_sensitivity']}")
    print(f"  - Specificity:         {vm['specificity']}")
    print(f"  - F1-Score:            {vm['f1_score']}")
    print(f"  - ROC-AUC:             {vm['roc_auc']}")
    print(f"  - PR-AUC:              {vm['pr_auc']}")
    print("\nOfficial 16-Image Validation Set (Informational):")
    om = summary["official_16_val_metrics"]
    print(f"  - Accuracy:            {om['accuracy']}")
    print(f"  - Recall:              {om['recall_sensitivity']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
