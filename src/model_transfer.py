"""
Transfer Learning Model Module for Pneumonia Detection.
Provides MobileNetV2 architecture with pretrained ImageNet weights, custom binary classification head,
freeze/unfreeze controls for two-phase transfer learning, and strict BatchNormalization freeze handling.
"""

from typing import Tuple, Optional, Dict, Any, List
import tensorflow as tf
import keras
from keras import layers

from src.config import config
from src.utils import get_logger

logger = get_logger("model_transfer")


def build_mobilenetv2_transfer_model(
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    dropout_rate: float = 0.4,
    learning_rate: float = config.LEARNING_RATE,
    weights: str = "imagenet",
    name: str = "mobilenetv2_transfer"
) -> keras.Model:
    """
    Constructs the MobileNetV2 Transfer Learning Model for binary pneumonia classification.
    
    Architecture:
    1. Input tensor: (224, 224, 3)
    2. MobileNetV2 base (include_top=False, weights='imagenet')
    3. GlobalAveragePooling2D
    4. Dropout(dropout_rate)
    5. Dense(1, activation='sigmoid')
    
    Initial state:
    - Entire MobileNetV2 base is FROZEN (Phase 1: Feature Extraction).
    - Only the classification head parameters are trainable.
    """
    logger.info(f"Building MobileNetV2 transfer model '{name}' (weights='{weights}', input_shape={input_shape})...")

    # Step 1: Input layer
    inputs = layers.Input(shape=input_shape, name="input_tensor")

    # Step 2: Pretrained base
    base_model = keras.applications.MobileNetV2(
        input_tensor=inputs,
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )

    # Step 3: Freeze entire base for Phase 1 feature extraction
    for layer in base_model.layers:
        layer.trainable = False

    # Step 4: Binary classification head
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = layers.Dropout(dropout_rate, name="dropout_head")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="dense_head")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)

    # Step 5: Compile with binary crossentropy and standard evaluation metrics
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.BinaryCrossentropy()
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc", curve="ROC")
        ]
    )

    total_params = int(model.count_params())
    trainable_params = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
    logger.info(
        f"MobileNetV2 built successfully: Total params={total_params:,}, "
        f"Trainable params={trainable_params:,} (Phase 1 Frozen Base)"
    )
    return model


def setup_phase2_fine_tuning(
    model: keras.Model,
    unfreeze_from_layer: str = "block_16_expand",
    learning_rate: float = config.FINE_TUNE_LEARNING_RATE
) -> keras.Model:
    """
    Reconfigures the MobileNetV2 transfer model for Phase 2: Limited Fine-Tuning.
    
    Rules enforced:
    - Base layers prior to unfreeze_from_layer remain FROZEN.
    - Convolutional/Depthwise layers from unfreeze_from_layer onwards are UNFROZEN.
    - ALL BatchNormalization layers remain STRICTLY FROZEN across the entire model
      to avoid corrupting pretrained batch statistics on small medical batches.
    - Recompiles model with a conservative fine-tuning learning rate (~1e-5).
    """
    logger.info(
        f"Configuring Phase 2 Fine-Tuning: unfreezing from '{unfreeze_from_layer}' "
        f"with lr={learning_rate} (BatchNormalization strictly frozen)..."
    )

    unfreeze = False
    unfrozen_layers: List[str] = []
    frozen_bn_layers: List[str] = []

    # Iterate over all layers in the model
    for layer in model.layers:
        if layer.name == unfreeze_from_layer:
            unfreeze = True

        is_bn = isinstance(layer, (keras.layers.BatchNormalization, tf.keras.layers.BatchNormalization)) or "BatchNormalization" in layer.__class__.__name__

        if unfreeze:
            if is_bn:
                layer.trainable = False
                frozen_bn_layers.append(layer.name)
            else:
                layer.trainable = True
                unfrozen_layers.append(layer.name)
        else:
            layer.trainable = False
            if is_bn:
                frozen_bn_layers.append(layer.name)

    # Recompile after altering trainable statuses
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.BinaryCrossentropy()
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc", curve="ROC")
        ]
    )

    total_params = int(model.count_params())
    trainable_params = int(sum(tf.size(w).numpy() for w in model.trainable_weights))
    logger.info(
        f"Phase 2 compiled: Total params={total_params:,}, Trainable params={trainable_params:,}, "
        f"Unfrozen layers={len(unfrozen_layers)}, Frozen BNs={len(frozen_bn_layers)}"
    )
    return model
