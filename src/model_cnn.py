"""
Baseline Convolutional Neural Network (CNN) for Binary Pneumonia Detection.
A lightweight, 4-block custom CNN architecture designed for CPU-only execution
with BatchNormalization, GlobalAveragePooling, and dedicated Grad-CAM target layers.
"""

from typing import Tuple, Optional
import io
import tensorflow as tf
import keras
from keras import layers

from src.config import config
from src.utils import get_logger

logger = get_logger("model_cnn")


def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
    learning_rate: float = config.LEARNING_RATE,
    dropout_rate: float = 0.4,
    compile_model: bool = True,
    name: str = "baseline_cnn"
) -> keras.Model:
    """
    Constructs and compiles the custom Baseline CNN architecture.
    
    Architecture:
    - Input: (224, 224, 3)
    - Block 1: Conv2D(32, 3x3) + BN + ReLU + MaxPool(2x2) -> (112, 112, 32)
    - Block 2: Conv2D(64, 3x3) + BN + ReLU + MaxPool(2x2) -> (56, 56, 64)
    - Block 3: Conv2D(128, 3x3) + BN + ReLU + MaxPool(2x2) -> (28, 28, 128)
    - Block 4: Conv2D(128, 3x3) + BN + ReLU + MaxPool(2x2) -> (14, 14, 128) [Grad-CAM Target: 'conv2d_block4_conv']
    - Classification Head: GlobalAveragePooling2D + Dense(64, ReLU) + Dropout(0.4) + Dense(1, Sigmoid)
    
    Total Parameters: ~250,561 (~978 KB) - ideal for CPU inference and training.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # -----------------------------
    # Convolutional Block 1
    # -----------------------------
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv2d_block1_conv")(inputs)
    x = layers.BatchNormalization(name="bn_block1")(x)
    x = layers.ReLU(name="relu_block1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_block1")(x)

    # -----------------------------
    # Convolutional Block 2
    # -----------------------------
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2d_block2_conv")(x)
    x = layers.BatchNormalization(name="bn_block2")(x)
    x = layers.ReLU(name="relu_block2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_block2")(x)

    # -----------------------------
    # Convolutional Block 3
    # -----------------------------
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv2d_block3_conv")(x)
    x = layers.BatchNormalization(name="bn_block3")(x)
    x = layers.ReLU(name="relu_block3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_block3")(x)

    # -----------------------------
    # Convolutional Block 4 (Target Layer for Grad-CAM Explainability)
    # -----------------------------
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv2d_block4_conv")(x)
    x = layers.BatchNormalization(name="bn_block4")(x)
    x = layers.ReLU(name="relu_block4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool_block4")(x)

    # -----------------------------
    # Classification Head
    # -----------------------------
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(64, activation="relu", name="dense_head")(x)
    x = layers.Dropout(dropout_rate, name="dropout_head", seed=config.RANDOM_SEED)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output_prediction")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=name)

    if compile_model:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = keras.losses.BinaryCrossentropy(name="binary_crossentropy")
        metrics = [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")
        ]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        logger.info(f"Model '{name}' compiled with Adam(lr={learning_rate}) and BinaryCrossentropy.")

    return model


def get_model_summary_str(model: keras.Model) -> str:
    """Returns the text summary of a Keras model."""
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    return stream.getvalue()


def main():
    model = build_baseline_cnn()
    print("=" * 65)
    print("BASELINE CNN MODEL ARCHITECTURE")
    print("=" * 65)
    print(get_model_summary_str(model))
    total_params = model.count_params()
    trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Target Grad-CAM Layer: {config.GRADCAM_TARGET_LAYERS['baseline_cnn']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
