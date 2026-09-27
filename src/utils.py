"""
Utility functions for Pneumonia Detection project.
Includes seed setting for reproducibility, logging setup, and I/O helpers.
"""

import os
import random
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, and TensorFlow for reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_logger(name: str = "pneumonia_detection") -> logging.Logger:
    """
    Configure and return a standardized console logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Save dictionary data to a JSON file, creating parent directories if needed.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_hardware_info() -> Dict[str, Any]:
    """
    Check and report available compute hardware (CPU / GPU).
    """
    info = {"gpu_available": False, "gpu_devices": []}
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        info["gpu_available"] = len(gpus) > 0
        info["gpu_devices"] = [gpu.name for gpu in gpus]
    except Exception as e:
        info["error"] = str(e)
    return info
