"""
Data Loading and Batching Module for Pneumonia Detection.
Provides memory-efficient, streaming tf.data.Dataset pipelines for CPU-only environments.
Ensures strict test-set isolation, dynamic RGB channel expansion, and on-the-fly resizing,
with training-only data augmentation, deterministic evaluation preprocessing, and
clinically rigorous patient-aware validation partitioning.
"""

import re
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit

from src.config import config
from src.utils import get_logger
from src.preprocessing import build_augmentation_layers, build_normalization_layer

logger = get_logger("data_loader")


def extract_patient_id(filename: str) -> str:
    """
    Extracts patient identifier from chest radiograph filename.
    Handles standard Kaggle Pneumonia naming patterns:
    - person100_bacteria_475.jpeg -> patient ID 'person100'
    - person1000_virus_1680.jpeg -> patient ID 'person1000'
    - NORMAL2-IM-0381-0001.jpeg -> patient series 'NORMAL2-IM-0381'
    - IM-0115-0001.jpeg -> patient series 'IM-0115'
    """
    m_pneumonia = re.match(r"(person\d+)_", filename)
    if m_pneumonia:
        return m_pneumonia.group(1)
    m_normal_prefix1 = re.match(r"(NORMAL2-IM-\d+)", filename)
    if m_normal_prefix1:
        return m_normal_prefix1.group(1)
    m_normal_prefix2 = re.match(r"(IM-\d+)", filename)
    if m_normal_prefix2:
        return m_normal_prefix2.group(1)
    return filename.split(".")[0]


def scan_dataset(dataset_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Scans the dataset directory structure and returns DataFrames for train, val, and test splits.
    Each DataFrame contains: [filepath, split, class_name, label, filename, file_size_kb, patient_id]
    """
    root = Path(dataset_dir) if dataset_dir else config.DATASET_DIR
    splits = ["train", "val", "test"]
    data: Dict[str, List[Dict[str, Any]]] = {s: [] for s in splits}

    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Required split directory does not exist: {split_dir}")

        for cls_name in config.CLASS_NAMES:
            cls_dir = split_dir / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(f"Required class directory does not exist: {cls_dir}")

            label = config.LABEL_MAPPING[cls_name]
            for file_path in sorted(cls_dir.glob("*.*")):
                if file_path.is_file() and file_path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    data[split].append({
                        "filepath": str(file_path.resolve()),
                        "filename": file_path.name,
                        "split": split,
                        "class_name": cls_name,
                        "label": label,
                        "file_size_kb": round(file_path.stat().st_size / 1024, 2),
                        "patient_id": extract_patient_id(file_path.name)
                    })

    dfs = {s: pd.DataFrame(data[s]) for s in splits}
    for s, df in dfs.items():
        logger.info(f"Loaded {len(df)} samples for split '{s}'")
    return dfs


def create_patient_aware_validation_split(
    train_df: pd.DataFrame,
    val_ratio: float = config.VAL_SPLIT_RATIO,
    seed: int = config.RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partitions the training set into development training and validation subsets using
    Patient-Aware Group Splitting (GroupShuffleSplit).
    Ensures zero patient ID overlap between dev_train and dev_val.
    """
    df = train_df.copy()
    if "patient_id" not in df.columns:
        df["patient_id"] = df["filename"].apply(extract_patient_id)

    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df["patient_id"]))

    dev_train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    dev_val_df = df.iloc[val_idx].copy().reset_index(drop=True)

    # Verification: assert zero patient overlap
    overlap = set(dev_train_df["patient_id"]) & set(dev_val_df["patient_id"])
    if overlap:
        raise ValueError(f"Patient overlap detected during validation split: {overlap}")

    logger.info(
        f"Created patient-aware validation split: dev_train={len(dev_train_df)}, "
        f"dev_val={len(dev_val_df)} (val_ratio={val_ratio}, patient_overlap=0)"
    )
    return dev_train_df, dev_val_df


def get_dataset_summary(splits_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Computes summary counts and class distributions across all splits.
    """
    summary: Dict[str, Any] = {"splits": {}, "total_images": 0, "overall_distribution": {}}
    total_images = sum(len(df) for df in splits_dict.values())
    summary["total_images"] = total_images

    overall_counts = {cls: 0 for cls in config.CLASS_NAMES}

    for split, df in splits_dict.items():
        split_counts = df["class_name"].value_counts().to_dict()
        for cls in config.CLASS_NAMES:
            split_counts.setdefault(cls, 0)
            overall_counts[cls] += split_counts[cls]

        split_total = len(df)
        summary["splits"][split] = {
            "total": split_total,
            "counts": split_counts,
            "percentages": {
                cls: round((count / split_total * 100), 2) if split_total > 0 else 0.0
                for cls, count in split_counts.items()
            }
        }

    summary["overall_distribution"] = {
        "counts": overall_counts,
        "percentages": {
            cls: round((count / total_images * 100), 2) if total_images > 0 else 0.0
            for cls, count in overall_counts.items()
        }
    }
    return summary


def compute_class_weights_dict(train_df: pd.DataFrame) -> Dict[int, float]:
    """
    Computes balanced class weights for training data to address class imbalance.
    Returns: {0: weight_for_normal, 1: weight_for_pneumonia}
    """
    labels = train_df["label"].values
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    class_weights = {int(c): float(round(w, 4)) for c, w in zip(classes, weights)}
    logger.info(f"Computed balanced class weights: {class_weights}")
    return class_weights


def load_raw_image_tensor(
    file_path: tf.Tensor,
    label: tf.Tensor,
    image_size: Tuple[int, int] = config.IMAGE_SIZE
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Reads image bytes from disk, decodes JPEG as 3 channels (converting grayscale automatically),
    and resizes to target image_size (default: 224x224).
    Outputs raw un-normalized float32 tensor in range [0.0, 255.0].
    """
    img_bytes = tf.io.read_file(file_path)
    # decode_jpeg handles JPEG and converts 1-channel grayscale to 3 channels
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, image_size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img, label


def create_tf_dataset(
    df: pd.DataFrame,
    batch_size: int = config.BATCH_SIZE,
    image_size: Tuple[int, int] = config.IMAGE_SIZE,
    shuffle: bool = False,
    augment: bool = False,
    normalize: bool = True,
    normalization_mode: str = "rescaling_0_1",
    seed: int = config.RANDOM_SEED,
    repeat: bool = False,
    config_obj=config
) -> tf.data.Dataset:
    """
    Creates a memory-efficient tf.data.Dataset from a manifest DataFrame.
    Images are loaded lazily on demand batch-by-batch without holding the dataset in RAM.
    """
    file_paths = df["filepath"].values
    labels = df["label"].values

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 1024), seed=seed, reshuffle_each_iteration=True)

    if repeat:
        ds = ds.repeat()

    # Step 1: Read, decode 3 channels, resize to 224x224
    ds = ds.map(
        lambda p, l: load_raw_image_tensor(p, l, image_size=image_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Step 2: Batching
    ds = ds.batch(batch_size, drop_remainder=False)

    # Step 3: Augmentation (applied only if augment=True, e.g. training set)
    if augment and config_obj.AUGMENTATION_ENABLED:
        aug_model = build_augmentation_layers(config_obj=config_obj)
        ds = ds.map(
            lambda bx, by: (aug_model(bx, training=True), by),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Step 4: Normalization (rescaling_0_1: [0, 1] for baseline CNN; mobilenetv2: [-1, 1] for MobileNetV2)
    if normalize:
        norm_layer = build_normalization_layer(mode=normalization_mode, config_obj=config_obj)
        ds = ds.map(
            lambda bx, by: (norm_layer(bx), by),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # Step 5: Prefetch
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def get_data_loaders(
    dataset_dir: Optional[Path] = None,
    batch_size: int = config.BATCH_SIZE,
    image_size: Tuple[int, int] = config.IMAGE_SIZE,
    seed: int = config.RANDOM_SEED,
    augment_train: bool = config.AUGMENTATION_ENABLED,
    normalize: bool = True,
    normalization_mode: str = "rescaling_0_1",
    validation_strategy: str = config.VALIDATION_STRATEGY,
    val_split_ratio: float = config.VAL_SPLIT_RATIO
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[str, Any]]:
    """
    Factory to instantiate train, validation, and test tf.data.Datasets.
    Supports either 'patient_aware_split' (derived from train, 0 patient overlap)
    or 'official' (official 16-image validation directory).
    Strictly preserves test-set separation and deterministic validation/test preprocessing.
    """
    splits = scan_dataset(dataset_dir)
    official_val_df = splits["val"]
    test_df = splits["test"]

    if validation_strategy == "patient_aware_split":
        dev_train_df, dev_val_df = create_patient_aware_validation_split(
            splits["train"], val_ratio=val_split_ratio, seed=seed
        )
        active_train_df = dev_train_df
        active_val_df = dev_val_df
    elif validation_strategy == "official":
        active_train_df = splits["train"]
        active_val_df = official_val_df
    else:
        raise ValueError(f"Unknown validation strategy: {validation_strategy}")

    class_weights = compute_class_weights_dict(active_train_df)
    summary = get_dataset_summary(splits)

    # 1. Training dataset
    train_ds = create_tf_dataset(
        active_train_df,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        augment=augment_train,
        normalize=normalize,
        normalization_mode=normalization_mode,
        seed=seed
    )

    # 2. Validation dataset: strictly deterministic, NO augmentation, NO shuffle
    val_ds = create_tf_dataset(
        active_val_df,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        augment=False,
        normalize=normalize,
        normalization_mode=normalization_mode,
        seed=seed
    )

    # 3. Test dataset: strictly isolated, deterministic, NO augmentation, NO shuffle
    test_ds = create_tf_dataset(
        test_df,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        augment=False,
        normalize=normalize,
        normalization_mode=normalization_mode,
        seed=seed
    )

    # Also build loader for official 16-image validation split so it is always accessible
    official_val_ds = create_tf_dataset(
        official_val_df,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        augment=False,
        normalize=normalize,
        normalization_mode=normalization_mode,
        seed=seed
    )

    metadata = {
        "manifests": splits,
        "summary": summary,
        "class_weights": class_weights,
        "batch_size": batch_size,
        "image_size": image_size,
        "validation_strategy": validation_strategy,
        "active_train_count": len(active_train_df),
        "active_val_count": len(active_val_df),
        "test_count": len(test_df),
        "official_val_count": len(official_val_df),
        "official_val_ds": official_val_ds,
        "active_train_df": active_train_df,
        "active_val_df": active_val_df,
        "normalization_mode": normalization_mode,
        "augmentation_applied_to_train": augment_train,
        "augmentation_applied_to_val": False,
        "augmentation_applied_to_test": False,
    }

    return train_ds, val_ds, test_ds, metadata
