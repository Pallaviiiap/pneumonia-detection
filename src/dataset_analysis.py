"""
Comprehensive Dataset Analysis & Quality Auditing Module for Pneumonia Detection.
Analyzes directory structure, counts, class distributions, file formats, image dimensions,
corruptions, hash duplicates, data leakage, and generates publication-grade visualizations.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / script execution
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from src.config import config
from src.utils import get_logger, save_json

logger = get_logger("dataset_analysis")


class DatasetAuditor:
    """
    Comprehensive auditor for the Chest X-Ray (Pneumonia) dataset.
    """

    def __init__(self, dataset_dir: Optional[Path] = None):
        self.dataset_dir = Path(dataset_dir) if dataset_dir else config.DATASET_DIR
        self.splits = ["train", "val", "test"]
        self.classes = list(config.CLASS_NAMES)
        
        # Results containers
        self.split_counts: Dict[str, Dict[str, int]] = {s: {c: 0 for c in self.classes} for s in self.splits}
        self.format_counts: Counter = Counter()
        self.pil_format_counts: Counter = Counter()
        self.pil_mode_counts: Counter = Counter()
        self.corrupted_files: List[Dict[str, str]] = []
        
        self.image_records: List[Dict[str, Any]] = []
        self.hash_to_files: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        self.patient_ids_by_split: Dict[str, Dict[str, set]] = {
            s: {c: set() for c in self.classes} for s in self.splits
        }
        self.unexpected_items: List[Dict[str, Any]] = []

    def scan_unexpected_items(self) -> List[Dict[str, Any]]:
        """
        Scans for unexpected archives, nested folders, macOS artifacts, and hidden files.
        """
        logger.info("Scanning for unexpected files and directory artifacts...")
        raw_root = config.RAW_DATASET_DIR
        unexpected = []

        if raw_root.exists():
            for item in raw_root.iterdir():
                if item.name == ".gitkeep":
                    unexpected.append({
                        "path": str(item.relative_to(config.PROJECT_ROOT)),
                        "type": "placeholder_file",
                        "description": "Git repository placeholder"
                    })
                elif item.is_dir() and item.name == "chest_xray":
                    # Check inside chest_xray for macOS and nested folders
                    for sub_item in item.iterdir():
                        if sub_item.name == "__MACOSX":
                            file_count = sum(1 for _ in sub_item.rglob("*"))
                            unexpected.append({
                                "path": str(sub_item.relative_to(config.PROJECT_ROOT)),
                                "type": "os_metadata_dir",
                                "description": f"macOS resource fork directory containing {file_count} '._*' files"
                            })
                        elif sub_item.name == "chest_xray" and sub_item.is_dir():
                            unexpected.append({
                                "path": str(sub_item.relative_to(config.PROJECT_ROOT)),
                                "type": "redundant_nested_archive",
                                "description": "Duplicate nested archive directory from Kaggle extraction"
                            })

        # Scan for .DS_Store or non-images in dataset tree
        for root, dirs, files in os.walk(raw_root):
            for file in files:
                if file == ".DS_Store" or file.startswith("._"):
                    rel = Path(root, file).relative_to(config.PROJECT_ROOT)
                    unexpected.append({
                        "path": str(rel),
                        "type": "os_metadata_file",
                        "description": f"OS metadata file: {file}"
                    })

        self.unexpected_items = unexpected
        logger.info(f"Identified {len(unexpected)} unexpected files/artifacts across dataset tree.")
        return unexpected

    def _extract_patient_id(self, filename: str) -> str:
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

    def audit_images(self) -> None:
        """
        Performs thorough per-image verification, dimension calculation, hash computation,
        and corruption testing across canonical train, val, and test splits.
        """
        logger.info(f"Auditing canonical dataset directory: {self.dataset_dir}")

        for split in self.splits:
            split_dir = self.dataset_dir / split
            if not split_dir.exists():
                raise FileNotFoundError(f"Missing split directory: {split_dir}")

            for cls in self.classes:
                cls_dir = split_dir / cls
                if not cls_dir.exists():
                    raise FileNotFoundError(f"Missing class directory: {cls_dir}")

                files = sorted([f for f in cls_dir.iterdir() if f.is_file()])
                self.split_counts[split][cls] = len(files)
                logger.info(f"Scanning {split}/{cls}: {len(files)} files...")

                for p in files:
                    ext = p.suffix.lower()
                    self.format_counts[ext] += 1
                    file_size_bytes = p.stat().st_size
                    patient_id = self._extract_patient_id(p.name)
                    self.patient_ids_by_split[split][cls].add(patient_id)

                    # Compute SHA-256 for exact duplicate & leakage detection
                    hasher = hashlib.sha256()
                    try:
                        with open(p, "rb") as f:
                            while chunk := f.read(65536):
                                hasher.update(chunk)
                        sha256_hash = hasher.hexdigest()
                    except Exception as e:
                        self.corrupted_files.append({"filepath": str(p), "error": f"IO read error: {e}"})
                        continue

                    # PIL integrity verification and dimension extraction
                    is_corrupt = False
                    pil_fmt = "UNKNOWN"
                    pil_mode = "UNKNOWN"
                    width, height = 0, 0
                    
                    try:
                        with Image.open(p) as img:
                            pil_fmt = img.format or "UNKNOWN"
                            pil_mode = img.mode or "UNKNOWN"
                            width, height = img.size
                            img.verify()

                        # Re-open and decode full pixel buffer to detect truncated / broken JPEG data
                        with Image.open(p) as img:
                            img.load()

                        self.pil_format_counts[pil_fmt] += 1
                        self.pil_mode_counts[pil_mode] += 1

                    except Exception as e:
                        is_corrupt = True
                        self.corrupted_files.append({"filepath": str(p), "error": f"Image decode error: {e}"})

                    if not is_corrupt:
                        aspect_ratio = round(width / height, 4) if height > 0 else 0.0
                        record = {
                            "filepath": str(p),
                            "filename": p.name,
                            "split": split,
                            "class_name": cls,
                            "label": config.LABEL_MAPPING[cls],
                            "width": width,
                            "height": height,
                            "aspect_ratio": aspect_ratio,
                            "file_size_kb": round(file_size_bytes / 1024, 2),
                            "pil_format": pil_fmt,
                            "pil_mode": pil_mode,
                            "sha256": sha256_hash,
                            "patient_id": patient_id
                        }
                        self.image_records.append(record)
                        self.hash_to_files[sha256_hash].append({
                            "split": split,
                            "class_name": cls,
                            "filename": p.name,
                            "filepath": str(p)
                        })

        logger.info(f"Audit completed: {len(self.image_records)} valid images, {len(self.corrupted_files)} corrupted.")

    def analyze_duplicates_and_leakage(self) -> Dict[str, Any]:
        """
        Analyzes hash-based duplicate files and patient ID cross-split overlap.
        """
        duplicate_groups = {h: files for h, files in self.hash_to_files.items() if len(files) > 1}
        intra_split_dups: List[Dict[str, Any]] = []
        cross_split_leakage: List[Dict[str, Any]] = []

        for h, files in duplicate_groups.items():
            splits_present = set(f["split"] for f in files)
            entry = {
                "sha256": h,
                "occurrences_count": len(files),
                "splits": list(splits_present),
                "files": files
            }
            if len(splits_present) > 1:
                cross_split_leakage.append(entry)
            else:
                intra_split_dups.append(entry)

        # Patient ID cross-split overlap analysis
        patient_overlap = {}
        for cls in self.classes:
            train_pts = self.patient_ids_by_split["train"][cls]
            val_pts = self.patient_ids_by_split["val"][cls]
            test_pts = self.patient_ids_by_split["test"][cls]

            train_val = sorted(list(train_pts & val_pts))
            train_test = sorted(list(train_pts & test_pts))
            val_test = sorted(list(val_pts & test_pts))

            patient_overlap[cls] = {
                "unique_patients_train": len(train_pts),
                "unique_patients_val": len(val_pts),
                "unique_patients_test": len(test_pts),
                "train_val_overlap_count": len(train_val),
                "train_val_overlap_patients": train_val,
                "train_test_overlap_count": len(train_test),
                "train_test_overlap_patients_sample": train_test[:10],
                "val_test_overlap_count": len(val_test),
                "val_test_overlap_patients": val_test,
            }

        return {
            "total_duplicate_groups": len(duplicate_groups),
            "intra_split_duplicates_count": len(intra_split_dups),
            "intra_split_duplicates": intra_split_dups,
            "cross_split_leakage_count": len(cross_split_leakage),
            "cross_split_leakage": cross_split_leakage,
            "patient_level_overlap": patient_overlap
        }

    def compute_statistics(self) -> Dict[str, Any]:
        """
        Calculates all required numerical statistics and dimensions metrics.
        """
        df = pd.DataFrame(self.image_records)
        total_images = len(df)

        # Split distributions
        split_summary = {}
        overall_counts = {cls: 0 for cls in self.classes}

        for split in self.splits:
            sub = df[df["split"] == split]
            split_total = len(sub)
            counts = sub["class_name"].value_counts().to_dict()
            for cls in self.classes:
                counts.setdefault(cls, 0)
                overall_counts[cls] += counts[cls]

            split_summary[split] = {
                "total": split_total,
                "percentage_of_dataset": round(split_total / total_images * 100, 2) if total_images > 0 else 0.0,
                "counts": counts,
                "class_percentages": {
                    cls: round(c / split_total * 100, 2) if split_total > 0 else 0.0
                    for cls, c in counts.items()
                }
            }

        overall_distribution = {
            "counts": overall_counts,
            "percentages": {
                cls: round(c / total_images * 100, 2) if total_images > 0 else 0.0
                for cls, c in overall_counts.items()
            }
        }

        # Dimension statistics
        widths = df["width"].values
        heights = df["height"].values
        aspect_ratios = df["aspect_ratio"].values
        file_sizes = df["file_size_kb"].values

        dim_pairs = list(zip(df["width"], df["height"]))
        dim_counter = Counter(dim_pairs)
        common_dims = [
            {"width": int(k[0]), "height": int(k[1]), "count": int(v)}
            for k, v in dim_counter.most_common(10)
        ]

        dimensions_stats = {
            "width": {
                "min": int(np.min(widths)),
                "max": int(np.max(widths)),
                "mean": float(round(np.mean(widths), 2)),
                "median": float(round(np.median(widths), 2)),
                "std": float(round(np.std(widths), 2))
            },
            "height": {
                "min": int(np.min(heights)),
                "max": int(np.max(heights)),
                "mean": float(round(np.mean(heights), 2)),
                "median": float(round(np.median(heights), 2)),
                "std": float(round(np.std(heights), 2))
            },
            "aspect_ratio": {
                "min": float(round(np.min(aspect_ratios), 4)),
                "max": float(round(np.max(aspect_ratios), 4)),
                "mean": float(round(np.mean(aspect_ratios), 4)),
                "median": float(round(np.median(aspect_ratios), 4)),
                "std": float(round(np.std(aspect_ratios), 4))
            },
            "file_size_kb": {
                "min": float(round(np.min(file_sizes), 2)),
                "max": float(round(np.max(file_sizes), 2)),
                "mean": float(round(np.mean(file_sizes), 2)),
                "median": float(round(np.median(file_sizes), 2))
            },
            "top_common_dimensions": common_dims
        }

        # Formats and modes
        formats_stats = {
            "file_extensions": dict(self.format_counts),
            "pil_formats": dict(self.pil_format_counts),
            "pil_color_modes": dict(self.pil_mode_counts)
        }

        # Duplicates and leakage
        dup_leakage = self.analyze_duplicates_and_leakage()

        return {
            "dataset_path": str(self.dataset_dir),
            "total_images": total_images,
            "corrupted_images_count": len(self.corrupted_files),
            "corrupted_files": self.corrupted_files,
            "splits": split_summary,
            "overall_distribution": overall_distribution,
            "dimensions_statistics": dimensions_stats,
            "format_statistics": formats_stats,
            "duplicates_and_leakage": dup_leakage,
            "unexpected_items_count": len(self.unexpected_items),
            "unexpected_items": self.unexpected_items
        }

    def save_metrics(self, stats: Dict[str, Any]) -> Tuple[Path, Path]:
        """
        Saves numerical analysis to JSON and summary to CSV in outputs/metrics/.
        """
        metrics_dir = config.METRICS_DIR
        metrics_dir.mkdir(parents=True, exist_ok=True)

        json_path = metrics_dir / "dataset_analysis.json"
        save_json(stats, json_path)
        logger.info(f"Saved full dataset analysis JSON to {json_path}")

        # Flatten split summary into tabular CSV
        rows = []
        for split, data in stats["splits"].items():
            for cls in self.classes:
                rows.append({
                    "split": split,
                    "class": cls,
                    "count": data["counts"][cls],
                    "class_percentage_in_split": data["class_percentages"][cls],
                    "split_total": data["total"],
                    "dataset_total": stats["total_images"]
                })
        csv_path = metrics_dir / "dataset_summary.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"Saved dataset summary CSV to {csv_path}")

        return json_path, csv_path

    # -------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------

    def plot_class_distribution(self, stats: Dict[str, Any]) -> Path:
        """
        Generates grouped bar chart showing class counts and percentages across splits.
        """
        plots_dir = config.PLOTS_DIR
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = plots_dir / "class_distribution.png"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
        fig.suptitle("Chest X-Ray Dataset Class Distribution & Imbalance Analysis", fontsize=15, fontweight="bold", y=1.02)

        # 1. Split counts
        splits = ["train", "val", "test"]
        x = np.arange(len(splits))
        width = 0.35

        normal_counts = [stats["splits"][s]["counts"]["NORMAL"] for s in splits]
        pneumonia_counts = [stats["splits"][s]["counts"]["PNEUMONIA"] for s in splits]

        rects1 = ax1.bar(x - width/2, normal_counts, width, label="NORMAL", color="#2b5c8f", edgecolor="black", linewidth=0.8)
        rects2 = ax1.bar(x + width/2, pneumonia_counts, width, label="PNEUMONIA", color="#d95f02", edgecolor="black", linewidth=0.8)

        ax1.set_ylabel("Number of Images", fontsize=12, fontweight="semibold")
        ax1.set_title("Image Counts by Split", fontsize=13, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{s.upper()}\n(Total: {stats['splits'][s]['total']})" for s in splits], fontsize=11)
        ax1.legend(frameon=True, facecolor="#f8f9fa")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        # Bar value annotations
        for rect in rects1:
            h = rect.get_height()
            ax1.annotate(f"{h}", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax1.annotate(f"{h}", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # 2. Overall proportion pie chart
        overall = stats["overall_distribution"]
        labels = [f"NORMAL\n{overall['counts']['NORMAL']} ({overall['percentages']['NORMAL']}%)",
                  f"PNEUMONIA\n{overall['counts']['PNEUMONIA']} ({overall['percentages']['PNEUMONIA']}%)"]
        colors = ["#2b5c8f", "#d95f02"]
        explode = (0.05, 0)

        ax2.pie([overall['counts']['NORMAL'], overall['counts']['PNEUMONIA']],
                explode=explode, labels=labels, colors=colors, autopct="",
                startangle=140, textprops={'fontsize': 11, 'fontweight': 'bold'},
                wedgeprops={'edgecolor': 'black', 'linewidth': 0.8})
        ax2.set_title(f"Overall Class Distribution\n(Total: {stats['total_images']} Images)", fontsize=13, fontweight="bold")

        plt.figtext(0.5, -0.02, config.DISCLAIMER, ha="center", fontsize=8, color="#555555", style="italic")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved class distribution plot to {save_path}")
        return save_path

    def plot_sample_images(self, cls_name: str, save_filename: str, grid: Tuple[int, int] = (3, 3)) -> Path:
        """
        Generates a high-resolution grid of sample chest radiographs for a given class.
        """
        plots_dir = config.PLOTS_DIR
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = plots_dir / save_filename

        cls_records = [r for r in self.image_records if r["class_name"] == cls_name and r["split"] == "train"]
        if not cls_records:
            cls_records = [r for r in self.image_records if r["class_name"] == cls_name]

        # Deterministic sample using seed
        np.random.seed(config.RANDOM_SEED)
        num_samples = grid[0] * grid[1]
        indices = np.random.choice(len(cls_records), size=min(num_samples, len(cls_records)), replace=False)
        samples = [cls_records[i] for i in indices]

        fig, axes = plt.subplots(grid[0], grid[1], figsize=(12, 12), dpi=300)
        title_cls = f"Sample Chest Radiographs: {cls_name} Class (Train Set)"
        fig.suptitle(title_cls, fontsize=15, fontweight="bold", y=0.98)

        for idx, ax in enumerate(axes.flat):
            if idx < len(samples):
                item = samples[idx]
                with Image.open(item["filepath"]) as img:
                    ax.imshow(img, cmap="gray")
                
                # Determine subtype if pneumonia
                subtype = ""
                if "bacteria" in item["filename"].lower():
                    subtype = "[Bacterial]"
                elif "virus" in item["filename"].lower():
                    subtype = "[Viral]"
                
                title_text = f"{item['filename'][:22]}...\n{subtype} {item['width']}x{item['height']} | AR: {item['aspect_ratio']}"
                ax.set_title(title_text.strip(), fontsize=9, fontweight="semibold")
            ax.axis("off")

        plt.figtext(0.5, 0.01, config.DISCLAIMER, ha="center", fontsize=8, color="#555555", style="italic")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {cls_name} sample grid to {save_path}")
        return save_path

    def plot_dimension_distributions(self, stats: Dict[str, Any]) -> Path:
        """
        Generates multi-panel dimension distributions: Width vs Height scatter,
        Width histogram, Height histogram, and Aspect Ratio distribution.
        """
        plots_dir = config.PLOTS_DIR
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_path = plots_dir / "image_dimensions_distribution.png"

        df = pd.DataFrame(self.image_records)

        fig = plt.figure(figsize=(15, 10), dpi=300)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        fig.suptitle("Chest Radiograph Image Dimensions & Aspect Ratio Analysis", fontsize=16, fontweight="bold", y=0.96)

        # 1. Width vs Height Scatter Plot colored by class
        ax1 = fig.add_subplot(gs[0, 0])
        sns.scatterplot(
            data=df, x="width", y="height", hue="class_name", alpha=0.4,
            palette={"NORMAL": "#2b5c8f", "PNEUMONIA": "#d95f02"},
            ax=ax1, s=25
        )
        ax1.axvline(224, color="red", linestyle="--", linewidth=1.2, label="Target 224x224")
        ax1.axhline(224, color="red", linestyle="--", linewidth=1.2)
        ax1.set_title("Resolution Scatter: Width vs Height", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Width (pixels)", fontsize=11)
        ax1.set_ylabel("Height (pixels)", fontsize=11)
        ax1.legend(loc="upper left", frameon=True)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Aspect Ratio Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(data=df, x="aspect_ratio", hue="class_name", kde=True, bins=40,
                     palette={"NORMAL": "#2b5c8f", "PNEUMONIA": "#d95f02"}, ax=ax2, alpha=0.5)
        mean_ar = stats["dimensions_statistics"]["aspect_ratio"]["mean"]
        ax2.axvline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Square (1.0)")
        ax2.axvline(mean_ar, color="black", linestyle="-.", linewidth=1.5, label=f"Mean AR ({mean_ar:.2f})")
        ax2.set_title("Aspect Ratio (Width / Height) Distribution", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Aspect Ratio", fontsize=11)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.legend(loc="upper right", frameon=True)
        ax2.grid(True, linestyle="--", alpha=0.5)

        # 3. Width Distribution Boxplot by Split
        ax3 = fig.add_subplot(gs[1, 0])
        sns.boxplot(data=df, x="split", y="width", hue="class_name",
                    palette={"NORMAL": "#2b5c8f", "PNEUMONIA": "#d95f02"}, ax=ax3)
        ax3.set_title("Image Width Variation Across Splits", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Split", fontsize=11)
        ax3.set_ylabel("Width (pixels)", fontsize=11)
        ax3.grid(True, linestyle="--", alpha=0.5)

        # 4. Height Distribution Boxplot by Split
        ax4 = fig.add_subplot(gs[1, 1])
        sns.boxplot(data=df, x="split", y="height", hue="class_name",
                    palette={"NORMAL": "#2b5c8f", "PNEUMONIA": "#d95f02"}, ax=ax4)
        ax4.set_title("Image Height Variation Across Splits", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Split", fontsize=11)
        ax4.set_ylabel("Height (pixels)", fontsize=11)
        ax4.grid(True, linestyle="--", alpha=0.5)

        plt.figtext(0.5, 0.01, config.DISCLAIMER, ha="center", fontsize=8, color="#555555", style="italic")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved dimensions distribution plot to {save_path}")
        return save_path

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Executes the entire auditing, metrics calculation, and visualization generation workflow.
        """
        self.scan_unexpected_items()
        self.audit_images()
        stats = self.compute_statistics()
        
        # Save metrics
        self.save_metrics(stats)
        
        # Create plots
        self.plot_class_distribution(stats)
        self.plot_sample_images("NORMAL", "sample_normal_xrays.png")
        self.plot_sample_images("PNEUMONIA", "sample_pneumonia_xrays.png")
        self.plot_dimension_distributions(stats)
        
        logger.info("Stage 2 full dataset audit pipeline completed successfully!")
        return stats


def main():
    auditor = DatasetAuditor()
    stats = auditor.run_full_pipeline()
    print("\n" + "=" * 65)
    print("STAGE 2 AUDIT SUMMARY")
    print("=" * 65)
    print(f"Dataset Path:            {stats['dataset_path']}")
    print(f"Total Images:            {stats['total_images']}")
    print(f"Corrupted Images:        {stats['corrupted_images_count']}")
    print("\nSplit Counts:")
    for split in ["train", "val", "test"]:
        s = stats["splits"][split]
        print(f"  - {split.upper():<5}: {s['total']} images (NORMAL: {s['counts']['NORMAL']}, PNEUMONIA: {s['counts']['PNEUMONIA']})")
    print("\nOverall Distribution:")
    ov = stats["overall_distribution"]
    print(f"  - NORMAL:    {ov['counts']['NORMAL']} ({ov['percentages']['NORMAL']}%)")
    print(f"  - PNEUMONIA: {ov['counts']['PNEUMONIA']} ({ov['percentages']['PNEUMONIA']}%)")
    print("\nDimension Summary:")
    dims = stats["dimensions_statistics"]
    print(f"  - Width:  min={dims['width']['min']}, max={dims['width']['max']}, mean={dims['width']['mean']}, median={dims['width']['median']}")
    print(f"  - Height: min={dims['height']['min']}, max={dims['height']['max']}, mean={dims['height']['mean']}, median={dims['height']['median']}")
    print(f"  - Aspect Ratio (w/h): min={dims['aspect_ratio']['min']}, max={dims['aspect_ratio']['max']}, mean={dims['aspect_ratio']['mean']}")
    print("\nIntegrity & Leakage:")
    dup = stats["duplicates_and_leakage"]
    print(f"  - Exact Duplicate Groups: {dup['total_duplicate_groups']}")
    print(f"  - Cross-Split Duplicate Leakage: {dup['cross_split_leakage_count']}")
    print(f"  - Intra-Split Duplicate Groups: {dup['intra_split_duplicates_count']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
