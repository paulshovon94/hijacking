"""
Modality analysis baseline using Random Forest classifiers.

This script mirrors exp_modality.py by evaluating incremental modality
combinations:
- x1
- x1+x2
- x1+x2+x3
- x1+x2+x3+x4
- x1+x2+x3+x4+x5
- x1+x2+x3+x4+x5+x6
- x1+x2+x3+x4+x5+x6+x7

For each combination, it trains one RandomForestClassifier per target head
and reports aggregated validation accuracy and macro-F1.
"""

import argparse
import ast
import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


TARGET_ORDER = ["model_family", "model_size", "optimizer", "learning_rate", "batch_size"]
LABEL_COLUMN_MAP = {
    "model_family": "model_family_label",
    "model_size": "model_size_label",
    "optimizer": "optimizer_label",
    "learning_rate": "learning_rate_label",
    "batch_size": "batch_size_label",
}
FEATURES_PER_MODALITY = 330


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _parse_label_vector(label_cell, expected_len: int) -> np.ndarray:
    if isinstance(label_cell, str):
        parsed = ast.literal_eval(label_cell)
    elif isinstance(label_cell, (list, tuple, np.ndarray)):
        parsed = label_cell
    else:
        raise ValueError(f"Unsupported label format: {type(label_cell)}")

    vector = np.asarray(parsed, dtype=np.float32).flatten()
    if vector.size != expected_len:
        raise ValueError(f"Expected label length {expected_len}, got {vector.size}")
    return vector


def _sanitize_feature_vector(feature_vector: np.ndarray, expected_size: int = FEATURES_PER_MODALITY) -> np.ndarray:
    feature_vector = np.asarray(feature_vector, dtype=np.float32).flatten()
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
    feature_vector = np.clip(feature_vector, -1e6, 1e6)
    if feature_vector.size < expected_size:
        feature_vector = np.pad(feature_vector, (0, expected_size - feature_vector.size))
    else:
        feature_vector = feature_vector[:expected_size]
    return feature_vector


def load_features_and_labels_from_dataloader(
    dataloader_path: str, label_mappings_path: str, show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[str]]]:
    dataloader_df = pd.read_csv(dataloader_path)
    with open(label_mappings_path, "r", encoding="utf-8") as handle:
        label_mappings = json.load(handle)

    for target in TARGET_ORDER:
        if target not in label_mappings:
            raise ValueError(f"Missing target mapping '{target}' in label_mappings.json")
        label_col = LABEL_COLUMN_MAP[target]
        if label_col not in dataloader_df.columns:
            raise ValueError(f"Missing label column '{label_col}' in dataloader.csv")

    required_feature_cols = [f"x{i}_file" for i in range(1, 8)]
    missing_feature_cols = [col for col in required_feature_cols if col not in dataloader_df.columns]
    if missing_feature_cols:
        raise ValueError(f"Missing feature columns: {missing_feature_cols}")

    features_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    skipped_entries = 0

    row_iterator = dataloader_df.iterrows()
    if show_progress:
        row_iterator = tqdm(
            row_iterator,
            total=len(dataloader_df),
            desc="Loading feature rows",
            unit="row",
        )

    for row_idx, row in row_iterator:
        try:
            feature_files = [row[f"x{i}_file"] for i in range(1, 8)]
            if any((not isinstance(path, str)) or (not path) or (not os.path.exists(path)) for path in feature_files):
                skipped_entries += 1
                continue

            chunks = []
            for file_path in feature_files:
                loaded = np.load(file_path)
                chunks.append(_sanitize_feature_vector(loaded, expected_size=FEATURES_PER_MODALITY))

            combined_features = np.concatenate(chunks)
            if combined_features.size < 2312:
                combined_features = np.pad(combined_features, (0, 2312 - combined_features.size))
            else:
                combined_features = combined_features[:2312]

            label_vectors = []
            for target in TARGET_ORDER:
                label_col = LABEL_COLUMN_MAP[target]
                expected_len = len(label_mappings[target])
                label_vectors.append(_parse_label_vector(row[label_col], expected_len))
            combined_labels = np.concatenate(label_vectors)

            features_list.append(combined_features.astype(np.float32))
            labels_list.append(combined_labels.astype(np.float32))
        except Exception as exc:
            logger.warning("Skipping row %s due to error: %s", row_idx, exc)
            skipped_entries += 1

    if not features_list:
        raise ValueError("No valid rows loaded from dataloader.csv")

    features = np.vstack(features_list)
    labels = np.vstack(labels_list)
    logger.info("Loaded %d samples, skipped %d samples", len(features), skipped_entries)
    logger.info("Feature shape: %s | Label shape: %s", features.shape, labels.shape)
    return features, labels, label_mappings


def build_label_indices(label_mappings: Dict[str, List[str]]) -> Dict[str, Tuple[int, int]]:
    label_indices: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for target in TARGET_ORDER:
        num_classes = len(label_mappings[target])
        label_indices[target] = (cursor, cursor + num_classes)
        cursor += num_classes
    return label_indices


def split_train_validation(
    features: np.ndarray, labels: np.ndarray, seed: int = 42, train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = features.shape[0]
    split_idx = int(train_ratio * n_samples)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    return features[train_idx], features[val_idx], labels[train_idx], labels[val_idx]


def select_modalities(features: np.ndarray, modality_indices: List[int]) -> np.ndarray:
    selected_chunks = []
    for modality_idx in modality_indices:
        start = modality_idx * FEATURES_PER_MODALITY
        end = start + FEATURES_PER_MODALITY
        selected_chunks.append(features[:, start:end])
    return np.concatenate(selected_chunks, axis=1)


def train_random_forest_heads(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    label_mappings: Dict[str, List[str]],
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    seed: int,
) -> Tuple[Dict[str, RandomForestClassifier], Dict[str, Dict[str, float]], Dict[str, Dict]]:
    label_indices = build_label_indices(label_mappings)
    head_models: Dict[str, RandomForestClassifier] = {}
    head_metrics: Dict[str, Dict[str, float]] = {}
    head_reports: Dict[str, Dict] = {}

    for target in TARGET_ORDER:
        start, end = label_indices[target]
        y_train_indices = np.argmax(y_train[:, start:end], axis=1)
        y_val_indices = np.argmax(y_val[:, start:end], axis=1)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(x_train, y_train_indices)
        val_predictions = model.predict(x_val)

        accuracy = accuracy_score(y_val_indices, val_predictions)
        macro_f1 = f1_score(y_val_indices, val_predictions, average="macro", zero_division=0)
        report = classification_report(
            y_val_indices,
            val_predictions,
            target_names=[str(name) for name in label_mappings[target]],
            output_dict=True,
            zero_division=0,
        )

        head_models[target] = model
        head_metrics[target] = {"accuracy": float(accuracy), "macro_f1": float(macro_f1)}
        head_reports[target] = report
        logger.info("%s -> accuracy: %.4f | macro_f1: %.4f", target, accuracy, macro_f1)

    return head_models, head_metrics, head_reports


def create_performance_plot(results: Dict[str, Dict[str, float]], save_path: Path) -> None:
    modalities = list(results.keys())
    num_modalities = [len(modality_name.split("+")) for modality_name in modalities]
    accuracies = [results[m]["val_accuracy"] for m in modalities]
    f1_scores = [results[m]["val_f1"] for m in modalities]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(num_modalities, accuracies, "o-", linewidth=2, markersize=8, label="Accuracy")
    plt.xlabel("Number of Modalities")
    plt.ylabel("Validation Accuracy")
    plt.title("Random Forest Baseline: Performance vs Number of Modalities")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(num_modalities, f1_scores, "o-", linewidth=2, markersize=8, color="orange", label="F1 Score")
    plt.xlabel("Number of Modalities")
    plt.ylabel("Validation F1 Score")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Performance plot saved to %s", save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Modality analysis with Random Forest baseline.")
    parser.add_argument("--dataloader_path", type=str, default="./dataloader/dataloader.csv")
    parser.add_argument("--label_mappings_path", type=str, default="./dataloader/label_mappings.json")
    parser.add_argument("--output_dir", type=str, default="./modality_results_baseline2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=0, help="Use 0 for None (unlimited depth).")
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--disable_progress_bar", action="store_true")
    args = parser.parse_args()

    show_progress = not args.disable_progress_bar
    set_seed(args.seed)

    dataloader_path = os.path.abspath(args.dataloader_path)
    label_mappings_path = os.path.abspath(args.label_mappings_path)
    output_dir = Path(os.path.abspath(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(dataloader_path):
        raise FileNotFoundError(f"dataloader.csv not found: {dataloader_path}")
    if not os.path.exists(label_mappings_path):
        raise FileNotFoundError(f"label_mappings.json not found: {label_mappings_path}")

    features, labels, label_mappings = load_features_and_labels_from_dataloader(
        dataloader_path, label_mappings_path, show_progress=show_progress
    )
    x_train_full, x_val_full, y_train, y_val = split_train_validation(features, labels, seed=args.seed, train_ratio=0.8)
    logger.info(
        "Train/Val split -> train: %d samples | val: %d samples",
        x_train_full.shape[0],
        x_val_full.shape[0],
    )

    modality_combinations = [
        ([0], "x1"),
        ([0, 1], "x1+x2"),
        ([0, 1, 2], "x1+x2+x3"),
        ([0, 1, 2, 3], "x1+x2+x3+x4"),
        ([0, 1, 2, 3, 4], "x1+x2+x3+x4+x5"),
        ([0, 1, 2, 3, 4, 5], "x1+x2+x3+x4+x5+x6"),
        ([0, 1, 2, 3, 4, 5, 6], "x1+x2+x3+x4+x5+x6+x7"),
    ]

    modality_results: Dict[str, Dict] = {}
    all_models: Dict[str, Dict[str, RandomForestClassifier]] = {}

    iterator = modality_combinations
    if show_progress:
        iterator = tqdm(modality_combinations, desc="Modality sweep", unit="modality")

    for modality_indices, modality_name in iterator:
        logger.info("=" * 60)
        logger.info("Training RF with %s (%d modalities)", modality_name, len(modality_indices))
        logger.info("=" * 60)

        x_train = select_modalities(x_train_full, modality_indices)
        x_val = select_modalities(x_val_full, modality_indices)

        head_models, head_metrics, head_reports = train_random_forest_heads(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            label_mappings=label_mappings,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            seed=args.seed,
        )

        val_accuracy = float(np.mean([metrics["accuracy"] for metrics in head_metrics.values()]))
        val_f1 = float(np.mean([metrics["macro_f1"] for metrics in head_metrics.values()]))

        modality_results[modality_name] = {
            "val_loss": None,
            "val_accuracy": val_accuracy,
            "val_f1": val_f1,
            "per_head_accuracy": {target: metrics["accuracy"] for target, metrics in head_metrics.items()},
            "per_head_f1": {target: metrics["macro_f1"] for target, metrics in head_metrics.items()},
            "classification_reports": head_reports,
            "num_modalities": len(modality_indices),
            "feature_dim": int(x_train.shape[1]),
        }
        all_models[modality_name] = head_models

        logger.info("Results for %s -> val_accuracy: %.4f | val_f1: %.4f", modality_name, val_accuracy, val_f1)

    results_path = output_dir / "modality_analysis_results.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(modality_results, handle, indent=2)

    models_path = output_dir / "modality_random_forest_heads.pkl"
    with open(models_path, "wb") as handle:
        pickle.dump(all_models, handle)

    mappings_path = output_dir / "label_mappings.json"
    with open(mappings_path, "w", encoding="utf-8") as handle:
        json.dump(label_mappings, handle, indent=2)

    plot_path = output_dir / "modality_performance.png"
    create_performance_plot(modality_results, plot_path)

    logger.info("=" * 80)
    logger.info("MODALITY ANALYSIS SUMMARY (RANDOM FOREST BASELINE)")
    logger.info("=" * 80)
    logger.info("%-20s %-10s %-10s %-10s", "Modality", "Accuracy", "F1 Score", "Loss")
    logger.info("%s", "-" * 80)
    for modality_name, metrics in modality_results.items():
        logger.info(
            "%-20s %-10.4f %-10.4f %-10s",
            modality_name,
            metrics["val_accuracy"],
            metrics["val_f1"],
            "N/A",
        )
    logger.info("Saved outputs to %s", output_dir)


if __name__ == "__main__":
    main()
