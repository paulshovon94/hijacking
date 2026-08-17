"""
Random Forest baseline for LoRA hyperparameter stealing.

This baseline mirrors the prediction task from experiment_lora.py:
- Input: concatenated x1-x7 features (2312 dim)
- Outputs:
  model_family, model_size, learning_rate,
  lora_r, lora_alpha, lora_dropout

Unlike the neural multi-head model, this script trains one
RandomForestClassifier per target head and reports per-head + overall metrics.
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

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm.auto import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


TARGET_ORDER = [
    "model_family",
    "model_size",
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
]
LABEL_COLUMN_MAP = {
    "model_family": "model_family_label",
    "model_size": "model_size_label",
    "learning_rate": "learning_rate_label",
    "lora_r": "lora_r_label",
    "lora_alpha": "lora_alpha_label",
    "lora_dropout": "lora_dropout_label",
}


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def _parse_label_vector(label_cell, expected_len: int) -> np.ndarray:
    """Parse one-hot label vector safely from CSV cell."""
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


def _sanitize_feature_vector(feature_vector: np.ndarray, expected_size: int = 330) -> np.ndarray:
    """Sanitize and normalize each x1..x7 feature vector length."""
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
    """Load features and one-hot labels from dataloader CSV."""
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
                chunks.append(_sanitize_feature_vector(loaded, expected_size=330))

            combined_features = np.concatenate(chunks)
            if combined_features.size < 2312:
                combined_features = np.pad(combined_features, (0, 2312 - combined_features.size))
            else:
                combined_features = combined_features[:2312]

            label_vectors = []
            for target in TARGET_ORDER:
                col = LABEL_COLUMN_MAP[target]
                expected_len = len(label_mappings[target])
                label_vectors.append(_parse_label_vector(row[col], expected_len))
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
    """Build index ranges for each target over concatenated one-hot labels."""
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
    """Split data using fixed random permutation (same style as experiment.py)."""
    n_samples = features.shape[0]
    split_idx = int(train_ratio * n_samples)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    return features[train_idx], features[val_idx], labels[train_idx], labels[val_idx]


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
    show_progress: bool = True,
) -> Tuple[Dict[str, RandomForestClassifier], Dict[str, Dict[str, float]], Dict[str, Dict[str, str]]]:
    """Train one RandomForest classifier per prediction head."""
    label_indices = build_label_indices(label_mappings)
    head_models: Dict[str, RandomForestClassifier] = {}
    head_metrics: Dict[str, Dict[str, float]] = {}
    head_reports: Dict[str, Dict[str, str]] = {}

    target_iterator = TARGET_ORDER
    if show_progress:
        target_iterator = tqdm(TARGET_ORDER, desc="Training RF heads", unit="head")

    for target in target_iterator:
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


def save_artifacts(
    output_dir: Path,
    head_models: Dict[str, RandomForestClassifier],
    metrics_payload: Dict,
    label_mappings: Dict[str, List[str]],
) -> None:
    """Persist models, mappings, and metrics to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    models_path = output_dir / "random_forest_multitask_heads_lora.pkl"
    with open(models_path, "wb") as handle:
        pickle.dump(head_models, handle)

    mappings_path = output_dir / "label_mappings_lora.json"
    with open(mappings_path, "w", encoding="utf-8") as handle:
        json.dump(label_mappings, handle, indent=2)

    metrics_path = output_dir / "random_forest_baseline2_lora_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    logger.info("Saved artifacts in %s", output_dir)
    logger.info("  - %s", models_path.name)
    logger.info("  - %s", mappings_path.name)
    logger.info("  - %s", metrics_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest baseline for LoRA multimodal hyperparameter prediction.")
    parser.add_argument("--dataloader_path", type=str, default="./dataloader/dataloader.csv")
    parser.add_argument("--label_mappings_path", type=str, default="./dataloader/label_mappings.json")
    parser.add_argument("--output_dir", type=str, default="./models/random_forest_baseline2_lora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=0, help="Use 0 for None (unlimited depth).")
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument(
        "--disable_progress_bar",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    args = parser.parse_args()
    show_progress = not args.disable_progress_bar

    set_seed(args.seed)

    dataloader_path = os.path.abspath(args.dataloader_path)
    label_mappings_path = os.path.abspath(args.label_mappings_path)
    output_dir = Path(os.path.abspath(args.output_dir))

    if not os.path.exists(dataloader_path):
        raise FileNotFoundError(f"dataloader.csv not found: {dataloader_path}")
    if not os.path.exists(label_mappings_path):
        raise FileNotFoundError(f"label_mappings.json not found: {label_mappings_path}")

    features, labels, label_mappings = load_features_and_labels_from_dataloader(
        dataloader_path, label_mappings_path, show_progress=show_progress
    )
    x_train, x_val, y_train, y_val = split_train_validation(features, labels, seed=args.seed, train_ratio=0.8)

    logger.info(
        "Train/Val split -> train: %d samples | val: %d samples",
        x_train.shape[0],
        x_val.shape[0],
    )

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
        show_progress=show_progress,
    )

    overall_accuracy = float(np.mean([m["accuracy"] for m in head_metrics.values()]))
    overall_macro_f1 = float(np.mean([m["macro_f1"] for m in head_metrics.values()]))

    metrics_payload = {
        "overall": {
            "accuracy_mean_across_heads": overall_accuracy,
            "macro_f1_mean_across_heads": overall_macro_f1,
        },
        "per_head": head_metrics,
        "classification_reports": head_reports,
        "config": {
            "seed": args.seed,
            "n_estimators": args.n_estimators,
            "max_depth": None if args.max_depth == 0 else args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "train_ratio": 0.8,
            "feature_dim": int(features.shape[1]),
            "num_samples": int(features.shape[0]),
            "targets": TARGET_ORDER,
        },
    }

    logger.info(
        "Overall mean across heads -> accuracy: %.4f | macro_f1: %.4f",
        overall_accuracy,
        overall_macro_f1,
    )
    best_accuracy_target, best_accuracy_metrics = max(
        head_metrics.items(), key=lambda item: item[1]["accuracy"]
    )
    best_f1_target, best_f1_metrics = max(
        head_metrics.items(), key=lambda item: item[1]["macro_f1"]
    )
    logger.info(
        "Best head by accuracy -> %s: %.4f",
        best_accuracy_target,
        best_accuracy_metrics["accuracy"],
    )
    logger.info(
        "Best head by macro_f1 -> %s: %.4f",
        best_f1_target,
        best_f1_metrics["macro_f1"],
    )

    save_artifacts(output_dir, head_models, metrics_payload, label_mappings)


if __name__ == "__main__":
    main()
