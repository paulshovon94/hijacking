"""
Random Forest cross-family attack baseline.

Train one RandomForestClassifier per prediction head on selected model families
and evaluate on different target families for cross-family generalization.
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


TARGET_ORDER = ["model_family", "model_size", "optimizer", "learning_rate", "batch_size"]
LABEL_COLUMN_MAP = {
    "model_family": "model_family_label",
    "model_size": "model_size_label",
    "optimizer": "optimizer_label",
    "learning_rate": "learning_rate_label",
    "batch_size": "batch_size_label",
}


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


def _sanitize_feature_vector(feature_vector: np.ndarray, expected_size: int = 330) -> np.ndarray:
    feature_vector = np.asarray(feature_vector, dtype=np.float32).flatten()
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
    feature_vector = np.clip(feature_vector, -1e6, 1e6)

    if feature_vector.size < expected_size:
        feature_vector = np.pad(feature_vector, (0, expected_size - feature_vector.size))
    else:
        feature_vector = feature_vector[:expected_size]
    return feature_vector


def build_label_indices(label_mappings: Dict[str, List[str]]) -> Dict[str, Tuple[int, int]]:
    label_indices: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for target in TARGET_ORDER:
        num_classes = len(label_mappings[target])
        label_indices[target] = (cursor, cursor + num_classes)
        cursor += num_classes
    return label_indices


def _normalize_families(values: List[str]) -> List[str]:
    return [str(v).strip().lower() for v in values]


def _load_split(
    split_df: pd.DataFrame,
    label_mappings: Dict[str, List[str]],
    split_name: str,
    show_progress: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    features_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    skipped_entries = 0

    row_iterator = split_df.iterrows()
    if show_progress:
        row_iterator = tqdm(
            row_iterator,
            total=len(split_df),
            desc=f"Loading {split_name} rows",
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
            logger.warning("Skipping %s row %s due to error: %s", split_name, row_idx, exc)
            skipped_entries += 1

    if not features_list:
        raise ValueError(f"No valid rows loaded for split: {split_name}")

    features = np.vstack(features_list)
    labels = np.vstack(labels_list)
    logger.info(
        "%s split -> loaded %d samples, skipped %d",
        split_name.capitalize(),
        len(features),
        skipped_entries,
    )
    return features, labels


def load_features_and_labels_from_dataloader_cross_family(
    dataloader_path: str,
    label_mappings_path: str,
    train_families: List[str],
    test_families: List[str],
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, List[str]]]:
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
    if "model_family" not in dataloader_df.columns:
        raise ValueError("dataloader.csv must contain 'model_family' for cross-family split.")

    train_families_norm = _normalize_families(train_families)
    test_families_norm = _normalize_families(test_families)

    family_norm_series = dataloader_df["model_family"].astype(str).str.strip().str.lower()
    train_df = dataloader_df[family_norm_series.isin(train_families_norm)].copy()
    test_df = dataloader_df[family_norm_series.isin(test_families_norm)].copy()

    if train_df.empty:
        raise ValueError(f"No rows found for train_families={train_families}")
    if test_df.empty:
        raise ValueError(f"No rows found for test_families={test_families}")

    logger.info("Cross-family split requested:")
    logger.info("  train_families=%s", train_families)
    logger.info("  test_families=%s", test_families)
    logger.info("Raw rows -> train: %d, test: %d", len(train_df), len(test_df))

    train_features, train_labels = _load_split(train_df, label_mappings, "train", show_progress)
    test_features, test_labels = _load_split(test_df, label_mappings, "test", show_progress)
    return train_features, train_labels, test_features, test_labels, label_mappings


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


def fit_rf_heads(
    x_train: np.ndarray,
    y_train: np.ndarray,
    label_mappings: Dict[str, List[str]],
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    seed: int,
    show_progress: bool = True,
) -> Dict[str, RandomForestClassifier]:
    label_indices = build_label_indices(label_mappings)
    head_models: Dict[str, RandomForestClassifier] = {}

    target_iterator = TARGET_ORDER
    if show_progress:
        target_iterator = tqdm(TARGET_ORDER, desc="Training RF heads", unit="head")

    for target in target_iterator:
        start, end = label_indices[target]
        y_train_indices = np.argmax(y_train[:, start:end], axis=1)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(x_train, y_train_indices)
        head_models[target] = model

    return head_models


def evaluate_rf_heads(
    head_models: Dict[str, RandomForestClassifier],
    features: np.ndarray,
    labels: np.ndarray,
    label_mappings: Dict[str, List[str]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict]]:
    label_indices = build_label_indices(label_mappings)
    head_metrics: Dict[str, Dict[str, float]] = {}
    head_reports: Dict[str, Dict] = {}

    for target in TARGET_ORDER:
        start, end = label_indices[target]
        y_true = np.argmax(labels[:, start:end], axis=1)
        y_pred = head_models[target].predict(features)

        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        report = classification_report(
            y_true,
            y_pred,
            target_names=[str(name) for name in label_mappings[target]],
            output_dict=True,
            zero_division=0,
        )

        head_metrics[target] = {
            "accuracy": float(accuracy),
            "f1_macro": float(macro_f1),
            "f1_weighted": float(weighted_f1),
        }
        head_reports[target] = report
        logger.info(
            "%s -> accuracy: %.4f | f1_macro: %.4f | f1_weighted: %.4f",
            target,
            accuracy,
            macro_f1,
            weighted_f1,
        )

    return head_metrics, head_reports


def aggregate_overall(head_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {
        "accuracy_mean_across_heads": float(np.mean([m["accuracy"] for m in head_metrics.values()])),
        "f1_macro_mean_across_heads": float(np.mean([m["f1_macro"] for m in head_metrics.values()])),
        "f1_weighted_mean_across_heads": float(np.mean([m["f1_weighted"] for m in head_metrics.values()])),
    }


def save_artifacts(
    output_dir: Path,
    head_models: Dict[str, RandomForestClassifier],
    payload: Dict,
    label_mappings: Dict[str, List[str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    models_path = output_dir / "random_forest_cross_family_heads.pkl"
    with open(models_path, "wb") as handle:
        pickle.dump(head_models, handle)

    mappings_path = output_dir / "label_mappings.json"
    with open(mappings_path, "w", encoding="utf-8") as handle:
        json.dump(label_mappings, handle, indent=2)

    metrics_path = output_dir / "random_forest_cross_family_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info("Saved artifacts in %s", output_dir)
    logger.info("  - %s", models_path.name)
    logger.info("  - %s", mappings_path.name)
    logger.info("  - %s", metrics_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-family Random Forest baseline attack.")
    parser.add_argument("--dataloader_path", type=str, default="./dataloader/dataloader.csv")
    parser.add_argument("--label_mappings_path", type=str, default="./dataloader/label_mappings.json")
    parser.add_argument("--output_dir", type=str, default="./models/random_forest_baseline2_cross_family")
    parser.add_argument("--train_families", nargs="+", default=["BART", "Pegasus"])
    parser.add_argument("--test_families", nargs="+", default=["GPT-2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
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

    if not os.path.exists(dataloader_path):
        raise FileNotFoundError(f"dataloader.csv not found: {dataloader_path}")
    if not os.path.exists(label_mappings_path):
        raise FileNotFoundError(f"label_mappings.json not found: {label_mappings_path}")

    logger.info(
        "Cross-family RF attack: train on %s | test on %s",
        args.train_families,
        args.test_families,
    )

    train_features, train_labels, test_features, test_labels, label_mappings = (
        load_features_and_labels_from_dataloader_cross_family(
            dataloader_path=dataloader_path,
            label_mappings_path=label_mappings_path,
            train_families=args.train_families,
            test_families=args.test_families,
            show_progress=show_progress,
        )
    )

    x_train, x_val, y_train, y_val = split_train_validation(
        train_features,
        train_labels,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )
    logger.info(
        "Split sizes -> train: %d | val: %d | test(cross-family): %d",
        x_train.shape[0],
        x_val.shape[0],
        test_features.shape[0],
    )

    head_models = fit_rf_heads(
        x_train=x_train,
        y_train=y_train,
        label_mappings=label_mappings,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        seed=args.seed,
        show_progress=show_progress,
    )

    logger.info("Validation metrics (same train families):")
    val_metrics, val_reports = evaluate_rf_heads(head_models, x_val, y_val, label_mappings)
    val_overall = aggregate_overall(val_metrics)

    logger.info("Cross-family test metrics:")
    test_metrics, test_reports = evaluate_rf_heads(head_models, test_features, test_labels, label_mappings)
    test_overall = aggregate_overall(test_metrics)

    payload = {
        "validation": {
            "overall": val_overall,
            "per_head": val_metrics,
            "classification_reports": val_reports,
        },
        "cross_family_test": {
            "overall": test_overall,
            "per_head": test_metrics,
            "classification_reports": test_reports,
        },
        "config": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "n_estimators": args.n_estimators,
            "max_depth": None if args.max_depth == 0 else args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "train_families": args.train_families,
            "test_families": args.test_families,
            "train_samples_total": int(train_features.shape[0]),
            "test_samples_total": int(test_features.shape[0]),
            "feature_dim": int(train_features.shape[1]),
        },
    }

    logger.info(
        "Validation overall -> accuracy: %.4f | f1_macro: %.4f | f1_weighted: %.4f",
        val_overall["accuracy_mean_across_heads"],
        val_overall["f1_macro_mean_across_heads"],
        val_overall["f1_weighted_mean_across_heads"],
    )
    logger.info(
        "Cross-family overall -> accuracy: %.4f | f1_macro: %.4f | f1_weighted: %.4f",
        test_overall["accuracy_mean_across_heads"],
        test_overall["f1_macro_mean_across_heads"],
        test_overall["f1_weighted_mean_across_heads"],
    )

    save_artifacts(output_dir, head_models, payload, label_mappings)


if __name__ == "__main__":
    main()
