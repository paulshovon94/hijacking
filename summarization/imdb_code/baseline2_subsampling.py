"""
Subsampling experiment with baseline2 Random Forest heads.

This script mirrors subsampling_exp.py protocol:
- Reserve a fixed set of model indices for testing.
- Sample different numbers of training shadow models.
- Run multiple trials with different seeds per subsample size.
- Evaluate on the same held-out test set to compare performance.
"""

import argparse
import ast
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def get_unique_model_indices(dataloader_path: str) -> List[int]:
    df = pd.read_csv(dataloader_path)
    if "model_index" not in df.columns:
        raise ValueError("dataloader.csv missing required column: model_index")
    model_indices = sorted(df["model_index"].dropna().astype(int).unique().tolist())
    if not model_indices:
        raise ValueError("No model_index values found in dataloader.csv")
    return model_indices


def load_features_and_labels_from_dataloader(
    dataloader_path: str,
    label_mappings_path: str,
    selected_model_indices: Optional[List[int]] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[str]]]:
    dataloader_df = pd.read_csv(dataloader_path)
    with open(label_mappings_path, "r", encoding="utf-8") as handle:
        label_mappings = json.load(handle)

    if selected_model_indices is not None:
        dataloader_df = dataloader_df[dataloader_df["model_index"].isin(selected_model_indices)]
        logger.info(
            "Filtered dataloader to %d rows from %d model indices",
            len(dataloader_df),
            len(selected_model_indices),
        )

    if dataloader_df.empty:
        raise ValueError("No rows available after filtering model indices.")

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
    logger.info("Loaded %d samples, skipped %d", len(features), skipped_entries)
    logger.info("Feature shape: %s | Label shape: %s", features.shape, labels.shape)
    return features, labels, label_mappings


def split_train_validation(
    features: np.ndarray, labels: np.ndarray, seed: int = 42, train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = features.shape[0]
    if n_samples <= 1:
        return features, features, labels, labels

    split_idx = int(train_ratio * n_samples)
    split_idx = max(1, min(split_idx, n_samples - 1))

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
) -> Dict[str, RandomForestClassifier]:
    label_indices = build_label_indices(label_mappings)
    head_models: Dict[str, RandomForestClassifier] = {}

    for target in TARGET_ORDER:
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


def evaluate_heads(
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
        class_labels = list(range(len(label_mappings[target])))

        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        report = classification_report(
            y_true,
            y_pred,
            labels=class_labels,
            target_names=[str(name) for name in label_mappings[target]],
            output_dict=True,
            zero_division=0,
        )

        head_metrics[target] = {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
        }
        head_reports[target] = report

    return head_metrics, head_reports


def aggregate_overall(head_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {
        "accuracy_mean_across_heads": float(np.mean([m["accuracy"] for m in head_metrics.values()])),
        "f1_macro_mean_across_heads": float(np.mean([m["f1_macro"] for m in head_metrics.values()])),
        "f1_weighted_mean_across_heads": float(np.mean([m["f1_weighted"] for m in head_metrics.values()])),
    }


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Subsampling experiment with baseline2 Random Forest.")
    parser.add_argument("--dataloader_path", type=str, default="./dataloader/dataloader.csv")
    parser.add_argument("--label_mappings_path", type=str, default="./dataloader/label_mappings.json")
    parser.add_argument("--output_dir", type=str, default="./subsampling_results_baseline2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsample_sizes", nargs="+", type=int, default=[10, 20, 50, 100, 150, 189])
    parser.add_argument("--seeds", nargs="+", type=int, default=[32, 42, 52])
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--min_test_models", type=int, default=20)
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

    all_model_indices = get_unique_model_indices(dataloader_path)
    total_models = len(all_model_indices)

    n_test_models = max(args.min_test_models, int(total_models * args.test_size))
    n_test_models = min(n_test_models, total_models - 1)
    if n_test_models <= 0:
        raise ValueError("Not enough models to reserve a test split.")

    rng_test = np.random.RandomState(args.seed)
    test_model_indices = sorted(
        rng_test.choice(all_model_indices, n_test_models, replace=False).astype(int).tolist()
    )
    train_model_indices = [idx for idx in all_model_indices if idx not in test_model_indices]

    valid_subsample_sizes = [size for size in args.subsample_sizes if size <= len(train_model_indices)]
    if not valid_subsample_sizes:
        raise ValueError(
            f"No valid subsample sizes. Max available training models: {len(train_model_indices)}"
        )

    logger.info("Subsampling baseline2 RF setup:")
    logger.info("  total models: %d", total_models)
    logger.info("  reserved test models: %d", len(test_model_indices))
    logger.info("  train models available: %d", len(train_model_indices))
    logger.info("  valid subsample sizes: %s", valid_subsample_sizes)
    logger.info("  trial seeds: %s", args.seeds)

    test_features, test_labels, label_mappings = load_features_and_labels_from_dataloader(
        dataloader_path=dataloader_path,
        label_mappings_path=label_mappings_path,
        selected_model_indices=test_model_indices,
        show_progress=show_progress,
    )
    logger.info("Fixed test set size: %d samples", len(test_features))

    all_results: Dict[int, List[Dict]] = {}
    head_names = TARGET_ORDER.copy()

    for subsample_size in valid_subsample_sizes:
        logger.info("=" * 60)
        logger.info("SUBSAMPLE SIZE: %d models", subsample_size)
        logger.info("=" * 60)
        size_results: List[Dict] = []

        for trial_idx, trial_seed in enumerate(args.seeds, start=1):
            set_seed(trial_seed)
            rng_trial = np.random.RandomState(trial_seed)
            selected_model_indices = sorted(
                rng_trial.choice(train_model_indices, subsample_size, replace=False).astype(int).tolist()
            )
            logger.info(
                "Trial %d/%d (seed=%d): selected %d models",
                trial_idx,
                len(args.seeds),
                trial_seed,
                len(selected_model_indices),
            )

            train_features_all, train_labels_all, _ = load_features_and_labels_from_dataloader(
                dataloader_path=dataloader_path,
                label_mappings_path=label_mappings_path,
                selected_model_indices=selected_model_indices,
                show_progress=show_progress,
            )

            x_train, x_val, y_train, y_val = split_train_validation(
                train_features_all,
                train_labels_all,
                seed=trial_seed,
                train_ratio=args.train_ratio,
            )

            head_models = fit_rf_heads(
                x_train=x_train,
                y_train=y_train,
                label_mappings=label_mappings,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                min_samples_leaf=args.min_samples_leaf,
                seed=trial_seed,
            )

            val_head_metrics, val_reports = evaluate_heads(head_models, x_val, y_val, label_mappings)
            test_head_metrics, test_reports = evaluate_heads(head_models, test_features, test_labels, label_mappings)

            trial_result = {
                "subsample_size": subsample_size,
                "trial": trial_idx,
                "seed": trial_seed,
                "selected_model_indices": selected_model_indices,
                "train_samples": int(x_train.shape[0]),
                "val_samples": int(x_val.shape[0]),
                "test_samples": int(test_features.shape[0]),
                "validation": {
                    "overall": aggregate_overall(val_head_metrics),
                    "per_head": val_head_metrics,
                    "classification_reports": val_reports,
                },
                "test": {
                    "overall": aggregate_overall(test_head_metrics),
                    "per_head": test_head_metrics,
                    "classification_reports": test_reports,
                },
            }
            size_results.append(trial_result)

            logger.info(
                "Trial %d test overall -> accuracy: %.4f | f1_macro: %.4f | f1_weighted: %.4f",
                trial_idx,
                trial_result["test"]["overall"]["accuracy_mean_across_heads"],
                trial_result["test"]["overall"]["f1_macro_mean_across_heads"],
                trial_result["test"]["overall"]["f1_weighted_mean_across_heads"],
            )

        all_results[subsample_size] = size_results

        test_accs = [r["test"]["overall"]["accuracy_mean_across_heads"] for r in size_results]
        test_f1s = [r["test"]["overall"]["f1_macro_mean_across_heads"] for r in size_results]
        logger.info(
            "Subsample %d summary -> test accuracy: %.4f ± %.4f | test f1_macro: %.4f ± %.4f",
            subsample_size,
            float(np.mean(test_accs)),
            float(np.std(test_accs)),
            float(np.mean(test_f1s)),
            float(np.std(test_f1s)),
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_payload = {
        "config": {
            "seed": args.seed,
            "subsample_sizes": valid_subsample_sizes,
            "seeds": args.seeds,
            "test_size": args.test_size,
            "min_test_models": args.min_test_models,
            "train_ratio": args.train_ratio,
            "n_estimators": args.n_estimators,
            "max_depth": None if args.max_depth == 0 else args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "dataloader_path": dataloader_path,
            "label_mappings_path": label_mappings_path,
            "total_models": total_models,
            "reserved_test_model_indices": test_model_indices,
            "available_train_model_count": len(train_model_indices),
            "fixed_test_samples": int(test_features.shape[0]),
        },
        "results": all_results,
    }

    detailed_path = output_dir / "baseline2_subsampling_results.json"
    with open(detailed_path, "w", encoding="utf-8") as handle:
        json.dump(convert_to_serializable(detailed_payload), handle, indent=2)
    logger.info("Saved detailed results to %s", detailed_path)

    summary_rows = []
    for subsample_size, trials in all_results.items():
        for trial in trials:
            row = {
                "subsample_size": subsample_size,
                "trial": trial["trial"],
                "seed": trial["seed"],
                "train_samples": trial["train_samples"],
                "val_samples": trial["val_samples"],
                "test_samples": trial["test_samples"],
                "val_overall_accuracy": trial["validation"]["overall"]["accuracy_mean_across_heads"],
                "val_overall_f1_macro": trial["validation"]["overall"]["f1_macro_mean_across_heads"],
                "test_overall_accuracy": trial["test"]["overall"]["accuracy_mean_across_heads"],
                "test_overall_f1_macro": trial["test"]["overall"]["f1_macro_mean_across_heads"],
                "test_overall_f1_weighted": trial["test"]["overall"]["f1_weighted_mean_across_heads"],
            }
            for head_name in head_names:
                row[f"test_{head_name}_accuracy"] = trial["test"]["per_head"][head_name]["accuracy"]
                row[f"test_{head_name}_f1_macro"] = trial["test"]["per_head"][head_name]["f1_macro"]
                row[f"test_{head_name}_f1_weighted"] = trial["test"]["per_head"][head_name]["f1_weighted"]
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "baseline2_subsampling_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved summary CSV to %s", summary_path)

    per_head_stats_rows = []
    for subsample_size, trials in all_results.items():
        for head_name in head_names:
            acc_values = [t["test"]["per_head"][head_name]["accuracy"] for t in trials]
            f1_macro_values = [t["test"]["per_head"][head_name]["f1_macro"] for t in trials]
            f1_weighted_values = [t["test"]["per_head"][head_name]["f1_weighted"] for t in trials]
            per_head_stats_rows.append(
                {
                    "subsample_size": subsample_size,
                    "head_name": head_name,
                    "accuracy_mean": float(np.mean(acc_values)),
                    "accuracy_std": float(np.std(acc_values)),
                    "f1_macro_mean": float(np.mean(f1_macro_values)),
                    "f1_macro_std": float(np.std(f1_macro_values)),
                    "f1_weighted_mean": float(np.mean(f1_weighted_values)),
                    "f1_weighted_std": float(np.std(f1_weighted_values)),
                }
            )

    per_head_stats_df = pd.DataFrame(per_head_stats_rows)
    per_head_stats_path = output_dir / "baseline2_subsampling_per_head_stats.csv"
    per_head_stats_df.to_csv(per_head_stats_path, index=False)
    logger.info("Saved per-head stats CSV to %s", per_head_stats_path)


if __name__ == "__main__":
    main()
