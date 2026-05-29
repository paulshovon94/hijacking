"""
Evaluate a trained baseline2 Random Forest model on another feature dataloader.

This script loads trained RandomForest heads from baseline2, reads x1-x7 features
from a dataloader CSV, and runs inference/evaluation on selected model indices.
"""

import argparse
import ast
import json
import logging
import os
import pickle
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


def parse_model_indices(indices_str: Optional[str]) -> Optional[List[int]]:
    """
    Parse model indices from comma/range format.

    Examples:
      "0,1,2"
      "0-5"
      "0,2-5,10"
    """
    if not indices_str:
        return None

    indices: List[int] = []
    for part in indices_str.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            start_idx = int(start.strip())
            end_idx = int(end.strip())
            indices.extend(range(start_idx, end_idx + 1))
        else:
            indices.append(int(item))
    return sorted(set(indices))


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


def load_features_and_labels_from_dataloader(
    dataloader_path: str,
    label_mappings_path: str,
    selected_model_indices: Optional[List[int]] = None,
    excluded_model_indices: Optional[List[int]] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[str]], pd.DataFrame]:
    dataloader_df = pd.read_csv(dataloader_path)
    with open(label_mappings_path, "r", encoding="utf-8") as handle:
        label_mappings = json.load(handle)

    if selected_model_indices is not None:
        dataloader_df = dataloader_df[dataloader_df["model_index"].isin(selected_model_indices)]
    if excluded_model_indices is not None:
        dataloader_df = dataloader_df[~dataloader_df["model_index"].isin(excluded_model_indices)]

    if dataloader_df.empty:
        raise ValueError("No rows left in dataloader after applying model index filters.")

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
    metadata_rows: List[Dict] = []
    skipped_entries = 0

    row_iterator = dataloader_df.iterrows()
    if show_progress:
        row_iterator = tqdm(
            row_iterator,
            total=len(dataloader_df),
            desc="Loading evaluation rows",
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
            metadata_rows.append(
                {
                    "row_idx": int(row_idx),
                    "model_index": int(row["model_index"]) if "model_index" in row else None,
                    "batch_index": int(row["batch_index"]) if "batch_index" in row and not pd.isna(row["batch_index"]) else None,
                    "model_family": row["model_family"] if "model_family" in row else None,
                    "model_size": row["model_size"] if "model_size" in row else None,
                }
            )
        except Exception as exc:
            logger.warning("Skipping row %s due to error: %s", row_idx, exc)
            skipped_entries += 1

    if not features_list:
        raise ValueError("No valid rows loaded from dataloader.csv")

    features = np.vstack(features_list)
    labels = np.vstack(labels_list)
    metadata_df = pd.DataFrame(metadata_rows)

    logger.info("Loaded %d samples, skipped %d samples", len(features), skipped_entries)
    logger.info("Feature shape: %s | Label shape: %s", features.shape, labels.shape)
    return features, labels, label_mappings, metadata_df


def load_trained_rf_heads(model_path: str) -> Dict[str, RandomForestClassifier]:
    with open(model_path, "rb") as handle:
        head_models = pickle.load(handle)
    if not isinstance(head_models, dict):
        raise ValueError("Loaded RF model file is not a dictionary of heads.")
    for target in TARGET_ORDER:
        if target not in head_models:
            raise ValueError(f"Missing head '{target}' in loaded model dictionary.")
    return head_models


def run_inference(
    head_models: Dict[str, RandomForestClassifier],
    features: np.ndarray,
    labels: np.ndarray,
    label_mappings: Dict[str, List[str]],
) -> Dict:
    label_indices = build_label_indices(label_mappings)
    predictions: Dict[str, np.ndarray] = {}
    probabilities: Dict[str, np.ndarray] = {}
    per_head_metrics: Dict[str, Dict[str, float]] = {}
    per_head_reports: Dict[str, Dict] = {}

    for target in TARGET_ORDER:
        model = head_models[target]
        start, end = label_indices[target]
        y_true = np.argmax(labels[:, start:end], axis=1)
        y_pred = model.predict(features)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(features)
        else:
            y_prob = np.zeros((len(features), len(label_mappings[target])), dtype=np.float32)

        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        report = classification_report(
            y_true,
            y_pred,
            target_names=[str(name) for name in label_mappings[target]],
            output_dict=True,
            zero_division=0,
        )

        predictions[target] = np.asarray(y_pred)
        probabilities[target] = np.asarray(y_prob)
        per_head_metrics[target] = {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
        }
        per_head_reports[target] = report
        logger.info(
            "%s -> accuracy: %.4f | f1_macro: %.4f | f1_weighted: %.4f",
            target,
            acc,
            f1_macro,
            f1_weighted,
        )

    overall = {
        "accuracy_mean_across_heads": float(np.mean([m["accuracy"] for m in per_head_metrics.values()])),
        "f1_macro_mean_across_heads": float(np.mean([m["f1_macro"] for m in per_head_metrics.values()])),
        "f1_weighted_mean_across_heads": float(np.mean([m["f1_weighted"] for m in per_head_metrics.values()])),
    }

    logger.info(
        "Overall mean across heads -> accuracy: %.4f | f1_macro: %.4f | f1_weighted: %.4f",
        overall["accuracy_mean_across_heads"],
        overall["f1_macro_mean_across_heads"],
        overall["f1_weighted_mean_across_heads"],
    )

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "per_head_metrics": per_head_metrics,
        "classification_reports": per_head_reports,
        "overall": overall,
    }


def save_inference_outputs(
    results: Dict,
    label_mappings: Dict[str, List[str]],
    metadata_df: pd.DataFrame,
    predictions_output_path: Path,
    metrics_output_path: Path,
    config_payload: Dict,
) -> None:
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df = metadata_df.copy()
    predictions = results["predictions"]
    probabilities = results["probabilities"]

    for target in TARGET_ORDER:
        pred = predictions[target]
        probs = probabilities[target]

        output_df[f"{target}_predicted_index"] = pred
        output_df[f"{target}_predicted"] = [label_mappings[target][int(idx)] for idx in pred]

        if probs.ndim == 2 and probs.shape[0] == len(output_df):
            output_df[f"{target}_confidence"] = probs.max(axis=1)
            for class_idx, class_name in enumerate(label_mappings[target]):
                safe_class_name = str(class_name).replace(" ", "_")
                output_df[f"{target}_prob_{safe_class_name}"] = probs[:, class_idx]

    output_df.to_csv(predictions_output_path, index=False)
    logger.info("Saved inference predictions to %s", predictions_output_path)

    metrics_payload = {
        "overall": results["overall"],
        "per_head": results["per_head_metrics"],
        "classification_reports": results["classification_reports"],
        "config": config_payload,
    }
    with open(metrics_output_path, "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    logger.info("Saved inference metrics to %s", metrics_output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline2 RandomForest heads on other models/features."
    )
    parser.add_argument("--dataloader_path", type=str, default="./dataloader/dataloader.csv")
    parser.add_argument("--label_mappings_path", type=str, default="./dataloader/label_mappings.json")
    parser.add_argument(
        "--rf_model_path",
        type=str,
        default="./models/random_forest_baseline2/random_forest_multitask_heads.pkl",
        help="Path to trained RF heads from baseline2.py",
    )
    parser.add_argument(
        "--predictions_output_path",
        type=str,
        default="./models/random_forest_baseline2/other_evaluation_predictions.csv",
    )
    parser.add_argument(
        "--metrics_output_path",
        type=str,
        default="./models/random_forest_baseline2/other_evaluation_metrics.json",
    )
    parser.add_argument(
        "--model_indices",
        type=str,
        default=None,
        help='Only evaluate these model indices, e.g. "0,1,2" or "10-30".',
    )
    parser.add_argument(
        "--exclude_model_indices",
        type=str,
        default=None,
        help='Exclude model indices, e.g. "0-50".',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_progress_bar", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    show_progress = not args.disable_progress_bar

    dataloader_path = os.path.abspath(args.dataloader_path)
    label_mappings_path = os.path.abspath(args.label_mappings_path)
    rf_model_path = os.path.abspath(args.rf_model_path)
    predictions_output_path = Path(os.path.abspath(args.predictions_output_path))
    metrics_output_path = Path(os.path.abspath(args.metrics_output_path))

    if not os.path.exists(dataloader_path):
        raise FileNotFoundError(f"dataloader.csv not found: {dataloader_path}")
    if not os.path.exists(label_mappings_path):
        raise FileNotFoundError(f"label_mappings.json not found: {label_mappings_path}")
    if not os.path.exists(rf_model_path):
        raise FileNotFoundError(f"trained RF model not found: {rf_model_path}")

    selected_model_indices = parse_model_indices(args.model_indices)
    excluded_model_indices = parse_model_indices(args.exclude_model_indices)

    if selected_model_indices is not None:
        logger.info("Evaluating only selected model indices: %s", selected_model_indices)
    if excluded_model_indices is not None:
        logger.info("Excluding model indices: %s", excluded_model_indices)

    features, labels, label_mappings, metadata_df = load_features_and_labels_from_dataloader(
        dataloader_path=dataloader_path,
        label_mappings_path=label_mappings_path,
        selected_model_indices=selected_model_indices,
        excluded_model_indices=excluded_model_indices,
        show_progress=show_progress,
    )

    head_models = load_trained_rf_heads(rf_model_path)
    results = run_inference(head_models, features, labels, label_mappings)

    config_payload = {
        "seed": args.seed,
        "rf_model_path": rf_model_path,
        "dataloader_path": dataloader_path,
        "label_mappings_path": label_mappings_path,
        "predictions_output_path": str(predictions_output_path),
        "metrics_output_path": str(metrics_output_path),
        "model_indices": selected_model_indices,
        "exclude_model_indices": excluded_model_indices,
        "num_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
    }

    save_inference_outputs(
        results=results,
        label_mappings=label_mappings,
        metadata_df=metadata_df,
        predictions_output_path=predictions_output_path,
        metrics_output_path=metrics_output_path,
        config_payload=config_payload,
    )


if __name__ == "__main__":
    main()
