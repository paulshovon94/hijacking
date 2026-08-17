#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-seed shadow-model evaluation for the translation hyperparameter-stealing attack.

Runs both classifiers -- the multi-head neural model and the Random Forest baseline --
across several seeds under a *shadow-model* split: the classifier trains on one set of
shadow models and is tested on a disjoint set, so every prediction concerns a model it
has never seen.

This file is orchestration only. The modelling code is imported and called unchanged
from experiment_lora.py and baseline2_lora.py.

Why this exists: both of those scripts split by row, and each shadow model contributes 96
rows to dataloader.csv. A row split puts ~77 of a model's rows in train and ~19 in
validation, so the classifier can learn a model's feature signature and then recognise
that same model's held-out rows. Stealing should mean predicting the hyperparameters of a
model never seen before, which is what `--split shadow` measures.

    python run_seed_sweep.py --seeds 32 42 52

The row split is kept only to reproduce the published single-seed numbers as a
correctness check on this driver (--verify).
"""

import argparse
import csv
import json
import logging
import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

import baseline2_lora as rf_base
import experiment_lora as nn_exp


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_ORDER = nn_exp.TARGET_ORDER

# Which dataloader.csv column identifies the shadow model. model_index is preferred (a
# compact integer, one per model); model_dir is the equivalent path and is accepted as a
# fallback. Both are 1:1 with the 216 models.
GROUP_COLUMN_CANDIDATES = ("model_index", "model_dir")

# Columns that, taken together, identify a "twin pair": the two configs identical in
# everything except lora_dropout. The grid is fully crossed, so every model has exactly
# one twin.
TWIN_KEY_COLUMNS = ("model_family", "learning_rate", "lora_r", "lora_alpha")

# Neural hyperparameters, copied from experiment_lora.main() so sweep runs stay
# comparable with the standalone ones. Changing anything here breaks that comparison.
NN_HIDDEN_DIMS = [512, 256, 128]
NN_DROPOUT = 0.2
NN_BATCH_SIZE = 32
NN_LR = 1e-4
NN_WEIGHT_DECAY = 1e-3
NN_EPOCHS = 100

# Forest hyperparameters, copied from baseline2_lora.main().
RF_N_ESTIMATORS = 400
RF_MAX_DEPTH = 0  # 0 means None (unlimited)
RF_MIN_SAMPLES_LEAF = 1


# --------------------------------------------------------------------------------------
# Feature loading and cache
# --------------------------------------------------------------------------------------

def _cache_key(dataloader_path: str) -> str:
    """Identify the dataloader.csv a cache was built from.

    Size plus mtime is enough: the file is rewritten wholesale by
    create_dataloader_lora.py, so any re-extraction changes both.
    """
    stat = os.stat(dataloader_path)
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def load_features(
    dataloader_path: str, label_mappings_path: str, cache_path: str, refresh: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[str]]]:
    """Load the feature matrix, caching it because loading dominates runtime.

    Reading 20,736 rows off Lustre takes 30-50 minutes; the forest then trains in about
    30 seconds per head. Loading once and reusing the cache is the difference between a
    1.5 hour sweep and an 8 hour one.

    experiment_lora's loader is used for *both* classifiers on purpose. The two scripts'
    loaders are not equivalent: experiment_lora._load_feature_file clips to +/-1e6 only
    when the array holds non-finite values, while baseline2_lora._sanitize_feature_vector
    always clips. A finite value above 1e6 would therefore reach the two models
    differently. Using one loader guarantees an identical matrix.
    """
    key = _cache_key(dataloader_path)

    if os.path.exists(cache_path) and not refresh:
        cached = np.load(cache_path, allow_pickle=True)
        if str(cached["cache_key"]) == key:
            label_mappings = json.loads(str(cached["label_mappings"]))
            logger.info("Loaded features from cache: %s", cache_path)
            return cached["features"], cached["labels"], label_mappings
        logger.info("Cache key mismatch, reloading features from disk.")

    start = time.time()
    features, labels, label_mappings = nn_exp.load_features_and_labels_from_dataloader(
        dataloader_path, label_mappings_path
    )
    logger.info("Loaded features in %.1f minutes", (time.time() - start) / 60.0)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        features=features,
        labels=labels,
        label_mappings=json.dumps(label_mappings),
        cache_key=key,
    )
    logger.info("Wrote feature cache: %s", cache_path)
    return features, labels, label_mappings


def load_groups(dataloader_path: str) -> np.ndarray:
    """Read the per-row shadow-model identity.

    The loaders return only (features, labels, mappings) -- no model identity -- so the
    grouping is recovered positionally from the same CSV.

    Called before the feature load on purpose: reading this header costs milliseconds and
    loading features costs half an hour, so a bad column name should fail immediately
    rather than after the expensive step.
    """
    header = pd.read_csv(dataloader_path, nrows=0)
    column = next((c for c in GROUP_COLUMN_CANDIDATES if c in header.columns), None)
    if column is None:
        raise ValueError(
            f"dataloader.csv has none of {list(GROUP_COLUMN_CANDIDATES)}. "
            f"Available columns: {list(header.columns)}"
        )

    frame = pd.read_csv(dataloader_path, usecols=[column])
    groups = frame[column].to_numpy()
    logger.info(
        "Grouping by '%s': %d rows across %d shadow models",
        column, len(groups), len(np.unique(groups)),
    )
    return groups


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------

def load_twin_keys(dataloader_path: str) -> np.ndarray:
    """Per-row twin-pair identity: family|lr|r|alpha, i.e. everything but dropout.

    Why this matters. lora_dropout scores *below* chance under a plain shadow split, and
    the cause is structural rather than a bug: because the grid is fully crossed, a
    model's nearest neighbour in behaviour space is usually its dropout-twin. Measured on
    these features, the twin is the single closest model 39.8% of the time (chance 0.5%)
    with a median rank of 2/215, and a model's nearest neighbour shares its dropout only
    29.2% of the time against a 50% baseline. A similarity-based classifier therefore
    predicts the opposite dropout systematically.

    Leaving the twin in training is itself leakage for the dropout head. Splitting on the
    pair holds both halves out together, which turns an artefactual below-chance number
    into an honest one.
    """
    frame = pd.read_csv(dataloader_path, usecols=list(TWIN_KEY_COLUMNS))
    keys = frame[list(TWIN_KEY_COLUMNS)].astype(str).agg("|".join, axis=1).to_numpy()
    logger.info("Twin pairs: %d distinct keys across %d rows", len(np.unique(keys)), len(keys))
    return keys


def row_split(n_samples: int, seed: int, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce the existing row-level split exactly (experiment_lora.main)."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_samples)
    split_idx = int(train_ratio * n_samples)
    return indices[:split_idx], indices[split_idx:]


def shadow_split(
    groups: np.ndarray, seed: int, train_ratio: float = 0.8, unit: str = "models"
) -> Tuple[np.ndarray, np.ndarray]:
    """Split by group, so no group straddles train and validation.

    Used with per-model ids for the shadow split, and with twin-pair keys for the
    twin-aware split. np.unique sorts, so the ordering is deterministic regardless of CSV
    row order and the permutation depends only on the seed.
    """
    unique_models = np.unique(groups)
    rng = np.random.RandomState(seed)
    permuted = unique_models[rng.permutation(len(unique_models))]
    n_train = int(train_ratio * len(unique_models))

    train_models = set(permuted[:n_train].tolist())
    val_models = set(permuted[n_train:].tolist())

    train_idx = np.flatnonzero(np.isin(groups, list(train_models)))
    val_idx = np.flatnonzero(np.isin(groups, list(val_models)))

    # The whole evaluation rests on this being a genuine holdout.
    overlap = train_models & val_models
    if overlap:
        raise AssertionError(f"{len(overlap)} models appear in both splits")
    if len(train_idx) + len(val_idx) != len(groups):
        raise AssertionError("split does not cover every row")

    logger.info(
        "Split seed %d -> train %d %s (%d rows) | val %d %s (%d rows)",
        seed, len(train_models), unit, len(train_idx), len(val_models), unit, len(val_idx),
    )
    return train_idx, val_idx


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

def _majority_vote(pred_idx: np.ndarray, groups: np.ndarray, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse each model's per-row predictions into one prediction by majority vote."""
    models = np.unique(groups)
    votes = np.empty(len(models), dtype=int)
    for position, model in enumerate(models):
        mask = groups == model
        votes[position] = np.bincount(pred_idx[mask], minlength=n_classes).argmax()
    return models, votes


def score_head(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    groups_val: np.ndarray,
    n_classes: int,
    do_vote: bool,
) -> Dict[str, float]:
    """Per-row metrics, plus the per-model vote when the split makes it meaningful."""
    result = {
        "row_acc": float(accuracy_score(y_true_idx, y_pred_idx)),
        "row_f1": float(f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)),
        "vote_acc": float("nan"),
        "vote_f1": float("nan"),
        "n_val_models": len(np.unique(groups_val)),
    }
    if not do_vote:
        # Under a row split the model was seen during training, so a per-model vote
        # would look strong and mean nothing. Left as NaN rather than computed.
        return result

    models, votes = _majority_vote(y_pred_idx, groups_val, n_classes)

    # Every row of a shadow model describes the same model, so its label must be constant.
    truth = np.empty(len(models), dtype=int)
    for position, model in enumerate(models):
        mask = groups_val == model
        model_labels = np.unique(y_true_idx[mask])
        if len(model_labels) != 1:
            raise AssertionError(
                f"model {model} has {len(model_labels)} distinct labels for one head"
            )
        truth[position] = model_labels[0]

    result["vote_acc"] = float(accuracy_score(truth, votes))
    result["vote_f1"] = float(f1_score(truth, votes, average="macro", zero_division=0))
    return result


def check_label_sanity(labels: np.ndarray, val_idx: np.ndarray, label_mappings) -> None:
    """Refuse to score a head whose validation set collapsed to one class."""
    label_indices = rf_base.build_label_indices(label_mappings)
    for target in TARGET_ORDER:
        start, end = label_indices[target]
        present = np.unique(np.argmax(labels[val_idx, start:end], axis=1))
        if len(present) < 2:
            raise AssertionError(
                f"head '{target}' has only {len(present)} class(es) in validation"
            )


# --------------------------------------------------------------------------------------
# Model runners
# --------------------------------------------------------------------------------------

def run_forest(
    x_train, y_train, x_val, y_val, label_mappings, seed, groups_val, do_vote
) -> Dict[str, Dict[str, float]]:
    """Train the Random Forest heads with baseline2_lora's own routine, unchanged."""
    head_models, _, _ = rf_base.train_random_forest_heads(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        label_mappings=label_mappings,
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        seed=seed,
        show_progress=False,
    )

    label_indices = rf_base.build_label_indices(label_mappings)
    scores = {}
    for target in TARGET_ORDER:
        start, end = label_indices[target]
        y_true_idx = np.argmax(y_val[:, start:end], axis=1)
        y_pred_idx = head_models[target].predict(x_val)
        scores[target] = score_head(
            y_true_idx, y_pred_idx, groups_val, end - start, do_vote
        )
    return scores


def run_neural(
    x_train, y_train, x_val, y_val, label_mappings, seed, groups_val, do_vote
) -> Dict[str, Dict[str, float]]:
    """Train the multi-head network with experiment_lora's own train_model, unchanged.

    train_model neither checkpoints nor restores best weights, so the model left behind
    is the final epoch. Metrics below therefore describe the final-epoch model, which is
    what the seed-42 verification numbers are taken from.
    """
    device = 0 if torch.cuda.is_available() else "cpu"
    nn_exp.set_seed(seed, deterministic=False)

    train_loader = DataLoader(
        nn_exp.MultimodalDataset(x_train, y_train),
        batch_size=NN_BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        nn_exp.MultimodalDataset(x_val, y_val),
        batch_size=NN_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True,
    )

    model = nn_exp.MultimodalHyperparameterClassifier(
        input_dim=x_train.shape[1],
        hidden_dims=NN_HIDDEN_DIMS,
        num_classes_per_head={name: len(m) for name, m in label_mappings.items()},
        dropout_rate=NN_DROPOUT,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=NN_LR, weight_decay=NN_WEIGHT_DECAY,
        betas=(0.9, 0.999), eps=1e-8,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.3, patience=2,
    )

    nn_exp.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NN_EPOCHS,
        device=device,
        label_mappings=label_mappings,
    )

    # train_model returns only aggregate history, so re-run inference to recover the
    # per-head predictions the vote needs.
    model.eval()
    head_logits: Dict[str, List[np.ndarray]] = {target: [] for target in TARGET_ORDER}
    with torch.no_grad():
        for features, _ in val_loader:
            outputs = model(features.to(device))
            for target in TARGET_ORDER:
                head_logits[target].append(outputs[target].cpu().numpy())

    label_indices = rf_base.build_label_indices(label_mappings)
    scores = {}
    for target in TARGET_ORDER:
        start, end = label_indices[target]
        y_true_idx = np.argmax(y_val[:, start:end], axis=1)
        y_pred_idx = np.concatenate(head_logits[target]).argmax(axis=1)
        scores[target] = score_head(
            y_true_idx, y_pred_idx, groups_val, end - start, do_vote
        )
    return scores


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def write_results(rows: List[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fields = [
        "seed", "model", "split", "head", "n_val_rows", "n_val_models",
        "row_acc", "row_f1", "vote_acc", "vote_f1",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d result rows to %s", len(rows), output_path)


def write_summary(rows: List[dict], label_mappings, output_path: str) -> None:
    """Mean +/- std across seeds per head, against each head's random baseline."""
    frame = pd.DataFrame(rows)
    lines = ["# Shadow-model seed sweep", ""]

    for split in sorted(frame["split"].unique()):
        subset = frame[frame["split"] == split]
        lines.append(f"## split = {split}")
        lines.append("")
        lines.append("| model | head | random | row acc (mean +/- std) | vote acc (mean +/- std) |")
        lines.append("|---|---|---|---|---|")
        for model_name in sorted(subset["model"].unique()):
            for head in TARGET_ORDER:
                cell = subset[(subset["model"] == model_name) & (subset["head"] == head)]
                if cell.empty:
                    continue
                baseline = 1.0 / len(label_mappings[head])
                row_txt = f"{cell['row_acc'].mean():.4f} +/- {cell['row_acc'].std(ddof=0):.4f}"
                if cell["vote_acc"].notna().any():
                    vote_txt = (
                        f"{cell['vote_acc'].mean():.4f} +/- {cell['vote_acc'].std(ddof=0):.4f}"
                    )
                else:
                    vote_txt = "n/a"
                lines.append(
                    f"| {model_name} | {head} | {baseline:.3f} | {row_txt} | {vote_txt} |"
                )
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    logger.info("Wrote summary to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[32, 42, 52])
    parser.add_argument("--dataloader_path", type=str, default="./dataloader/dataloader.csv")
    parser.add_argument(
        "--label_mappings_path", type=str, default="./dataloader/label_mappings.json"
    )
    parser.add_argument("--cache_path", type=str, default="./cache/features_labels.npz")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--refresh_cache", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Also run seed 42 row-level, to reproduce the published single-seed numbers.",
    )
    parser.add_argument(
        "--twin",
        action="store_true",
        help=(
            "Also run a twin-aware split, holding each dropout pair out together. "
            "Removes the structural leak that drives lora_dropout below chance."
        ),
    )
    args = parser.parse_args()

    # Cheap checks first: the group column is read in milliseconds, the features take
    # about half an hour.
    groups = load_groups(os.path.abspath(args.dataloader_path))

    features, labels, label_mappings = load_features(
        os.path.abspath(args.dataloader_path),
        os.path.abspath(args.label_mappings_path),
        os.path.abspath(args.cache_path),
        refresh=args.refresh_cache,
    )

    # Positional alignment between the CSV and the feature matrix is load-bearing: the
    # loader skips unreadable rows, and one skipped row would shift every subsequent
    # group assignment, silently corrupting the split.
    if len(groups) != len(features):
        raise ValueError(
            f"dataloader.csv has {len(groups)} rows but the loader returned "
            f"{len(features)} feature rows. Positional alignment is unsafe."
        )

    logger.info(
        "Features %s | labels %s | %d shadow models | max|feature| = %.4g",
        features.shape, labels.shape, len(np.unique(groups)), np.abs(features).max(),
    )

    twin_keys = load_twin_keys(os.path.abspath(args.dataloader_path)) if args.twin else None

    runs: List[Tuple[int, str]] = [(seed, "shadow") for seed in args.seeds]
    if args.twin:
        runs += [(seed, "twin") for seed in args.seeds]
    if args.verify:
        # Row-level is not an evaluation protocol here -- this exists purely to prove the
        # driver reproduces the standalone scripts.
        runs.append((42, "row"))

    rows: List[dict] = []
    for seed, split in runs:
        if split == "shadow":
            train_idx, val_idx = shadow_split(groups, seed, unit="models")
        elif split == "twin":
            train_idx, val_idx = shadow_split(twin_keys, seed, unit="twin pairs")
        else:
            train_idx, val_idx = row_split(len(features), seed)

        check_label_sanity(labels, val_idx, label_mappings)
        # Both grouped splits hold whole models out, so the per-model vote is meaningful;
        # only the row split contaminates it.
        do_vote = split != "row"

        x_train, x_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        groups_val = groups[val_idx]

        for model_name, runner in (("forest", run_forest), ("neural", run_neural)):
            logger.info("=== seed %d | split %s | %s ===", seed, split, model_name)
            started = time.time()
            scores = runner(
                x_train, y_train, x_val, y_val, label_mappings, seed, groups_val, do_vote
            )
            for head, metrics in scores.items():
                logger.info(
                    "  %-14s row_acc=%.4f  vote_acc=%s",
                    head,
                    metrics["row_acc"],
                    "n/a" if np.isnan(metrics["vote_acc"]) else f"{metrics['vote_acc']:.4f}",
                )
                rows.append(
                    {
                        "seed": seed,
                        "model": model_name,
                        "split": split,
                        "head": head,
                        "n_val_rows": len(val_idx),
                        **metrics,
                    }
                )
            logger.info("    finished in %.1f min", (time.time() - started) / 60.0)

    output_dir = os.path.abspath(args.output_dir)
    write_results(rows, os.path.join(output_dir, "seed_sweep_results.csv"))
    write_summary(rows, label_mappings, os.path.join(output_dir, "seed_sweep_summary.md"))


if __name__ == "__main__":
    main()
