"""
Multimodal multi-head classifier for LoRA hyperparameter stealing.

This mirrors experiment.py and trains on concatenated x1-x7 features (2312 dim)
from dataloader rows. It predicts:
- model_family
- model_size
- learning_rate
- lora_r
- lora_alpha
- lora_dropout
"""

import ast
import os
import random
import argparse
import json
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_ORDER = [
    "model_family",
    "model_size",
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
]

CACHE_DIR = "/work/shovon/LLM/"
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(CACHE_DIR, "sentence-transformers")
os.environ["NLTK_DATA"] = os.path.join(CACHE_DIR, "nltk_data")
os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")

for cache_path in [
    os.environ["TRANSFORMERS_CACHE"],
    os.environ["HF_HOME"],
    os.environ["HF_DATASETS_CACHE"],
    os.environ["SENTENCE_TRANSFORMERS_HOME"],
    os.environ["NLTK_DATA"],
    os.environ["TORCH_HOME"],
]:
    os.makedirs(cache_path, exist_ok=True)
    logger.info("Using cache directory: %s", cache_path)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        logger.info("Set random seed to %s with deterministic algorithms", seed)
    else:
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        logger.info("Set random seed to %s without deterministic algorithms", seed)


class MultimodalDataset(Dataset):
    """Dataset holding feature vectors and concatenated one-hot labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        assert len(self.features) == len(self.labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class MultimodalHyperparameterClassifier(nn.Module):
    """Shared MLP encoder with one linear classification head per target."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_classes_per_head: Dict[str, int],
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim
        self.shared_encoder = nn.Sequential(*layers)
        self.classification_heads = nn.ModuleDict(
            {name: nn.Linear(prev_dim, n_classes) for name, n_classes in num_classes_per_head.items()}
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.shared_encoder(x)
        return {name: head(encoded) for name, head in self.classification_heads.items()}


def _safe_parse_one_hot(value) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _load_feature_file(file_path: str, expected_size: int = 330) -> np.ndarray:
    feature_array = np.load(file_path)
    if not np.all(np.isfinite(feature_array)):
        feature_array = np.nan_to_num(feature_array, nan=0.0)
        feature_array = np.clip(feature_array, -1e6, 1e6)
    feature_array = feature_array.flatten()
    if len(feature_array) < expected_size:
        feature_array = np.pad(feature_array, (0, expected_size - len(feature_array)))
    else:
        feature_array = feature_array[:expected_size]
    return feature_array


def _select_target_mappings(label_mappings: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Keep only targets we want to predict for LoRA runs."""
    missing_targets = [target for target in TARGET_ORDER if target not in label_mappings]
    if missing_targets:
        raise ValueError(f"Missing target mappings in label_mappings.json: {missing_targets}")
    return {target: label_mappings[target] for target in TARGET_ORDER}


def load_features_and_labels_from_dataloader(
    dataloader_path: str, label_mappings_path: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[str]]]:
    logger.info("Loading features and labels from dataloader...")
    dataloader_df = pd.read_csv(dataloader_path)
    with open(label_mappings_path, "r", encoding="utf-8") as handle:
        label_mappings = _select_target_mappings(json.load(handle))

    logger.info("Loaded dataloader with %d entries", len(dataloader_df))
    logger.info("Loaded label mappings for %d targets", len(label_mappings))

    required_feature_columns = [f"x{i}_file" for i in range(1, 8)]
    required_label_columns = [f"{target}_label" for target in TARGET_ORDER]
    required_columns = required_feature_columns + required_label_columns
    missing_columns = [column for column in required_columns if column not in dataloader_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataloader.csv: {missing_columns}")

    label_indices = {}
    start_idx = 0
    for target_name, mapping in label_mappings.items():
        end_idx = start_idx + len(mapping)
        label_indices[target_name] = (start_idx, end_idx)
        start_idx = end_idx

    logger.info("\nLabel indices:")
    for target_name, (start, end) in label_indices.items():
        logger.info("%s: %s-%s (%s classes)", target_name, start, end, end - start)

    features_list = []
    labels_list = []
    processed_entries = 0
    skipped_entries = 0

    for row_idx, row in dataloader_df.iterrows():
        try:
            feature_paths = [row[f"x{i}_file"] for i in range(1, 8)]
            missing_files = [path for path in feature_paths if not path or not os.path.exists(path)]
            if missing_files:
                skipped_entries += 1
                continue

            feature_chunks = [_load_feature_file(path) for path in feature_paths]
            combined_features = np.concatenate(feature_chunks)
            if len(combined_features) < 2312:
                combined_features = np.pad(combined_features, (0, 2312 - len(combined_features)))
            else:
                combined_features = combined_features[:2312]

            one_hot_chunks = [_safe_parse_one_hot(row[f"{target}_label"]) for target in TARGET_ORDER]
            combined_labels = np.concatenate(one_hot_chunks)

            features_list.append(combined_features)
            labels_list.append(combined_labels)
            processed_entries += 1

            if processed_entries % 100 == 0:
                logger.info("Processed %d entries...", processed_entries)
        except Exception as exc:
            logger.warning("Error processing entry %s: %s", row_idx, str(exc))
            skipped_entries += 1
            continue

    if not features_list:
        raise ValueError("No valid features found in dataloader")

    features = np.vstack(features_list)
    labels = np.vstack(labels_list)

    assert features.shape[0] == labels.shape[0]
    assert features.shape[1] == 2312

    logger.info("\nSummary:")
    logger.info("Loaded %d samples", len(labels))
    logger.info("Features shape: %s", features.shape)
    logger.info("Labels shape: %s", labels.shape)
    logger.info("Processed %d entries", processed_entries)
    logger.info("Skipped %d entries", skipped_entries)

    return features, labels, label_mappings


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler],
    num_epochs: int,
    device: int,
    label_mappings: Dict[str, List[str]],
) -> Dict[str, List[float]]:
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_epoch_accuracy = 0.0
    best_epoch_f1 = 0.0
    best_accuracy = 0.0
    best_accuracy_epoch = 0
    best_f1 = 0.0
    best_f1_epoch = 0

    label_indices = {}
    start_idx = 0
    for target_name, mapping in label_mappings.items():
        end_idx = start_idx + len(mapping)
        label_indices[target_name] = (start_idx, end_idx)
        start_idx = end_idx

    def multi_head_loss(predictions: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for target_name, pred in predictions.items():
            start, end = label_indices[target_name]
            target = labels[:, start:end]
            target_indices = target.argmax(dim=1)
            head_loss = F.cross_entropy(pred, target_indices, reduction="mean", label_smoothing=0.05)
            if torch.isnan(head_loss) or torch.isinf(head_loss):
                logger.warning("Invalid loss detected for %s", target_name)
                head_loss = torch.tensor(0.0, device=labels.device, requires_grad=True)
            total_loss += head_loss
        return total_loss

    logger.info("Starting training on %s for %d epochs", device, num_epochs)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            if torch.isnan(features).any() or torch.isinf(features).any():
                continue

            optimizer.zero_grad()
            predictions = model(features)
            loss = multi_head_loss(predictions, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        train_loss = train_loss / max(train_batches, 1)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_predictions = {name: [] for name in label_mappings.keys()}
        all_targets = {name: [] for name in label_mappings.keys()}

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                predictions = model(features)
                loss = multi_head_loss(predictions, labels)
                val_loss += loss.item()
                val_batches += 1

                for target_name, pred in predictions.items():
                    start, end = label_indices[target_name]
                    target = labels[:, start:end]
                    pred_indices = pred.argmax(dim=1).cpu().numpy()
                    target_indices = target.argmax(dim=1).cpu().numpy()
                    all_predictions[target_name].extend(pred_indices)
                    all_targets[target_name].extend(target_indices)

        val_loss = val_loss / max(val_batches, 1)
        if scheduler is not None:
            scheduler.step(val_loss)

        per_head_accuracy = {}
        per_head_f1 = {}
        overall_accuracy = 0.0
        overall_f1 = 0.0

        for target_name in label_mappings.keys():
            if not all_targets[target_name]:
                continue
            acc = accuracy_score(all_targets[target_name], all_predictions[target_name])
            f1 = f1_score(all_targets[target_name], all_predictions[target_name], average="macro")
            per_head_accuracy[target_name] = acc
            per_head_f1[target_name] = f1
            overall_accuracy += acc
            overall_f1 += f1

        num_heads = max(len(label_mappings), 1)
        overall_accuracy /= num_heads
        overall_f1 /= num_heads

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(overall_accuracy)
        history["val_f1"].append(overall_f1)

        logger.info(
            "Epoch %d/%d | Train Loss %.4f | Val Loss %.4f | Val Acc %.4f | Val F1 %.4f",
            epoch + 1,
            num_epochs,
            train_loss,
            val_loss,
            overall_accuracy,
            overall_f1,
        )
        for target_name in label_mappings.keys():
            if target_name in per_head_accuracy:
                logger.info(
                    "  %s: Acc=%.4f F1=%.4f",
                    target_name,
                    per_head_accuracy[target_name],
                    per_head_f1[target_name],
                )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_epoch_accuracy = overall_accuracy
            best_epoch_f1 = overall_f1

        if overall_accuracy > best_accuracy:
            best_accuracy = overall_accuracy
            best_accuracy_epoch = epoch

        if overall_f1 > best_f1:
            best_f1 = overall_f1
            best_f1_epoch = epoch

    logger.info(
        "Training completed. Best (by val loss) epoch: %d | val_loss: %.4f | val_acc: %.4f | val_f1: %.4f",
        best_epoch + 1,
        best_val_loss,
        best_epoch_accuracy,
        best_epoch_f1,
    )
    logger.info(
        "Best accuracy epoch: %d | val_acc: %.4f",
        best_accuracy_epoch + 1,
        best_accuracy,
    )
    logger.info(
        "Best macro-F1 epoch: %d | val_f1: %.4f",
        best_f1_epoch + 1,
        best_f1,
    )
    return history


def setup_distributed() -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        gpu = 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    return rank, world_size, gpu


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    rank, world_size, gpu = setup_distributed()

    dataloader_path = os.path.abspath("./dataloader/dataloader.csv")
    label_mappings_path = os.path.abspath("./dataloader/label_mappings.json")
    if not os.path.exists(dataloader_path):
        raise FileNotFoundError(f"Dataloader file not found: {dataloader_path}")
    if not os.path.exists(label_mappings_path):
        raise FileNotFoundError(f"Label mappings file not found: {label_mappings_path}")

    try:
        features, labels, label_mappings = load_features_and_labels_from_dataloader(
            dataloader_path, label_mappings_path
        )

        n_samples = len(features)
        train_size = int(0.8 * n_samples)
        rng = np.random.RandomState(args.seed)
        indices = rng.permutation(n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        x_train, x_val = features[train_indices], features[val_indices]
        y_train, y_val = labels[train_indices], labels[val_indices]

        if rank == 0:
            logger.info("Training samples: %d, Validation samples: %d", len(y_train), len(y_val))

        train_dataset = MultimodalDataset(x_train, y_train)
        val_dataset = MultimodalDataset(x_val, y_val)

        train_sampler = DistributedSampler(train_dataset, seed=args.seed) if world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset, seed=args.seed) if world_size > 1 else None

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            sampler=val_sampler,
            num_workers=2,
            pin_memory=True,
        )

        input_dim = 2312
        hidden_dims = [512, 256, 128]
        num_classes_per_head = {name: len(mapping) for name, mapping in label_mappings.items()}

        model = MultimodalHyperparameterClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes_per_head=num_classes_per_head,
            dropout_rate=0.2,
        ).to(gpu)
        if world_size > 1:
            model = DDP(model, device_ids=[gpu])

        optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.3,
            patience=2,
            verbose=True,
            min_lr=1e-7,
        )

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=100,
            device=gpu,
            label_mappings=label_mappings,
        )

        if rank == 0:
            model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            os.makedirs(model_save_dir, exist_ok=True)

            model_path = os.path.join(model_save_dir, "multimodal_hyperparameter_classifier_lora.pt")
            if isinstance(model, DDP):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)

            history_path = os.path.join(model_save_dir, "training_history_multimodal_classifier_lora.json")
            with open(history_path, "w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)

            mappings_path = os.path.join(model_save_dir, "label_mappings_lora.json")
            with open(mappings_path, "w", encoding="utf-8") as handle:
                json.dump(label_mappings, handle, indent=2)

            logger.info("Saved LoRA experiment artifacts in %s", model_save_dir)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
