"""
DataLoader creator for BART + Pegasus + GPT-2 feature datasets.

Creates:
- dataloader/dataloader.csv
- dataloader/label_mappings.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataLoaderCreator:
    """Create a dataloader CSV for BART, Pegasus, and GPT-2 models."""

    def __init__(
        self,
        data_dir: str = "./multimodal_dataset",
        config_path: str = "./configs/config_summary.csv",
        output_file: str = "./dataloader/dataloader.csv",
    ):
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.model_family_mapping = ["BART", "Pegasus", "GPT-2"]
        self.model_size_mapping = ["base", "large", "xsum", "small", "medium"]
        self.optimizer_mapping = ["adamw", "sgd", "adafactor"]
        self.lr_mapping = [1e-5, 5e-5, 1e-4]
        self.bs_mapping = [4, 8, 16]

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config summary not found: {self.config_path}")

    @staticmethod
    def _normalize_family(value: str) -> str:
        text = str(value).strip().lower()
        if text in {"gpt2", "gpt-2"}:
            return "GPT-2"
        if text == "bart":
            return "BART"
        if text == "pegasus":
            return "Pegasus"
        return str(value).strip()

    def _load_config(self) -> pd.DataFrame:
        config_df = pd.read_csv(self.config_path)
        required_cols = [
            "model_index",
            "model_family",
            "model_size",
            "optimizer",
            "learning_rate",
            "batch_size",
            "num_train_epochs",
            "model_output_dir",
        ]
        missing = [col for col in required_cols if col not in config_df.columns]
        if missing:
            raise ValueError(f"Missing required columns in config summary: {missing}")

        config_df = config_df.copy()
        config_df["model_family"] = config_df["model_family"].apply(self._normalize_family)
        config_df = config_df[config_df["model_family"].isin(self.model_family_mapping)]
        logger.info("Loaded %d config rows for BART/Pegasus/GPT-2", len(config_df))
        return config_df

    @staticmethod
    def _extract_model_index_from_dirname(dirname: str) -> Optional[int]:
        parts = dirname.split("_")
        if not parts:
            return None
        try:
            return int(parts[0])
        except ValueError:
            return None

    def _collect_feature_files(self, model_dir: Path) -> Optional[Dict[str, List[str]]]:
        feature_files: Dict[str, List[str]] = {}
        for i in range(1, 8):
            matches = sorted(model_dir.glob(f"x{i}_batch_*.npy"))
            if not matches:
                return None
            feature_files[f"x{i}"] = [str(path) for path in matches]
        return feature_files

    def _one_hot(self, value, mapping: List) -> List[float]:
        arr = np.zeros(len(mapping), dtype=np.float32)
        if value in mapping:
            arr[mapping.index(value)] = 1.0
        return arr.tolist()

    def _create_label_encodings(self, row: pd.Series) -> Dict[str, List[float]]:
        family = self._normalize_family(row["model_family"])
        optimizer = str(row["optimizer"]).lower()
        return {
            "model_family_label": self._one_hot(family, self.model_family_mapping),
            "model_size_label": self._one_hot(row["model_size"], self.model_size_mapping),
            "optimizer_label": self._one_hot(optimizer, self.optimizer_mapping),
            "learning_rate_label": self._one_hot(row["learning_rate"], self.lr_mapping),
            "batch_size_label": self._one_hot(int(row["batch_size"]), self.bs_mapping),
        }

    def create_dataloader_csv(self) -> pd.DataFrame:
        config_df = self._load_config()

        records: List[Dict] = []
        missing_dirs = 0
        missing_features = 0

        for _, cfg in config_df.iterrows():
            rel_output_dir = str(cfg["model_output_dir"]).replace("./results/", "")
            model_dir = self.data_dir / rel_output_dir
            if not model_dir.exists():
                missing_dirs += 1
                continue

            feature_files = self._collect_feature_files(model_dir)
            if feature_files is None:
                missing_features += 1
                continue

            num_batches = min(len(files) for files in feature_files.values())
            labels = self._create_label_encodings(cfg)

            for batch_idx in range(num_batches):
                row = {
                    "model_index": int(cfg["model_index"]),
                    "model_family": self._normalize_family(cfg["model_family"]),
                    "model_size": cfg["model_size"],
                    "optimizer": str(cfg["optimizer"]).lower(),
                    "learning_rate": float(cfg["learning_rate"]),
                    "batch_size": int(cfg["batch_size"]),
                    "num_train_epochs": int(cfg["num_train_epochs"]),
                    "model_dir": str(model_dir),
                    "batch_index": batch_idx,
                }
                for i in range(1, 8):
                    row[f"x{i}_file"] = feature_files[f"x{i}"][batch_idx]
                row.update(labels)
                records.append(row)

        dataloader_df = pd.DataFrame(records)
        dataloader_df.to_csv(self.output_file, index=False)
        logger.info("Created %s with %d rows", self.output_file, len(dataloader_df))
        logger.info("Skipped %d models (missing dirs), %d models (missing x1..x7)", missing_dirs, missing_features)
        return dataloader_df

    def save_label_mappings(self) -> None:
        payload = {
            "model_family": self.model_family_mapping,
            "model_size": self.model_size_mapping,
            "optimizer": self.optimizer_mapping,
            "learning_rate": [str(v) for v in self.lr_mapping],
            "batch_size": self.bs_mapping,
        }
        mappings_path = self.output_file.parent / "label_mappings.json"
        with open(mappings_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Saved label mappings to %s", mappings_path)


def main() -> None:
    creator = DataLoaderCreator()
    dataloader_df = creator.create_dataloader_csv()
    creator.save_label_mappings()
    logger.info("Done. Rows: %d | Output dir: %s", len(dataloader_df), creator.output_file.parent)


if __name__ == "__main__":
    main()
