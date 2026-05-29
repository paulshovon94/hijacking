"""
Data Loader Creator for BART-only models.

This script scans the multimodal dataset and config_summary.csv to create
`dataloader/dataloader.csv` using only BART model configurations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BartOnlyDataLoaderCreator:
    """Creates dataloader.csv for multi-label, multi-class BART-only training."""

    def __init__(self, data_dir: str = "./multimodal_dataset", config_path: str = "./configs/config_summary.csv"):
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)

        self.dataloader_dir = Path("dataloader")
        self.dataloader_dir.mkdir(exist_ok=True)
        logger.info(f"Using dataloader directory: {self.dataloader_dir.absolute()}")

        self.output_file = self.dataloader_dir / "dataloader.csv"

        # BART-only label space.
        self.model_family_mapping = ["BART"]
        self.model_size_mapping = ["base", "large"]
        self.optimizer_mapping = ["adamw", "sgd", "adafactor"]
        self.lr_mapping = [1e-5, 5e-5, 1e-4]
        self.bs_mapping = [4, 8, 16]

        if not self.data_dir.exists():
            raise FileNotFoundError(f"multimodal_dataset directory not found: {self.data_dir}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

    def load_config_summary(self) -> pd.DataFrame:
        """Load and validate config summary, then keep only BART rows."""
        logger.info(f"Loading config summary from {self.config_path}")
        config_df = pd.read_csv(self.config_path)
        logger.info(f"Loaded {len(config_df)} configurations")

        required_columns = [
            "model_index",
            "model_family",
            "model_size",
            "optimizer",
            "learning_rate",
            "batch_size",
            "num_train_epochs",
            "model_output_dir",
        ]
        missing_columns = [col for col in required_columns if col not in config_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in config_summary.csv: {missing_columns}")

        config_df = config_df[config_df["model_family"].astype(str).str.upper() == "BART"].copy()
        logger.info(f"Filtered config to {len(config_df)} BART models")
        return config_df

    def scan_data_directory(self) -> List[Dict]:
        """Scan dataset for model directories containing x1-x7 .npy features."""
        logger.info(f"Scanning multimodal dataset directory: {self.data_dir}")

        model_data: List[Dict] = []
        total_dirs = 0
        dirs_with_features = 0

        for model_dir in self.data_dir.rglob("*"):
            if not model_dir.is_dir():
                continue
            total_dirs += 1

            feature_files: Dict[str, List[str]] = {}
            for i in range(1, 8):
                patterns_to_try = [
                    f"x{i}_batch_*.npy",
                    f"x{i}_*.npy",
                    f"*x{i}*.npy",
                    f"features_x{i}*.npy",
                    f"*features*{i}*.npy",
                ]

                x_files = []
                for pattern in patterns_to_try:
                    x_files = list(model_dir.glob(pattern))
                    if x_files:
                        break

                if x_files:
                    feature_files[f"x{i}"] = [str(path) for path in x_files]

            if not feature_files:
                continue
            dirs_with_features += 1

            try:
                model_info = self.extract_model_info_from_path(model_dir)
                if model_info is None:
                    continue
                model_info["feature_files"] = feature_files
                model_info["model_dir"] = str(model_dir)
                model_data.append(model_info)
                logger.info(
                    f"Found BART model directory: {model_dir} "
                    f"(Index: {model_info.get('model_index')})"
                )
            except Exception as exc:
                logger.warning(f"Could not extract model info from {model_dir}: {str(exc)}")

        logger.info(f"Scanned {total_dirs} directories")
        logger.info(f"Found {dirs_with_features} directories with feature files")
        logger.info(f"Successfully processed {len(model_data)} BART model directories")
        return model_data

    def extract_model_info_from_path(self, model_dir: Path) -> Optional[Dict]:
        """Extract model metadata from directory name and keep BART only."""
        try:
            dir_name = model_dir.name
            if "_" not in dir_name:
                return None

            parts = dir_name.split("_")
            if len(parts) < 4:
                return None

            model_index = int(parts[0])
            model_family = parts[1].upper()
            if model_family != "BART":
                return None

            model_size = parts[2]
            optimizer = parts[3]

            lr_part = next((part for part in parts if part.startswith("lr")), None)
            learning_rate = float(lr_part.replace("lr", "")) if lr_part else None

            bs_part = next((part for part in parts if part.startswith("bs")), None)
            batch_size = int(bs_part.replace("bs", "")) if bs_part else None

            return {
                "model_index": model_index,
                "model_family": model_family,
                "model_size": model_size,
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_train_epochs": 3,
                "model_output_dir": str(model_dir),
            }
        except Exception as exc:
            logger.warning(f"Error extracting model info from {model_dir}: {str(exc)}")
            return None

    def create_label_encodings(self, model_info: Dict) -> Dict:
        """Create one-hot labels for each predicted hyperparameter."""
        family_label = np.zeros(len(self.model_family_mapping))
        size_label = np.zeros(len(self.model_size_mapping))
        optimizer_label = np.zeros(len(self.optimizer_mapping))
        lr_label = np.zeros(len(self.lr_mapping))
        bs_label = np.zeros(len(self.bs_mapping))

        family = str(model_info.get("model_family", "")).upper()
        if family in self.model_family_mapping:
            family_label[self.model_family_mapping.index(family)] = 1
        else:
            logger.warning(f"Model family '{family}' not in mapping {self.model_family_mapping}")

        size = str(model_info.get("model_size", ""))
        if size in self.model_size_mapping:
            size_label[self.model_size_mapping.index(size)] = 1
        else:
            logger.warning(f"Model size '{size}' not in mapping {self.model_size_mapping}")

        optimizer = str(model_info.get("optimizer", "")).lower()
        if optimizer in self.optimizer_mapping:
            optimizer_label[self.optimizer_mapping.index(optimizer)] = 1
        else:
            logger.warning(f"Optimizer '{optimizer}' not in mapping {self.optimizer_mapping}")

        learning_rate = model_info.get("learning_rate")
        if learning_rate is not None and learning_rate in self.lr_mapping:
            lr_label[self.lr_mapping.index(learning_rate)] = 1
        else:
            logger.warning(f"Learning rate '{learning_rate}' not in mapping {self.lr_mapping}")

        batch_size = model_info.get("batch_size")
        if batch_size is not None and batch_size in self.bs_mapping:
            bs_label[self.bs_mapping.index(batch_size)] = 1
        else:
            logger.warning(f"Batch size '{batch_size}' not in mapping {self.bs_mapping}")

        return {
            "model_family_label": family_label.tolist(),
            "model_size_label": size_label.tolist(),
            "optimizer_label": optimizer_label.tolist(),
            "learning_rate_label": lr_label.tolist(),
            "batch_size_label": bs_label.tolist(),
        }

    def match_with_config(self, model_data: List[Dict], config_df: pd.DataFrame) -> List[Dict]:
        """Match scanned model metadata with BART config_summary rows."""
        logger.info("Matching model data with BART config summary...")
        matched_data: List[Dict] = []

        for model_info in model_data:
            model_index = model_info.get("model_index")
            model_dir = str(model_info.get("model_output_dir", ""))

            if model_index is not None:
                matching_rows = config_df[config_df["model_index"] == model_index]
                if not matching_rows.empty:
                    config_row = matching_rows.iloc[0]
                    model_info.update(
                        {
                            "model_family": config_row["model_family"],
                            "model_size": config_row["model_size"],
                            "optimizer": config_row["optimizer"],
                            "learning_rate": config_row["learning_rate"],
                            "batch_size": config_row["batch_size"],
                            "num_train_epochs": config_row["num_train_epochs"],
                        }
                    )
                else:
                    logger.warning(f"No BART config found for model index {model_index}")

            if str(model_info.get("model_family", "")).upper() != "BART":
                for _, config_row in config_df.iterrows():
                    config_model_dir = str(config_row["model_output_dir"])
                    if model_dir in config_model_dir or config_model_dir in model_dir:
                        model_info.update(
                            {
                                "model_family": config_row["model_family"],
                                "model_size": config_row["model_size"],
                                "optimizer": config_row["optimizer"],
                                "learning_rate": config_row["learning_rate"],
                                "batch_size": config_row["batch_size"],
                                "num_train_epochs": config_row["num_train_epochs"],
                            }
                        )
                        break

            if str(model_info.get("model_family", "")).upper() == "BART":
                matched_data.append(model_info)

        logger.info(f"Matched {len(matched_data)} BART models")
        return matched_data

    def validate_label_mappings(self):
        """Validate BART-only mappings used for one-hot labels."""
        logger.info("Validating BART-only label mappings...")

        expected_families = ["BART"]
        if self.model_family_mapping != expected_families:
            logger.warning(f"Model family mapping mismatch: {self.model_family_mapping}")
            self.model_family_mapping = expected_families

        expected_sizes = ["base", "large"]
        if self.model_size_mapping != expected_sizes:
            logger.warning(f"Model size mapping mismatch: {self.model_size_mapping}")
            self.model_size_mapping = expected_sizes

        expected_optimizers = ["adamw", "sgd", "adafactor"]
        if self.optimizer_mapping != expected_optimizers:
            logger.warning(f"Optimizer mapping mismatch: {self.optimizer_mapping}")
            self.optimizer_mapping = expected_optimizers

        expected_lrs = [1e-5, 5e-5, 1e-4]
        if self.lr_mapping != expected_lrs:
            logger.warning(f"Learning rate mapping mismatch: {self.lr_mapping}")
            self.lr_mapping = expected_lrs

        expected_bs = [4, 8, 16]
        if self.bs_mapping != expected_bs:
            logger.warning(f"Batch size mapping mismatch: {self.bs_mapping}")
            self.bs_mapping = expected_bs

    def create_dataloader_csv(self) -> pd.DataFrame:
        """Create and save BART-only dataloader.csv."""
        logger.info("Creating BART-only dataloader.csv...")
        self.validate_label_mappings()

        config_df = self.load_config_summary()
        model_data = self.scan_data_directory()
        matched_data = self.match_with_config(model_data, config_df)

        matched_data = [
            model_info
            for model_info in matched_data
            if str(model_info.get("model_family", "")).upper() == "BART"
        ]
        logger.info(f"After filtering, {len(matched_data)} BART models remain")

        dataloader_entries: List[Dict] = []
        for model_info in matched_data:
            label_encodings = self.create_label_encodings(model_info)
            feature_files = model_info.get("feature_files", {})

            num_batches = min(len(files) for files in feature_files.values()) if feature_files else 0
            for batch_idx in range(num_batches):
                entry = {
                    "model_index": model_info.get("model_index"),
                    "model_family": model_info.get("model_family"),
                    "model_size": model_info.get("model_size"),
                    "optimizer": model_info.get("optimizer"),
                    "learning_rate": model_info.get("learning_rate"),
                    "batch_size": model_info.get("batch_size"),
                    "num_train_epochs": model_info.get("num_train_epochs"),
                    "model_dir": model_info.get("model_dir"),
                    "batch_index": batch_idx,
                }

                for i in range(1, 8):
                    key = f"x{i}"
                    entry[f"{key}_file"] = feature_files.get(key, [""] * (batch_idx + 1))[batch_idx]

                entry.update(label_encodings)
                dataloader_entries.append(entry)

        dataloader_df = pd.DataFrame(dataloader_entries)
        dataloader_df.to_csv(self.output_file, index=False)
        logger.info(f"Created dataloader.csv with {len(dataloader_df)} entries")

        self.print_summary_statistics(dataloader_df)
        return dataloader_df

    def print_summary_statistics(self, df: pd.DataFrame):
        """Print summary stats of the generated dataloader."""
        logger.info("\n" + "=" * 50)
        logger.info("BART-ONLY DATALOADER SUMMARY STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total entries: {len(df)}")

        if len(df) == 0:
            logger.warning("No data found. Check dataset directories and config_summary.csv.")
            logger.info("=" * 50)
            return

        logger.info(f"Unique models: {df['model_index'].nunique() if 'model_index' in df.columns else 0}")

        if "model_family" in df.columns:
            logger.info("\nModel Family Distribution:")
            for family, count in df["model_family"].value_counts().items():
                logger.info(f"  {family}: {count}")

        if "model_size" in df.columns:
            logger.info("\nModel Size Distribution:")
            for size, count in df["model_size"].value_counts().items():
                logger.info(f"  {size}: {count}")

        if "optimizer" in df.columns:
            logger.info("\nOptimizer Distribution:")
            for optimizer, count in df["optimizer"].value_counts().items():
                logger.info(f"  {optimizer}: {count}")

        if "learning_rate" in df.columns:
            logger.info("\nLearning Rate Distribution:")
            for learning_rate, count in df["learning_rate"].value_counts().items():
                logger.info(f"  {learning_rate}: {count}")

        if "batch_size" in df.columns:
            logger.info("\nBatch Size Distribution:")
            for batch_size, count in df["batch_size"].value_counts().items():
                logger.info(f"  {batch_size}: {count}")

        logger.info("\nFeature File Availability:")
        for i in range(1, 8):
            column_name = f"x{i}_file"
            if column_name in df.columns:
                available = df[column_name].astype(str).str.len() > 0
                logger.info(f"  x{i}: {available.sum()}/{len(df)} ({available.sum() / len(df) * 100:.1f}%)")

        logger.info("=" * 50)

    def save_label_mappings(self):
        """Save label mappings to JSON for downstream loading."""
        label_mappings = {
            "model_family": self.model_family_mapping,
            "model_size": self.model_size_mapping,
            "optimizer": self.optimizer_mapping,
            "learning_rate": [str(learning_rate) for learning_rate in self.lr_mapping],
            "batch_size": self.bs_mapping,
        }

        label_mappings_path = self.dataloader_dir / "label_mappings.json"
        with open(label_mappings_path, "w", encoding="utf-8") as file:
            json.dump(label_mappings, file, indent=2)

        logger.info(f"Saved label mappings to {label_mappings_path}")


def main():
    """Main entry point for standalone BART-only dataloader creation."""
    try:
        creator = BartOnlyDataLoaderCreator()
        logger.info("Expected model families: BART")
        logger.info("Expected model sizes: base, large")

        dataloader_df = creator.create_dataloader_csv()
        creator.save_label_mappings()

        logger.info("Successfully created BART-only dataloader.csv and label_mappings.json")
        logger.info(f"Files created in {creator.dataloader_dir}:")
        logger.info(f"  - dataloader.csv ({len(dataloader_df)} entries)")
        logger.info("  - label_mappings.json")
    except Exception as exc:
        logger.error(f"Error creating BART-only dataloader: {str(exc)}")
        raise


if __name__ == "__main__":
    main()
