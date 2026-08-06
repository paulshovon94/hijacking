"""
Data Loader Creator for LoRA models across the translation model zoo.

This script scans the multimodal dataset directory and LoRA config summary CSV
to build `dataloader/dataloader.csv` for downstream hyperparameter stealing.

LoRA hyperparameter policy (from generate_configs_lora.py):
- learning_rate: [1e-5, 5e-5, 1e-4]
- lora_r: [4, 8, 16]
- lora_alpha: [8, 16, 32]
- lora_dropout: [0.05, 0.1]
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LoRADataLoaderCreator:
    """Creates dataloader.csv for LoRA runs across all translation-capable families."""

    # Directory-name slug -> canonical family label used in the config summary.
    FAMILY_BY_SLUG = {
        "marian": "Marian",
        "bart": "BART",
        "llama": "LLaMA",
        "qwen": "Qwen",
    }

    def __init__(
        self,
        data_dir: str = "./multimodal_dataset",
        config_path: str = "./configs_lora/config_summary.csv",
    ):
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)

        self.dataloader_dir = Path("dataloader")
        self.dataloader_dir.mkdir(exist_ok=True)
        logger.info("Using dataloader directory: %s", self.dataloader_dir.absolute())

        self.output_file = self.dataloader_dir / "dataloader.csv"

        # Label mappings aligned with the LoRA grid in generate_configs_lora.py. Only
        # families that can actually translate De->En are included -- see the viability
        # gate results recorded in that file.
        self.model_family_mapping = ["Marian", "BART", "Qwen", "LLaMA"]
        self.model_size_mapping = [
            "base", "large", "0.5B", "1.5B", "1B", "3B", "7B", "8B",
        ]
        self.lr_mapping = [1e-5, 5e-5, 1e-4]
        self.lora_r_mapping = [4, 8, 16]
        self.lora_alpha_mapping = [8, 16, 32]
        self.lora_dropout_mapping = [0.05, 0.1]

        if not self.data_dir.exists():
            raise FileNotFoundError(f"multimodal_dataset directory not found: {self.data_dir}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

    def load_config_summary(self) -> pd.DataFrame:
        """Load and validate LoRA config summary CSV."""
        logger.info("Loading config summary from %s", self.config_path)
        config_df = pd.read_csv(self.config_path)
        logger.info("Loaded %d configurations", len(config_df))

        required_columns = [
            "model_index",
            "model_family",
            "model_size",
            "learning_rate",
            "num_train_epochs",
            "model_output_dir",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
        ]
        missing_columns = [column for column in required_columns if column not in config_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in config_summary.csv: {missing_columns}")

        known = set(self.model_family_mapping)
        config_df = config_df[config_df["model_family"].isin(known)].copy()
        logger.info(
            "Kept %d LoRA models across families: %s",
            len(config_df),
            ", ".join(sorted(config_df["model_family"].unique())),
        )
        return config_df

    def scan_data_directory(self) -> List[Dict]:
        """Scan multimodal_dataset for directories containing x1-x7 feature files."""
        logger.info("Scanning multimodal dataset directory: %s", self.data_dir)

        model_data: List[Dict] = []
        total_dirs = 0
        dirs_with_features = 0

        for model_dir in self.data_dir.rglob("*"):
            if not model_dir.is_dir():
                continue

            total_dirs += 1
            feature_files: Dict[str, List[str]] = {}

            for feature_idx in range(1, 8):
                patterns = [
                    f"x{feature_idx}_batch_*.npy",
                    f"x{feature_idx}_*.npy",
                    f"*x{feature_idx}*.npy",
                    f"features_x{feature_idx}*.npy",
                    f"*features*{feature_idx}*.npy",
                ]

                matched_files = []
                for pattern in patterns:
                    matched_files = sorted(model_dir.glob(pattern))
                    if matched_files:
                        break

                if matched_files:
                    feature_files[f"x{feature_idx}"] = [str(path) for path in matched_files]

            if not feature_files:
                continue

            dirs_with_features += 1
            model_info = self.extract_model_info_from_path(model_dir)
            if model_info is None:
                continue

            model_info["feature_files"] = feature_files
            model_info["model_dir"] = str(model_dir)
            model_data.append(model_info)
            logger.info(
                "Found model directory %s (family=%s, index=%s)",
                model_dir,
                model_info.get("model_family"),
                model_info.get("model_index"),
            )

        logger.info("Scanned %d directories", total_dirs)
        logger.info("Found %d directories with feature files", dirs_with_features)
        logger.info("Successfully processed %d model directories", len(model_data))
        return model_data

    def extract_model_info_from_path(self, model_dir: Path) -> Optional[Dict]:
        """Extract model metadata from directory naming convention."""
        try:
            parts = model_dir.name.split("_")
            if len(parts) < 8:
                return None

            model_index = int(parts[0])
            # Run names embed the family as family.lower() (see generate_configs_lora.py),
            # so map the slug back to the canonical label rather than upper-casing it --
            # "qwen".upper() would give "QWEN", which is not a label in the mapping.
            model_family = self.FAMILY_BY_SLUG.get(parts[1].lower())
            if model_family is None:
                return None

            model_size = parts[2]

            lr_part = next((part for part in parts if part.startswith("lr")), None)
            learning_rate = float(lr_part.replace("lr", "")) if lr_part else None

            r_part = next((part for part in parts if part.startswith("r")), None)
            lora_r = int(r_part.replace("r", "")) if r_part else None

            alpha_part = next((part for part in parts if part.startswith("alpha")), None)
            lora_alpha = int(alpha_part.replace("alpha", "")) if alpha_part else None

            dropout_part = next((part for part in parts if part.startswith("dropout")), None)
            lora_dropout = float(dropout_part.replace("dropout", "")) if dropout_part else None

            return {
                "model_index": model_index,
                "model_family": model_family,
                "model_size": model_size,
                "learning_rate": learning_rate,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "num_train_epochs": 3,
                "model_output_dir": str(model_dir),
            }
        except Exception as exc:
            logger.warning("Error extracting model info from %s: %s", model_dir, str(exc))
            return None

    def create_label_encodings(self, model_info: Dict) -> Dict:
        """Create one-hot label vectors for model + LoRA hyperparameters."""
        family_label = np.zeros(len(self.model_family_mapping))
        size_label = np.zeros(len(self.model_size_mapping))
        lr_label = np.zeros(len(self.lr_mapping))
        lora_r_label = np.zeros(len(self.lora_r_mapping))
        lora_alpha_label = np.zeros(len(self.lora_alpha_mapping))
        lora_dropout_label = np.zeros(len(self.lora_dropout_mapping))

        family = str(model_info.get("model_family", ""))
        if family in self.model_family_mapping:
            family_label[self.model_family_mapping.index(family)] = 1

        size = str(model_info.get("model_size", ""))
        if size in self.model_size_mapping:
            size_label[self.model_size_mapping.index(size)] = 1

        learning_rate = model_info.get("learning_rate")
        if learning_rate in self.lr_mapping:
            lr_label[self.lr_mapping.index(learning_rate)] = 1

        lora_r = model_info.get("lora_r")
        if lora_r in self.lora_r_mapping:
            lora_r_label[self.lora_r_mapping.index(lora_r)] = 1

        lora_alpha = model_info.get("lora_alpha")
        if lora_alpha in self.lora_alpha_mapping:
            lora_alpha_label[self.lora_alpha_mapping.index(lora_alpha)] = 1

        lora_dropout = model_info.get("lora_dropout")
        if lora_dropout in self.lora_dropout_mapping:
            lora_dropout_label[self.lora_dropout_mapping.index(lora_dropout)] = 1

        return {
            "model_family_label": family_label.tolist(),
            "model_size_label": size_label.tolist(),
            "learning_rate_label": lr_label.tolist(),
            "lora_r_label": lora_r_label.tolist(),
            "lora_alpha_label": lora_alpha_label.tolist(),
            "lora_dropout_label": lora_dropout_label.tolist(),
        }

    def match_with_config(self, model_data: List[Dict], config_df: pd.DataFrame) -> List[Dict]:
        """Match scanned models with config summary to populate exact metadata."""
        logger.info("Matching model data with LoRA config summary...")
        matched_data: List[Dict] = []

        for model_info in model_data:
            model_index = model_info.get("model_index")
            model_dir = str(model_info.get("model_output_dir", ""))

            if model_index is not None:
                by_index = config_df[config_df["model_index"] == model_index]
                if not by_index.empty:
                    row = by_index.iloc[0]
                    model_info.update(
                        {
                            "model_family": row["model_family"],
                            "model_size": row["model_size"],
                            "learning_rate": row["learning_rate"],
                            "num_train_epochs": row["num_train_epochs"],
                            "lora_r": row["lora_r"],
                            "lora_alpha": row["lora_alpha"],
                            "lora_dropout": row["lora_dropout"],
                        }
                    )
                else:
                    logger.warning("No config row found for model_index=%s", model_index)

            # If the directory name did not yield a known family, fall back to matching
            # the config summary by output directory.
            if model_info.get("model_family") not in self.model_family_mapping:
                for _, row in config_df.iterrows():
                    cfg_dir = str(row["model_output_dir"])
                    if model_dir in cfg_dir or cfg_dir in model_dir:
                        model_info.update(
                            {
                                "model_family": row["model_family"],
                                "model_size": row["model_size"],
                                "learning_rate": row["learning_rate"],
                                "num_train_epochs": row["num_train_epochs"],
                                "lora_r": row["lora_r"],
                                "lora_alpha": row["lora_alpha"],
                                "lora_dropout": row["lora_dropout"],
                            }
                        )
                        break

            if model_info.get("model_family") in self.model_family_mapping:
                matched_data.append(model_info)

        logger.info("Matched %d LoRA model directories", len(matched_data))
        return matched_data

    def validate_label_mappings(self) -> None:
        """Ensure mapping arrays match generate_configs_lora.py policy."""
        expected = {
            "model_family_mapping": ["Marian", "BART", "Qwen", "LLaMA"],
            "model_size_mapping": [
                "base", "large", "0.5B", "1.5B", "1B", "3B", "7B", "8B",
            ],
            "lr_mapping": [1e-5, 5e-5, 1e-4],
            "lora_r_mapping": [4, 8, 16],
            "lora_alpha_mapping": [8, 16, 32],
            "lora_dropout_mapping": [0.05, 0.1],
        }

        for attr_name, expected_value in expected.items():
            if getattr(self, attr_name) != expected_value:
                logger.warning(
                    "Mapping mismatch for %s: got %s expected %s. Resetting.",
                    attr_name,
                    getattr(self, attr_name),
                    expected_value,
                )
                setattr(self, attr_name, expected_value)

    def create_dataloader_csv(self) -> pd.DataFrame:
        """Create dataloader.csv with feature paths and one-hot labels."""
        logger.info("Creating LoRA dataloader.csv...")
        self.validate_label_mappings()

        config_df = self.load_config_summary()
        model_data = self.scan_data_directory()
        matched_data = self.match_with_config(model_data, config_df)

        dataloader_entries: List[Dict] = []
        for model_info in matched_data:
            feature_files = model_info.get("feature_files", {})
            if not feature_files:
                continue

            min_batches = min(len(files) for files in feature_files.values())
            labels = self.create_label_encodings(model_info)

            for batch_idx in range(min_batches):
                entry = {
                    "model_index": model_info.get("model_index"),
                    "model_family": model_info.get("model_family"),
                    "model_size": model_info.get("model_size"),
                    "learning_rate": model_info.get("learning_rate"),
                    "lora_r": model_info.get("lora_r"),
                    "lora_alpha": model_info.get("lora_alpha"),
                    "lora_dropout": model_info.get("lora_dropout"),
                    "num_train_epochs": model_info.get("num_train_epochs"),
                    "model_dir": model_info.get("model_dir"),
                    "batch_index": batch_idx,
                }

                for feature_idx in range(1, 8):
                    key = f"x{feature_idx}"
                    files = feature_files.get(key, [])
                    entry[f"{key}_file"] = files[batch_idx] if batch_idx < len(files) else ""

                entry.update(labels)
                dataloader_entries.append(entry)

        dataloader_df = pd.DataFrame(dataloader_entries)
        dataloader_df.to_csv(self.output_file, index=False)
        logger.info("Created dataloader.csv with %d entries", len(dataloader_df))
        self.print_summary_statistics(dataloader_df)
        return dataloader_df

    def print_summary_statistics(self, df: pd.DataFrame) -> None:
        """Print key summary statistics for generated dataloader."""
        logger.info("\n" + "=" * 60)
        logger.info("LORA DATALOADER SUMMARY STATISTICS")
        logger.info("=" * 60)
        logger.info("Total entries: %d", len(df))

        if df.empty:
            logger.warning("No entries found. Check feature generation and config paths.")
            logger.info("=" * 60)
            return

        for column in [
            "model_family",
            "model_size",
            "learning_rate",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
        ]:
            if column in df.columns:
                logger.info("\n%s distribution:", column)
                for value, count in df[column].value_counts().items():
                    logger.info("  %s: %s", value, count)

        logger.info("\nFeature file availability:")
        for feature_idx in range(1, 8):
            column = f"x{feature_idx}_file"
            available = df[column].astype(str).str.len() > 0
            logger.info(
                "  x%s: %s/%s (%.1f%%)",
                feature_idx,
                int(available.sum()),
                len(df),
                float(available.mean() * 100.0),
            )
        logger.info("=" * 60)

    def save_label_mappings(self) -> None:
        """Save label mappings used in dataloader label vectors."""
        label_mappings = {
            "model_family": self.model_family_mapping,
            "model_size": self.model_size_mapping,
            "learning_rate": [str(value) for value in self.lr_mapping],
            "lora_r": self.lora_r_mapping,
            "lora_alpha": self.lora_alpha_mapping,
            "lora_dropout": self.lora_dropout_mapping,
        }

        output_path = self.dataloader_dir / "label_mappings.json"
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(label_mappings, file, indent=2)
        logger.info("Saved label mappings to %s", output_path)


def main():
    """Entry point for creating LoRA-aware dataloader artifacts."""
    try:
        creator = LoRADataLoaderCreator()
        dataloader_df = creator.create_dataloader_csv()
        creator.save_label_mappings()

        logger.info("Successfully created LoRA dataloader.csv and label_mappings.json")
        logger.info("Files created in %s", creator.dataloader_dir)
        logger.info("  - dataloader.csv (%d entries)", len(dataloader_df))
        logger.info("  - label_mappings.json")
    except Exception as exc:
        logger.error("Error creating LoRA dataloader: %s", str(exc))
        raise


if __name__ == "__main__":
    main()
