"""
Data Loader Creator for Multimodal Multi-label Multi-class Classification

Builds dataloader.csv + label_mappings.json from ./multimodal_dataset and
configs/config_summary.csv — same pipeline as create_dataloader_only_gpt2_phi_llama3-1_qwen2_5.py,
extended to include BART and Pegasus alongside GPT-2, Phi, LLaMA (3.1 8B), and Qwen (2.5 7B).

Families and sizes follow generate_configs.py:
- BART: base, large
- Pegasus: xsum, large
- GPT-2: small, medium, large
- Phi: 1.5
- LLaMA: 8B
- Qwen: 7B
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPPORTED_FAMILIES = ['BART', 'Pegasus', 'GPT-2', 'Phi', 'LLaMA', 'Qwen']

# All distinct model_size values across those families (shared 'large' slot for BART / Pegasus / GPT-2)
MODEL_FAMILY_MAPPING = ['BART', 'Pegasus', 'GPT-2', 'Phi', 'LLaMA', 'Qwen']
MODEL_SIZE_MAPPING = ['base', 'large', 'xsum', 'small', 'medium', '1.5', '8B', '7B']


class DataLoaderCreator:
    """Creates dataloader.csv for BART, Pegasus, GPT-2, Phi, LLaMA, and Qwen."""

    def __init__(self, data_dir: str = "./multimodal_dataset", config_path: str = "./configs/config_summary.csv"):
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)

        self.dataloader_dir = Path("dataloader")
        self.dataloader_dir.mkdir(exist_ok=True)
        logger.info(f"Using dataloader directory: {self.dataloader_dir.absolute()}")

        self.output_file = self.dataloader_dir / "dataloader.csv"

        self.model_family_mapping = list(MODEL_FAMILY_MAPPING)
        self.model_size_mapping = list(MODEL_SIZE_MAPPING)
        self.optimizer_mapping = ['adamw', 'sgd', 'adafactor']
        self.lr_mapping = [1e-5, 5e-5, 1e-4]
        self.bs_mapping = [4, 8, 16]

        if not self.data_dir.exists():
            raise FileNotFoundError(f"multimodal_dataset directory not found: {self.data_dir}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        logger.info("Multimodal dataset directory contents:")
        if self.data_dir.exists():
            for item in list(self.data_dir.iterdir())[:10]:
                if item.is_dir():
                    logger.info(f"  Directory: {item.name}")
                else:
                    logger.info(f"  File: {item.name}")
            if len(list(self.data_dir.iterdir())) > 10:
                logger.info(f"  ... and {len(list(self.data_dir.iterdir())) - 10} more items")
        else:
            logger.warning("Multimodal dataset directory does not exist!")

    def load_config_summary(self) -> pd.DataFrame:
        logger.info(f"Loading config summary from {self.config_path}")

        config_df = pd.read_csv(self.config_path)
        logger.info(f"Loaded {len(config_df)} configurations")

        filtered_configs = config_df[config_df['model_family'].isin(SUPPORTED_FAMILIES)]
        logger.info(f"Found {len(filtered_configs)} rows for BART, Pegasus, GPT-2, Phi, LLaMA, Qwen")
        for fam in SUPPORTED_FAMILIES:
            n = len(filtered_configs[filtered_configs['model_family'] == fam])
            logger.info(f"  {fam}: {n}")

        required_columns = [
            'model_index', 'model_family', 'model_size', 'optimizer',
            'learning_rate', 'batch_size', 'num_train_epochs', 'model_output_dir'
        ]

        missing_columns = [col for col in required_columns if col not in filtered_configs.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in config_summary.csv: {missing_columns}")

        return filtered_configs

    def scan_data_directory(self) -> List[Dict]:
        logger.info(f"Scanning multimodal dataset directory: {self.data_dir}")

        model_data = []
        total_dirs = 0
        dirs_with_features = 0

        for model_dir in self.data_dir.rglob("*"):
            if not model_dir.is_dir():
                continue

            total_dirs += 1

            if not self.is_supported_model_directory(model_dir):
                continue

            feature_files = {}
            search_dir = model_dir

            for i in range(1, 8):
                patterns_to_try = [
                    f"x{i}_batch_*.npy",
                    f"x{i}_*.npy",
                    f"*x{i}*.npy",
                    f"features_x{i}*.npy",
                    f"*features*{i}*.npy"
                ]

                x_files = []
                for pattern in patterns_to_try:
                    x_files = list(search_dir.glob(pattern))
                    if x_files:
                        break

                if x_files:
                    feature_files[f'x{i}'] = [str(f) for f in x_files]

            if not feature_files:
                npy_files = list(search_dir.glob("*.npy"))
                if npy_files:
                    logger.info(f"Found potential .npy feature files in {search_dir}:")
                    logger.info(f"  .npy files: {[f.name for f in npy_files[:5]]}")

            if not feature_files:
                continue

            dirs_with_features += 1

            try:
                model_info = self.extract_model_info_from_path(model_dir)
                if model_info:
                    model_info['feature_files'] = feature_files
                    model_info['model_dir'] = str(model_dir)
                    model_data.append(model_info)

                    family = model_info.get('model_family', 'Unknown')
                    model_index = model_info.get('model_index', 'Unknown')
                    logger.info(f"Found model directory: {model_dir} (Family: {family}, Index: {model_index})")
                else:
                    logger.warning(f"Could not extract model info from {model_dir}")
            except Exception as e:
                logger.warning(f"Could not extract model info from {model_dir}: {str(e)}")
                continue

        logger.info(f"Scanned {total_dirs} directories")
        logger.info(f"Found {dirs_with_features} supported model directories with feature files")
        logger.info(f"Successfully processed {len(model_data)} model directories")

        if model_data:
            family_counts = {}
            for model in model_data:
                family = model.get('model_family', 'Unknown')
                family_counts[family] = family_counts.get(family, 0) + 1

            logger.info("Model family distribution from scanning:")
            for family, count in family_counts.items():
                logger.info(f"  {family}: {count}")

        if total_dirs > 0 and dirs_with_features == 0:
            logger.warning("No supported directories with feature files found. Checking first few directories:")
            count = 0
            for model_dir in self.data_dir.rglob("*"):
                if not model_dir.is_dir() or count >= 5:
                    break
                count += 1
                files = list(model_dir.glob("*"))
                logger.info(f"  {model_dir}: {len(files)} files")
                if files:
                    logger.info(f"    Sample files: {[f.name for f in files[:5]]}")

                    for i in range(1, 8):
                        x_files = list(model_dir.glob(f"*x{i}*.npy"))
                        if x_files:
                            logger.info(f"      Found x{i} .npy files: {[f.name for f in x_files[:3]]}")

        return model_data

    def is_supported_model_directory(self, model_dir: Path) -> bool:
        """Path hints at one of the six families (output dirs from generate_configs)."""
        p = str(model_dir).lower()
        return (
            'bart' in p
            or 'pegasus' in p
            or 'gpt-2' in p
            or 'gpt2' in p
            or 'phi' in p
            or 'llama' in p
            or 'qwen' in p
        )

    def extract_model_info_from_path(self, model_dir: Path) -> Optional[Dict]:
        """Parse leaf dir names like {idx}_{family}_{size}_{opt}_lr{_}_bs{_} (generate_configs pattern)."""
        try:
            dir_name = model_dir.name

            if '_' in dir_name:
                parts = dir_name.split('_')
                if len(parts) >= 4:
                    try:
                        model_index = int(parts[0])

                        family_part = parts[1].lower()
                        if family_part == 'bart':
                            model_family = 'BART'
                        elif family_part == 'pegasus':
                            model_family = 'Pegasus'
                        elif family_part == 'gpt-2' or family_part == 'gpt2':
                            model_family = 'GPT-2'
                        elif 'phi' in family_part:
                            model_family = 'Phi'
                        elif family_part == 'llama':
                            model_family = 'LLaMA'
                        elif family_part == 'qwen':
                            model_family = 'Qwen'
                        else:
                            return None

                        model_size = parts[2]
                        optimizer = parts[3]

                        lr_part = next((p for p in parts if p.startswith('lr')), None)
                        lr = float(lr_part.replace('lr', '')) if lr_part else None

                        bs_part = next((p for p in parts if p.startswith('bs')), None)
                        batch_size = int(bs_part.replace('bs', '')) if bs_part else None

                        return {
                            'model_index': model_index,
                            'model_family': model_family,
                            'model_size': model_size,
                            'optimizer': optimizer,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'num_train_epochs': 3,
                            'model_output_dir': str(model_dir)
                        }
                    except (ValueError, IndexError):
                        pass

            return None

        except Exception as e:
            logger.warning(f"Error extracting model info from {model_dir}: {str(e)}")
            return None

    def create_label_encodings(self, model_info: Dict) -> Dict:
        family_label = np.zeros(len(self.model_family_mapping))
        size_label = np.zeros(len(self.model_size_mapping))
        optimizer_label = np.zeros(len(self.optimizer_mapping))
        lr_label = np.zeros(len(self.lr_mapping))
        bs_label = np.zeros(len(self.bs_mapping))

        family = model_info.get('model_family', '')

        if family in self.model_family_mapping:
            family_idx = self.model_family_mapping.index(family)
            family_label[family_idx] = 1
        else:
            family_lower = family.lower()
            for i, mapped_family in enumerate(self.model_family_mapping):
                if family_lower == mapped_family.lower():
                    family_label[i] = 1
                    break
            else:
                for i, mapped_family in enumerate(self.model_family_mapping):
                    if family.lower() in mapped_family.lower() or mapped_family.lower() in family.lower():
                        family_label[i] = 1
                        break
                else:
                    logger.warning(f"Model family '{family}' not found in mapping: {self.model_family_mapping}")

        size = model_info.get('model_size', '')
        if size in self.model_size_mapping:
            size_idx = self.model_size_mapping.index(size)
            size_label[size_idx] = 1
        else:
            logger.warning(f"Model size '{size}' not found in mapping: {self.model_size_mapping}")

        opt = model_info.get('optimizer', '').lower()
        if opt in self.optimizer_mapping:
            opt_idx = self.optimizer_mapping.index(opt)
            optimizer_label[opt_idx] = 1
        else:
            logger.warning(f"Optimizer '{opt}' not found in mapping: {self.optimizer_mapping}")

        lr = model_info.get('learning_rate')
        if lr is not None and lr in self.lr_mapping:
            lr_idx = self.lr_mapping.index(lr)
            lr_label[lr_idx] = 1
        else:
            logger.warning(f"Learning rate '{lr}' not found in mapping: {self.lr_mapping}")

        bs = model_info.get('batch_size')
        if bs is not None and bs in self.bs_mapping:
            bs_idx = self.bs_mapping.index(bs)
            bs_label[bs_idx] = 1
        else:
            logger.warning(f"Batch size '{bs}' not found in mapping: {self.bs_mapping}")

        return {
            'model_family_label': family_label.tolist(),
            'model_size_label': size_label.tolist(),
            'optimizer_label': optimizer_label.tolist(),
            'learning_rate_label': lr_label.tolist(),
            'batch_size_label': bs_label.tolist()
        }

    def match_with_config(self, model_data: List[Dict], config_df: pd.DataFrame) -> List[Dict]:
        logger.info("Matching model data with config summary...")

        matched_data = []

        for model_info in model_data:
            model_index = model_info.get('model_index')
            model_dir = model_info.get('model_output_dir', '')

            if model_index is not None:
                matching_configs = config_df[config_df['model_index'] == model_index]
                if not matching_configs.empty:
                    config_row = matching_configs.iloc[0]
                    old_family = model_info.get('model_family')
                    model_info.update({
                        'model_family': config_row['model_family'],
                        'model_size': config_row['model_size'],
                        'optimizer': config_row['optimizer'],
                        'learning_rate': config_row['learning_rate'],
                        'batch_size': config_row['batch_size'],
                        'num_train_epochs': config_row['num_train_epochs']
                    })
                    logger.info(f"Matched model {model_index} with config: {old_family} -> {config_row['model_family']}")
                else:
                    logger.warning(f"No config found for model index {model_index}")

            if model_info.get('model_family') is None:
                for _, config_row in config_df.iterrows():
                    if model_dir in str(config_row['model_output_dir']) or str(config_row['model_output_dir']) in model_dir:
                        model_info.update({
                            'model_family': config_row['model_family'],
                            'model_size': config_row['model_size'],
                            'optimizer': config_row['optimizer'],
                            'learning_rate': config_row['learning_rate'],
                            'batch_size': config_row['batch_size'],
                            'num_train_epochs': config_row['num_train_epochs']
                        })
                        logger.info(f"Matched model directory with config: -> {config_row['model_family']}")
                        break

            matched_data.append(model_info)

        return matched_data

    def validate_label_mappings(self):
        logger.info("Validating label mappings...")

        if self.model_family_mapping != MODEL_FAMILY_MAPPING:
            logger.warning("Resetting model_family_mapping to canonical list")
            self.model_family_mapping = list(MODEL_FAMILY_MAPPING)

        if self.model_size_mapping != MODEL_SIZE_MAPPING:
            logger.warning("Resetting model_size_mapping to canonical list")
            self.model_size_mapping = list(MODEL_SIZE_MAPPING)

        expected_optimizers = ['adamw', 'sgd', 'adafactor']
        if self.optimizer_mapping != expected_optimizers:
            self.optimizer_mapping = expected_optimizers

        expected_lrs = [1e-5, 5e-5, 1e-4]
        if self.lr_mapping != expected_lrs:
            self.lr_mapping = expected_lrs

        expected_bs = [4, 8, 16]
        if self.bs_mapping != expected_bs:
            self.bs_mapping = expected_bs

        logger.info(f"Final model family mapping: {self.model_family_mapping}")
        logger.info(f"Final model size mapping: {self.model_size_mapping}")

    def create_dataloader_csv(self) -> pd.DataFrame:
        logger.info(
            "Creating dataloader.csv for BART, Pegasus, GPT-2, Phi, LLaMA (3.1 8B), Qwen (2.5 7B)..."
        )

        self.validate_label_mappings()

        config_df = self.load_config_summary()
        model_data = self.scan_data_directory()
        matched_data = self.match_with_config(model_data, config_df)

        matched_data = [
            m for m in matched_data
            if m.get('model_family') in SUPPORTED_FAMILIES
        ]
        logger.info(f"After filtering, {len(matched_data)} models remain")

        dataloader_entries = []
        family_counts = {}

        for model_info in matched_data:
            label_encodings = self.create_label_encodings(model_info)

            family = model_info.get('model_family', 'Unknown')
            family_counts[family] = family_counts.get(family, 0) + 1

            feature_files = model_info.get('feature_files', {})

            n_batches = min(len(files) for files in feature_files.values()) if feature_files else 0
            for batch_idx in range(n_batches):
                entry = {
                    'model_index': model_info.get('model_index'),
                    'model_family': model_info.get('model_family'),
                    'model_size': model_info.get('model_size'),
                    'optimizer': model_info.get('optimizer'),
                    'learning_rate': model_info.get('learning_rate'),
                    'batch_size': model_info.get('batch_size'),
                    'num_train_epochs': model_info.get('num_train_epochs'),
                    'model_dir': model_info.get('model_dir'),
                    'batch_index': batch_idx,
                }

                for i in range(1, 8):
                    x_key = f'x{i}'
                    if x_key in feature_files and batch_idx < len(feature_files[x_key]):
                        entry[f'{x_key}_file'] = feature_files[x_key][batch_idx]
                    else:
                        entry[f'{x_key}_file'] = ''

                entry.update(label_encodings)
                dataloader_entries.append(entry)

        logger.info("Model family distribution in matched data:")
        for family, count in family_counts.items():
            logger.info(f"  {family}: {count}")

        dataloader_df = pd.DataFrame(dataloader_entries)

        dataloader_df.to_csv(self.output_file, index=False)
        logger.info(f"Created dataloader.csv with {len(dataloader_df)} entries")

        self.print_summary_statistics(dataloader_df)

        return dataloader_df

    def print_summary_statistics(self, df: pd.DataFrame):
        logger.info("\n" + "=" * 50)
        logger.info("DATALOADER SUMMARY (BART, Pegasus, GPT-2, Phi, LLaMA, Qwen)")
        logger.info("=" * 50)

        logger.info(f"Total entries: {len(df)}")

        if len(df) == 0:
            logger.warning("No data found. Check multimodal_dataset paths, feature .npy files,")
            logger.warning("and configs/config_summary.csv for the six families.")
            logger.info("=" * 50)
            return

        if 'model_index' in df.columns:
            logger.info(f"Unique models: {df['model_index'].nunique()}")

        if 'model_family' in df.columns:
            logger.info("\nModel Family Distribution:")
            for family, count in df['model_family'].value_counts().items():
                logger.info(f"  {family}: {count}")

        if 'model_size' in df.columns:
            logger.info("\nModel Size Distribution:")
            for size, count in df['model_size'].value_counts().items():
                logger.info(f"  {size}: {count}")

        if 'optimizer' in df.columns:
            logger.info("\nOptimizer Distribution:")
            for opt, count in df['optimizer'].value_counts().items():
                logger.info(f"  {opt}: {count}")

        if 'learning_rate' in df.columns:
            logger.info("\nLearning Rate Distribution:")
            for lr, count in df['learning_rate'].value_counts().items():
                logger.info(f"  {lr}: {count}")

        if 'batch_size' in df.columns:
            logger.info("\nBatch Size Distribution:")
            for bs, count in df['batch_size'].value_counts().items():
                logger.info(f"  {bs}: {count}")

        logger.info("\nFeature File Availability:")
        for i in range(1, 8):
            x_key = f'x{i}_file'
            if x_key in df.columns:
                available = df[x_key].astype(str).str.len() > 0
                logger.info(f"  x{i}: {available.sum()}/{len(df)} ({available.sum()/len(df)*100:.1f}%)")

        logger.info("=" * 50)

    def save_label_mappings(self):
        label_mappings = {
            'model_family': self.model_family_mapping,
            'model_size': self.model_size_mapping,
            'optimizer': self.optimizer_mapping,
            'learning_rate': [str(lr) for lr in self.lr_mapping],
            'batch_size': self.bs_mapping
        }

        label_mappings_path = self.dataloader_dir / "label_mappings.json"
        with open(label_mappings_path, 'w') as f:
            json.dump(label_mappings, f, indent=2)

        logger.info(f"Saved label mappings to {label_mappings_path}")


def main():
    try:
        creator = DataLoaderCreator()

        logger.info("Families: BART, Pegasus, GPT-2, Phi, LLaMA, Qwen (generate_configs.py)")
        logger.info("LLaMA 3.1: Meta-Llama-3.1-8B (size 8B); Qwen 2.5: Qwen2.5-7B (size 7B)")

        dataloader_df = creator.create_dataloader_csv()
        creator.save_label_mappings()

        logger.info("Successfully created dataloader.csv and label_mappings.json")
        logger.info(f"Output directory: {creator.dataloader_dir}")
        logger.info(f"  - dataloader.csv ({len(dataloader_df)} rows)")
        logger.info("  - label_mappings.json")

    except Exception as e:
        logger.error(f"Error creating dataloader: {str(e)}")
        raise


if __name__ == "__main__":
    main()
