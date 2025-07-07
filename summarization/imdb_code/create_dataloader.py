"""
Data Loader Creator for Multimodal Multi-label Multi-class Classification

This script scans the ./results folder and config_summary.csv file to create a dataloader.csv
file that contains all the necessary information for training the hyperparameter stealing model.

The dataloader.csv will include:
- Model index
- Model family (BART, GPT-2, Pegasus, Mistral, Qwen, LLaMA)
- Model size (base, large, small, medium, 0.5B, 1.8B, 7B, 13B, xsum)
- Optimizer type (AdamW, SGD, Adafactor)
- Learning rate (1e-5, 5e-5, 1e-4)
- Batch size (4, 8, 16)
- Number of epochs (3)
- Feature file paths (x1-x7)
- Label encodings for multi-label classification
"""

import os
import pandas as pd
import numpy as np
import json
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoaderCreator:
    """Creates dataloader.csv for multi-label multi-class classification."""
    
    def __init__(self, data_dir: str = "./multimodal_dataset", config_path: str = "./configs/config_summary.csv"):
        """
        Initialize the DataLoaderCreator.
        
        Args:
            data_dir (str): Directory containing multimodal dataset with features
            config_path (str): Path to config_summary.csv file
        """
        self.data_dir = Path(data_dir)
        self.config_path = Path(config_path)
        
        # Create dataloader directory if it doesn't exist
        self.dataloader_dir = Path("dataloader")
        self.dataloader_dir.mkdir(exist_ok=True)
        logger.info(f"Using dataloader directory: {self.dataloader_dir.absolute()}")
        
        self.output_file = self.dataloader_dir / "dataloader.csv"
        
        # Define label mappings for each hyperparameter
        self.model_family_mapping = ['BART', 'GPT-2', 'Pegasus', 'Mistral', 'Qwen', 'LLaMA']
        self.model_size_mapping = ['base', 'large', 'small', 'medium', '0.5B', '1.8B', '7B', '13B', 'xsum']
        self.optimizer_mapping = ['adamw', 'sgd', 'adafactor']
        self.lr_mapping = [1e-5, 5e-5, 1e-4]
        self.bs_mapping = [4, 8, 16]
        
        # Verify paths exist
        if not self.data_dir.exists():
            raise FileNotFoundError(f"multimodal_dataset directory not found: {self.data_dir}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        # Debug: Check what's in the data directory
        logger.info(f"Multimodal dataset directory contents:")
        if self.data_dir.exists():
            for item in list(self.data_dir.iterdir())[:10]:  # Show first 10 items
                if item.is_dir():
                    logger.info(f"  Directory: {item.name}")
                else:
                    logger.info(f"  File: {item.name}")
            if len(list(self.data_dir.iterdir())) > 10:
                logger.info(f"  ... and {len(list(self.data_dir.iterdir())) - 10} more items")
        else:
            logger.warning("Multimodal dataset directory does not exist!")
    
    def load_config_summary(self) -> pd.DataFrame:
        """Load and validate the config summary CSV file."""
        logger.info(f"Loading config summary from {self.config_path}")
        
        config_df = pd.read_csv(self.config_path)
        logger.info(f"Loaded {len(config_df)} configurations")
        
        # Validate required columns
        required_columns = [
            'model_index', 'model_family', 'model_size', 'optimizer', 
            'learning_rate', 'batch_size', 'num_train_epochs', 'model_output_dir'
        ]
        
        missing_columns = [col for col in required_columns if col not in config_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in config_summary.csv: {missing_columns}")
        
        return config_df
    
    def scan_data_directory(self) -> List[Dict]:
        """Scan the multimodal dataset directory to find all model directories and their feature files."""
        logger.info(f"Scanning multimodal dataset directory: {self.data_dir}")
        
        model_data = []
        total_dirs = 0
        dirs_with_features = 0
        
        # Walk through all subdirectories in multimodal dataset folder
        for model_dir in self.data_dir.rglob("*"):
            if not model_dir.is_dir():
                continue
            
            total_dirs += 1
            
            # Check if this directory contains feature files (x1-x7)
            feature_files = {}
            
            # Search directly in the model directory (no need for final_model subdirectory)
            search_dir = model_dir
            
            for i in range(1, 8):  # x1 to x7
                # Try different possible patterns for feature files (only .npy)
                patterns_to_try = [
                    f"x{i}_batch_*.npy",  # Original pattern with .npy extension
                    f"x{i}_*.npy",        # Simplified pattern with .npy extension
                    f"*x{i}*.npy",        # Contains x{i} with .npy extension
                    f"features_x{i}*.npy", # Features with x{i} and .npy extension
                    f"*features*{i}*.npy"  # Features containing {i} with .npy extension
                ]
                
                x_files = []
                for pattern in patterns_to_try:
                    x_files = list(search_dir.glob(pattern))
                    if x_files:
                        break
                
                if x_files:
                    feature_files[f'x{i}'] = [str(f) for f in x_files]
            
            # Also check if there are any .npy files that might be features
            if not feature_files:
                npy_files = list(search_dir.glob("*.npy"))
                if npy_files:
                    logger.info(f"Found potential .npy feature files in {search_dir}:")
                    logger.info(f"  .npy files: {[f.name for f in npy_files[:5]]}")
            
            if not feature_files:
                continue
            
            dirs_with_features += 1
            
            # Extract model information from directory path
            try:
                model_info = self.extract_model_info_from_path(model_dir)
                if model_info:
                    model_info['feature_files'] = feature_files
                    model_info['model_dir'] = str(model_dir)
                    model_data.append(model_info)
                    logger.info(f"Found model directory: {model_dir}")
                else:
                    logger.warning(f"Could not extract model info from {model_dir}")
            except Exception as e:
                logger.warning(f"Could not extract model info from {model_dir}: {str(e)}")
                continue
        
        logger.info(f"Scanned {total_dirs} directories")
        logger.info(f"Found {dirs_with_features} directories with feature files")
        logger.info(f"Successfully processed {len(model_data)} model directories")
        
        # Debug: List some directories that were found
        if total_dirs > 0 and dirs_with_features == 0:
            logger.warning("No directories with feature files found. Checking first few directories:")
            count = 0
            for model_dir in self.data_dir.rglob("*"):
                if not model_dir.is_dir() or count >= 5:
                    break
                count += 1
                files = list(model_dir.glob("*"))
                logger.info(f"  {model_dir}: {len(files)} files")
                if files:
                    logger.info(f"    Sample files: {[f.name for f in files[:5]]}")
                    
                    # Check for feature files directly in the directory (only .npy)
                    for i in range(1, 8):
                        x_files = list(model_dir.glob(f"*x{i}*.npy"))
                        if x_files:
                            logger.info(f"      Found x{i} .npy files: {[f.name for f in x_files[:3]]}")
        
        return model_data
    
    def extract_model_info_from_path(self, model_dir: Path) -> Optional[Dict]:
        """Extract model information from directory path."""
        try:
            # Get the relative path from data directory
            rel_path = model_dir.relative_to(self.data_dir)
            path_parts = rel_path.parts
            
            # Try to extract model index from directory name
            dir_name = model_dir.name
            
            # Look for patterns like "0_bart_base_adamw_lr1e-05_bs4"
            if '_' in dir_name:
                parts = dir_name.split('_')
                if len(parts) >= 4:
                    try:
                        model_index = int(parts[0])
                        model_family = parts[1].upper()
                        model_size = parts[2]
                        optimizer = parts[3]
                        
                        # Extract learning rate and batch size
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
                            'num_train_epochs': 3,  # Default value
                            'model_output_dir': str(model_dir)
                        }
                    except (ValueError, IndexError):
                        pass
            
            # If we can't extract from directory name, try to match with config
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting model info from {model_dir}: {str(e)}")
            return None
    
    def create_label_encodings(self, model_info: Dict) -> Dict:
        """Create one-hot encoded labels for multi-label classification."""
        # Initialize label arrays
        family_label = np.zeros(len(self.model_family_mapping))
        size_label = np.zeros(len(self.model_size_mapping))
        optimizer_label = np.zeros(len(self.optimizer_mapping))
        lr_label = np.zeros(len(self.lr_mapping))
        bs_label = np.zeros(len(self.bs_mapping))
        
        # Set model family label
        family = model_info.get('model_family', '').upper()
        if family in self.model_family_mapping:
            family_idx = self.model_family_mapping.index(family)
            family_label[family_idx] = 1
        
        # Set model size label
        size = model_info.get('model_size', '')
        if size in self.model_size_mapping:
            size_idx = self.model_size_mapping.index(size)
            size_label[size_idx] = 1
        
        # Set optimizer label
        opt = model_info.get('optimizer', '').lower()
        if opt in self.optimizer_mapping:
            opt_idx = self.optimizer_mapping.index(opt)
            optimizer_label[opt_idx] = 1
        
        # Set learning rate label
        lr = model_info.get('learning_rate')
        if lr is not None and lr in self.lr_mapping:
            lr_idx = self.lr_mapping.index(lr)
            lr_label[lr_idx] = 1
        
        # Set batch size label
        bs = model_info.get('batch_size')
        if bs is not None and bs in self.bs_mapping:
            bs_idx = self.bs_mapping.index(bs)
            bs_label[bs_idx] = 1
        
        return {
            'model_family_label': family_label.tolist(),
            'model_size_label': size_label.tolist(),
            'optimizer_label': optimizer_label.tolist(),
            'learning_rate_label': lr_label.tolist(),
            'batch_size_label': bs_label.tolist()
        }
    
    def match_with_config(self, model_data: List[Dict], config_df: pd.DataFrame) -> List[Dict]:
        """Match model data with config summary to fill missing information."""
        logger.info("Matching model data with config summary...")
        
        matched_data = []
        
        for model_info in model_data:
            model_index = model_info.get('model_index')
            model_dir = model_info.get('model_output_dir', '')
            
            # Try to find matching config by model index
            if model_index is not None:
                matching_configs = config_df[config_df['model_index'] == model_index]
                if not matching_configs.empty:
                    config_row = matching_configs.iloc[0]
                    # Update model info with config data
                    model_info.update({
                        'model_family': config_row['model_family'],
                        'model_size': config_row['model_size'],
                        'optimizer': config_row['optimizer'],
                        'learning_rate': config_row['learning_rate'],
                        'batch_size': config_row['batch_size'],
                        'num_train_epochs': config_row['num_train_epochs']
                    })
                    logger.info(f"Matched model {model_index} with config")
                else:
                    logger.warning(f"No config found for model index {model_index}")
            
            # If still no match, try matching by directory path
            if model_info.get('model_family') is None:
                for _, config_row in config_df.iterrows():
                    if model_dir in config_row['model_output_dir'] or config_row['model_output_dir'] in model_dir:
                        model_info.update({
                            'model_family': config_row['model_family'],
                            'model_size': config_row['model_size'],
                            'optimizer': config_row['optimizer'],
                            'learning_rate': config_row['learning_rate'],
                            'batch_size': config_row['batch_size'],
                            'num_train_epochs': config_row['num_train_epochs']
                        })
                        logger.info(f"Matched model directory with config")
                        break
            
            matched_data.append(model_info)
        
        return matched_data
    
    def create_dataloader_csv(self) -> pd.DataFrame:
        """Create the dataloader.csv file with all necessary information."""
        logger.info("Creating dataloader.csv...")
        
        # Load config summary
        config_df = self.load_config_summary()
        
        # Scan multimodal dataset directory
        model_data = self.scan_data_directory()
        
        # Match with config
        matched_data = self.match_with_config(model_data, config_df)
        
        # Create dataloader entries
        dataloader_entries = []
        
        for model_info in matched_data:
            # Create label encodings
            label_encodings = self.create_label_encodings(model_info)
            
            # Get feature files
            feature_files = model_info.get('feature_files', {})
            
            # Create entry for each batch
            for batch_idx in range(min(len(files) for files in feature_files.values()) if feature_files else 0):
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
                
                # Add feature file paths
                for i in range(1, 8):
                    x_key = f'x{i}'
                    if x_key in feature_files and batch_idx < len(feature_files[x_key]):
                        entry[f'{x_key}_file'] = feature_files[x_key][batch_idx]
                    else:
                        entry[f'{x_key}_file'] = ''
                
                # Add label encodings
                entry.update(label_encodings)
                
                dataloader_entries.append(entry)
        
        # Create DataFrame
        dataloader_df = pd.DataFrame(dataloader_entries)
        
        # Save to CSV
        dataloader_df.to_csv(self.output_file, index=False)
        logger.info(f"Created dataloader.csv with {len(dataloader_df)} entries")
        
        # Print summary statistics
        self.print_summary_statistics(dataloader_df)
        
        return dataloader_df
    
    def print_summary_statistics(self, df: pd.DataFrame):
        """Print summary statistics of the created dataloader."""
        logger.info("\n" + "="*50)
        logger.info("DATALOADER SUMMARY STATISTICS")
        logger.info("="*50)
        
        logger.info(f"Total entries: {len(df)}")
        
        # Check if DataFrame is empty
        if len(df) == 0:
            logger.warning("No data found! Please check:")
            logger.warning("1. The 'multimodal_dataset' directory exists and contains model directories")
            logger.warning("2. Model directories contain feature files (x1_batch_*, x2_batch_*, etc.)")
            logger.warning("3. The config_summary.csv file is properly formatted")
            logger.info("="*50)
            return
        
        # Only proceed with statistics if DataFrame has data
        if 'model_index' in df.columns:
            logger.info(f"Unique models: {df['model_index'].nunique()}")
        
        # Model family distribution
        if 'model_family' in df.columns:
            logger.info("\nModel Family Distribution:")
            family_counts = df['model_family'].value_counts()
            for family, count in family_counts.items():
                logger.info(f"  {family}: {count}")
        
        # Model size distribution
        if 'model_size' in df.columns:
            logger.info("\nModel Size Distribution:")
            size_counts = df['model_size'].value_counts()
            for size, count in size_counts.items():
                logger.info(f"  {size}: {count}")
        
        # Optimizer distribution
        if 'optimizer' in df.columns:
            logger.info("\nOptimizer Distribution:")
            opt_counts = df['optimizer'].value_counts()
            for opt, count in opt_counts.items():
                logger.info(f"  {opt}: {count}")
        
        # Learning rate distribution
        if 'learning_rate' in df.columns:
            logger.info("\nLearning Rate Distribution:")
            lr_counts = df['learning_rate'].value_counts()
            for lr, count in lr_counts.items():
                logger.info(f"  {lr}: {count}")
        
        # Batch size distribution
        if 'batch_size' in df.columns:
            logger.info("\nBatch Size Distribution:")
            bs_counts = df['batch_size'].value_counts()
            for bs, count in bs_counts.items():
                logger.info(f"  {bs}: {count}")
        
        # Feature file availability
        logger.info("\nFeature File Availability:")
        for i in range(1, 8):
            x_key = f'x{i}_file'
            if x_key in df.columns:
                available = df[x_key].str.len() > 0
                logger.info(f"  x{i}: {available.sum()}/{len(df)} ({available.sum()/len(df)*100:.1f}%)")
        
        logger.info("="*50)
    
    def save_label_mappings(self):
        """Save label mappings to JSON file for reference."""
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
    """Main function to create the dataloader."""
    try:
        # Initialize creator
        creator = DataLoaderCreator()
        
        # Create dataloader CSV
        dataloader_df = creator.create_dataloader_csv()
        
        # Save label mappings
        creator.save_label_mappings()
        
        logger.info("Successfully created dataloader.csv and label_mappings.json")
        logger.info(f"Files created in {creator.dataloader_dir}:")
        logger.info(f"  - dataloader.csv ({len(dataloader_df)} entries)")
        logger.info(f"  - label_mappings.json")
        
    except Exception as e:
        logger.error(f"Error creating dataloader: {str(e)}")
        raise

if __name__ == "__main__":
    main() 