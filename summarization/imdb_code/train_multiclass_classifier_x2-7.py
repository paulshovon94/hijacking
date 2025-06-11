"""
Script to train a multiclass classifier using x2-x7 features.
This version uses a specialized architecture to handle different feature types.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import logging
from typing import List, Dict, Tuple, Optional
import json
import argparse
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
CACHE_DIR = "/work/shovon/LLM/"  # Base directory for caching models and data

# Set up cache directories
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'transformers')
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, 'huggingface')
os.environ['HF_DATASETS_CACHE'] = os.path.join(CACHE_DIR, 'datasets')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(CACHE_DIR, 'sentence-transformers')
os.environ['NLTK_DATA'] = os.path.join(CACHE_DIR, 'nltk_data')
os.environ['TORCH_HOME'] = os.path.join(CACHE_DIR, 'torch')

# Create cache directories
for cache_path in [
    os.environ['TRANSFORMERS_CACHE'],
    os.environ['HF_HOME'],
    os.environ['HF_DATASETS_CACHE'],
    os.environ['SENTENCE_TRANSFORMERS_HOME'],
    os.environ['NLTK_DATA'],
    os.environ['TORCH_HOME']
]:
    os.makedirs(cache_path, exist_ok=True)
    logger.info(f"Using cache directory: {cache_path}")

class MultiFeatureDataset(Dataset):
    """Dataset for handling multiple feature types (x2 and x4)."""
    
    def __init__(self, features_dict: Dict[str, np.ndarray], labels: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            features_dict (Dict[str, np.ndarray]): Dictionary of feature matrices for x2 and x4
            labels (np.ndarray): Label matrix
        """
        self.features = {
            k: torch.FloatTensor(v) for k, v in features_dict.items()
        }
        self.labels = torch.FloatTensor(labels)
        
        # Verify all features have same number of samples
        n_samples = len(labels)
        for k, v in self.features.items():
            assert len(v) == n_samples, f"Feature {k} has {len(v)} samples, expected {n_samples}"
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return {k: v[idx] for k, v in self.features.items()}, self.labels[idx]

class multiclassClassifier(nn.Module):
    """Neural network for multiclass classification."""
    
    def __init__(self, feature_dims: Dict[str, int], hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.3):
        """
        Initialize the classifier.
        
        Args:
            feature_dims (Dict[str, int]): Dictionary of input dimensions for each feature type
            hidden_dims (List[int]): List of hidden layer dimensions
            output_dim (int): Output dimension (number of labels)
            dropout_rate (float): Dropout rate for regularization
        """
        super().__init__()
        
        # Feature-specific encoders with residual connections
        self.feature_encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dims[0]),
                nn.LayerNorm(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dims[0], hidden_dims[0]),
                nn.LayerNorm(hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for name, dim in feature_dims.items()
        })
        
        # Feature attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dims[0] * len(feature_dims), len(feature_dims)),
            nn.Softmax(dim=1)
        )
        
        # Combined feature processing with residual connections
        self.combined_encoder = nn.Sequential(
            nn.Linear(hidden_dims[0] * len(feature_dims), hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Main processing layers
        layers = []
        prev_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.model = nn.Sequential(*layers)
        
        # Shared feature extractor for all heads
        self.shared_extractor = nn.Sequential(
            nn.Linear(prev_dim, prev_dim),
            nn.LayerNorm(prev_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Category-specific feature extractors
        self.category_extractors = nn.ModuleDict({
            'family': nn.Sequential(
                nn.Linear(prev_dim, prev_dim),
                nn.LayerNorm(prev_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ),
            'size': nn.Sequential(
                nn.Linear(prev_dim, prev_dim),
                nn.LayerNorm(prev_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ),
            'optimizer': nn.Sequential(
                nn.Linear(prev_dim, prev_dim),
                nn.LayerNorm(prev_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ),
            'lr': nn.Sequential(
                nn.Linear(prev_dim, prev_dim),
                nn.LayerNorm(prev_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ),
            'bs': nn.Sequential(
                nn.Linear(prev_dim, prev_dim),
                nn.LayerNorm(prev_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        })
        
        # Separate output heads for each category
        self.family_head = nn.Sequential(
            nn.Linear(prev_dim, 2),  # 2 classes: BART, Qwen
            nn.Softmax(dim=1)
        )
        
        self.size_head = nn.Sequential(
            nn.Linear(prev_dim, 2),  # 2 classes: base, large
            nn.Softmax(dim=1)
        )
        
        self.optimizer_head = nn.Sequential(
            nn.Linear(prev_dim, 3),  # 3 classes: adamw, sgd, adafactor
            nn.Softmax(dim=1)
        )
        
        self.lr_head = nn.Sequential(
            nn.Linear(prev_dim, 3),  # 3 classes: 1e-5, 5e-5, 1e-4
            nn.Softmax(dim=1)
        )
        
        self.bs_head = nn.Sequential(
            nn.Linear(prev_dim, 3),  # 3 classes: 4, 8, 16
            nn.Softmax(dim=1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode each feature type
        encoded_features = []
        for name, features in x.items():
            encoded = self.feature_encoders[name](features)
            encoded_features.append(encoded)
        
        # Concatenate encoded features
        combined = torch.cat(encoded_features, dim=1)
        
        # Apply attention mechanism
        attention_weights = self.attention(combined)
        attended_features = []
        for i, encoded in enumerate(encoded_features):
            attended = encoded * attention_weights[:, i:i+1]
            attended_features.append(attended)
        
        # Combine attended features
        combined = torch.cat(attended_features, dim=1)
        
        # Process combined features
        features = self.combined_encoder(combined)
        features = self.model(features)
        
        # Extract shared features
        shared_features = self.shared_extractor(features)
        
        # Extract category-specific features
        family_features = self.category_extractors['family'](shared_features)
        size_features = self.category_extractors['size'](shared_features)
        optimizer_features = self.category_extractors['optimizer'](shared_features)
        lr_features = self.category_extractors['lr'](shared_features)
        bs_features = self.category_extractors['bs'](shared_features)
        
        # Get predictions from each head
        family_pred = self.family_head(family_features)
        size_pred = self.size_head(size_features)
        optimizer_pred = self.optimizer_head(optimizer_features)
        lr_pred = self.lr_head(lr_features)
        bs_pred = self.bs_head(bs_features)
        
        # Concatenate all predictions
        return torch.cat([family_pred, size_pred, optimizer_pred, lr_pred, bs_pred], dim=1)

def load_features_and_labels(data_dir: str, config_path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """
    Load features and corresponding labels from the dataset.
    
    Args:
        data_dir (str): Directory containing the feature files
        config_path (str): Path to the config summary CSV file
        
    Returns:
        Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]: Features dictionary, labels, and label names
    """
    logger.info("Loading features and labels...")
    
    # Load config summary
    config_df = pd.read_csv(config_path)
    logger.info(f"Loaded config summary with {len(config_df)} entries")
    logger.info(f"Available model indices: {config_df['model_index'].tolist()}")
    
    # Get all model directories recursively
    model_dirs = []
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            # Check if this directory contains x2_batch_*.npy files
            dir_path = os.path.join(root, dir_name)
            if any(f.startswith('x2_batch_') and f.endswith('.npy') for f in os.listdir(dir_path)):
                model_dirs.append(dir_path)
    
    logger.info(f"Found {len(model_dirs)} model directories with features")
    
    features_dict = {'x2': [], 'x4': []}
    labels_list = []
    processed_dirs = []
    skipped_dirs = []
    
    for dir_path in model_dirs:
        try:
            model_dir = os.path.basename(dir_path)
            logger.info(f"\nProcessing directory: {dir_path}")
            
            # List all files in the directory
            all_files = os.listdir(dir_path)
            logger.info(f"Files in {model_dir}: {all_files}")
            
            # Extract model configuration from directory name
            try:
                parts = model_dir.split('_')
                model_index = int(parts[0])
                model_family = parts[1].upper()  # BART or QWEN
                model_size = parts[2]  # base or large
                optimizer = parts[3]  # adamw, sgd, or adafactor
                
                # Extract learning rate and batch size
                lr_part = next(p for p in parts if p.startswith('lr'))
                lr = float(lr_part.replace('lr', ''))
                
                bs_part = next(p for p in parts if p.startswith('bs'))
                batch_size = int(bs_part.replace('bs', ''))
                
                # Create a config row that matches the format in config_df
                model_config = pd.DataFrame([{
                    'model_index': model_index,
                    'model_family': model_family,
                    'model_size': model_size,
                    'optimizer': optimizer,
                    'learning_rate': lr,
                    'batch_size': batch_size
                }])
                
                logger.info(f"Extracted config from directory name: {model_config.iloc[0].to_dict()}")
                
            except Exception as e:
                logger.warning(f"Could not extract config from directory name {model_dir}: {str(e)}")
                # Try to find model index in config by matching directory name
                matching_configs = config_df[config_df['model_output_dir'].str.contains(model_dir, case=False)]
                if not matching_configs.empty:
                    model_config = matching_configs.iloc[0:1]
                    logger.info(f"Found matching config for directory {model_dir}")
                else:
                    logger.warning(f"Skipping directory {model_dir} - could not determine model config")
                    skipped_dirs.append(model_dir)
                    continue
            
            # Load x2 and x4 features for this model
            x2_files = sorted([f for f in all_files 
                             if f.startswith('x2_batch_') and f.endswith('.npy')])
            x4_files = sorted([f for f in all_files 
                             if f.startswith('x4_batch_') and f.endswith('.npy')])
            
            if not x2_files or not x4_files:
                logger.warning(f"Missing feature files in {model_dir}")
                logger.info(f"Available files: {all_files}")
                skipped_dirs.append(model_dir)
                continue
            
            logger.info(f"Found {len(x2_files)} x2 files and {len(x4_files)} x4 files in {model_dir}")
            
            # Create labels for this model
            labels, label_names = create_categorical_labels(model_config)
            
            # Load features from matching batch files
            for x2_file, x4_file in zip(x2_files, x4_files):
                try:
                    x2_path = os.path.join(dir_path, x2_file)
                    x4_path = os.path.join(dir_path, x4_file)
                    
                    logger.info(f"Loading features from {x2_file} and {x4_file}")
                    x2_features = np.load(x2_path)
                    x4_features = np.load(x4_path)
                    
                    logger.info(f"Loaded features with shapes: x2={x2_features.shape}, x4={x4_features.shape}")
                    
                    # Ensure features and labels have the same number of samples
                    num_samples = x2_features.shape[0]
                    features_dict['x2'].append(x2_features)
                    features_dict['x4'].append(x4_features)
                    labels_list.append(np.tile(labels, (num_samples, 1)))
                    
                    logger.info(f"Successfully loaded features from {x2_file} and {x4_file}")
                    
                except Exception as e:
                    logger.error(f"Error loading features: {str(e)}")
                    continue
            
            processed_dirs.append(model_dir)
                    
        except Exception as e:
            logger.error(f"Error processing directory {model_dir}: {str(e)}")
            skipped_dirs.append(model_dir)
            continue
    
    if not features_dict['x2'] or not features_dict['x4']:
        logger.error("No valid features found in any directory")
        logger.error(f"Processed directories: {processed_dirs}")
        logger.error(f"Skipped directories: {skipped_dirs}")
        raise ValueError("No valid features found in any directory")
    
    # Combine all features and labels
    features_dict = {
        'x2': np.vstack(features_dict['x2']),
        'x4': np.vstack(features_dict['x4'])
    }
    y = np.vstack(labels_list)
    
    # Verify shapes match
    n_samples = len(y)
    for name, features in features_dict.items():
        if features.shape[0] != n_samples:
            raise ValueError(f"Feature {name} has {features.shape[0]} samples, expected {n_samples}")
    
    logger.info(f"\nSummary:")
    logger.info(f"Loaded {n_samples} samples")
    for name, features in features_dict.items():
        logger.info(f"{name} features shape: {features.shape}")
    logger.info(f"Labels shape: {y.shape}")
    logger.info(f"Processed {len(processed_dirs)} directories: {processed_dirs}")
    logger.info(f"Skipped {len(skipped_dirs)} directories: {skipped_dirs}")
    logger.info(f"Label names: {label_names}")
    
    return features_dict, y, label_names

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                scheduler: Optional[optim.lr_scheduler._LRScheduler],
                num_epochs: int,
                device: str,
                label_names: List[str]) -> Dict[str, List[float]]:
    """
    Train the model.
    """
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Calculate class weights for each category
    with torch.no_grad():
        all_labels = []
        for _, labels in train_loader:
            all_labels.append(labels)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Calculate weights for each category
        category_weights = {}
        category_indices = {
            'family': [0, 1],
            'size': [2, 3],
            'optimizer': [4, 5, 6],
            'lr': [7, 8, 9],
            'bs': [10, 11, 12]
        }
        
        for category, indices in category_indices.items():
            category_labels = all_labels[:, indices]
            pos_counts = category_labels.sum(dim=0)
            neg_counts = len(category_labels) - pos_counts
            weights = neg_counts / (pos_counts + 1e-6)  # Add small epsilon to avoid division by zero
            category_weights[category] = weights
    
    logger.info("\nCategory weights:")
    for category, weights in category_weights.items():
        logger.info(f"{category}: {weights.tolist()}")
    
    # Create weighted loss function for each category
    def weighted_loss(outputs, labels):
        total_loss = 0.0
        
        # Family loss
        family_outputs = outputs[:, category_indices['family']]
        family_labels = labels[:, category_indices['family']]
        family_loss = F.cross_entropy(
            family_outputs, 
            family_labels.argmax(dim=1),
            reduction='mean'
        )
        total_loss += family_loss
        
        # Size loss
        size_outputs = outputs[:, category_indices['size']]
        size_labels = labels[:, category_indices['size']]
        size_loss = F.cross_entropy(
            size_outputs,
            size_labels.argmax(dim=1),
            reduction='mean'
        )
        total_loss += size_loss
        
        # Optimizer loss
        optimizer_outputs = outputs[:, category_indices['optimizer']]
        optimizer_labels = labels[:, category_indices['optimizer']]
        optimizer_loss = F.cross_entropy(
            optimizer_outputs,
            optimizer_labels.argmax(dim=1),
            reduction='mean'
        )
        total_loss += optimizer_loss
        
        # Learning rate loss
        lr_outputs = outputs[:, category_indices['lr']]
        lr_labels = labels[:, category_indices['lr']]
        lr_loss = F.cross_entropy(
            lr_outputs,
            lr_labels.argmax(dim=1),
            reduction='mean'
        )
        total_loss += lr_loss
        
        # Batch size loss
        bs_outputs = outputs[:, category_indices['bs']]
        bs_labels = labels[:, category_indices['bs']]
        bs_loss = F.cross_entropy(
            bs_outputs,
            bs_labels.argmax(dim=1),
            reduction='mean'
        )
        total_loss += bs_loss
        
        return total_loss
    
    logger.info("\nStarting Training...")
    logger.info(f"Training on {device} for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for features, labels in train_loader:
            features = {k: v.to(device) for k, v in features.items()}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            
            loss = weighted_loss(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        # Per-label metrics
        true_positives = torch.zeros(len(label_names), device=device)
        false_positives = torch.zeros(len(label_names), device=device)
        false_negatives = torch.zeros(len(label_names), device=device)
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = {k: v.to(device) for k, v in features.items()}
                labels = labels.to(device)
                
                outputs = model(features)
                loss = weighted_loss(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
                
                # Convert outputs to binary predictions
                predictions = torch.zeros_like(outputs)
                for category, indices in category_indices.items():
                    category_outputs = outputs[:, indices]
                    category_preds = torch.zeros_like(category_outputs)
                    category_preds[torch.arange(len(category_outputs)), category_outputs.argmax(dim=1)] = 1
                    predictions[:, indices] = category_preds
                
                # Calculate metrics
                true_positives += (predictions * labels).sum(dim=0)
                false_positives += (predictions * (1 - labels)).sum(dim=0)
                false_negatives += ((1 - predictions) * labels).sum(dim=0)
        
        val_loss /= val_batches
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Calculate per-label F1 score
        per_label_f1 = torch.zeros(len(label_names), device=device)
        for i in range(len(label_names)):
            tp = true_positives[i]
            fp = false_positives[i]
            fn = false_negatives[i]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            per_label_f1[i] = f1
        
        # Calculate overall F1 score
        total_tp = true_positives.sum()
        total_fp = false_positives.sum()
        total_fn = false_negatives.sum()
        
        total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0.0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(total_f1.item())
        
        # Log results
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val F1: {total_f1:.4f}")
        
        # Log per-label metrics
        logger.info("\nPer-label Metrics:")
        for i, name in enumerate(label_names):
            tp = true_positives[i].item()
            fp = false_positives[i].item()
            fn = false_negatives[i].item()
            f1 = per_label_f1[i].item()
            
            if (tp + fp + fn) > 0:
                logger.info(f"{name:20s}: F1={f1:.4f} (TP={int(tp)}, FP={int(fp)}, FN={int(fn)})")
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            logger.info(f"\nNew best model at epoch {epoch+1}!")
    
    logger.info(f"\nTraining completed. Best model at epoch {best_epoch + 1}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final validation F1: {total_f1:.4f}")
    
    return history

def setup_distributed():
    """Set up distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank = 0
        world_size = 1
        gpu = 0

    torch.cuda.set_device(gpu)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    return rank, world_size, gpu

def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def create_categorical_labels(config_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Create categorical labels from config DataFrame.
    
    Args:
        config_df (pd.DataFrame): Config DataFrame
        
    Returns:
        Tuple[np.ndarray, List[str]]: Categorical labels and label names
    """
    # Create label encoders for each categorical variable
    model_family_encoder = {family: i for i, family in enumerate(['BART', 'Qwen'])}
    model_size_encoder = {size: i for i, size in enumerate(['base', 'large'])}
    optimizer_encoder = {opt: i for i, opt in enumerate(['adamw', 'sgd', 'adafactor'])}
    lr_encoder = {lr: i for i, lr in enumerate([1e-5, 5e-5, 1e-4])}
    bs_encoder = {bs: i for i, bs in enumerate([4, 8, 16])}
    
    # Create binary labels for each category
    family_labels = np.zeros((len(config_df), len(model_family_encoder)))
    size_labels = np.zeros((len(config_df), len(model_size_encoder)))
    optimizer_labels = np.zeros((len(config_df), len(optimizer_encoder)))
    lr_labels = np.zeros((len(config_df), len(lr_encoder)))
    bs_labels = np.zeros((len(config_df), len(bs_encoder)))
    
    for i, row in config_df.iterrows():
        # Set model family label
        family_idx = model_family_encoder[row['model_family']]
        family_labels[i, family_idx] = 1
        
        # Set model size label
        size_idx = model_size_encoder[row['model_size']]
        size_labels[i, size_idx] = 1
        
        # Set optimizer label
        opt_idx = optimizer_encoder[row['optimizer']]
        optimizer_labels[i, opt_idx] = 1
        
        # Set learning rate label
        lr = float(row['learning_rate'])
        lr_idx = lr_encoder[lr]
        lr_labels[i, lr_idx] = 1
        
        # Set batch size label
        bs = int(row['batch_size'])
        bs_idx = bs_encoder[bs]
        bs_labels[i, bs_idx] = 1
    
    # Combine all labels
    labels = np.hstack([family_labels, size_labels, optimizer_labels, lr_labels, bs_labels])
    
    # Create label names
    label_names = (
        [f'family_{family}' for family in model_family_encoder.keys()] +
        [f'size_{size}' for size in model_size_encoder.keys()] +
        [f'optimizer_{opt}' for opt in optimizer_encoder.keys()] +
        [f'lr_{lr}' for lr in lr_encoder.keys()] +
        [f'bs_{bs}' for bs in bs_encoder.keys()]
    )
    
    return labels, label_names

def main():
    """Main function to train the classifier."""
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    
    # Set up distributed training
    rank, world_size, gpu = setup_distributed()
    
    # Configuration
    DATA_DIR = os.path.abspath("./multimodal_dataset")
    CONFIG_PATH = os.path.abspath("./configs/config_summary.csv")
    
    # Verify paths exist
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    
    if rank == 0:
        logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    try:
        # Load data
        features_dict, y, label_names = load_features_and_labels(DATA_DIR, CONFIG_PATH)
        
        if rank == 0:
            logger.info(f"Loaded {len(y)} samples")
            for name, features in features_dict.items():
                logger.info(f"{name} features shape: {features.shape}")
        
        # Calculate split indices
        n_samples = len(y)
        train_size = int(0.8 * n_samples)
        
        # Create indices for shuffling
        indices = np.random.permutation(n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Split data using indices
        X_train = {k: v[train_indices] for k, v in features_dict.items()}
        X_val = {k: v[val_indices] for k, v in features_dict.items()}
        y_train = y[train_indices]
        y_val = y[val_indices]
        
        if rank == 0:
            logger.info(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}")
            for name, features in X_train.items():
                logger.info(f"Train {name} features shape: {features.shape}")
            for name, features in X_val.items():
                logger.info(f"Val {name} features shape: {features.shape}")
        
        # Create datasets and dataloaders
        train_dataset = MultiFeatureDataset(X_train, y_train)
        val_dataset = MultiFeatureDataset(X_val, y_val)
        
        train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset) if world_size > 1 else None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=128,  # Increased batch size
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=128,  # Increased batch size
            shuffle=False,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model with feature-specific encoders
        feature_dims = {name: features.shape[1] for name, features in features_dict.items()}
        output_dim = y.shape[1]
        
        model = multiclassClassifier(
            feature_dims=feature_dims,
            hidden_dims=[512, 256, 128],  # Increased model capacity
            output_dim=output_dim,
            dropout_rate=0.4  # Adjusted dropout
        ).to(gpu)
        
        if world_size > 1:
            model = DDP(model, device_ids=[gpu])
        
        # Initialize optimizer with lower learning rate
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.00005,  # Further reduced learning rate
            weight_decay=0.02,  # Increased weight decay
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.0005,  # Reduced max learning rate
            epochs=10,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,  # Added learning rate range
            final_div_factor=1000.0
        )
        
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.BCELoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=10,
            device=gpu,
            label_names=label_names
        )
        
        if rank == 0:  # Only save on main process
            # Create models directory
            model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Unimodel")
            os.makedirs(model_save_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_save_dir, "multiclass_classifier_x2_x4.pt")
            if isinstance(model, DDP):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            
            # Save training history
            history_path = os.path.join(model_save_dir, "training_history_multiclass_classifier_x2_x4.json")
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"Model and training history saved in {model_save_dir}")
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main() 