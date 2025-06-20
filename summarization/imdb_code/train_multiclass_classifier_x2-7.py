"""
Script to train a multiclass classifier using x2, x3, x4, x5, x6, and x7 features.
This version uses a specialized architecture to handle multiple feature types with attention mechanisms.
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
import datetime
import torch.multiprocessing as mp
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
CACHE_DIR = "/work/shovon/LLM/"  # Base directory for caching models and data

def setup_cache_directories():
    """Set up cache directories and environment variables."""
    # Set up cache directories
    cache_paths = {
        'TRANSFORMERS_CACHE': 'transformers',
        'HF_HOME': 'huggingface',
        'HF_DATASETS_CACHE': 'datasets',
        'SENTENCE_TRANSFORMERS_HOME': 'sentence-transformers',
        'NLTK_DATA': 'nltk_data',
        'TORCH_HOME': 'torch'
    }
    
    for env_var, dir_name in cache_paths.items():
        cache_path = os.path.join(CACHE_DIR, dir_name)
        os.environ[env_var] = cache_path
        os.makedirs(cache_path, exist_ok=True)
        logger.info(f"Using cache directory: {cache_path}")

class MultiFeatureDataset(Dataset):
    """Dataset for handling multiple feature types (x2, x3, x4, x5, x6, x7)."""
    
    def __init__(self, features_dict: Dict[str, np.ndarray], labels: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            features_dict (Dict[str, np.ndarray]): Dictionary of feature matrices for x2, x3, x4, x5, x6, x7
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

class TemperatureScaling(nn.Module):
    """Temperature scaling layer for calibration."""
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, x):
        return x / self.temperature

class multiclassClassifier(nn.Module):
    """Neural network for multiclass classification with multiple feature types (x2, x3, x4, x5, x6, x7)."""
    
    def __init__(self, feature_dims: Dict[str, int], hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.3):
        """
        Initialize the classifier.
        
        Args:
            feature_dims (Dict[str, int]): Dictionary of input dimensions for each feature type (x2, x3, x4, x5, x6, x7)
            hidden_dims (List[int]): List of hidden layer dimensions
            output_dim (int): Output dimension (number of labels)
            dropout_rate (float): Dropout rate for regularization
        """
        super().__init__()
        
        # Feature-specific encoders with residual connections for all feature types
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
        
        # Feature attention mechanism for 6 feature types
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
        
        # Main processing layers with residual connections
        self.layers = nn.ModuleList()
        prev_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            # Add projection layer for residual connection
            self.layers.append(nn.ModuleDict({
                'main': nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ),
                'residual': nn.Linear(prev_dim, hidden_dim) if prev_dim != hidden_dim else nn.Identity()
            }))
            prev_dim = hidden_dim
        
        # Category-specific feature extractors with residual connections
        self.category_extractors = nn.ModuleDict({
            'family': nn.ModuleDict({
                'main': nn.Sequential(
                    nn.Linear(prev_dim, prev_dim),
                    nn.LayerNorm(prev_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(prev_dim, prev_dim),
                    nn.LayerNorm(prev_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ),
                'residual': nn.Identity()
            }),
            'size': nn.ModuleDict({
                'main': nn.Sequential(
                    nn.Linear(prev_dim, prev_dim),
                    nn.LayerNorm(prev_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(prev_dim, prev_dim),
                    nn.LayerNorm(prev_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ),
                'residual': nn.Identity()
            }),
            'optimizer': nn.ModuleDict({
                'main': nn.Sequential(
                    nn.Linear(prev_dim, prev_dim),
                    nn.LayerNorm(prev_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(prev_dim, prev_dim),
                    nn.LayerNorm(prev_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ),
                'residual': nn.Identity()
            }),
            'lr': nn.ModuleDict({
                'main': nn.Sequential(
                    nn.Linear(prev_dim, prev_dim),
                    nn.LayerNorm(prev_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(prev_dim, prev_dim),
                    nn.LayerNorm(prev_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ),
                'residual': nn.Identity()
            }),
            'bs': nn.ModuleDict({
                'main': nn.Sequential(
                    nn.Linear(prev_dim, prev_dim),
                    nn.LayerNorm(prev_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(prev_dim, prev_dim),
                    nn.LayerNorm(prev_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ),
                'residual': nn.Identity()
            })
        })
        
        # Separate output heads for each category with temperature scaling
        self.family_head = nn.Sequential(
            nn.Linear(prev_dim, 2),  # 2 classes: BART, Qwen
            TemperatureScaling(2.0)  # Add temperature scaling
        )
        
        self.size_head = nn.Sequential(
            nn.Linear(prev_dim, 2),  # 2 classes: base, large
            TemperatureScaling(2.0)
        )
        
        self.optimizer_head = nn.Sequential(
            nn.Linear(prev_dim, 3),  # 3 classes: adamw, sgd, adafactor
            TemperatureScaling(2.0)
        )
        
        self.lr_head = nn.Sequential(
            nn.Linear(prev_dim, 3),  # 3 classes: 1e-5, 5e-5, 1e-4
            TemperatureScaling(2.0)
        )
        
        self.bs_head = nn.Sequential(
            nn.Linear(prev_dim, 3),  # 3 classes: 4, 8, 16
            TemperatureScaling(2.0)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Encode each feature type (x2, x3, x4, x5, x6, x7)
        encoded_features = []
        for name, features in x.items():
            encoded = self.feature_encoders[name](features)
            encoded_features.append(encoded)
        
        # Concatenate encoded features
        combined = torch.cat(encoded_features, dim=1)
        
        # Apply attention mechanism across all 6 feature types
        attention_weights = self.attention(combined)
        attended_features = []
        for i, encoded in enumerate(encoded_features):
            attended = encoded * attention_weights[:, i:i+1]
            attended_features.append(attended)
        
        # Combine attended features
        combined = torch.cat(attended_features, dim=1)
        
        # Process combined features with residual connections
        features = self.combined_encoder(combined)
        for layer in self.layers:
            residual = layer['residual'](features)
            features = layer['main'](features)
            features = features + residual  # Residual connection
        
        # Extract category-specific features with residual connections
        family_features = self.category_extractors['family']['main'](features) + self.category_extractors['family']['residual'](features)
        size_features = self.category_extractors['size']['main'](features) + self.category_extractors['size']['residual'](features)
        optimizer_features = self.category_extractors['optimizer']['main'](features) + self.category_extractors['optimizer']['residual'](features)
        lr_features = self.category_extractors['lr']['main'](features) + self.category_extractors['lr']['residual'](features)
        bs_features = self.category_extractors['bs']['main'](features) + self.category_extractors['bs']['residual'](features)
        
        # Get predictions from each head (raw logits)
        family_pred = self.family_head(family_features)
        size_pred = self.size_head(size_features)
        optimizer_pred = self.optimizer_head(optimizer_features)
        lr_pred = self.lr_head(lr_features)
        bs_pred = self.bs_head(bs_features)
        
        # Concatenate all predictions
        return torch.cat([family_pred, size_pred, optimizer_pred, lr_pred, bs_pred], dim=1)

def preprocess_x3_features(x3_features: np.ndarray) -> np.ndarray:
    """
    Preprocess x3 features with shape (100, 3) to make them compatible with other features.
    
    Args:
        x3_features (np.ndarray): x3 features with shape (100, 3)
        
    Returns:
        np.ndarray: Preprocessed x3 features
    """
    # x3 features have shape (100, 3), we can flatten them or use pooling
    # Option 1: Flatten to (300,) - simple approach
    if len(x3_features.shape) == 2 and x3_features.shape[1] == 3:
        # Flatten the features: (100, 3) -> (300,)
        return x3_features.flatten()
    else:
        # If already flattened or different shape, return as is
        return x3_features

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
    
    # Get all model directories recursively
    model_dirs = []
    for root, dirs, files in os.walk(data_dir):
        for dir_name in dirs:
            # Check if this directory contains x2_batch_*.npy files
            dir_path = os.path.join(root, dir_name)
            if any(f.startswith('x2_batch_') and f.endswith('.npy') for f in os.listdir(dir_path)):
                model_dirs.append(dir_path)
    
    logger.info(f"Found {len(model_dirs)} model directories with features")
    
    # Initialize features dictionary for all feature types
    features_dict = {'x2': [], 'x3': [], 'x4': [], 'x5': [], 'x6': [], 'x7': []}
    labels_list = []
    processed_dirs = []
    skipped_dirs = []
    
    for dir_path in model_dirs:
        try:
            model_dir = os.path.basename(dir_path)
            
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
                
            except Exception as e:
                # Try to find model index in config by matching directory name
                matching_configs = config_df[config_df['model_output_dir'].str.contains(model_dir, case=False)]
                if not matching_configs.empty:
                    model_config = matching_configs.iloc[0:1]
                else:
                    skipped_dirs.append(model_dir)
                    continue
            
            # Load all feature types for this model
            all_files = os.listdir(dir_path)
            
            # Get sorted file lists for each feature type
            feature_files = {}
            for feature_type in ['x2', 'x3', 'x4', 'x5', 'x6', 'x7']:
                feature_files[feature_type] = sorted([f for f in all_files 
                                                   if f.startswith(f'{feature_type}_batch_') and f.endswith('.npy')])
            
            # Check if we have files for all feature types
            missing_features = [ft for ft, files in feature_files.items() if not files]
            if missing_features:
                logger.warning(f"Missing feature files for {missing_features} in {model_dir}")
                skipped_dirs.append(model_dir)
                continue
            
            # Create labels for this model
            labels, label_names = create_categorical_labels(model_config)
            
            # Load features from matching batch files
            # Use x2 files as reference for number of batches
            num_batches = len(feature_files['x2'])
            
            for batch_idx in range(num_batches):
                try:
                    batch_features = {}
                    
                    # Load each feature type
                    for feature_type in ['x2', 'x3', 'x4', 'x5', 'x6', 'x7']:
                        if batch_idx < len(feature_files[feature_type]):
                            feature_file = feature_files[feature_type][batch_idx]
                            feature_path = os.path.join(dir_path, feature_file)
                            features = np.load(feature_path)
                            
                            # Special handling for x3 features
                            if feature_type == 'x3':
                                # Process each sample in the batch
                                processed_features = []
                                for i in range(features.shape[0]):
                                    if len(features.shape) == 3:  # (batch_size, 100, 3)
                                        processed_sample = preprocess_x3_features(features[i])
                                    elif len(features.shape) == 2 and features.shape[1] == 3:  # (100, 3)
                                        processed_sample = preprocess_x3_features(features)
                                    else:
                                        processed_sample = features[i] if len(features.shape) > 1 else features
                                    processed_features.append(processed_sample)
                                batch_features[feature_type] = np.array(processed_features)
                            else:
                                batch_features[feature_type] = features
                        else:
                            logger.warning(f"Missing {feature_type} batch {batch_idx} in {model_dir}")
                            continue
                    
                    # Ensure all features have the same number of samples
                    sample_counts = [features.shape[0] for features in batch_features.values()]
                    if len(set(sample_counts)) > 1:
                        logger.warning(f"Inconsistent sample counts in {model_dir} batch {batch_idx}: {sample_counts}")
                        continue
                    
                    # Add features to the dictionary
                    for feature_type, features in batch_features.items():
                        features_dict[feature_type].append(features)
                    
                    # Add labels
                    num_samples = sample_counts[0]
                    labels_list.append(np.tile(labels, (num_samples, 1)))
                    
                except Exception as e:
                    logger.error(f"Error loading batch {batch_idx} from {model_dir}: {str(e)}")
                    continue
            
            processed_dirs.append(model_dir)
                    
        except Exception as e:
            skipped_dirs.append(model_dir)
            continue
    
    if not features_dict['x2'] or not features_dict['x3'] or not features_dict['x4'] or not features_dict['x5'] or not features_dict['x6'] or not features_dict['x7']:
        raise ValueError("No valid features found in any directory")
    
    # Combine all features and labels
    features_dict = {
        feature_type: np.vstack(features_list) for feature_type, features_list in features_dict.items()
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
    logger.info(f"Processed {len(processed_dirs)} directories")
    logger.info(f"Skipped {len(skipped_dirs)} directories")
    
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
    
    # Define category indices and names
    category_indices = {
        'family': {'indices': [0, 1], 'name': 'Model Family'},
        'size': {'indices': [2, 3], 'name': 'Model Size'},
        'optimizer': {'indices': [4, 5, 6], 'name': 'Optimizer'},
        'lr': {'indices': [7, 8, 9], 'name': 'Learning Rate'},
        'bs': {'indices': [10, 11, 12], 'name': 'Batch Size'}
    }
    
    # Calculate class weights for each category
    with torch.no_grad():
        all_labels = []
        for _, labels in train_loader:
            all_labels.append(labels)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Calculate weights for each category
        category_weights = {}
        for category, info in category_indices.items():
            category_labels = all_labels[:, info['indices']]
            pos_counts = category_labels.sum(dim=0)
            neg_counts = len(category_labels) - pos_counts
            weights = neg_counts / (pos_counts + 1e-6)  # Add small epsilon to avoid division by zero
            # Clamp weights to prevent instability
            weights = torch.clamp(weights, min=0.1, max=10.0)
            category_weights[category] = weights
    
    logger.info("\nCategory weights:")
    for category, weights in category_weights.items():
        logger.info(f"{category}: {weights.tolist()}")
    
    # Create weighted loss function for each category
    def weighted_loss(outputs, labels):
        total_loss = 0.0
        
        for category, info in category_indices.items():
            category_outputs = outputs[:, info['indices']]
            category_labels = labels[:, info['indices']]
            category_loss = F.cross_entropy(
                category_outputs,
                category_labels.argmax(dim=1),
                reduction='mean',
                weight=category_weights[category].to(device)
            )
            total_loss += category_loss
        
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
            
            # Monitor only high gradients before clipping
            if train_batches % 50 == 0:  # Reduced frequency
                high_grads = []
                for name, param in model.named_parameters():
                    if param.grad is not None and param.grad.norm().item() > 1.0:
                        high_grads.append((name, param.grad.norm().item()))
                if high_grads:
                    logger.info(f"\nHigh gradients detected in batch {train_batches}:")
                    for name, norm in high_grads:
                        logger.info(f"{name}: {norm:.4f}")
            
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
        
        # Per-category metrics
        category_metrics = {
            category: {
                'correct': 0,
                'total': 0,
                'class_correct': torch.zeros(len(info['indices']), device=device),
                'class_total': torch.zeros(len(info['indices']), device=device)
            }
            for category, info in category_indices.items()
        }
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = {k: v.to(device) for k, v in features.items()}
                labels = labels.to(device)
                
                outputs = model(features)
                loss = weighted_loss(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
                
                # Calculate per-category metrics
                for category, info in category_indices.items():
                    category_outputs = outputs[:, info['indices']]
                    category_labels = labels[:, info['indices']]
                    
                    # Get predictions
                    preds = category_outputs.argmax(dim=1)
                    targets = category_labels.argmax(dim=1)
                    
                    # Update metrics
                    category_metrics[category]['correct'] += (preds == targets).sum().item()
                    category_metrics[category]['total'] += len(preds)
                    
                    # Update per-class metrics
                    for i in range(len(info['indices'])):
                        mask = targets == i
                        category_metrics[category]['class_correct'][i] += (preds[mask] == i).sum()
                        category_metrics[category]['class_total'][i] += mask.sum()
        
        val_loss /= val_batches
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Calculate and log per-category metrics
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        logger.info("\nPer-Category Metrics:")
        logger.info(f"{'Category':<15} {'Accuracy':<10} {'Macro F1':<10}")
        logger.info("-" * 35)
        
        for category, info in category_indices.items():
            metrics = category_metrics[category]
            accuracy = metrics['correct'] / metrics['total']
            
            # Calculate per-class F1 scores
            f1_scores = []
            for i in range(len(info['indices'])):
                tp = metrics['class_correct'][i]
                fp = metrics['class_total'][i] - tp
                fn = metrics['total'] - metrics['class_total'][i] - (metrics['correct'] - tp)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                f1_scores.append(f1)
            
            macro_f1 = sum(f1_scores) / len(f1_scores)
            
            logger.info(f"{info['name']:<15} {accuracy:.4f}     {macro_f1:.4f}")
            
            # Log per-class metrics
            for i, f1 in enumerate(f1_scores):
                class_name = label_names[info['indices'][i]]
                logger.info(f"  {class_name:<20} F1={f1:.4f}")
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            logger.info(f"\nNew best model at epoch {epoch+1}!")
    
    logger.info(f"\nTraining completed. Best model at epoch {best_epoch + 1}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    return history

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

def main(rank: int):
    """Main function to train the classifier.
    
    Args:
        rank (int): Process rank for distributed training
    """
    try:
        # Set up distributed training
        world_size = int(os.environ.get('WORLD_SIZE', torch.cuda.device_count()))
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        
        # Set environment variables for distributed training if not already set
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = '127.0.0.1'  # Use localhost IP
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'  # Changed port to avoid conflicts
        
        # Set device before process group initialization
        torch.cuda.set_device(local_rank)
        
        # Initialize process group with GLOO backend
        dist.init_process_group(
            backend='gloo',  # Use GLOO backend instead of NCCL
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=60)  # Increased timeout
        )
        
        # Configuration
        DATA_DIR = os.path.abspath("./multimodal_dataset")
        CONFIG_PATH = os.path.abspath("./configs/config_summary.csv")
        
        # Verify paths exist
        if not os.path.exists(DATA_DIR):
            raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
        
        # Only log from main process
        if rank == 0:
            logger.info(f"Using {world_size} GPUs")
            logger.info(f"Main process using device: cuda:{local_rank}")
            logger.info(f"Master address: {os.environ['MASTER_ADDR']}")
            logger.info(f"Master port: {os.environ['MASTER_PORT']}")
        
        # Load data
        features_dict, y, label_names = load_features_and_labels(DATA_DIR, CONFIG_PATH)
        
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
        
        # Create datasets and dataloaders
        train_dataset = MultiFeatureDataset(X_train, y_train)
        val_dataset = MultiFeatureDataset(X_val, y_val)
        
        # Use DistributedSampler for multi-GPU training
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        
        # Calculate batch size per GPU based on world size
        base_batch_size = 32
        batch_size = max(1, base_batch_size // world_size)  # Ensure at least 1 sample per batch
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle when using DistributedSampler
            sampler=train_sampler,
            num_workers=0,  # Reduced workers to prevent semaphore issues
            pin_memory=True,
            drop_last=True  # Drop last incomplete batch
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=0,  # Reduced workers to prevent semaphore issues
            pin_memory=True,
            drop_last=True  # Drop last incomplete batch
        )
        
        # Initialize model with feature-specific encoders for all feature types
        feature_dims = {name: features.shape[1] for name, features in features_dict.items()}
        output_dim = y.shape[1]
        
        model = multiclassClassifier(
            feature_dims=feature_dims,
            hidden_dims=[256, 128, 64],
            output_dim=output_dim,
            dropout_rate=0.5
        ).to(local_rank)
        
        # Wrap model with DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
        # Initialize optimizer with lower learning rate
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.0001,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with warmup
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001,
            epochs=10,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=100.0
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
            device=local_rank,
            label_names=label_names
        )
        
        if rank == 0:  # Only save on main process
            # Create models directory
            model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Unimodel")
            os.makedirs(model_save_dir, exist_ok=True)
            
            # Save model with updated name to reflect all feature types
            model_path = os.path.join(model_save_dir, "multiclass_classifier_x2_x3_x4_x5_x6_x7.pt")
            torch.save(model.module.state_dict(), model_path)  # Save DDP model's module
            
            # Save training history
            history_path = os.path.join(model_save_dir, "training_history_multiclass_classifier_x2_x3_x4_x5_x6_x7.json")
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"Model and training history saved in {model_save_dir}")
            logger.info(f"Model saved as: {model_path}")
            logger.info(f"History saved as: {history_path}")
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        # Ensure proper cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()
        # Clean up any remaining resources
        if 'train_loader' in locals():
            del train_loader
        if 'val_loader' in locals():
            del val_loader
        if 'model' in locals():
            del model
        if 'optimizer' in locals():
            del optimizer
        if 'scheduler' in locals():
            del scheduler
        gc.collect()  # Force garbage collection

if __name__ == "__main__":
    # Set up cache directories only once
    setup_cache_directories()
    
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No GPUs available for training")
    
    # Set environment variables for distributed training
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Method already set
    
    # Launch distributed training
    mp.spawn(main, nprocs=world_size, args=(), join=True)  # Use all available GPUs 