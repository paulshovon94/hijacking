"""
Multimodal Multi-label Multi-class Classifier for Hyperparameter Stealing

This model takes as input a 2312-dimensional feature vector (x1–x7) extracted from a language model's outputs on a transformed dataset.
It predicts multiple hyperparameters used to train that language model, including:

- Model family (e.g., BART, GPT-2, Pegasus, Mistral, Qwen, LLaMA)
- Model size (e.g., base, large, small, medium, 0.5B, 1.8B, 7B, 13B)
- Optimizer type (e.g., AdamW, SGD, Adafactor)
- Learning rate (e.g., 1e-5, 5e-5, 1e-4)
- Batch size (e.g., 4, 8, 16)

Architecture:
- Input layer of size 2312 (concatenated x1-x7 features)
- Shared encoder: two dense layers with ReLU, BatchNorm, and Dropout
- Multiple classification heads (one per hyperparameter), each using a Linear layer followed by softmax
- Loss function: sum of CrossEntropy losses from each head
- Evaluation metrics (during validation): accuracy and macro-F1 per head

This model supports adversarial analysis by predicting the training-time hyperparameters from only the generated summary behavior of language models.
"""

import os
import sys

# Set CuBLAS environment variable for deterministic behavior (must be set before importing torch)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

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
from sklearn.metrics import accuracy_score, f1_score, classification_report
import random

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

def set_seed(seed=42, deterministic=True):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
        deterministic (bool): Whether to use deterministic algorithms (may be slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        logger.info(f"Set random seed to {seed} with deterministic algorithms")
    else:
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
        logger.info(f"Set random seed to {seed} without deterministic algorithms")

class MultimodalDataset(Dataset):
    """Dataset for handling multimodal features (x1-x7)."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            features (np.ndarray): Feature matrix of shape (n_samples, 2312)
            labels (np.ndarray): Label matrix of shape (n_samples, n_labels)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
        # Verify shapes match
        assert len(self.features) == len(self.labels), f"Features: {len(self.features)}, Labels: {len(self.labels)}"
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

class MultimodalHyperparameterClassifier(nn.Module):
    """Neural network for multimodal hyperparameter classification."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes_per_head: Dict[str, int], dropout_rate: float = 0.3):
        """
        Initialize the classifier.
        
        Args:
            input_dim (int): Input dimension (2312 for x1-x7 features)
            hidden_dims (List[int]): List of hidden layer dimensions
            num_classes_per_head (Dict[str, int]): Number of classes for each hyperparameter head
            dropout_rate (float): Dropout rate for regularization
        """
        super().__init__()
        
        # Shared encoder
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.shared_encoder = nn.Sequential(*layers)
        
        # Separate classification heads for each hyperparameter
        self.classification_heads = nn.ModuleDict({
            name: nn.Linear(prev_dim, num_classes) for name, num_classes in num_classes_per_head.items()
        })
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use Xavier initialization for more stable training
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode features
        encoded = self.shared_encoder(x)
        
        # Get predictions from each head
        predictions = {}
        for name, head in self.classification_heads.items():
            predictions[name] = head(encoded)
        
        return predictions

def load_features_only_from_dataloader(dataloader_path: str) -> np.ndarray:
    """
    Load only features (without labels) from the dataloader CSV file for inference.
    
    Args:
        dataloader_path (str): Path to the dataloader.csv file
        
    Returns:
        np.ndarray: Features array of shape (n_samples, 2312)
    """
    logger.info("Loading features from dataloader for inference...")
    
    # Load dataloader CSV
    dataloader_df = pd.read_csv(dataloader_path)
    logger.info(f"Loaded dataloader with {len(dataloader_df)} entries")
    
    # Verify required columns exist
    required_columns = [
        'x1_file', 'x2_file', 'x3_file', 'x4_file', 'x5_file', 'x6_file', 'x7_file'
    ]
    
    missing_columns = [col for col in required_columns if col not in dataloader_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataloader.csv: {missing_columns}")
    
    features_list = []
    processed_entries = 0
    skipped_entries = 0
    
    for idx, row in dataloader_df.iterrows():
        try:
            # Load x1-x7 features for this entry
            batch_features = []
            feature_files = [
                row['x1_file'], row['x2_file'], row['x3_file'], 
                row['x4_file'], row['x5_file'], row['x6_file'], row['x7_file']
            ]
            
            # Check if all feature files exist
            missing_files = [f for f in feature_files if not f or not os.path.exists(f)]
            if missing_files:
                logger.warning(f"Missing feature files for entry {idx}: {missing_files}")
                skipped_entries += 1
                continue
            
            # Load each feature file
            for i, file_path in enumerate(feature_files, 1):
                try:
                    if file_path.endswith('.npy'):
                        x_features = np.load(file_path)
                        
                        # Sanitize corrupted features at load time
                        if not np.all(np.isfinite(x_features)):
                            logger.warning(f"Corrupted features detected in {file_path}, sanitizing...")
                            
                            # Replace NaN with 0, clip Inf values
                            x_features = np.nan_to_num(x_features, nan=0.0)
                            x_features = np.clip(x_features, -1e6, 1e6)
                            
                            logger.info(f"Sanitized {file_path}: NaN/Inf values replaced")
                    else:
                        logger.warning(f"Unexpected file format: {file_path}")
                        skipped_entries += 1
                        continue
                    
                    # Flatten and ensure consistent shape
                    x_features = x_features.flatten()
                    
                    # Pad or truncate to expected size (adjust based on actual feature sizes)
                    expected_size = 330  # Adjust this based on your actual feature sizes
                    if len(x_features) < expected_size:
                        x_features = np.pad(x_features, (0, expected_size - len(x_features)))
                    else:
                        x_features = x_features[:expected_size]
                    
                    batch_features.append(x_features)
                    
                except Exception as e:
                    logger.error(f"Error loading feature file {file_path}: {str(e)}")
                    skipped_entries += 1
                    continue
            
            if len(batch_features) != 7:
                logger.warning(f"Expected 7 features, got {len(batch_features)} for entry {idx}")
                skipped_entries += 1
                continue
            
            # Concatenate all features (x1-x7)
            combined_features = np.concatenate(batch_features)
            
            # Ensure we have the expected total size (2312)
            if len(combined_features) < 2312:
                combined_features = np.pad(combined_features, (0, 2312 - len(combined_features)))
            else:
                combined_features = combined_features[:2312]
            
            features_list.append(combined_features)
            processed_entries += 1
            
            if processed_entries % 100 == 0:
                logger.info(f"Processed {processed_entries} entries...")
                
        except Exception as e:
            logger.error(f"Error processing entry {idx}: {str(e)}")
            skipped_entries += 1
            continue
    
    if not features_list:
        logger.error("No valid features found in dataloader")
        raise ValueError("No valid features found in dataloader")
    
    # Combine all features
    features = np.vstack(features_list)
    
    # Verify shape
    assert features.shape[1] == 2312, f"Expected 2312 features, got {features.shape[1]}"
    
    logger.info(f"\nSummary:")
    logger.info(f"Loaded {len(features)} samples")
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Processed {processed_entries} entries")
    logger.info(f"Skipped {skipped_entries} entries")
    
    return features

def load_features_and_labels_from_dataloader(dataloader_path: str, label_mappings_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[str]]]:
    """
    Load features and corresponding labels from the dataloader CSV file.
    
    Args:
        dataloader_path (str): Path to the dataloader.csv file
        label_mappings_path (str): Path to the label_mappings.json file
        
    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, List[str]]]: Features, labels, and label mappings
    """
    logger.info("Loading features and labels from dataloader...")
    
    # Load dataloader CSV
    dataloader_df = pd.read_csv(dataloader_path)
    logger.info(f"Loaded dataloader with {len(dataloader_df)} entries")
    
    # Load label mappings
    with open(label_mappings_path, 'r') as f:
        label_mappings = json.load(f)
    logger.info(f"Loaded label mappings for {len(label_mappings)} categories")
    
    # Verify required columns exist
    required_columns = [
        'x1_file', 'x2_file', 'x3_file', 'x4_file', 'x5_file', 'x6_file', 'x7_file',
        'model_family_label', 'model_size_label', 'optimizer_label', 'learning_rate_label', 'batch_size_label'
    ]
    
    missing_columns = [col for col in required_columns if col not in dataloader_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataloader.csv: {missing_columns}")
    
    features_list = []
    labels_list = []
    processed_entries = 0
    skipped_entries = 0
    
    # Calculate label indices for each head
    label_indices = {}
    start_idx = 0
    for name, mapping in label_mappings.items():
        end_idx = start_idx + len(mapping)
        label_indices[name] = (start_idx, end_idx)
        start_idx = end_idx
    
    logger.info("\nLabel indices:")
    for name, (start, end) in label_indices.items():
        logger.info(f"{name}: {start}-{end} ({end-start} classes)")
    
    for idx, row in dataloader_df.iterrows():
        try:
            # Load x1-x7 features for this entry
            batch_features = []
            feature_files = [
                row['x1_file'], row['x2_file'], row['x3_file'], 
                row['x4_file'], row['x5_file'], row['x6_file'], row['x7_file']
            ]
            
            # Check if all feature files exist
            missing_files = [f for f in feature_files if not f or not os.path.exists(f)]
            if missing_files:
                logger.warning(f"Missing feature files for entry {idx}: {missing_files}")
                skipped_entries += 1
                continue
            
            # Load each feature file
            for i, file_path in enumerate(feature_files, 1):
                try:
                    if file_path.endswith('.npy'):
                        x_features = np.load(file_path)
                        
                        # Sanitize corrupted features at load time
                        if not np.all(np.isfinite(x_features)):
                            logger.warning(f"Corrupted features detected in {file_path}, sanitizing...")
                            
                            # Replace NaN with 0, clip Inf values
                            x_features = np.nan_to_num(x_features, nan=0.0)
                            x_features = np.clip(x_features, -1e6, 1e6)
                            
                            logger.info(f"Sanitized {file_path}: NaN/Inf values replaced")
                    else:
                        logger.warning(f"Unexpected file format: {file_path}")
                        skipped_entries += 1
                        continue
                    
                    # Flatten and ensure consistent shape
                    x_features = x_features.flatten()
                    
                    # Pad or truncate to expected size (adjust based on actual feature sizes)
                    expected_size = 330  # Adjust this based on your actual feature sizes
                    if len(x_features) < expected_size:
                        x_features = np.pad(x_features, (0, expected_size - len(x_features)))
                    else:
                        x_features = x_features[:expected_size]
                    
                    batch_features.append(x_features)
                    
                except Exception as e:
                    logger.error(f"Error loading feature file {file_path}: {str(e)}")
                    skipped_entries += 1
                    continue
            
            if len(batch_features) != 7:
                logger.warning(f"Expected 7 features, got {len(batch_features)} for entry {idx}")
                skipped_entries += 1
                continue
            
            # Concatenate all features (x1-x7)
            combined_features = np.concatenate(batch_features)
            
            # Ensure we have the expected total size (2312)
            if len(combined_features) < 2312:
                combined_features = np.pad(combined_features, (0, 2312 - len(combined_features)))
            else:
                combined_features = combined_features[:2312]
            
            # Create labels from the label columns
            label_columns = [
                'model_family_label', 'model_size_label', 'optimizer_label', 
                'learning_rate_label', 'batch_size_label'
            ]
            
            # Parse label strings back to arrays
            labels = []
            for col in label_columns:
                label_str = row[col]
                if isinstance(label_str, str):
                    # Convert string representation back to array
                    label_array = np.array(eval(label_str))
                else:
                    label_array = np.array(label_str)
                labels.append(label_array)
            
            # Combine all labels
            combined_labels = np.concatenate(labels)
            
            features_list.append(combined_features)
            labels_list.append(combined_labels)
            processed_entries += 1
            
            if processed_entries % 100 == 0:
                logger.info(f"Processed {processed_entries} entries...")
                
        except Exception as e:
            logger.error(f"Error processing entry {idx}: {str(e)}")
            skipped_entries += 1
            continue
    
    if not features_list:
        logger.error("No valid features found in dataloader")
        raise ValueError("No valid features found in dataloader")
    
    # Combine all features and labels
    features = np.vstack(features_list)
    labels = np.vstack(labels_list)
    
    # Verify shapes match
    n_samples = len(labels)
    assert features.shape[0] == n_samples, f"Features: {features.shape[0]}, Labels: {n_samples}"
    assert features.shape[1] == 2312, f"Expected 2312 features, got {features.shape[1]}"
    
    # Calculate and display class distribution
    logger.info(f"\nClass Distribution Analysis:")
    total_samples = len(labels)
    
    for name, mapping in label_mappings.items():
        start_idx, end_idx = label_indices[name]
        class_counts = np.zeros(len(mapping))
        
        # Count samples for each class
        for i in range(total_samples):
            label_slice = labels[i, start_idx:end_idx]
            class_idx = np.argmax(label_slice)
            class_counts[class_idx] += 1
        
        logger.info(f"\nClass distribution for {name}:")
        for i, class_name in enumerate(mapping):
            count = int(class_counts[i])
            percentage = (count / total_samples) * 100
            logger.info(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Calculate class weights for balanced training
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        weights = total_samples / (len(mapping) * (class_counts + epsilon))
        logger.info(f"Class weights for {name}: {weights.tolist()}")
    
    logger.info(f"\nSummary:")
    logger.info(f"Loaded {n_samples} samples")
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    logger.info(f"Processed {processed_entries} entries")
    logger.info(f"Skipped {skipped_entries} entries")
    
    return features, labels, label_mappings



def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: optim.Optimizer,
                scheduler: Optional[optim.lr_scheduler._LRScheduler],
                num_epochs: int,
                device: str,
                label_mappings: Dict[str, List[str]]) -> Dict[str, List[float]]:
    """
    Train the model.
    """
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Calculate label indices for each head
    label_indices = {}
    start_idx = 0
    for name, mapping in label_mappings.items():
        end_idx = start_idx + len(mapping)
        label_indices[name] = (start_idx, end_idx)
        start_idx = end_idx
    
    logger.info("\nLabel indices:")
    for name, (start, end) in label_indices.items():
        logger.info(f"{name}: {start}-{end} ({end-start} classes)")
    
            # Custom loss function for multi-head classification
        def multi_head_loss(predictions, labels):
            total_loss = 0.0
            
            for name, pred in predictions.items():
                start_idx, end_idx = label_indices[name]
                target = labels[:, start_idx:end_idx]
                
                # Convert to class indices
                target_indices = target.argmax(dim=1)
                
                # Calculate cross entropy loss with label smoothing
                loss = F.cross_entropy(pred, target_indices, reduction='mean', label_smoothing=0.05)  # Reduced label smoothing
                
                # Add gradient clipping to individual losses
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss detected for {name}: {loss}")
                    loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
                
                total_loss += loss
            
            return total_loss
    
    logger.info("\nStarting Training...")
    logger.info(f"Training on {device} for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            # Check for invalid inputs
            if torch.isnan(features).any() or torch.isinf(features).any():
                logger.warning(f"Invalid features detected in batch, skipping...")
                continue
            
            optimizer.zero_grad()
            predictions = model(features)
            
            loss = multi_head_loss(predictions, labels)
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss detected: {loss}, skipping batch...")
                continue
            
            loss.backward()
            
            # Gradient clipping - more aggressive to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        # Per-head metrics
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
                
                # Collect predictions and targets for each head
                for name, pred in predictions.items():
                    start_idx, end_idx = label_indices[name]
                    target = labels[:, start_idx:end_idx]
                    
                    pred_indices = pred.argmax(dim=1).cpu().numpy()
                    target_indices = target.argmax(dim=1).cpu().numpy()
                    
                    all_predictions[name].extend(pred_indices)
                    all_targets[name].extend(target_indices)
        
        val_loss /= val_batches
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Calculate per-head metrics
        per_head_accuracy = {}
        per_head_f1 = {}
        overall_accuracy = 0.0
        overall_f1 = 0.0
        
        for name in label_mappings.keys():
            if all_predictions[name] and all_targets[name]:
                accuracy = accuracy_score(all_targets[name], all_predictions[name])
                f1 = f1_score(all_targets[name], all_predictions[name], average='macro')
                
                per_head_accuracy[name] = accuracy
                per_head_f1[name] = f1
                overall_accuracy += accuracy
                overall_f1 += f1
        
        # Average metrics across heads
        num_heads = len(label_mappings)
        overall_accuracy /= num_heads
        overall_f1 /= num_heads
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(overall_accuracy)
        history['val_f1'].append(overall_f1)
        
        # Log results
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        logger.info(f"Overall Val Accuracy: {overall_accuracy:.4f} - Overall Val F1: {overall_f1:.4f}")
        
        # Log per-head metrics
        logger.info("\nPer-head Metrics:")
        for name in label_mappings.keys():
            if name in per_head_accuracy:
                logger.info(f"{name:15s}: Acc={per_head_accuracy[name]:.4f}, F1={per_head_f1[name]:.4f}")
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            logger.info(f"\nNew best model at epoch {epoch+1}!")
    
    logger.info(f"\nTraining completed. Best model at epoch {best_epoch + 1}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final validation accuracy: {overall_accuracy:.4f}")
    logger.info(f"Final validation F1: {overall_f1:.4f}")
    
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

def load_trained_model(model_path: str, label_mappings_path: str, device: str) -> Tuple[nn.Module, Dict[str, List[str]]]:
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model state dict
        label_mappings_path (str): Path to the label mappings JSON file
        device (str): Device to load the model on
        
    Returns:
        Tuple[nn.Module, Dict[str, List[str]]]: Loaded model and label mappings
    """
    logger.info(f"Loading trained model from {model_path}")
    
    # Load label mappings
    with open(label_mappings_path, 'r') as f:
        label_mappings = json.load(f)
    logger.info(f"Loaded label mappings for {len(label_mappings)} categories")
    
    # Load state dict first to infer architecture
    state_dict = torch.load(model_path, map_location=device)
    
    # Infer hidden dimensions from the state dict
    # The shared_encoder has layers: Linear, BatchNorm, ReLU, Dropout for each hidden layer
    # Structure: Linear(0) -> BatchNorm(1) -> ReLU(2) -> Dropout(3) -> Linear(4) -> BatchNorm(5) -> ...
    hidden_dims = []
    layer_idx = 0
    
    # Check Linear layers at indices 0, 4, 8, etc.
    while f'shared_encoder.{layer_idx}.weight' in state_dict:
        weight_shape = state_dict[f'shared_encoder.{layer_idx}.weight'].shape
        # Linear layer: (out_features, in_features)
        out_features = weight_shape[0]
        hidden_dims.append(out_features)
        layer_idx += 4  # Move to next Linear layer (skip BatchNorm, ReLU, Dropout)
    
    # Alternative: infer from classification head if shared_encoder inference failed
    if not hidden_dims:
        # Check classification head to get last hidden dimension
        for head_name in ['model_family', 'model_size', 'optimizer', 'learning_rate', 'batch_size']:
            if f'classification_heads.{head_name}.weight' in state_dict:
                weight_shape = state_dict[f'classification_heads.{head_name}.weight'].shape
                # Classification head: (num_classes, in_features)
                last_hidden_dim = weight_shape[1]
                # Try to infer from the structure - if we have 256, likely [1024, 512, 256] or [512, 256]
                # Check if there's a layer with 1024
                if 'shared_encoder.0.weight' in state_dict:
                    first_dim = state_dict['shared_encoder.0.weight'].shape[0]
                    if first_dim == 1024:
                        hidden_dims = [1024, 512, last_hidden_dim]
                    else:
                        hidden_dims = [first_dim, last_hidden_dim]
                else:
                    hidden_dims = [512, last_hidden_dim]
                logger.info(f"Inferred architecture from classification head: {hidden_dims}")
                break
    
    # If we still couldn't infer, use default architecture
    if not hidden_dims:
        logger.warning("Could not infer architecture from state dict, using default [1024, 512, 256]")
        hidden_dims = [1024, 512, 256]
    else:
        logger.info(f"Inferred hidden_dims from model: {hidden_dims}")
    
    # Initialize model architecture (must match training configuration)
    input_dim = 2312  # x1-x7 features
    num_classes_per_head = {
        'model_family': len(label_mappings['model_family']),
        'model_size': len(label_mappings['model_size']),
        'optimizer': len(label_mappings['optimizer']),
        'learning_rate': len(label_mappings['learning_rate']),
        'batch_size': len(label_mappings['batch_size'])
    }
    
    # Create model instance
    model = MultimodalHyperparameterClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes_per_head=num_classes_per_head,
        dropout_rate=0.2
    )
    
    # Load state dict
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model, label_mappings

def run_inference(model: nn.Module, features: np.ndarray, label_mappings: Dict[str, List[str]], 
                  device: str, batch_size: int = 32, labels: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Run inference on all features using the trained model.
    
    Args:
        model: Trained model
        features: Feature array of shape (n_samples, 2312)
        label_mappings: Label mappings for each head
        device: Device to run inference on
        batch_size: Batch size for inference
        labels: Optional ground truth labels for evaluation (shape: n_samples, n_total_labels)
        
    Returns:
        Dictionary containing predictions for each head (class indices and probabilities), and optionally metrics
    """
    logger.info(f"Running inference on {len(features)} samples...")
    
    model.eval()
    
    # Create dataset and dataloader
    # For inference, we don't need labels, so we create dummy labels
    if labels is None:
        dummy_labels = np.zeros((len(features), sum(len(mapping) for mapping in label_mappings.values())))
    else:
        dummy_labels = labels
    inference_dataset = MultimodalDataset(features, dummy_labels)
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Calculate label indices for each head (needed for metrics)
    label_indices = {}
    start_idx = 0
    for name, mapping in label_mappings.items():
        end_idx = start_idx + len(mapping)
        label_indices[name] = (start_idx, end_idx)
        start_idx = end_idx
    
    # Store predictions for each head
    all_predictions = {name: [] for name in label_mappings.keys()}
    all_probabilities = {name: [] for name in label_mappings.keys()}
    all_targets = {name: [] for name in label_mappings.keys()} if labels is not None else None
    
    with torch.no_grad():
        for batch_idx, (batch_features, batch_labels) in enumerate(inference_loader):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device) if labels is not None else None
            
            # Check for invalid inputs
            if torch.isnan(batch_features).any() or torch.isinf(batch_features).any():
                logger.warning(f"Invalid features detected in batch {batch_idx}, skipping...")
                continue
            
            # Get predictions
            predictions = model(batch_features)
            
            # Process predictions for each head
            for name, pred_logits in predictions.items():
                # Get class indices (argmax)
                pred_indices = pred_logits.argmax(dim=1).cpu().numpy()
                all_predictions[name].extend(pred_indices)
                
                # Get probabilities (softmax)
                probs = F.softmax(pred_logits, dim=1).cpu().numpy()
                all_probabilities[name].extend(probs)
                
                # Store targets if labels are provided
                if labels is not None and all_targets is not None:
                    start_idx, end_idx = label_indices[name]
                    target = batch_labels[:, start_idx:end_idx]
                    target_indices = target.argmax(dim=1).cpu().numpy()
                    all_targets[name].extend(target_indices)
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1} batches...")
    
    # Convert to numpy arrays
    predictions_dict = {}
    probabilities_dict = {}
    for name in label_mappings.keys():
        predictions_dict[name] = np.array(all_predictions[name])
        probabilities_dict[name] = np.array(all_probabilities[name])
    
    result = {
        'predictions': predictions_dict,
        'probabilities': probabilities_dict
    }
    
    # Calculate metrics if labels were provided
    if labels is not None and all_targets is not None:
        per_head_accuracy = {}
        per_head_f1 = {}
        
        for name in label_mappings.keys():
            if all_targets[name] and all_predictions[name]:
                accuracy = accuracy_score(all_targets[name], all_predictions[name])
                f1 = f1_score(all_targets[name], all_predictions[name], average='macro')
                
                per_head_accuracy[name] = accuracy
                per_head_f1[name] = f1
        
        result['metrics'] = {
            'accuracy': per_head_accuracy,
            'f1': per_head_f1
        }
        
        # Display metrics in the requested format
        logger.info("\nPer-head Metrics:")
        for name in label_mappings.keys():
            if name in per_head_accuracy:
                logger.info(f"{name:15s}: Acc={per_head_accuracy[name]:.4f}, F1={per_head_f1[name]:.4f}")
    
    logger.info("Inference completed successfully")
    
    return result

def save_inference_results(results: Dict[str, np.ndarray], label_mappings: Dict[str, List[str]], 
                           output_path: str):
    """
    Save inference results to a CSV file.
    
    Args:
        results: Dictionary containing predictions and probabilities
        label_mappings: Label mappings for each head
        output_path: Path to save the results CSV
    """
    logger.info(f"Saving inference results to {output_path}")
    
    predictions = results['predictions']
    probabilities = results['probabilities']
    
    # Create DataFrame with predictions
    data = {}
    n_samples = len(predictions[list(predictions.keys())[0]])
    
    for name in label_mappings.keys():
        # Add predicted class index
        data[f'{name}_predicted_index'] = predictions[name]
        
        # Add predicted class name
        data[f'{name}_predicted'] = [label_mappings[name][int(idx)] for idx in predictions[name]]
        
        # Add probability of predicted class
        max_probs = probabilities[name].max(axis=1)
        data[f'{name}_confidence'] = max_probs
        
        # Add probabilities for all classes
        for i, class_name in enumerate(label_mappings[name]):
            data[f'{name}_prob_{class_name}'] = probabilities[name][:, i]
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    # Print summary statistics
    logger.info("\nInference Results Summary:")
    for name in label_mappings.keys():
        pred_counts = pd.Series(predictions[name]).value_counts().sort_index()
        logger.info(f"\n{name} predictions:")
        for idx, count in pred_counts.items():
            class_name = label_mappings[name][int(idx)]
            percentage = (count / n_samples) * 100
            logger.info(f"  {class_name}: {count} samples ({percentage:.1f}%)")

def main():
    """Main function to train or run inference with the classifier."""
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic algorithms (may be slower)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'], 
                       help='Mode: train or inference')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='Path to trained model (required for inference mode)')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save inference results (default: ./models/inference_results.csv)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed, deterministic=args.deterministic)
    
    # Set up distributed training
    rank, world_size, gpu = setup_distributed()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if rank == 0:
        logger.info(f"Using device: {device}")
    
    # Configuration
    DATALOADER_PATH = os.path.abspath("./dataloader/dataloader.csv")
    LABEL_MAPPINGS_PATH = os.path.abspath("./dataloader/label_mappings.json")
    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    # Verify paths exist
    if not os.path.exists(DATALOADER_PATH):
        raise FileNotFoundError(f"Dataloader file not found: {DATALOADER_PATH}")
    
    # Handle inference mode
    if args.mode == 'inference':
        if rank == 0:
            logger.info("=" * 80)
            logger.info("RUNNING INFERENCE MODE")
            logger.info("=" * 80)
        
        # Set model path
        if args.model_path is None:
            model_path = os.path.join(MODELS_DIR, "multimodal_hyperparameter_classifier.pt")
        else:
            model_path = args.model_path
        
        # Set label mappings path (try models directory first, then dataloader directory)
        if os.path.exists(os.path.join(MODELS_DIR, "label_mappings.json")):
            label_mappings_path = os.path.join(MODELS_DIR, "label_mappings.json")
            if rank == 0:
                logger.info(f"Using label_mappings.json from models directory: {label_mappings_path}")
        else:
            label_mappings_path = LABEL_MAPPINGS_PATH
            if rank == 0:
                logger.info(f"Using label_mappings.json from dataloader directory: {label_mappings_path}")
        
        if rank == 0:
            logger.info(f"Model path: {model_path}")
            logger.info(f"Label mappings path: {label_mappings_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(label_mappings_path):
            raise FileNotFoundError(f"Label mappings file not found: {label_mappings_path}")
        
        try:
            # Load trained model
            model, label_mappings = load_trained_model(model_path, label_mappings_path, device)
            
            # Load all features and labels from dataloader for evaluation
            features, labels, _ = load_features_and_labels_from_dataloader(DATALOADER_PATH, label_mappings_path)
            
            if rank == 0:
                logger.info(f"Loaded {len(features)} samples for inference")
                logger.info(f"Features shape: {features.shape}")
                logger.info(f"Labels shape: {labels.shape}")
            
            # Run inference with labels for metric calculation
            results = run_inference(model, features, label_mappings, device, batch_size=32, labels=labels)
            
            if rank == 0:
                # Save results
                if args.output_path is None:
                    output_path = os.path.join(MODELS_DIR, "inference_results.csv")
                else:
                    output_path = args.output_path
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_inference_results(results, label_mappings, output_path)
                
                logger.info("=" * 80)
                logger.info("INFERENCE COMPLETED SUCCESSFULLY")
                logger.info("=" * 80)
        
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise
        finally:
            cleanup_distributed()
        
        return
    
    # Training mode (original code)
    if rank == 0:
        logger.info("=" * 80)
        logger.info("RUNNING TRAINING MODE")
        logger.info("=" * 80)
    
    if not os.path.exists(LABEL_MAPPINGS_PATH):
        raise FileNotFoundError(f"Label mappings file not found: {LABEL_MAPPINGS_PATH}")
    
    try:
        # Load data from dataloader
        features, labels, label_mappings = load_features_and_labels_from_dataloader(DATALOADER_PATH, LABEL_MAPPINGS_PATH)
        
        if rank == 0:
            logger.info(f"Loaded {len(features)} samples")
            logger.info(f"Features shape: {features.shape}")
            logger.info(f"Labels shape: {labels.shape}")
        
        # Calculate split indices
        n_samples = len(features)
        train_size = int(0.8 * n_samples)
        
        # Create indices for shuffling (using fixed seed for reproducibility)
        rng = np.random.RandomState(args.seed)
        indices = rng.permutation(n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Split data using indices
        X_train = features[train_indices]
        X_val = features[val_indices]
        y_train = labels[train_indices]
        y_val = labels[val_indices]
        
        if rank == 0:
            logger.info(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}")
        
        # Create datasets and dataloaders
        train_dataset = MultimodalDataset(X_train, y_train)
        val_dataset = MultimodalDataset(X_val, y_val)
        
        train_sampler = DistributedSampler(train_dataset, seed=args.seed) if world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset, seed=args.seed) if world_size > 1 else None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,  # Reduced batch size for stability
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=2,  # Reduced workers
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,  # Reduced batch size for stability
            shuffle=False,
            sampler=val_sampler,
            num_workers=2,  # Reduced workers
            pin_memory=True
        )
        
        # Initialize model with smaller architecture for stability
        input_dim = 2312  # x1-x7 features
        hidden_dims = [512, 256, 128]  # Reduced dimensions
        num_classes_per_head = {
            'model_family': len(label_mappings['model_family']),
            'model_size': len(label_mappings['model_size']),
            'optimizer': len(label_mappings['optimizer']),
            'learning_rate': len(label_mappings['learning_rate']),
            'batch_size': len(label_mappings['batch_size'])
        }
        
        model = MultimodalHyperparameterClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes_per_head=num_classes_per_head,
            dropout_rate=0.2  # Reduced dropout
        ).to(gpu)
        
        if world_size > 1:
            model = DDP(model, device_ids=[gpu])
        
        # Initialize optimizer with more conservative settings
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.0001,  # Reduced from 0.001 to prevent instability
            weight_decay=0.001,  # Reduced from 0.01
            betas=(0.9, 0.999),
            eps=1e-8  # Increased epsilon for numerical stability
        )
        
        # Set optimizer seed for reproducibility
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.requires_grad:
                    param.data = param.data.to(torch.float32)  # Ensure consistent dtype
        
        # Learning rate scheduler - more aggressive
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.3,  # Reduce LR more aggressively
            patience=2,   # Reduce patience
            verbose=True,
            min_lr=1e-7   # Set minimum learning rate
        )
        
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=50,
            device=gpu,
            label_mappings=label_mappings
        )
        
        if rank == 0:  # Only save on main process
            # Create models directory
            model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            os.makedirs(model_save_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_save_dir, "multimodal_hyperparameter_classifier.pt")
            if isinstance(model, DDP):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            
            # Save training history
            history_path = os.path.join(model_save_dir, "training_history_multimodal_classifier.json")
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            
            # Save label mappings
            mappings_path = os.path.join(model_save_dir, "label_mappings.json")
            with open(mappings_path, "w") as f:
                json.dump(label_mappings, f, indent=2)
            
            logger.info(f"Model, training history, and label mappings saved in {model_save_dir}")
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
