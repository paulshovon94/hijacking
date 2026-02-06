"""
Subsampling Experiment: Can the attacker fine-tune fewer than 189 shadows and still match performance?

This script investigates whether an attacker can achieve similar attack performance
with fewer shadow models by:
1. Creating subsets of shadow models with different sizes (e.g., 10, 20, 50, 100, 150, 189)
2. Training attack models on each subset
3. Evaluating on the same test set (from reserved test models)
4. Comparing performance across different subsample sizes

The goal is to determine the minimum number of shadow models needed to achieve
comparable attack performance.
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

def load_features_and_labels_from_dataloader(dataloader_path: str, label_mappings_path: str,
                                             selected_model_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[str]]]:
    """
    Load features and corresponding labels from the dataloader CSV file.
    
    Args:
        dataloader_path (str): Path to the dataloader.csv file
        label_mappings_path (str): Path to the label_mappings.json file
        selected_model_indices (Optional[List[int]]): If provided, only load data from these model indices
        
    Returns:
        Tuple containing features, labels, and label_mappings
    """
    logger.info("Loading features and labels from dataloader...")
    
    # Load dataloader CSV
    dataloader_df = pd.read_csv(dataloader_path)
    logger.info(f"Loaded dataloader with {len(dataloader_df)} entries")
    
    # Filter by selected model indices if provided
    if selected_model_indices is not None:
        dataloader_df = dataloader_df[dataloader_df['model_index'].isin(selected_model_indices)]
        logger.info(f"Filtered to {len(dataloader_df)} entries from {len(selected_model_indices)} shadow models")
    
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
            loss = F.cross_entropy(pred, target_indices, reduction='mean', label_smoothing=0.05)
            
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
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        if train_batches > 0:
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
        
        if val_batches > 0:
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
        if num_heads > 0:
            overall_accuracy /= num_heads
            overall_f1 /= num_heads
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(overall_accuracy)
        history['val_f1'].append(overall_f1)
        
        # Log results
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            logger.info(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            logger.info(f"Overall Val Accuracy: {overall_accuracy:.4f} - Overall Val F1: {overall_f1:.4f}")
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
    
    logger.info(f"\nTraining completed. Best model at epoch {best_epoch + 1}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Final validation accuracy: {overall_accuracy:.4f}")
    logger.info(f"Final validation F1: {overall_f1:.4f}")
    
    return history

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str, 
                  label_mappings: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        label_mappings: Label mappings for each head
        
    Returns:
        Dictionary containing evaluation metrics for each head
    """
    model.eval()
    
    # Calculate label indices for each head
    label_indices = {}
    start_idx = 0
    for name, mapping in label_mappings.items():
        end_idx = start_idx + len(mapping)
        label_indices[name] = (start_idx, end_idx)
        start_idx = end_idx
    
    # Per-head metrics
    all_predictions = {name: [] for name in label_mappings.keys()}
    all_targets = {name: [] for name in label_mappings.keys()}
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            predictions = model(features)
            
            # Collect predictions and targets for each head
            for name, pred in predictions.items():
                start_idx, end_idx = label_indices[name]
                target = labels[:, start_idx:end_idx]
                
                pred_indices = pred.argmax(dim=1).cpu().numpy()
                target_indices = target.argmax(dim=1).cpu().numpy()
                
                all_predictions[name].extend(pred_indices)
                all_targets[name].extend(target_indices)
    
    # Calculate metrics for each head
    results = {}
    for name in label_mappings.keys():
        if all_predictions[name] and all_targets[name]:
            accuracy = accuracy_score(all_targets[name], all_predictions[name])
            f1_macro = f1_score(all_targets[name], all_predictions[name], average='macro')
            f1_weighted = f1_score(all_targets[name], all_predictions[name], average='weighted')
            
            results[name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted
            }
    
    return results

def get_unique_model_indices(dataloader_path: str) -> List[int]:
    """
    Get unique model indices from the dataloader CSV.
    
    Args:
        dataloader_path (str): Path to the dataloader.csv file
        
    Returns:
        List of unique model indices
    """
    df = pd.read_csv(dataloader_path)
    unique_indices = sorted(df['model_index'].unique().tolist())
    logger.info(f"Found {len(unique_indices)} unique shadow models (indices: {min(unique_indices)} to {max(unique_indices)})")
    return unique_indices

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

    if world_size > 1:
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

def main():
    """Main function to run subsampling experiment."""
    parser = argparse.ArgumentParser(description='Subsampling experiment for shadow models')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic algorithms (may be slower)')
    parser.add_argument('--subsample_sizes', nargs='+', type=int, default=[10, 20, 50, 100, 150, 189],
                       help='List of subsample sizes to test')
    parser.add_argument('--seeds', nargs='+', type=int, default=[32, 42, 52],
                       help='List of specific seeds to use for trials (default: 32, 42, 52)')
    parser.add_argument('--num_epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of models to reserve for testing')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed, deterministic=args.deterministic)
    
    # Set up distributed training
    rank, world_size, gpu = setup_distributed()
    
    # Configuration
    DATALOADER_PATH = os.path.abspath("./dataloader/dataloader.csv")
    LABEL_MAPPINGS_PATH = os.path.abspath("./dataloader/label_mappings.json")
    
    # Verify paths exist
    if not os.path.exists(DATALOADER_PATH):
        raise FileNotFoundError(f"Dataloader file not found: {DATALOADER_PATH}")
    if not os.path.exists(LABEL_MAPPINGS_PATH):
        raise FileNotFoundError(f"Label mappings file not found: {LABEL_MAPPINGS_PATH}")
    
    if rank == 0:
        logger.info(f"Subsampling Experiment: Testing performance with different numbers of shadow models")
        logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    try:
        # Get all unique model indices
        all_model_indices = get_unique_model_indices(DATALOADER_PATH)
        max_models = len(all_model_indices)
        
        # Reserve test models first to know how many are available for training
        np.random.seed(args.seed)
        n_test_models = max(20, int(len(all_model_indices) * args.test_size))
        max_train_models = max_models - n_test_models
        
        # Filter subsample sizes to be valid (must be <= available training models)
        valid_subsample_sizes = [s for s in args.subsample_sizes if s <= max_train_models]
        if not valid_subsample_sizes:
            raise ValueError(f"No valid subsample sizes. Max available for training: {max_train_models} (after reserving {n_test_models} for testing)")
        
        if rank == 0:
            logger.info(f"Total models: {max_models}, Reserved for testing: {n_test_models}, Available for training: {max_train_models}")
            logger.info(f"Running subsampling experiment with sizes: {valid_subsample_sizes}")
            logger.info(f"Using seeds: {args.seeds} ({len(args.seeds)} trials per size)")
        
        # Reserve a fixed set of model indices for testing (separate from training)
        # This ensures no data leakage - test models are never used for training
        test_model_indices = np.random.choice(all_model_indices, n_test_models, replace=False).tolist()
        test_model_indices = sorted(test_model_indices)
        train_model_indices = [idx for idx in all_model_indices if idx not in test_model_indices]
        
        if rank == 0:
            logger.info(f"\nReserved {len(test_model_indices)} models for testing (indices: {test_model_indices[:10]}...)" if len(test_model_indices) > 10 else f"\nReserved {len(test_model_indices)} models for testing (indices: {test_model_indices})")
            logger.info(f"Available {len(train_model_indices)} models for training")
        
        # Load test set from reserved test models
        if rank == 0:
            logger.info("\n" + "="*50)
            logger.info("Loading test set from reserved test models...")
            logger.info("="*50)
        test_features, test_labels, label_mappings = load_features_and_labels_from_dataloader(
            DATALOADER_PATH, LABEL_MAPPINGS_PATH, selected_model_indices=test_model_indices
        )
        
        if rank == 0:
            logger.info(f"Test set size: {len(test_features)} samples from {len(test_model_indices)} models")
        
        # Store results for each subsample size
        all_results = {}
        
        # Run experiment for each subsample size
        for subsample_size in valid_subsample_sizes:
            if rank == 0:
                logger.info("\n" + "="*50)
                logger.info(f"SUBSAMPLE SIZE: {subsample_size} shadow models")
                logger.info("="*50)
            
            size_results = []
            
            # Run multiple trials for this subsample size using specific seeds
            for trial_idx, trial_seed in enumerate(args.seeds):
                if rank == 0:
                    logger.info(f"\n--- Trial {trial_idx + 1}/{len(args.seeds)} (seed: {trial_seed}) ---")
                
                # Set seed for this trial to ensure reproducibility
                set_seed(trial_seed, deterministic=args.deterministic)
                
                # Randomly sample model indices from TRAINING models only (exclude test models)
                np.random.seed(trial_seed)
                
                # Ensure we don't sample more models than available
                available_train_models = len(train_model_indices)
                actual_subsample_size = min(subsample_size, available_train_models)
                
                if actual_subsample_size < subsample_size and rank == 0:
                    logger.warning(f"Requested {subsample_size} models but only {available_train_models} available. Using {actual_subsample_size}.")
                
                selected_model_indices = np.random.choice(train_model_indices, actual_subsample_size, replace=False).tolist()
                selected_model_indices = sorted(selected_model_indices)
                
                if rank == 0:
                    logger.info(f"Selected model indices: {selected_model_indices[:10]}..." if len(selected_model_indices) > 10 else f"Selected model indices: {selected_model_indices}")
                
                # Load training data for this subsample (guaranteed no overlap with test set)
                train_features, train_labels, _ = load_features_and_labels_from_dataloader(
                    DATALOADER_PATH, LABEL_MAPPINGS_PATH, selected_model_indices=selected_model_indices
                )
                
                if rank == 0:
                    logger.info(f"Training samples: {len(train_features)} (from {len(selected_model_indices)} models, no overlap with test set)")
                
                # Split training data into train/validation
                n_train_samples = len(train_features)
                train_size = int(0.8 * n_train_samples)
                
                # Create indices for shuffling (using fixed seed for reproducibility)
                rng = np.random.RandomState(trial_seed)
                indices = rng.permutation(n_train_samples)
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
                
                X_train = train_features[train_indices]
                X_val = train_features[val_indices]
                y_train = train_labels[train_indices]
                y_val = train_labels[val_indices]
                
                if rank == 0:
                    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
                
                # Create datasets and dataloaders
                train_dataset = MultimodalDataset(X_train, y_train)
                val_dataset = MultimodalDataset(X_val, y_val)
                test_dataset = MultimodalDataset(test_features, test_labels)
                
                train_sampler = DistributedSampler(train_dataset, seed=trial_seed) if world_size > 1 else None
                val_sampler = DistributedSampler(val_dataset, seed=trial_seed) if world_size > 1 else None
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=32,
                    shuffle=(train_sampler is None),
                    sampler=train_sampler,
                    num_workers=2,
                    pin_memory=True
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=32,
                    shuffle=False,
                    sampler=val_sampler,
                    num_workers=2,
                    pin_memory=True
                )
                
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True
                )
                
                # Initialize model
                input_dim = 2312
                hidden_dims = [512, 256, 128]
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
                    dropout_rate=0.2
                ).to(gpu)
                
                if world_size > 1:
                    model = DDP(model, device_ids=[gpu])
                
                # Initialize optimizer
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=0.0001,
                    weight_decay=0.001,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
                
                # Learning rate scheduler
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.3,
                    patience=2,
                    verbose=False,
                    min_lr=1e-7
                )
                
                # Train model
                history = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    num_epochs=args.num_epochs,
                    device=gpu,
                    label_mappings=label_mappings
                )
                
                # Evaluate on test set
                test_results = evaluate_model(model, test_loader, gpu, label_mappings)
                
                # Calculate overall metrics
                overall_accuracy = np.mean([r['accuracy'] for r in test_results.values()])
                overall_f1_macro = np.mean([r['f1_macro'] for r in test_results.values()])
                overall_f1_weighted = np.mean([r['f1_weighted'] for r in test_results.values()])
                
                trial_result = {
                    'subsample_size': subsample_size,
                    'trial': trial_idx + 1,
                    'seed': trial_seed,
                    'selected_model_indices': selected_model_indices,
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(test_features),
                    'test_results': test_results,
                    'overall_accuracy': overall_accuracy,
                    'overall_f1_macro': overall_f1_macro,
                    'overall_f1_weighted': overall_f1_weighted,
                    'training_history': history
                }
                
                if rank == 0:
                    size_results.append(trial_result)
                    
                    logger.info(f"\nTrial {trial_idx + 1} (seed: {trial_seed}) Results:")
                    logger.info(f"  Overall Test Accuracy: {overall_accuracy:.4f}")
                    logger.info(f"  Overall Test F1-Macro: {overall_f1_macro:.4f}")
                    logger.info(f"  Overall Test F1-Weighted: {overall_f1_weighted:.4f}")
            
            if rank == 0:
                all_results[subsample_size] = size_results
                
                # Calculate statistics across trials for overall metrics
                accuracies = [r['overall_accuracy'] for r in size_results]
                f1_macros = [r['overall_f1_macro'] for r in size_results]
                f1_weighteds = [r['overall_f1_weighted'] for r in size_results]
                
                logger.info(f"\nSubsample Size {subsample_size} Summary (across {len(args.seeds)} trials with seeds {args.seeds}):")
                logger.info(f"  Overall Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
                logger.info(f"  Overall F1-Macro: {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
                logger.info(f"  Overall F1-Weighted: {np.mean(f1_weighteds):.4f} ± {np.std(f1_weighteds):.4f}")
                
                # Calculate per-head statistics
                head_names = ['model_family', 'model_size', 'optimizer', 'learning_rate', 'batch_size']
                per_head_stats = {}
                
                for head_name in head_names:
                    # Extract metrics for this head across all trials
                    head_accuracies = [r['test_results'][head_name]['accuracy'] for r in size_results if head_name in r['test_results']]
                    head_f1_macros = [r['test_results'][head_name]['f1_macro'] for r in size_results if head_name in r['test_results']]
                    head_f1_weighteds = [r['test_results'][head_name]['f1_weighted'] for r in size_results if head_name in r['test_results']]
                    
                    if head_accuracies:
                        per_head_stats[head_name] = {
                            'accuracy_mean': np.mean(head_accuracies),
                            'accuracy_std': np.std(head_accuracies),
                            'f1_macro_mean': np.mean(head_f1_macros),
                            'f1_macro_std': np.std(head_f1_macros),
                            'f1_weighted_mean': np.mean(head_f1_weighteds),
                            'f1_weighted_std': np.std(head_f1_weighteds)
                        }
                
                # Display per-head statistics
                logger.info(f"\n  Per-Head Statistics:")
                for head_name in head_names:
                    if head_name in per_head_stats:
                        stats = per_head_stats[head_name]
                        logger.info(f"    {head_name}:")
                        logger.info(f"      Accuracy: {stats['accuracy_mean']:.4f} ± {stats['accuracy_std']:.4f}")
                        logger.info(f"      F1-Macro: {stats['f1_macro_mean']:.4f} ± {stats['f1_macro_std']:.4f}")
                        logger.info(f"      F1-Weighted: {stats['f1_weighted_mean']:.4f} ± {stats['f1_weighted_std']:.4f}")
        
        if rank == 0:  # Only save on main process
            # Save results
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subsampling_results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save detailed results
            results_path = os.path.join(results_dir, "subsampling_experiment_results.json")
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                return obj
            
            serializable_results = convert_to_serializable(all_results)
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"\nResults saved to {results_path}")
            
            # Create summary CSV with overall and per-head metrics
            summary_data = []
            head_names = ['model_family', 'model_size', 'optimizer', 'learning_rate', 'batch_size']
            
            for subsample_size, trials in all_results.items():
                for trial in trials:
                    row = {
                        'subsample_size': subsample_size,
                        'trial': trial['trial'],
                        'seed': trial.get('seed', 'N/A'),
                        'train_samples': trial['train_samples'],
                        'test_samples': trial['test_samples'],
                        'overall_accuracy': trial['overall_accuracy'],
                        'overall_f1_macro': trial['overall_f1_macro'],
                        'overall_f1_weighted': trial['overall_f1_weighted']
                    }
                    
                    # Add per-head metrics
                    for head_name in head_names:
                        if head_name in trial['test_results']:
                            row[f'{head_name}_accuracy'] = trial['test_results'][head_name]['accuracy']
                            row[f'{head_name}_f1_macro'] = trial['test_results'][head_name]['f1_macro']
                            row[f'{head_name}_f1_weighted'] = trial['test_results'][head_name]['f1_weighted']
                        else:
                            row[f'{head_name}_accuracy'] = None
                            row[f'{head_name}_f1_macro'] = None
                            row[f'{head_name}_f1_weighted'] = None
                    
                    summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(results_dir, "subsampling_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Summary saved to {summary_path}")
            
            # Create per-head statistics CSV
            per_head_stats_data = []
            for subsample_size in valid_subsample_sizes:
                trials = all_results[subsample_size]
                for head_name in head_names:
                    head_accuracies = [r['test_results'][head_name]['accuracy'] for r in trials if head_name in r['test_results']]
                    head_f1_macros = [r['test_results'][head_name]['f1_macro'] for r in trials if head_name in r['test_results']]
                    head_f1_weighteds = [r['test_results'][head_name]['f1_weighted'] for r in trials if head_name in r['test_results']]
                    
                    if head_accuracies:
                        per_head_stats_data.append({
                            'subsample_size': subsample_size,
                            'head_name': head_name,
                            'accuracy_mean': np.mean(head_accuracies),
                            'accuracy_std': np.std(head_accuracies),
                            'f1_macro_mean': np.mean(head_f1_macros),
                            'f1_macro_std': np.std(head_f1_macros),
                            'f1_weighted_mean': np.mean(head_f1_weighteds),
                            'f1_weighted_std': np.std(head_f1_weighteds)
                        })
            
            per_head_stats_df = pd.DataFrame(per_head_stats_data)
            per_head_stats_path = os.path.join(results_dir, "subsampling_per_head_stats.csv")
            per_head_stats_df.to_csv(per_head_stats_path, index=False)
            logger.info(f"Per-head statistics saved to {per_head_stats_path}")
            
            # Print final summary
            logger.info("\n" + "="*50)
            logger.info("FINAL SUMMARY")
            logger.info("="*50)
            
            head_names = ['model_family', 'model_size', 'optimizer', 'learning_rate', 'batch_size']
            
            for subsample_size in valid_subsample_sizes:
                trials = all_results[subsample_size]
                accuracies = [r['overall_accuracy'] for r in trials]
                f1_macros = [r['overall_f1_macro'] for r in trials]
                
                logger.info(f"\nSubsample Size {subsample_size}:")
                logger.info(f"  Overall Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
                logger.info(f"  Overall Mean F1-Macro: {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
                logger.info(f"  Overall Min Accuracy: {np.min(accuracies):.4f}")
                logger.info(f"  Overall Max Accuracy: {np.max(accuracies):.4f}")
                
                # Calculate and display per-head statistics
                logger.info(f"\n  Per-Head Statistics for Subsample Size {subsample_size}:")
                for head_name in head_names:
                    # Extract metrics for this head across all trials
                    head_accuracies = [r['test_results'][head_name]['accuracy'] for r in trials if head_name in r['test_results']]
                    head_f1_macros = [r['test_results'][head_name]['f1_macro'] for r in trials if head_name in r['test_results']]
                    head_f1_weighteds = [r['test_results'][head_name]['f1_weighted'] for r in trials if head_name in r['test_results']]
                    
                    if head_accuracies:
                        logger.info(f"    {head_name}:")
                        logger.info(f"      Accuracy: {np.mean(head_accuracies):.4f} ± {np.std(head_accuracies):.4f}")
                        logger.info(f"      F1-Macro: {np.mean(head_f1_macros):.4f} ± {np.std(head_f1_macros):.4f}")
                        logger.info(f"      F1-Weighted: {np.mean(head_f1_weighteds):.4f} ± {np.std(head_f1_weighteds):.4f}")
            
            # Print cross-subsample summary for each head
            logger.info("\n" + "="*50)
            logger.info("CROSS-SUBSAMPLE PER-HEAD SUMMARY")
            logger.info("="*50)
            
            for head_name in head_names:
                logger.info(f"\n{head_name.upper()}:")
                logger.info(f"  {'Subsample Size':<15} {'Accuracy (mean ± std)':<25} {'F1-Macro (mean ± std)':<25} {'F1-Weighted (mean ± std)':<25}")
                logger.info(f"  {'-'*15} {'-'*25} {'-'*25} {'-'*25}")
                
                for subsample_size in valid_subsample_sizes:
                    trials = all_results[subsample_size]
                    head_accuracies = [r['test_results'][head_name]['accuracy'] for r in trials if head_name in r['test_results']]
                    head_f1_macros = [r['test_results'][head_name]['f1_macro'] for r in trials if head_name in r['test_results']]
                    head_f1_weighteds = [r['test_results'][head_name]['f1_weighted'] for r in trials if head_name in r['test_results']]
                    
                    if head_accuracies:
                        acc_str = f"{np.mean(head_accuracies):.4f} ± {np.std(head_accuracies):.4f}"
                        f1_macro_str = f"{np.mean(head_f1_macros):.4f} ± {np.std(head_f1_macros):.4f}"
                        f1_weighted_str = f"{np.mean(head_f1_weighteds):.4f} ± {np.std(head_f1_weighteds):.4f}"
                        logger.info(f"  {subsample_size:<15} {acc_str:<25} {f1_macro_str:<25} {f1_weighted_str:<25}")
    
    except Exception as e:
        logger.error(f"Error during subsampling experiment: {str(e)}")
        raise
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
