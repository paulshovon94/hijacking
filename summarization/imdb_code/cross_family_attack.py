"""
Cross-Family Attack: Train on BART and Pegasus, Test on GPT-2

This model trains a multimodal hyperparameter classifier using only BART and Pegasus models
from the training data, but evaluates its performance on GPT-2 models to test cross-family
generalization capabilities.

The attack tests whether hyperparameter information learned from one model family
can be used to predict hyperparameters of a different model family.
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

def load_features_and_labels_from_dataloader_cross_family(dataloader_path: str, label_mappings_path: str, 
                                                         train_families: List[str], test_families: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, List[str]]]:
    """
    Load features and corresponding labels from the dataloader CSV file with cross-family split.
    
    Args:
        dataloader_path (str): Path to the dataloader.csv file
        label_mappings_path (str): Path to the label_mappings.json file
        train_families (List[str]): Model families to use for training (e.g., ['BART', 'Pegasus'])
        test_families (List[str]): Model families to use for testing (e.g., ['GPT-2'])
        
    Returns:
        Tuple containing train_features, train_labels, test_features, test_labels, and label_mappings
    """
    logger.info("Loading features and labels from dataloader with cross-family split...")
    logger.info(f"Training families: {train_families}")
    logger.info(f"Testing families: {test_families}")
    
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
    
    # Split data by model family
    train_df = dataloader_df[dataloader_df['model_family'].isin(train_families)]
    test_df = dataloader_df[dataloader_df['model_family'].isin(test_families)]
    
    logger.info(f"Training data: {len(train_df)} samples from families {train_families}")
    logger.info(f"Testing data: {len(test_df)} samples from families {test_families}")
    
    # Process training data
    train_features_list = []
    train_labels_list = []
    train_processed_entries = 0
    train_skipped_entries = 0
    
    # Process testing data
    test_features_list = []
    test_labels_list = []
    test_processed_entries = 0
    test_skipped_entries = 0
    
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
    
    def process_dataframe(df, features_list, labels_list, processed_entries, skipped_entries, data_type):
        """Helper function to process a dataframe"""
        for idx, row in df.iterrows():
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
                    logger.warning(f"Missing feature files for {data_type} entry {idx}: {missing_files}")
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
                    logger.warning(f"Expected 7 features, got {len(batch_features)} for {data_type} entry {idx}")
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
                    logger.info(f"Processed {processed_entries} {data_type} entries...")
                    
            except Exception as e:
                logger.error(f"Error processing {data_type} entry {idx}: {str(e)}")
                skipped_entries += 1
                continue
        
        return processed_entries, skipped_entries
    
    # Process training data
    train_processed_entries, train_skipped_entries = process_dataframe(
        train_df, train_features_list, train_labels_list, train_processed_entries, train_skipped_entries, "training"
    )
    
    # Process testing data
    test_processed_entries, test_skipped_entries = process_dataframe(
        test_df, test_features_list, test_labels_list, test_processed_entries, test_skipped_entries, "testing"
    )
    
    if not train_features_list:
        logger.error("No valid training features found in dataloader")
        raise ValueError("No valid training features found in dataloader")
    
    if not test_features_list:
        logger.error("No valid testing features found in dataloader")
        raise ValueError("No valid testing features found in dataloader")
    
    # Combine all features and labels
    train_features = np.vstack(train_features_list)
    train_labels = np.vstack(train_labels_list)
    test_features = np.vstack(test_features_list)
    test_labels = np.vstack(test_labels_list)
    
    # Verify shapes match
    assert train_features.shape[0] == len(train_labels), f"Train Features: {train_features.shape[0]}, Train Labels: {len(train_labels)}"
    assert test_features.shape[0] == len(test_labels), f"Test Features: {test_features.shape[0]}, Test Labels: {len(test_labels)}"
    assert train_features.shape[1] == 2312, f"Expected 2312 features, got {train_features.shape[1]}"
    assert test_features.shape[1] == 2312, f"Expected 2312 features, got {test_features.shape[1]}"
    
    # Calculate and display class distribution for training data
    logger.info(f"\nTraining Data Class Distribution Analysis:")
    train_total_samples = len(train_labels)
    
    for name, mapping in label_mappings.items():
        start_idx, end_idx = label_indices[name]
        class_counts = np.zeros(len(mapping))
        
        # Count samples for each class
        for i in range(train_total_samples):
            label_slice = train_labels[i, start_idx:end_idx]
            class_idx = np.argmax(label_slice)
            class_counts[class_idx] += 1
        
        logger.info(f"\nClass distribution for {name} (training):")
        for i, class_name in enumerate(mapping):
            count = int(class_counts[i])
            percentage = (count / train_total_samples) * 100
            logger.info(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    # Calculate and display class distribution for testing data
    logger.info(f"\nTesting Data Class Distribution Analysis:")
    test_total_samples = len(test_labels)
    
    for name, mapping in label_mappings.items():
        start_idx, end_idx = label_indices[name]
        class_counts = np.zeros(len(mapping))
        
        # Count samples for each class
        for i in range(test_total_samples):
            label_slice = test_labels[i, start_idx:end_idx]
            class_idx = np.argmax(label_slice)
            class_counts[class_idx] += 1
        
        logger.info(f"\nClass distribution for {name} (testing):")
        for i, class_name in enumerate(mapping):
            count = int(class_counts[i])
            percentage = (count / test_total_samples) * 100
            logger.info(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    logger.info(f"\nCross-Family Split Summary:")
    logger.info(f"Training samples: {len(train_labels)} from families {train_families}")
    logger.info(f"Testing samples: {len(test_labels)} from families {test_families}")
    logger.info(f"Train features shape: {train_features.shape}")
    logger.info(f"Train labels shape: {train_labels.shape}")
    logger.info(f"Test features shape: {test_features.shape}")
    logger.info(f"Test labels shape: {test_labels.shape}")
    logger.info(f"Training processed: {train_processed_entries}, skipped: {train_skipped_entries}")
    logger.info(f"Testing processed: {test_processed_entries}, skipped: {test_skipped_entries}")
    
    return train_features, train_labels, test_features, test_labels, label_mappings

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

def evaluate_cross_family(model: nn.Module, test_loader: DataLoader, device: str, 
                         label_mappings: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the model on cross-family test data.
    
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
            
            logger.info(f"\nCross-Family Test Results for {name}:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  F1-Macro: {f1_macro:.4f}")
            logger.info(f"  F1-Weighted: {f1_weighted:.4f}")
    
    return results

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

def main():
    """Main function to train the cross-family attack classifier."""
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic algorithms (may be slower)')
    parser.add_argument('--train_families', nargs='+', default=['BART', 'Pegasus'], 
                       help='Model families to use for training')
    parser.add_argument('--test_families', nargs='+', default=['GPT-2'], 
                       help='Model families to use for testing')
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
        logger.info(f"Cross-Family Attack: Train on {args.train_families}, Test on {args.test_families}")
        logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    try:
        # Load data with cross-family split
        train_features, train_labels, test_features, test_labels, label_mappings = load_features_and_labels_from_dataloader_cross_family(
            DATALOADER_PATH, LABEL_MAPPINGS_PATH, args.train_families, args.test_families
        )
        
        if rank == 0:
            logger.info(f"Loaded {len(train_features)} training samples and {len(test_features)} test samples")
            logger.info(f"Train features shape: {train_features.shape}")
            logger.info(f"Test features shape: {test_features.shape}")
        
        # Split training data into train/validation
        n_train_samples = len(train_features)
        train_size = int(0.8 * n_train_samples)
        
        # Create indices for shuffling (using fixed seed for reproducibility)
        rng = np.random.RandomState(args.seed)
        indices = rng.permutation(n_train_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Split training data using indices
        X_train = train_features[train_indices]
        X_val = train_features[val_indices]
        y_train = train_labels[train_indices]
        y_val = train_labels[val_indices]
        
        if rank == 0:
            logger.info(f"Training samples: {len(y_train)}, Validation samples: {len(y_val)}")
            logger.info(f"Cross-family test samples: {len(test_features)}")
        
        # Create datasets and dataloaders
        train_dataset = MultimodalDataset(X_train, y_train)
        val_dataset = MultimodalDataset(X_val, y_val)
        test_dataset = MultimodalDataset(test_features, test_labels)
        
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
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=2,
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
            # Evaluate on cross-family test data
            logger.info("\n" + "="*50)
            logger.info("CROSS-FAMILY EVALUATION")
            logger.info("="*50)
            test_results = evaluate_cross_family(model, test_loader, gpu, label_mappings)
            
            # Create models directory
            model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            os.makedirs(model_save_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_save_dir, "cross_family_hyperparameter_classifier.pt")
            if isinstance(model, DDP):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            
            # Save training history
            history_path = os.path.join(model_save_dir, "training_history_cross_family_classifier.json")
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            
            # Save test results
            test_results_path = os.path.join(model_save_dir, "cross_family_test_results.json")
            with open(test_results_path, "w") as f:
                json.dump(test_results, f, indent=2)
            
            # Save label mappings
            mappings_path = os.path.join(model_save_dir, "label_mappings.json")
            with open(mappings_path, "w") as f:
                json.dump(label_mappings, f, indent=2)
            
            logger.info(f"Model, training history, test results, and label mappings saved in {model_save_dir}")
    
    except Exception as e:
        logger.error(f"Error during cross-family attack training: {str(e)}")
        raise
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
