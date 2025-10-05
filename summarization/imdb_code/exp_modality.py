"""
Modality Analysis for Multimodal Hyperparameter Classification

This script trains the hyperparameter classifier incrementally with different feature combinations:
- x1 only
- x1 + x2
- x1 + x2 + x3
- x1 + x2 + x3 + x4
- x1 + x2 + x3 + x4 + x5
- x1 + x2 + x3 + x4 + x5 + x6
- x1 + x2 + x3 + x4 + x5 + x6 + x7

This allows us to analyze how each modality contributes to the overall performance
in predicting hyperparameters from language model outputs.
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
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
except ImportError:
    logger.warning("Seaborn not available, using matplotlib defaults")

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

class ModalityDataset(Dataset):
    """Dataset for handling different modality combinations."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, modality_indices: List[int]):
        """
        Initialize the dataset.
        
        Args:
            features (np.ndarray): Feature matrix of shape (n_samples, total_features)
            labels (np.ndarray): Label matrix of shape (n_samples, n_labels)
            modality_indices (List[int]): List of modality indices to include (0-6 for x1-x7)
        """
        self.labels = torch.FloatTensor(labels)
        
        # Extract only the specified modalities
        feature_size_per_modality = 330  # Based on the original code
        start_idx = 0
        selected_features = []
        
        for modality_idx in modality_indices:
            start = modality_idx * feature_size_per_modality
            end = start + feature_size_per_modality
            selected_features.append(features[:, start:end])
        
        # Concatenate selected modalities
        self.features = torch.FloatTensor(np.concatenate(selected_features, axis=1))
        
        # Verify shapes match
        assert len(self.features) == len(self.labels), f"Features: {len(self.features)}, Labels: {len(self.labels)}"
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

class ModalityHyperparameterClassifier(nn.Module):
    """Neural network for modality-specific hyperparameter classification."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes_per_head: Dict[str, int], dropout_rate: float = 0.3):
        """
        Initialize the classifier.
        
        Args:
            input_dim (int): Input dimension (varies based on number of modalities)
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
                    
                    # Pad or truncate to expected size
                    expected_size = 330
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

def train_modality_model(model: nn.Module,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        optimizer: optim.Optimizer,
                        scheduler: Optional[optim.lr_scheduler._LRScheduler],
                        num_epochs: int,
                        device: str,
                        label_mappings: Dict[str, List[str]],
                        modality_name: str) -> Dict[str, float]:
    """
    Train the model for a specific modality combination.
    
    Returns:
        Dict[str, float]: Final performance metrics
    """
    # Calculate label indices for each head
    label_indices = {}
    start_idx = 0
    for name, mapping in label_mappings.items():
        end_idx = start_idx + len(mapping)
        label_indices[name] = (start_idx, end_idx)
        start_idx = end_idx
    
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
    
    logger.info(f"\nStarting Training for {modality_name}...")
    logger.info(f"Training on {device} for {num_epochs} epochs")
    
    best_val_loss = float('inf')
    best_metrics = {}
    
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
        
        # Log results every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info(f"\nEpoch {epoch+1}/{num_epochs} - {modality_name}")
            logger.info(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            logger.info(f"Overall Val Accuracy: {overall_accuracy:.4f} - Overall Val F1: {overall_f1:.4f}")
        
        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = {
                'val_loss': val_loss,
                'val_accuracy': overall_accuracy,
                'val_f1': overall_f1,
                'per_head_accuracy': per_head_accuracy.copy(),
                'per_head_f1': per_head_f1.copy()
            }
    
    logger.info(f"\nTraining completed for {modality_name}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best validation accuracy: {best_metrics['val_accuracy']:.4f}")
    logger.info(f"Best validation F1: {best_metrics['val_f1']:.4f}")
    
    return best_metrics

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

def create_performance_plot(results: Dict[str, Dict[str, float]], save_path: str):
    """Create visualization of performance vs number of modalities."""
    modalities = list(results.keys())
    num_modalities = [int(m.split('+')[-1].replace('x', '')) for m in modalities]
    
    # Extract metrics
    accuracies = [results[m]['val_accuracy'] for m in modalities]
    f1_scores = [results[m]['val_f1'] for m in modalities]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(num_modalities, accuracies, 'o-', linewidth=2, markersize=8, label='Accuracy')
    plt.xlabel('Number of Modalities')
    plt.ylabel('Validation Accuracy')
    plt.title('Performance vs Number of Modalities')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot F1 score
    plt.subplot(2, 1, 2)
    plt.plot(num_modalities, f1_scores, 'o-', linewidth=2, markersize=8, color='orange', label='F1 Score')
    plt.xlabel('Number of Modalities')
    plt.ylabel('Validation F1 Score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance plot saved to {save_path}")

def main():
    """Main function to train the classifier with different modality combinations."""
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic algorithms (may be slower)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs per modality')
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
        logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
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
        
        # Define modality combinations
        modality_combinations = [
            ([0], "x1"),
            ([0, 1], "x1+x2"),
            ([0, 1, 2], "x1+x2+x3"),
            ([0, 1, 2, 3], "x1+x2+x3+x4"),
            ([0, 1, 2, 3, 4], "x1+x2+x3+x4+x5"),
            ([0, 1, 2, 3, 4, 5], "x1+x2+x3+x4+x5+x6"),
            ([0, 1, 2, 3, 4, 5, 6], "x1+x2+x3+x4+x5+x6+x7")
        ]
        
        # Store results for each modality combination
        modality_results = {}
        
        # Train model for each modality combination
        for modality_indices, modality_name in modality_combinations:
            if rank == 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"Training with {modality_name} ({len(modality_indices)} modalities)")
                logger.info(f"{'='*60}")
            
            # Create datasets for this modality combination
            train_dataset = ModalityDataset(X_train, y_train, modality_indices)
            val_dataset = ModalityDataset(X_val, y_val, modality_indices)
            
            train_sampler = DistributedSampler(train_dataset, seed=args.seed) if world_size > 1 else None
            val_sampler = DistributedSampler(val_dataset, seed=args.seed) if world_size > 1 else None
            
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
            
            # Calculate input dimension for this modality combination
            input_dim = len(modality_indices) * 330  # 330 features per modality
            
            # Initialize model
            hidden_dims = [512, 256, 128] if input_dim >= 512 else [256, 128, 64]
            num_classes_per_head = {
                'model_family': len(label_mappings['model_family']),
                'model_size': len(label_mappings['model_size']),
                'optimizer': len(label_mappings['optimizer']),
                'learning_rate': len(label_mappings['learning_rate']),
                'batch_size': len(label_mappings['batch_size'])
            }
            
            model = ModalityHyperparameterClassifier(
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
                verbose=True,
                min_lr=1e-7
            )
            
            # Train model
            best_metrics = train_modality_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=args.epochs,
                device=gpu,
                label_mappings=label_mappings,
                modality_name=modality_name
            )
            
            # Store results
            modality_results[modality_name] = best_metrics
            
            if rank == 0:
                logger.info(f"\nResults for {modality_name}:")
                logger.info(f"  Validation Accuracy: {best_metrics['val_accuracy']:.4f}")
                logger.info(f"  Validation F1: {best_metrics['val_f1']:.4f}")
                logger.info(f"  Validation Loss: {best_metrics['val_loss']:.4f}")
                
                # Log per-head results
                logger.info(f"\nPer-head Results for {modality_name}:")
                for head_name in label_mappings.keys():
                    if head_name in best_metrics['per_head_accuracy']:
                        acc = best_metrics['per_head_accuracy'][head_name]
                        f1 = best_metrics['per_head_f1'][head_name]
                        logger.info(f"  {head_name:15s}: Acc={acc:.4f}, F1={f1:.4f}")
        
        if rank == 0:  # Only save on main process
            # Create results directory
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modality_results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Save results
            results_path = os.path.join(results_dir, "modality_analysis_results.json")
            with open(results_path, "w") as f:
                json.dump(modality_results, f, indent=2)
            
            # Create performance plot
            plot_path = os.path.join(results_dir, "modality_performance.png")
            create_performance_plot(modality_results, plot_path)
            
            # Print summary table
            logger.info(f"\n{'='*80}")
            logger.info("MODALITY ANALYSIS SUMMARY")
            logger.info(f"{'='*80}")
            logger.info(f"{'Modality':<20} {'Accuracy':<10} {'F1 Score':<10} {'Loss':<10}")
            logger.info(f"{'-'*80}")
            
            for modality_name, metrics in modality_results.items():
                logger.info(f"{modality_name:<20} {metrics['val_accuracy']:<10.4f} {metrics['val_f1']:<10.4f} {metrics['val_loss']:<10.4f}")
            
            logger.info(f"\nResults saved to {results_dir}")
    
    except Exception as e:
        logger.error(f"Error during modality analysis: {str(e)}")
        raise
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
