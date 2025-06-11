"""
Script to train a multilabel classifier using the x1 features (combined embeddings).
The classifier predicts model configuration parameters based on the features.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import logging
from typing import List, Dict, Tuple
import json
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
CACHE_DIR = "/work/shovon/LLM/"  # Base directory for caching models and data

# Set up cache directories for different components
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'transformers')
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, 'huggingface')
os.environ['HF_DATASETS_CACHE'] = os.path.join(CACHE_DIR, 'datasets')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(CACHE_DIR, 'sentence-transformers')
os.environ['NLTK_DATA'] = os.path.join(CACHE_DIR, 'nltk_data')
os.environ['TORCH_HOME'] = os.path.join(CACHE_DIR, 'torch')  # PyTorch cache directory

# Create cache directories if they don't exist
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

class FeatureDataset(Dataset):
    """PyTorch Dataset for loading and processing features."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize the dataset.
        
        Args:
            features (np.ndarray): Feature matrix
            labels (np.ndarray): Label matrix
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

class MultilabelClassifier(nn.Module):
    """Neural network for multilabel classification."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.3):
        """
        Initialize the classifier.
        
        Args:
            input_dim (int): Input feature dimension
            hidden_dims (List[int]): List of hidden layer dimensions
            output_dim (int): Output dimension (number of labels)
            dropout_rate (float): Dropout rate for regularization
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers with residual connections
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # LayerNorm instead of BatchNorm
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Add output layer with separate heads for each category
        self.model = nn.Sequential(*layers)
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model(x)
        
        # Get predictions from each head
        family_pred = self.family_head(features)
        size_pred = self.size_head(features)
        optimizer_pred = self.optimizer_head(features)
        lr_pred = self.lr_head(features)
        bs_pred = self.bs_head(features)
        
        # Concatenate all predictions
        return torch.cat([family_pred, size_pred, optimizer_pred, lr_pred, bs_pred], dim=1)

def normalize_labels(labels: np.ndarray) -> np.ndarray:
    """
    Normalize labels to be between 0 and 1 for binary cross-entropy loss.
    
    Args:
        labels (np.ndarray): Raw label values
        
    Returns:
        np.ndarray: Normalized labels between 0 and 1
    """
    # Create a copy to avoid modifying the original
    normalized = labels.copy()
    
    # Normalize each column independently
    for i in range(labels.shape[1]):
        col = labels[:, i]
        min_val = col.min()
        max_val = col.max()
        if max_val > min_val:  # Avoid division by zero
            normalized[:, i] = (col - min_val) / (max_val - min_val)
        else:
            normalized[:, i] = 0.5  # If all values are the same, set to 0.5
    
    return normalized

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

def load_features_and_labels(data_dir: str, config_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load features and corresponding labels from the dataset.
    
    Args:
        data_dir (str): Directory containing the feature files
        config_path (str): Path to the config summary CSV file
        
    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: Features, labels, and label names
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
            # Check if this directory contains x1_batch_*.npy files
            dir_path = os.path.join(root, dir_name)
            if any(f.startswith('x1_batch_') and f.endswith('.npy') for f in os.listdir(dir_path)):
                model_dirs.append(dir_path)
    
    logger.info(f"Found {len(model_dirs)} model directories with features")
    
    features_list = []
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
            # Format: {index}_{model_family}_{model_size}_{optimizer}_lr{lr}_bs{batch_size}
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
            
            # Load all x1 features for this model
            x1_files = sorted([f for f in all_files 
                             if f.startswith('x1_batch_') and f.endswith('.npy')])
            
            if not x1_files:
                logger.warning(f"No x1 feature files found in {model_dir}")
                logger.info(f"Available files: {all_files}")
                skipped_dirs.append(model_dir)
                continue
            
            logger.info(f"Found {len(x1_files)} x1 feature files in {model_dir}: {x1_files}")
            
            # Create labels for this model
            labels, label_names = create_categorical_labels(model_config)
            
            for x1_file in x1_files:
                try:
                    file_path = os.path.join(dir_path, x1_file)
                    logger.info(f"Loading features from {file_path}")
                    features = np.load(file_path)
                    logger.info(f"Loaded features with shape {features.shape}")
                    
                    # Ensure features and labels have the same number of samples
                    num_samples = features.shape[0]
                    features_list.append(features)
                    labels_list.append(np.tile(labels, (num_samples, 1)))
                    
                    logger.info(f"Successfully loaded features from {x1_file}")
                    
                except Exception as e:
                    logger.error(f"Error loading features from {x1_file}: {str(e)}")
                    continue
            
            processed_dirs.append(model_dir)
                    
        except Exception as e:
            logger.error(f"Error processing directory {model_dir}: {str(e)}")
            skipped_dirs.append(model_dir)
            continue
    
    if not features_list:
        logger.error("No valid features found in any directory")
        logger.error(f"Processed directories: {processed_dirs}")
        logger.error(f"Skipped directories: {skipped_dirs}")
        raise ValueError("No valid features found in any directory")
    
    # Combine all features and labels
    X = np.vstack(features_list)
    y = np.vstack(labels_list)
    
    # Verify shapes match
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Feature and label shapes don't match: X={X.shape}, y={y.shape}")
    
    logger.info(f"\nSummary:")
    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features and {y.shape[1]} labels")
    logger.info(f"Processed {len(processed_dirs)} directories: {processed_dirs}")
    logger.info(f"Skipped {len(skipped_dirs)} directories: {skipped_dirs}")
    logger.info(f"Label names: {label_names}")
    
    return X, y, label_names

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                num_epochs: int,
                device: str,
                label_names: List[str]) -> Dict[str, List[float]]:
    """
    Train the model.
    
    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        num_epochs (int): Number of epochs to train
        device (str): Device to train on ('cuda' or 'cpu')
        label_names (List[str]): Names of the labels
        
    Returns:
        Dict[str, List[float]]: Training history
    """
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Initialize label weights for weighted loss
    label_weights = torch.ones(len(label_names), device=device)
    
    # Calculate class weights based on inverse frequency
    with torch.no_grad():
        all_labels = []
        for _, labels in train_loader:
            all_labels.append(labels)
        all_labels = torch.cat(all_labels, dim=0)
        
        for i in range(len(label_names)):
            pos_count = all_labels[:, i].sum()
            neg_count = len(all_labels) - pos_count
            label_weights[i] = (neg_count / pos_count) if pos_count > 0 else 1.0
    
    logger.info("\nLabel weights:")
    for i, name in enumerate(label_names):
        logger.info(f"{name:20s}: {label_weights[i]:.4f}")
    
    # Create weighted loss function
    weighted_criterion = nn.BCELoss(weight=label_weights)
    
    logger.info("\nStarting Training...")
    logger.info(f"Training on {device} for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            
            loss = weighted_criterion(outputs, labels)
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
                features, labels = features.to(device), labels.to(device)
                
                outputs = model(features)
                loss = weighted_criterion(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
                
                predictions = (outputs > 0.5).float()
                
                # Calculate metrics
                true_positives += (predictions * labels).sum(dim=0)
                false_positives += (predictions * (1 - labels)).sum(dim=0)
                false_negatives += ((1 - predictions) * labels).sum(dim=0)
        
        val_loss /= val_batches
        
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
        
        # Log results for every epoch
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val F1: {total_f1:.4f}")
        
        # Log per-label metrics
        logger.info("\nPer-label Metrics:")
        for i, name in enumerate(label_names):
            tp = true_positives[i].item()
            fp = false_positives[i].item()
            fn = false_negatives[i].item()
            f1 = per_label_f1[i].item()
            
            if (tp + fp + fn) > 0:  # Only show metrics for labels that have samples
                logger.info(f"{name:20s}: F1={f1:.4f} (TP={int(tp)}, FP={int(fp)}, FN={int(fn)})")
        
        # Check if this is the best model so far
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
        X, y, label_names = load_features_and_labels(DATA_DIR, CONFIG_PATH)
        
        if rank == 0:
            logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features and {y.shape[1]} labels")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if rank == 0:
            logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Create datasets and dataloaders
        train_dataset = FeatureDataset(X_train, y_train)
        val_dataset = FeatureDataset(X_val, y_val)
        
        train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
        val_sampler = DistributedSampler(val_dataset) if world_size > 1 else None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model with larger architecture
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        
        model = MultilabelClassifier(
            input_dim=input_dim,
            hidden_dims=[1024, 512, 256],  # Increased network capacity
            output_dim=output_dim,
            dropout_rate=0.4  # Increased dropout for better regularization
        ).to(gpu)
        
        if world_size > 1:
            model = DDP(model, device_ids=[gpu])
        
        # Initialize optimizer with weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01,  # L2 regularization
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=nn.BCELoss(),  # Base criterion, will be weighted in train_model
            optimizer=optimizer,
            num_epochs=100,  # Increased epochs
            device=gpu,
            label_names=label_names
        )
        
        if rank == 0:  # Only save on main process
            # Create models directory in current directory
            model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Unimodel")
            os.makedirs(model_save_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_save_dir, "multilabel_classifier_x1.pt")
            if isinstance(model, DDP):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            
            # Save training history
            history_path = os.path.join(model_save_dir, "training_history_multilabel_classifier_x1.json")
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