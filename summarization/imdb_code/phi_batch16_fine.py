#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train Phi shadow models using generated YAML configurations.
Specialized for Phi models with specific configurations:
- max_source_length = 1024
- fp16 = False  # A100: prefer bf16 over fp16
- bf16 = True   # A100: use bf16 for better performance and memory efficiency
- Decoder-only architecture for text generation
- Optimized for A100 GPUs with gradient checkpointing and group_by_length
"""

import os
import yaml
import json
import torch
import logging
import wandb
import multiprocessing
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import timedelta
import time
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AdamW,
    Adafactor,
    get_scheduler
)
from datasets import Dataset as HFDataset, load_from_disk
import numpy as np
import random
import argparse
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set cache directory for all model downloads and caching
CACHE_DIR = "/work/shovon/LLM/"
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'transformers')
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, 'huggingface')
os.environ['HF_DATASETS_CACHE'] = os.path.join(CACHE_DIR, 'datasets')

# Create cache directories if they don't exist
for cache_path in [os.environ['TRANSFORMERS_CACHE'], os.environ['HF_HOME'], os.environ['HF_DATASETS_CACHE']]:
    os.makedirs(cache_path, exist_ok=True)
    logger.info(f"Using cache directory: {cache_path}")

# Get CPU count for multiprocessing
CPU_COUNT = multiprocessing.cpu_count()
NUM_PROC = min(16, CPU_COUNT)  # Limit to 16 processes or CPU count, whichever is lower
logger.info(f"Using {NUM_PROC} processes for dataset processing")

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SummarizationDataset:
    """Dataset class for handling summarization data for Phi text generation."""
    
    def __init__(self, file_path: str, tokenizer, max_source_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.file_path = file_path
        
        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.texts = []
        self.summaries = []
        
        for item in data['summarization']:
            self.texts.append(item['real'])
            self.summaries.append(item['summarize'])

    def preprocess_function(self, examples: Dict) -> Dict:
        """Preprocess and tokenize input examples for Phi text generation."""
        # For Phi, we'll format the input as: "Text: {input_text} Summary: {summary}"
        # This allows the model to learn the summarization task
        inputs = []
        for text, summary in zip(examples["input_text"], examples["target_text"]):
            formatted_input = f"Text: {text} Summary: {summary}"
            inputs.append(formatted_input)
        
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # For language modeling, the labels are the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].clone()
        return model_inputs

    def create_dataset(self) -> HFDataset:
        """Create HuggingFace dataset from processed data."""
        # Create cache directory for preprocessed datasets
        cache_dir = os.path.join(CACHE_DIR, 'preprocessed_datasets')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate unique cache key based on tokenizer and dataset parameters
        tokenizer_name = self.tokenizer.name_or_path
        file_hash = hash(self.file_path)
        cache_key = f"phi_{tokenizer_name}_{file_hash}_{self.max_source_length}"
        cache_path = os.path.join(cache_dir, f"{cache_key}.hf")
        
        # Try to load from cache
        if os.path.exists(cache_path):
            logger.info(f"Loading preprocessed dataset from cache: {cache_path}")
            return load_from_disk(cache_path)
        
        # If not in cache, process and save
        logger.info("Processing dataset and saving to cache...")
        dataset = HFDataset.from_dict({
            "input_text": self.texts,
            "target_text": self.summaries
        })
        
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=NUM_PROC,
            remove_columns=["input_text", "target_text"],
            desc="Processing dataset",
        )
        
        # Save to cache
        logger.info(f"Saving processed dataset to cache: {cache_path}")
        processed_dataset.save_to_disk(cache_path)
        
        return processed_dataset

def get_phi_model_and_tokenizer(config: Dict[str, Any]):
    """Get Phi model and tokenizer based on configuration."""
    model_name = config['model']['name']
    
    # Validate that this is a Phi model
    if 'phi' not in model_name.lower():
        raise ValueError(f"Expected Phi model, but got: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=os.environ['TRANSFORMERS_CACHE'],
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=os.environ['TRANSFORMERS_CACHE'],
        trust_remote_code=True
    )
    
    # Set pad token for Phi models if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def get_optimizer(model, config: Dict[str, Any]):
    """Get optimizer based on configuration."""
    optimizer_name = config['training']['optimizer']
    learning_rate = config['training']['learning_rate']
    
    if optimizer_name == 'adamw':
        return AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'adafactor':
        return Adafactor(model.parameters(), lr=learning_rate, scale_parameter=True, relative_step=False)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def calculate_gradient_accumulation_steps(per_device_batch_size: int, target_effective_batch_size: int = 64) -> int:
    """Calculate gradient accumulation steps to achieve target effective batch size."""
    # Get number of GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # Calculate required gradient accumulation steps
    gradient_accumulation_steps = max(1, target_effective_batch_size // (per_device_batch_size * num_gpus))
    
    logger.info(f"Calculating gradient accumulation steps:")
    logger.info(f"- Target effective batch size: {target_effective_batch_size}")
    logger.info(f"- Per device batch size: {per_device_batch_size}")
    logger.info(f"- Number of GPUs: {num_gpus}")
    logger.info(f"- Calculated gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"- Actual effective batch size: {per_device_batch_size * gradient_accumulation_steps * num_gpus}")
    
    return gradient_accumulation_steps

def train_phi_model(config_path: str, model_index: int) -> None:
    """Train a Phi model using the specified configuration."""
    # Get local rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate that this is a Phi configuration
    if config['model']['family'] != 'Phi':
        raise ValueError(f"Expected Phi model configuration, but got: {config['model']['family']}")
    
    # Set up logging
    model_name = config['model']['name']
    output_dir = config['output']['output_dir']
    
    # Check if model already exists
    if os.path.exists(os.path.join(output_dir, "final_model")):
        logger.info(f"Model already exists at {output_dir}/final_model. Skipping training.")
        return
    
    logger.info(f"\nTraining Phi model: {model_name}")
    logger.info(f"Configuration: {config['training']}")
    
    # Initialize wandb only on main process
    if local_rank == 0:
        wandb.init(
            project="phi-shadow-model-training",
            name=f"phi_{model_index}_{config['model']['size']}_{config['training']['optimizer']}_lr{config['training']['learning_rate']}_bs{config['training']['batch_size']}",
            config=config
        )
    
    try:
        # Set random seed
        set_seed(42)
        
        # Get Phi model and tokenizer
        model, tokenizer = get_phi_model_and_tokenizer(config)
        
        # Load datasets with Phi-specific configurations
        train_dataset = SummarizationDataset(
            config['data']['train_file'],
            tokenizer,
            config['data']['max_source_length']  # Should be 1024 for Phi
        )
        test_dataset = SummarizationDataset(
            config['data']['test_file'],
            tokenizer,
            config['data']['max_source_length']  # Should be 1024 for Phi
        )
        
        # Convert to HuggingFace datasets with cache directory
        train_hf = train_dataset.create_dataset()
        test_hf = test_dataset.create_dataset()
        
        # Split training data
        train_val_datasets = train_hf.train_test_split(test_size=0.2, seed=42)
        
        # Calculate gradient accumulation steps for effective batch size of 64
        gradient_accumulation_steps = calculate_gradient_accumulation_steps(
            per_device_batch_size=config['training']['batch_size']
        )
        
        # Note: We override fp16/bf16 settings from config for A100 optimization
        # A100 GPUs perform better with bf16 and these specific optimizations
        logger.info("Using A100-optimized training settings: bf16=True, gradient_checkpointing=True, group_by_length=True")
        
        # Configure training arguments with Phi-specific settings and A100 optimizations
        training_args = TrainingArguments(
            output_dir=config['output']['output_dir'],
            num_train_epochs=int(config['training']['num_train_epochs']),
            per_device_train_batch_size=config['training']['batch_size'],
            per_device_eval_batch_size=config['training']['batch_size'],
            warmup_steps=config['training']['warmup_steps'],
            weight_decay=config['training']['weight_decay'],
            logging_dir=config['output']['logging_dir'],
            logging_steps=config['training']['logging_steps'],
            eval_steps=config['training']['eval_steps'],
            save_steps=config['training']['save_steps'],
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=False,  # A100: prefer bf16 over fp16
            bf16=True,   # A100: use bf16 for better performance and memory efficiency
            gradient_checkpointing=True,  # Reduce memory usage
            group_by_length=True,  # Group batches by length to reduce padding and memory usage
            ddp_find_unused_parameters=False,  # Optimize distributed training
            report_to="wandb" if local_rank == 0 else "none",
            learning_rate=config['training']['learning_rate'],
            lr_scheduler_type=config['training'].get('lr_scheduler_type', 'linear'),
            max_steps=-1,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_accumulation_steps=1,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            #optim="adamw_torch_fused",  # Use fused optimizer if available for better performance
            tf32=True  # Enable TF32 for speed on A100 GPUs
        )
        
        # Initialize data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Phi is not a masked language model
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_val_datasets["train"],
            eval_dataset=train_val_datasets["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Train model
        start_time = time.time()
        logger.info("Starting Phi training...")
        
        trainer.train()
        
        # Calculate and log training duration
        end_time = time.time()
        training_duration = timedelta(seconds=int(end_time - start_time))
        logger.info(f"Phi training completed in {training_duration}")
        
        # Save model
        trainer.save_model(os.path.join(config['output']['output_dir'], "final_model"))
        
        # Evaluate on test set
        logger.info("Evaluating Phi model on test set...")
        test_results = trainer.evaluate(test_hf)
        logger.info(f"Phi test results: {test_results}")
        
        # Log final metrics only on main process
        if local_rank == 0:
            wandb.log(test_results)
        
    except Exception as e:
        logger.error(f"Error training Phi model {model_name}: {str(e)}")
        raise
    finally:
        if local_rank == 0:
            wandb.finish()

def main():
    """Main function to train Phi shadow models."""
    import argparse
    import pandas as pd
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train Phi shadow models')
    parser.add_argument('--model_indices', type=str, nargs='+', 
                      help='Indices of Phi models to train. Can be individual indices (e.g., 1 2 3) or ranges (e.g., 0-9).')
    args = parser.parse_args()
    
    # Read config summary CSV
    config_summary_path = "./configs/config_summary.csv"
    if not os.path.exists(config_summary_path):
        raise FileNotFoundError(f"Config summary file not found at {config_summary_path}")
    
    # Read CSV file and filter for Phi models only
    config_df = pd.read_csv(config_summary_path)
    phi_df = config_df[config_df['model_family'] == 'Phi'].copy()
    
    if len(phi_df) == 0:
        logger.warning("No Phi models found in the configuration summary.")
        return
    
    logger.info(f"Found {len(phi_df)} Phi models in configuration")
    
    # If model indices are provided, filter the dataframe
    if args.model_indices:
        selected_indices = set()
        
        for idx_str in args.model_indices:
            if '-' in idx_str:
                try:
                    start, end = map(int, idx_str.split('-'))
                    selected_indices.update(range(start, end + 1))
                except ValueError:
                    logger.warning(f"Invalid range format: {idx_str}. Skipping...")
            else:
                try:
                    selected_indices.add(int(idx_str))
                except ValueError:
                    logger.warning(f"Invalid index: {idx_str}. Skipping...")
        
        # Filter for Phi models that match the selected indices
        phi_df = phi_df[phi_df['model_index'].isin(selected_indices)]
        
        invalid_indices = [idx for idx in selected_indices if idx not in config_df['model_index'].values]
        if invalid_indices:
            logger.warning(f"Invalid model indices: {invalid_indices}. These will be skipped.")
        
        logger.info(f"Processing {len(phi_df)} specified Phi models")
    
    logger.info(f"Training {len(phi_df)} Phi shadow models")
    
    # Train Phi models
    for _, row in phi_df.iterrows():
        config_path = row['config_path']
        model_index = row['model_index']
        model_name = row['model_name']
        logger.info(f"\nProcessing Phi model index {model_index}: {model_name}")
        logger.info(f"Config path: {config_path}")
        try:
            train_phi_model(config_path, model_index)
        except Exception as e:
            logger.error(f"Failed to train Phi model with config {config_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
