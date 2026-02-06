#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train Pegasus shadow models using generated YAML configurations.
Specialized for Pegasus models with specific configurations:
- max_source_length = 512
- fp16 = False
- bf16 = True
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
    PegasusTokenizer, PegasusForConditionalGeneration,
    Trainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
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
    """Dataset class for handling summarization data."""
    
    def __init__(self, file_path: str, tokenizer, max_source_length: int = 512, max_target_length: int = 128):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
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
        """Preprocess and tokenize input examples."""
        inputs = ["summarize: " + doc for doc in examples["input_text"]]
        
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
        )

        labels = self.tokenizer(
            text_target=examples["target_text"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def create_dataset(self) -> HFDataset:
        """Create HuggingFace dataset from processed data."""
        # Create cache directory for preprocessed datasets
        cache_dir = os.path.join(CACHE_DIR, 'preprocessed_datasets')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate unique cache key based on tokenizer and dataset parameters
        tokenizer_name = self.tokenizer.name_or_path
        file_hash = hash(self.file_path)
        cache_key = f"{tokenizer_name}_{file_hash}_{self.max_source_length}_{self.max_target_length}"
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

def get_pegasus_model_and_tokenizer(config: Dict[str, Any]):
    """Get Pegasus model and tokenizer based on configuration."""
    model_name = config['model']['name']
    
    # Validate that this is a Pegasus model
    if 'pegasus' not in model_name.lower():
        raise ValueError(f"Expected Pegasus model, but got: {model_name}")
    
    tokenizer = PegasusTokenizer.from_pretrained(
        model_name,
        cache_dir=os.environ['TRANSFORMERS_CACHE']
    )
    model = PegasusForConditionalGeneration.from_pretrained(
        model_name,
        cache_dir=os.environ['TRANSFORMERS_CACHE']
    )
    
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

def train_pegasus_model(config_path: str, model_index: int) -> None:
    """Train a Pegasus model using the specified configuration."""
    # Get local rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate that this is a Pegasus configuration
    if config['model']['family'] != 'Pegasus':
        raise ValueError(f"Expected Pegasus model configuration, but got: {config['model']['family']}")
    
    # Set up logging
    model_name = config['model']['name']
    output_dir = config['output']['output_dir']
    
    # Check if model already exists
    if os.path.exists(os.path.join(output_dir, "final_model")):
        logger.info(f"Model already exists at {output_dir}/final_model. Skipping training.")
        return
    
    logger.info(f"\nTraining Pegasus model: {model_name}")
    logger.info(f"Configuration: {config['training']}")
    
    # Initialize wandb only on main process
    if local_rank == 0:
        wandb.init(
            project="pegasus-shadow-model-training",
            name=f"pegasus_{model_index}_{config['model']['size']}_{config['training']['optimizer']}_lr{config['training']['learning_rate']}_bs{config['training']['batch_size']}",
            config=config
        )
    
    try:
        # Set random seed
        set_seed(42)
        
        # Get Pegasus model and tokenizer
        model, tokenizer = get_pegasus_model_and_tokenizer(config)
        
        # Load datasets with Pegasus-specific configurations
        train_dataset = SummarizationDataset(
            config['data']['train_file'],
            tokenizer,
            config['data']['max_source_length'],  # Should be 512 for Pegasus
            config['data']['max_target_length']
        )
        test_dataset = SummarizationDataset(
            config['data']['test_file'],
            tokenizer,
            config['data']['max_source_length'],  # Should be 512 for Pegasus
            config['data']['max_target_length']
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
        
        # Configure training arguments with Pegasus-specific settings
        training_args = Seq2SeqTrainingArguments(
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
            fp16=config['training']['fp16'],  # Should be False for Pegasus
            bf16=config['training'].get('bf16', True),  # Should be True for Pegasus
            report_to="wandb" if local_rank == 0 else "none",
            generation_max_length=config['training']['generation_max_length'],
            predict_with_generate=True,
            generation_num_beams=config['training']['generation_num_beams'],
            learning_rate=config['training']['learning_rate'],
            lr_scheduler_type=config['training'].get('lr_scheduler_type', 'linear'),
            max_steps=-1,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_accumulation_steps=1
        )
        
        # Initialize data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            return_tensors="pt"
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
        logger.info("Starting Pegasus training...")
        
        trainer.train()
        
        # Calculate and log training duration
        end_time = time.time()
        training_duration = timedelta(seconds=int(end_time - start_time))
        logger.info(f"Pegasus training completed in {training_duration}")
        
        # Save model
        trainer.save_model(os.path.join(config['output']['output_dir'], "final_model"))
        
        # Evaluate on test set
        logger.info("Evaluating Pegasus model on test set...")
        test_results = trainer.evaluate(test_hf)
        logger.info(f"Pegasus test results: {test_results}")
        
        # Log final metrics only on main process
        if local_rank == 0:
            wandb.log(test_results)
        
    except Exception as e:
        logger.error(f"Error training Pegasus model {model_name}: {str(e)}")
        raise
    finally:
        if local_rank == 0:
            wandb.finish()

def main():
    """Main function to train Pegasus shadow models."""
    import argparse
    import pandas as pd
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train Pegasus shadow models')
    parser.add_argument('--model_indices', type=str, nargs='+', 
                      help='Indices of Pegasus models to train. Can be individual indices (e.g., 1 2 3) or ranges (e.g., 0-9).')
    args = parser.parse_args()
    
    # Read config summary CSV
    config_summary_path = "./configs/config_summary.csv"
    if not os.path.exists(config_summary_path):
        raise FileNotFoundError(f"Config summary file not found at {config_summary_path}")
    
    # Read CSV file and filter for Pegasus models only
    config_df = pd.read_csv(config_summary_path)
    pegasus_df = config_df[config_df['model_family'] == 'Pegasus'].copy()
    
    if len(pegasus_df) == 0:
        logger.warning("No Pegasus models found in the configuration summary.")
        return
    
    logger.info(f"Found {len(pegasus_df)} Pegasus models in configuration")
    
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
        
        # Filter for Pegasus models that match the selected indices
        pegasus_df = pegasus_df[pegasus_df['model_index'].isin(selected_indices)]
        
        invalid_indices = [idx for idx in selected_indices if idx not in config_df['model_index'].values]
        if invalid_indices:
            logger.warning(f"Invalid model indices: {invalid_indices}. These will be skipped.")
        
        logger.info(f"Processing {len(pegasus_df)} specified Pegasus models")
    
    logger.info(f"Training {len(pegasus_df)} Pegasus shadow models")
    
    # Train Pegasus models
    for _, row in pegasus_df.iterrows():
        config_path = row['config_path']
        model_index = row['model_index']
        model_name = row['model_name']
        logger.info(f"\nProcessing Pegasus model index {model_index}: {model_name}")
        logger.info(f"Config path: {config_path}")
        try:
            train_pegasus_model(config_path, model_index)
        except Exception as e:
            logger.error(f"Failed to train Pegasus model with config {config_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 