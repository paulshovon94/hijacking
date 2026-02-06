#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train shadow models using LoRA (Low-Rank Adaptation) fine-tuning.
Trains multiple models with different hyperparameter combinations using LoRA.
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
    BartTokenizer, BartForConditionalGeneration,
    PegasusTokenizer, PegasusForConditionalGeneration,
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForCausalLM,
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
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
from peft.tuners.lora import LoraLayer
import bitsandbytes as bnb

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
    
    def __init__(self, file_path: str, tokenizer, max_source_length: int = 1024, max_target_length: int = 128):
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

def get_model_and_tokenizer(config: Dict[str, Any]):
    """Get appropriate model and tokenizer based on configuration."""
    model_name = config['model']['name']
    model_type = config['model']['type']
    
    # Determine precision settings
    use_bf16 = config['training'].get('bf16', False)
    use_fp16 = config['training'].get('fp16', False)
    
    # Set torch dtype based on availability and configuration
    if use_bf16 and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
        logger.info("Using bfloat16 precision")
    elif use_fp16 and torch.cuda.is_available():
        torch_dtype = torch.float16
        logger.info("Using float16 precision")
    else:
        torch_dtype = torch.float32
        logger.info("Using float32 precision")
    
    if model_type == 'encoder-decoder':
        if 'bart' in model_name.lower():
            tokenizer = BartTokenizer.from_pretrained(
                model_name,
                cache_dir=os.environ['TRANSFORMERS_CACHE']
            )
            model = BartForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=os.environ['TRANSFORMERS_CACHE'],
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None
            )
        elif 'pegasus' in model_name.lower():
            tokenizer = PegasusTokenizer.from_pretrained(
                model_name,
                cache_dir=os.environ['TRANSFORMERS_CACHE']
            )
            model = PegasusForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=os.environ['TRANSFORMERS_CACHE'],
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None
            )
        else:
            raise ValueError(f"Unsupported encoder-decoder model: {model_name}")
    else:  # decoder-only
        if 'gpt2' in model_name.lower():
            tokenizer = GPT2Tokenizer.from_pretrained(
                model_name,
                cache_dir=os.environ['TRANSFORMERS_CACHE']
            )
            model = GPT2LMHeadModel.from_pretrained(
                model_name,
                cache_dir=os.environ['TRANSFORMERS_CACHE'],
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=os.environ['TRANSFORMERS_CACHE']
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=os.environ['TRANSFORMERS_CACHE'],
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None
            )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def create_lora_config(config: Dict[str, Any]) -> LoraConfig:
    """Create LoRA configuration based on model type and training config."""
    model_type = config['model']['type']
    
    # Default LoRA parameters
    lora_config = {
        "r": config['lora'].get('r', 16),
        "lora_alpha": config['lora'].get('lora_alpha', 32),
        "target_modules": config['lora'].get('target_modules', None),
        "lora_dropout": config['lora'].get('lora_dropout', 0.1),
        "bias": config['lora'].get('bias', "none"),
        "task_type": TaskType.CAUSAL_LM if model_type == 'decoder-only' else TaskType.SEQ_2_SEQ_LM,
    }
    
    # Set target modules based on model type if not specified
    if lora_config["target_modules"] is None:
        if 'bart' in config['model']['name'].lower():
            lora_config["target_modules"] = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        elif 'pegasus' in config['model']['name'].lower():
            lora_config["target_modules"] = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        elif 'gpt2' in config['model']['name'].lower():
            lora_config["target_modules"] = ["c_attn", "c_proj", "c_fc"]
        else:
            # Generic target modules for other models
            lora_config["target_modules"] = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
    
    return LoraConfig(**lora_config)

def apply_lora_to_model(model, lora_config: LoraConfig, config: Dict[str, Any]):
    """Apply LoRA to the model."""
    # Prepare model for k-bit training if using quantization
    if config['training'].get('use_4bit', False):
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

def get_optimizer(model, config: Dict[str, Any]):
    """Get optimizer based on configuration."""
    optimizer_name = config['training']['optimizer']
    learning_rate = config['training']['learning_rate']
    
    # For LoRA, we typically use a higher learning rate
    lora_learning_rate = config['lora'].get('learning_rate', learning_rate * 2)
    
    if optimizer_name == 'adamw':
        return AdamW(model.parameters(), lr=lora_learning_rate)
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lora_learning_rate)
    elif optimizer_name == 'adafactor':
        return Adafactor(model.parameters(), lr=lora_learning_rate, scale_parameter=True, relative_step=False)
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

def train_model(config_path: str, model_index: int) -> None:
    """Train a model using LoRA fine-tuning with the specified configuration."""
    # Get local rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add default LoRA configuration if not present
    if 'lora' not in config:
        config['lora'] = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'bias': 'none',
            'learning_rate': config['training']['learning_rate'] * 2
        }
    
    # Set up logging
    model_name = config['model']['name']
    output_dir = config['output']['output_dir']
    
    # Check if model already exists
    if os.path.exists(os.path.join(output_dir, "final_model")):
        logger.info(f"Model already exists at {output_dir}/final_model. Skipping training.")
        return
    
    logger.info(f"\nTraining model with LoRA: {model_name}")
    logger.info(f"Configuration: {config['training']}")
    logger.info(f"LoRA Configuration: {config['lora']}")
    
    # Initialize wandb only on main process
    if local_rank == 0:
        wandb.init(
            project="shadow-model-training-lora",
            name=f"lora_model_{model_index}_{config['model']['family']}_{config['model']['size']}_{config['training']['optimizer']}_lr{config['lora']['learning_rate']}_bs{config['training']['batch_size']}",
            config=config
        )
    
    try:
        # Set random seed
        set_seed(42)
        
        # Get model and tokenizer
        model, tokenizer = get_model_and_tokenizer(config)
        
        # Create LoRA configuration
        lora_config = create_lora_config(config)
        
        # Apply LoRA to model
        model = apply_lora_to_model(model, lora_config, config)
        
        # Load datasets with cache directory
        train_dataset = SummarizationDataset(
            config['data']['train_file'],
            tokenizer,
            config['data']['max_source_length'],
            config['data']['max_target_length']
        )
        test_dataset = SummarizationDataset(
            config['data']['test_file'],
            tokenizer,
            config['data']['max_source_length'],
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
        
        # Determine precision settings for training
        use_bf16 = config['training'].get('bf16', False)
        use_fp16 = config['training'].get('fp16', False)
        
        # Set precision flags based on availability and configuration
        if use_bf16 and torch.cuda.is_bf16_supported():
            bf16 = True
            fp16 = False
            logger.info("Training with bfloat16 precision")
        elif use_fp16 and torch.cuda.is_available():
            bf16 = False
            fp16 = True
            logger.info("Training with float16 precision")
        else:
            bf16 = False
            fp16 = False
            logger.info("Training with float32 precision")
        
        # Configure training arguments
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
            bf16=bf16,
            fp16=fp16,
            report_to="wandb" if local_rank == 0 else "none",
            generation_max_length=config['training']['generation_max_length'],
            predict_with_generate=True,
            generation_num_beams=config['training']['generation_num_beams'],
            learning_rate=config['lora']['learning_rate'],  # Use LoRA learning rate
            lr_scheduler_type=config['training'].get('lr_scheduler_type', 'linear'),
            max_steps=-1,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_accumulation_steps=1,
            # LoRA-specific settings
            remove_unused_columns=False,
            dataloader_pin_memory=False,
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
        logger.info("Starting LoRA training...")
        
        trainer.train()
        
        # Calculate and log training duration
        end_time = time.time()
        training_duration = timedelta(seconds=int(end_time - start_time))
        logger.info(f"LoRA training completed in {training_duration}")
        
        # Save model (this will save the LoRA adapters)
        trainer.save_model(os.path.join(config['output']['output_dir'], "final_model"))
        
        # Save the base model and LoRA adapters separately for easier loading
        model.save_pretrained(os.path.join(config['output']['output_dir'], "lora_adapters"))
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_hf)
        logger.info(f"Test results: {test_results}")
        
        # Log final metrics only on main process
        if local_rank == 0:
            wandb.log(test_results)
        
    except Exception as e:
        logger.error(f"Error training LoRA model {model_name}: {str(e)}")
        raise
    finally:
        if local_rank == 0:
            wandb.finish()

def main():
    """Main function to train shadow models with LoRA."""
    import argparse
    import pandas as pd
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train shadow models with LoRA')
    parser.add_argument('--model_indices', type=str, nargs='+', 
                      help='Indices of models to train. Can be individual indices (e.g., 1 2 3) or ranges (e.g., 0-9).')
    args = parser.parse_args()
    
    # Read config summary CSV
    config_summary_path = "./configs/config_summary.csv"
    if not os.path.exists(config_summary_path):
        raise FileNotFoundError(f"Config summary file not found at {config_summary_path}")
    
    # Read CSV file
    config_df = pd.read_csv(config_summary_path)
    
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
        
        invalid_indices = [idx for idx in selected_indices if idx not in config_df['model_index'].values]
        if invalid_indices:
            logger.warning(f"Invalid model indices: {invalid_indices}. These will be skipped.")
        
        config_df = config_df[config_df['model_index'].isin(selected_indices)]
        logger.info(f"Processing {len(config_df)} specified models")
    
    logger.info(f"Training {len(config_df)} shadow models with LoRA")
    
    # Train models
    for _, row in config_df.iterrows():
        config_path = row['config_path']
        model_index = row['model_index']
        logger.info(f"\nProcessing model index {model_index}: {config_path}")
        try:
            train_model(config_path, model_index)
        except Exception as e:
            logger.error(f"Failed to train LoRA model with config {config_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
