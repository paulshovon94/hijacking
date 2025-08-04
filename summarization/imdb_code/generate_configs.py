#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to generate configuration files for shadow model training.
Creates YAML configuration files for all combinations of model architectures and hyperparameters.
Designed to be easily extensible for adding new models and hyperparameters.
"""

import os
import yaml
import csv
from itertools import product
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Base class for model configurations."""
    name: str
    type: str
    size: str
    family: str
    optimizers: List[str]
    learning_rates: List[float]
    batch_sizes: List[int]

@dataclass
class TrainingConfig:
    """Base class for training configurations."""
    optimizer: str
    learning_rate: float
    batch_size: int
    num_train_epochs: int = 3  # Fixed value
    warmup_steps: int = 0
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    generation_max_length: int = 128
    generation_num_beams: int = 4
    lr_scheduler_type: str = "constant"

class HyperparameterSpace:
    """Class to manage hyperparameter spaces and their combinations."""
    
    def __init__(self):
        self.spaces: Dict[str, List[Any]] = {}
    
    def add_space(self, name: str, values: List[Any]) -> None:
        """Add a new hyperparameter space."""
        self.spaces[name] = values
    
    def get_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters."""
        if not self.spaces:
            return []
        
        keys = list(self.spaces.keys())
        values = list(self.spaces.values())
        
        combinations = []
        for combination in product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations

class ModelRegistry:
    """Class to manage model configurations."""
    
    def __init__(self):
        self.models: Dict[str, List[ModelConfig]] = {}
        self._initialize_default_configs()
    
    def _initialize_default_configs(self) -> None:
        """Initialize default model configurations with their specific hyperparameters."""
        # BART models
        self.models['BART'] = [
            ModelConfig(
                name='facebook/bart-base',
                type='encoder-decoder',
                size='base',
                family='BART',
                optimizers=['adamw', 'sgd', 'adafactor'],
                learning_rates=[1e-5, 5e-5, 1e-4],
                batch_sizes=[4, 8, 16]
            ),
            ModelConfig(
                name='facebook/bart-large',
                type='encoder-decoder',
                size='large',
                family='BART',
                optimizers=['adamw', 'sgd', 'adafactor'],
                learning_rates=[1e-5, 5e-5, 1e-4],
                batch_sizes=[4, 8, 16]
            )
        ]
        
        # PEGASUS models
        self.models['Pegasus'] = [
            ModelConfig(
                name='google/pegasus-xsum',
                type='encoder-decoder',
                size='xsum',
                family='Pegasus',
                optimizers=['adamw', 'sgd', 'adafactor'],
                learning_rates=[1e-5, 5e-5, 1e-4],
                batch_sizes=[4, 8, 16]
            ),
            ModelConfig(
                name='google/pegasus-large',
                type='encoder-decoder',
                size='large',
                family='Pegasus',
                optimizers=['adamw', 'sgd', 'adafactor'],
                learning_rates=[1e-5, 5e-5, 1e-4],
                batch_sizes=[4, 8, 16]
            )
        ]

        # GPT-2 models
        self.models['GPT-2'] = [
            ModelConfig(
                name='gpt2',
                type='decoder-only',
                size='small',
                family='GPT-2',
                optimizers=['adamw', 'sgd', 'adafactor'],
                learning_rates=[1e-5, 5e-5, 1e-4],
                batch_sizes=[4, 8, 16]
            ),
            ModelConfig(
                name='gpt2-medium',
                type='decoder-only',
                size='medium',
                family='GPT-2',
                optimizers=['adamw', 'sgd', 'adafactor'],
                learning_rates=[1e-5, 5e-5, 1e-4],
                batch_sizes=[4, 8, 16]
            ),
            ModelConfig(
                name='gpt2-large',
                type='decoder-only',
                size='large',
                family='GPT-2',
                optimizers=['adamw', 'sgd', 'adafactor'],
                learning_rates=[1e-5, 5e-5, 1e-4],
                batch_sizes=[4, 8, 16]
            )
        ]

        # Mistral models
        self.models['Mistral'] = [
            ModelConfig(
                name='mistralai/Mistral-7B-v0.1',
                type='decoder-only',
                size='7B',
                family='Mistral',
                optimizers=['adamw', 'sgd', 'adafactor'],
                learning_rates=[1e-5, 5e-5, 1e-4],
                batch_sizes=[4, 8, 16]
            )
        ]

        # Qwen models
        self.models['Qwen'] = [
            ModelConfig(
                name='Qwen/Qwen1.5-0.5B',
                type='decoder-only',
                size='0.5B',
                family='Qwen',
                optimizers=['adamw', 'sgd', 'adafactor'],
                learning_rates=[1e-5, 5e-5, 1e-4],
                batch_sizes=[4, 8, 16]
            ),
            ModelConfig(
                name='Qwen/Qwen1.5-1.8B',
                type='decoder-only',
                size='1.8B',
                family='Qwen',
                optimizers=['adamw', 'sgd', 'adafactor'],
                learning_rates=[1e-5, 5e-5, 1e-4],
                batch_sizes=[4, 8, 16]
            ),
            ModelConfig(
                name='Qwen/Qwen1.5-7B',
                type='decoder-only',
                size='7B',
                family='Qwen',
                optimizers=['adamw', 'sgd', 'adafactor'],
                learning_rates=[1e-5, 5e-5, 1e-4],
                batch_sizes=[4, 8, 16]
            )
        ]

        # LLaMA models
        self.models['LLaMA'] = [
            ModelConfig(
                name='meta-llama/Llama-2-7b-hf',
                type='decoder-only',
                size='7B',
                family='LLaMA',
                optimizers=['adamw', 'sgd', 'adafactor'],
                learning_rates=[1e-5, 5e-5, 1e-4],
                batch_sizes=[4, 8, 16]
            ),
            ModelConfig(
                name='meta-llama/Llama-2-13b-hf',
                type='decoder-only',
                size='13B',
                family='LLaMA',
                optimizers=['adamw', 'sgd', 'adafactor'],
                learning_rates=[1e-5, 5e-5, 1e-4],
                batch_sizes=[4, 8, 16]
            )
        ]
    
    def get_all_models(self) -> List[ModelConfig]:
        """Get all registered models."""
        return [
            model
            for family_models in self.models.values()
            for model in family_models
        ]

class ConfigGenerator:
    """Class to generate configuration files."""
    
    def __init__(self, output_dir: str = './configs'):
        self.output_dir = output_dir
        self.model_registry = ModelRegistry()
    
    def create_config(self, model: ModelConfig, hp_combination: Dict[str, Any]) -> Dict[str, Any]:
        """Create a configuration dictionary for a specific model and hyperparameter combination."""
        # Set model-specific configurations
        max_source_length = 512 if model.family == "Pegasus" else 1024
        fp16 = False if model.family == "Pegasus" else True
        bf16 = True if model.family == "Pegasus" else False
        
        training_config = TrainingConfig(
            optimizer=hp_combination['optimizer'],
            learning_rate=hp_combination['learning_rate'],
            batch_size=hp_combination['batch_size'],
            fp16=fp16
        )
        
        # Create the full model name for the output directory
        full_model_name = f"{model.family.lower()}_{model.size}_{hp_combination['optimizer']}_lr{hp_combination['learning_rate']}_bs{hp_combination['batch_size']}"
        
        config = {
            'model': {
                'name': model.name,
                'type': model.type,
                'size': model.size,
                'family': model.family
            },
            'training': asdict(training_config),
            'data': {
                'train_file': '../transformed_data/imdb/train.json',
                'test_file': '../transformed_data/imdb/test.json',
                'max_source_length': max_source_length,
                'max_target_length': 128
            },
            'output': {
                'output_dir': f'./results/{model.family.lower()}/{model.size}/{full_model_name}',
                'logging_dir': f'./logs/{model.family.lower()}/{model.size}/{full_model_name}'
            }
        }
        
        # Add bf16 configuration for Pegasus models
        if model.family == "Pegasus":
            config['training']['bf16'] = bf16
        
        return config
    
    def get_model_hp_combinations(self, model: ModelConfig) -> List[Dict[str, Any]]:
        """Generate all hyperparameter combinations for a specific model."""
        combinations = []
        for optimizer in model.optimizers:
            for lr in model.learning_rates:
                for bs in model.batch_sizes:
                    combinations.append({
                        'optimizer': optimizer,
                        'learning_rate': lr,
                        'batch_size': bs
                    })
        return combinations
    
    def generate_configs(self) -> None:
        """Generate all configuration files."""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Using directory: {self.output_dir}")
        
        total_configs = 0
        model_counts = {}
        skipped_configs = 0
        
        # Create CSV file for config information
        csv_path = os.path.join(self.output_dir, 'config_summary.csv')
        csv_fields = [
            'model_index',
            'config_filename',
            'config_path',
            'model_family',
            'model_size',
            'model_name',
            'optimizer',
            'learning_rate',
            'batch_size',
            'num_train_epochs',
            'warmup_steps',
            'weight_decay',
            'gradient_accumulation_steps',
            'fp16',
            'bf16',
            'logging_steps',
            'eval_steps',
            'save_steps',
            'generation_max_length',
            'generation_num_beams',
            'lr_scheduler_type',
            'model_output_dir'
        ]
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
            writer.writeheader()
            
            # Generate configurations for each model
            model_index = 0  # Initialize model index counter
            for model in self.model_registry.get_all_models():
                # Track model counts
                if model.family not in model_counts:
                    model_counts[model.family] = 0
                model_counts[model.family] += 1
                
                # Create model family and size directories
                family_dir = os.path.join(self.output_dir, model.family.lower())
                size_dir = os.path.join(family_dir, model.size)
                
                # Create directories
                os.makedirs(size_dir, exist_ok=True)
                logger.info(f"Using directory for {model.family} {model.size}")
                
                # Get hyperparameter combinations for this model
                hp_combinations = self.get_model_hp_combinations(model)
                logger.info(f"Number of hyperparameter combinations for {model.family} {model.size}: {len(hp_combinations)}")
                
                for hp_combination in hp_combinations:
                    # Generate filename with model index
                    filename = f"{model_index}_{model.family.lower()}_{model.size}_{hp_combination['optimizer']}_lr{hp_combination['learning_rate']}_bs{hp_combination['batch_size']}.yaml"
                    filepath = os.path.join(size_dir, filename)
                    
                    # Create configuration (even if file exists, we need it for CSV)
                    config = self.create_config(model, hp_combination)
                    
                    # Update output directories to include model index
                    full_model_name = f"{model_index}_{model.family.lower()}_{model.size}_{hp_combination['optimizer']}_lr{hp_combination['learning_rate']}_bs{hp_combination['batch_size']}"
                    config['output']['output_dir'] = f'./results/{model.family.lower()}/{model.size}/{full_model_name}'
                    config['output']['logging_dir'] = f'./logs/{model.family.lower()}/{model.size}/{full_model_name}'
                    
                    # Write to CSV regardless of whether file exists
                    csv_row = {
                        'model_index': model_index,
                        'config_filename': filename,
                        'config_path': filepath,
                        'model_family': model.family,
                        'model_size': model.size,
                        'model_name': model.name,
                        'optimizer': hp_combination['optimizer'],
                        'learning_rate': hp_combination['learning_rate'],
                        'batch_size': hp_combination['batch_size'],
                        'num_train_epochs': config['training']['num_train_epochs'],
                        'warmup_steps': config['training']['warmup_steps'],
                        'weight_decay': config['training']['weight_decay'],
                        'gradient_accumulation_steps': config['training']['gradient_accumulation_steps'],
                        'fp16': config['training']['fp16'],
                        'bf16': config['training'].get('bf16', False),
                        'logging_steps': config['training']['logging_steps'],
                        'eval_steps': config['training']['eval_steps'],
                        'save_steps': config['training']['save_steps'],
                        'generation_max_length': config['training']['generation_max_length'],
                        'generation_num_beams': config['training']['generation_num_beams'],
                        'lr_scheduler_type': config['training']['lr_scheduler_type'],
                        'model_output_dir': config['output']['output_dir']
                    }
                    writer.writerow(csv_row)
                    model_index += 1  # Increment model index
                    
                    # Check if config file already exists
                    if os.path.exists(filepath):
                        logger.info(f"Config file already exists: {filepath}")
                        skipped_configs += 1
                        continue
                    
                    # Save configuration
                    with open(filepath, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    
                    total_configs += 1
        
        # Print detailed summary
        logger.info("\nConfiguration Generation Summary:")
        logger.info(f"Total models: {sum(model_counts.values())}")
        logger.info("Models per family:")
        for family, count in model_counts.items():
            logger.info(f"  - {family}: {count} models")
        logger.info(f"New configuration files created: {total_configs}")
        logger.info(f"Existing configuration files skipped: {skipped_configs}")
        logger.info(f"Files generated in: {self.output_dir}")
        logger.info(f"Config summary CSV created at: {csv_path}")

def main():
    """Main function to generate configuration files."""
    try:
        # Initialize and run config generator
        generator = ConfigGenerator()
        generator.generate_configs()
        
        # Print summary
        logger.info("\nConfiguration Generation Summary:")
        logger.info(f"Total model families: {len(generator.model_registry.models)}")
        logger.info(f"Total models: {len(generator.model_registry.get_all_models())}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 