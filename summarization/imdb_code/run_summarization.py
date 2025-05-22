"""
Script for fine-tuning BART model for text summarization on IMDB dataset.
This implementation includes data preprocessing, model training, and evaluation.
"""

import json
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
import os
from typing import Dict, List
import logging
import wandb
import time
from datetime import timedelta

# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across different libraries.
    
    Args:
        seed (int): Random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(directory: str) -> str:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory (str): Path to the directory to create
        
    Returns:
        str: Path to the created/existing directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
    return directory

class SummarizationDataset:
    """
    Custom dataset class for handling summarization data.
    Handles data loading, preprocessing, and tokenization for BART model.
    """
    
    def __init__(self, file_path: str, tokenizer, max_source_length: int = 1024, max_target_length: int = 128):
        """
        Initialize the dataset with tokenizer and length constraints.
        
        Args:
            file_path (str): Path to the JSON file containing the dataset
            tokenizer: BART tokenizer instance
            max_source_length (int): Maximum length for source text
            max_target_length (int): Maximum length for target summary
        """
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # Load and preprocess the data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text and summaries
        self.texts = []
        self.summaries = []
        
        for item in data['summarization']:
            self.texts.append(item['real'])
            self.summaries.append(item['summarize'])
        
        # Take only 100% of the data for faster experimentation
        num_samples = len(self.texts)
        indices = list(range(num_samples))
        selected_indices = random.sample(indices, int(1 * num_samples))
        
        self.texts = [self.texts[i] for i in selected_indices]
        self.summaries = [self.summaries[i] for i in selected_indices]

    def preprocess_function(self, examples: Dict) -> Dict:
        """
        Preprocess and tokenize the input examples for the model.
        
        Args:
            examples (Dict): Dictionary containing input text and target text
            
        Returns:
            Dict: Tokenized inputs with labels
        """
        # Add prefix to the input for better generation
        inputs = ["summarize: " + doc for doc in examples["input_text"]]
        
        # Tokenize inputs with specified max length and padding
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
        )

        # Tokenize targets using text_target parameter
        labels = self.tokenizer(
            text_target=examples["target_text"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def create_dataset(self) -> HFDataset:
        """
        Create a HuggingFace dataset from the processed data.
        
        Returns:
            HFDataset: Processed dataset ready for training
        """
        # Create initial dataset with proper column names
        dataset = HFDataset.from_dict({
            "input_text": self.texts,
            "target_text": self.summaries
        })
        
        # Process the dataset to create model inputs
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=["input_text", "target_text"],
            desc="Processing dataset",
        )
        
        return processed_dataset

def main() -> None:
    """
    Main function to orchestrate the training process.
    Handles model initialization, training, and evaluation.
    """
    # Initialize distributed training if available
    if torch.cuda.is_available():
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    else:
        local_rank = 0
        world_size = 1
    
    # Initialize wandb only in the main process (rank 0)
    if local_rank == 0:
        wandb.init(project="bart-summarization-finetuning")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create necessary directories for outputs and logs
    output_dir = ensure_dir("./bart-summarization-results")
    final_model_dir = ensure_dir("./bart-summarization-final")
    log_dir = ensure_dir("./logs")
    
    # Initialize BART tokenizer and model
    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Load and preprocess training and test datasets
    train_dataset = SummarizationDataset(
        "../transformed_data/imdb/train.json",
        tokenizer
    )
    test_dataset = SummarizationDataset(
        "../transformed_data/imdb/test.json",
        tokenizer
    )
    
    # Convert to HuggingFace datasets
    train_hf = train_dataset.create_dataset()
    test_hf = test_dataset.create_dataset()
    
    # Split training data into train and validation sets (80/20 split)
    train_val_datasets = train_hf.train_test_split(test_size=0.2, seed=42)
    
    # Configure training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=log_dir,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        gradient_accumulation_steps=4,
        fp16=True,  # Enable mixed precision training
        report_to="wandb",
        run_name="full_3_epochs",
        overwrite_output_dir=True,
        generation_max_length=128,
        predict_with_generate=True,
        generation_num_beams=4,
        local_rank=local_rank,  # Add local rank for distributed training
        ddp_find_unused_parameters=False,  # Optimize for distributed training
    )
    
    # Initialize data collator for sequence-to-sequence tasks
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # Initialize the trainer with model, training arguments, and datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_val_datasets["train"],
        eval_dataset=train_val_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    try:
        # Start timing the training process
        start_time = time.time()
        logger.info("Starting training...")
        
        # Train the model
        trainer.train()
        
        # Calculate and log training duration
        end_time = time.time()
        training_duration = timedelta(seconds=int(end_time - start_time))
        logger.info(f"Training completed in {training_duration}")
        
        # Save the trained model
        logger.info(f"Saving final model to {final_model_dir}")
        trainer.save_model(final_model_dir)
        
        # Evaluate the model on the test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_hf)
        logger.info(f"Test results: {test_results}")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise
    finally:
        # Ensure wandb run is properly closed in the main process
        if local_rank == 0:
            wandb.finish()
        
        # Clean up distributed training
        if torch.cuda.is_available():
            dist.destroy_process_group()

if __name__ == "__main__":
    main() 