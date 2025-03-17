import json
import random
import numpy as np
import torch
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
    return directory

class SummarizationDataset:
    def __init__(self, file_path: str, tokenizer, max_source_length: int = 1024, max_target_length: int = 128):
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
        
        # Take only 10% of the data
        num_samples = len(self.texts)
        indices = list(range(num_samples))
        selected_indices = random.sample(indices, int(0.1 * num_samples))
        
        self.texts = [self.texts[i] for i in selected_indices]
        self.summaries = [self.summaries[i] for i in selected_indices]

    def preprocess_function(self, examples):
        # Add prefix to the input for better generation
        inputs = ["summarize: " + doc for doc in examples["input_text"]]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
        )

        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer( 
                examples["target_text"],
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def create_dataset(self):
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

def main():
    # Initialize wandb for experiment tracking
    wandb.init(project="bart-summarization-finetuning")
    
    # Set random seed
    set_seed(42)
    
    # Create output directories
    output_dir = ensure_dir("./bart-summarization-results")
    final_model_dir = ensure_dir("./bart-summarization-final")
    log_dir = ensure_dir("./logs")
    
    # Initialize tokenizer and model
    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    # Load and preprocess datasets
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
    
    # Split train into train and validation (80/20)
    train_val_datasets = train_hf.train_test_split(test_size=0.2, seed=42)
    
    # Training arguments using Seq2SeqTrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=log_dir,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        gradient_accumulation_steps=4,
        fp16=True,
        report_to="wandb",
        overwrite_output_dir=True,
        generation_max_length=128,
        predict_with_generate=True,
        generation_num_beams=4,
    )
    
    # Data collator
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
    
    try:
        # Train the model
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        logger.info(f"Saving final model to {final_model_dir}")
        trainer.save_model(final_model_dir)
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_hf)
        logger.info(f"Test results: {test_results}")
        
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise
    finally:
        # Close wandb run
        wandb.finish()

if __name__ == "__main__":
    main() 