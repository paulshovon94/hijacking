#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to prepare IMDB dataset summaries using PEGASUS large model.
Processes the dataset and saves summaries in pseudo_data directory.
"""

import os
from dotenv import load_dotenv
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    set_seed,
    logging
)
from tqdm import tqdm
from huggingface_hub import login

# Load environment variables from .env file
load_dotenv()

# Constants
MAX_INPUT_LENGTH = 1024
MAX_OUTPUT_LENGTH = 128
BATCH_SIZE = 8  # Adjust based on your GPU memory
MODEL_NAME = "google/pegasus-large"
CACHE_DIR = "/work/shovon/LLM/"
OUTPUT_DIR = "../pseudo_data/imdb"  # Path relative to current file location
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

def setup():
    """Setup directories, model, tokenizer, and dataset."""
    # Login to Hugging Face Hub using token from environment variable
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        raise EnvironmentError(
            "HUGGINGFACE_TOKEN environment variable not found. "
            "Please set it with your Hugging Face access token. "
            "You can get your token from https://huggingface.co/settings/tokens"
        )
    
    login(token=hf_token)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(SEED)
    
    # Load model and tokenizer
    print("Loading PEGASUS model and tokenizer...")
    model = PegasusForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        device_map="auto" if torch.cuda.is_available() else None
    ).to(DEVICE)
    
    tokenizer = PegasusTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR
    )
    
    # Load IMDB dataset
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb", cache_dir=CACHE_DIR)
    
    return model, tokenizer, dataset

def process_batch(texts, model, tokenizer):
    """Process a batch of texts through the model."""
    # Tokenize inputs
    inputs = tokenizer(
        texts,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    ).to(DEVICE)
    
    # Generate summaries
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_OUTPUT_LENGTH,
            min_length=30,  # Ensure summaries aren't too short
            num_beams=4,    # Beam search for better quality
            length_penalty=2.0,  # Encourage longer summaries
            no_repeat_ngram_size=3,  # Avoid repetition
            early_stopping=True
        )
    
    # Decode summaries
    summaries = tokenizer.batch_decode(
        summary_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    
    return summaries

def process_dataset(dataset_split, model, tokenizer, output_file):
    """Process entire dataset split and save summaries."""
    print(f"Processing {len(dataset_split)} examples...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(0, len(dataset_split), BATCH_SIZE)):
            # Get batch of texts
            batch_texts = dataset_split[i:i + BATCH_SIZE]["text"]
            
            try:
                # Generate summaries for batch
                summaries = process_batch(batch_texts, model, tokenizer)
                
                # Write summaries to file
                for summary in summaries:
                    f.write(summary.strip() + "\n")
                    f.flush()
                    
            except Exception as e:
                print(f"Error processing batch starting at index {i}: {str(e)}")
                # Write empty lines for failed batch to maintain alignment
                for _ in range(len(batch_texts)):
                    f.write("\n")

def main():
    """Main function to process IMDB dataset."""
    # Setup
    model, tokenizer, dataset = setup()
    
    # Process train split
    train_output_file = os.path.join(OUTPUT_DIR, "train.pegasus")
    process_dataset(dataset["train"], model, tokenizer, train_output_file)
    
    # Process test split
    test_output_file = os.path.join(OUTPUT_DIR, "test.pegasus")
    process_dataset(dataset["test"], model, tokenizer, test_output_file)
    
    print("Processing complete!")
    print(f"Summaries saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 