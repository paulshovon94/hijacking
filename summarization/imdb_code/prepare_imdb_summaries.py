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
import pandas as pd
from datasets import load_dataset
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    set_seed,
    logging
)
from tqdm import tqdm
from huggingface_hub import login
import gc
import re
from collections import Counter
import langdetect
import string

# Load environment variables from .env file
load_dotenv()

# Constants
MAX_INPUT_LENGTH = 1024
MAX_OUTPUT_LENGTH = 128
BATCH_SIZE = 4  # Reduced batch size for better memory management
MODEL_NAME = "google/pegasus-large"
CACHE_DIR = "/work/shovon/LLM/"
OUTPUT_DIR = "../pseudo_data/imdb"  # Path relative to current file location
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
DATA_PERCENTAGE = 1  # Process 10% of the data

# Filtering constants
MIN_CHARS = 20
MAX_CHARS = 600
MAX_WORD_FREQUENCY = 0.5
MIN_WORDS = 5
MAX_PUNCTUATION_RATIO = 0.2

def is_valid_summary(summary):
    """
    Apply multiple filters to check if a summary is valid.
    Returns: (bool, str) - (is_valid, reason_if_invalid)
    """
    if not summary or not summary.strip():
        return False, "empty"
    
    # Length filtering
    if len(summary) < MIN_CHARS:
        return False, "too_short"
    if len(summary) > MAX_CHARS:
        return False, "too_long"
        
    # Basic text cleaning
    text = summary.lower().strip()
    words = text.split()
    
    if len(words) < MIN_WORDS:
        return False, "too_few_words"
    
    # Repetition filtering
    word_counts = Counter(words)
    most_common_count = word_counts.most_common(1)[0][1]
    if most_common_count / len(words) > MAX_WORD_FREQUENCY:
        return False, "repetitive"
    
    # Language filtering
    try:
        if langdetect.detect(text) != 'en':
            return False, "non_english"
    except:
        return False, "language_detection_failed"
    
    # Garbage filtering
    # Check for excessive punctuation
    punct_count = sum(1 for char in text if char in string.punctuation)
    if punct_count / len(text) > MAX_PUNCTUATION_RATIO:
        return False, "excessive_punctuation"
    
    # Check for broken words and weird characters
    if re.search(r'[^a-zA-Z0-9\s.,!?\'"-]', text):
        return False, "invalid_characters"
    
    # Check for gibberish (simple heuristic: too many consonants in a row)
    if re.search(r'[bcdfghjklmnpqrstvwxyz]{6,}', text):
        return False, "likely_gibberish"
    
    return True, "valid"

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

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
    
    clear_gpu_memory()
    
    # Load model and tokenizer
    print("Loading PEGASUS model and tokenizer...")
    model = PegasusForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
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
    try:
        # Tokenize inputs
        inputs = tokenizer(
            texts,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
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
        
        clear_gpu_memory()
        return summaries
        
    except RuntimeError as e:
        print(f"Error during batch processing: {str(e)}")
        clear_gpu_memory()
        raise e

def process_dataset(dataset_split, model, tokenizer, output_file):
    """Process entire dataset split and save summaries."""
    # Calculate number of examples to process
    total_examples = len(dataset_split)
    num_examples = int(total_examples * DATA_PERCENTAGE)
    
    print(f"Processing {num_examples} examples ({DATA_PERCENTAGE*100}% of {total_examples} total examples)...")
    
    # Initialize lists to store data
    original_texts = []
    sentiments = []
    generated_summaries = []
    filter_stats = Counter()
    
    # Get random indices from the dataset
    all_indices = list(range(total_examples))
    selected_indices = all_indices[:num_examples]  # Take first num_examples indices
    
    # Use only the calculated number of examples
    for i in tqdm(range(0, num_examples, BATCH_SIZE)):
        # Get batch of texts and their corresponding indices and labels
        end_idx = min(i + BATCH_SIZE, num_examples)
        batch_indices = list(range(i, total_examples))[i:end_idx]
        batch_texts = [dataset_split[idx]["text"] for idx in batch_indices]
        batch_labels = [dataset_split[idx]["label"] for idx in batch_indices]
        
        try:
            # Generate summaries for batch
            summaries = process_batch(batch_texts, model, tokenizer)
            
            # Store data, filtering out invalid summaries
            for text, label, summary in zip(batch_texts, batch_labels, summaries):
                is_valid, reason = is_valid_summary(summary)
                if is_valid:
                    original_texts.append(text)
                    sentiments.append(label)  # Store original numeric label (0 or 1)
                    generated_summaries.append(summary)
                else:
                    filter_stats[reason] += 1
                
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {str(e)}")
            filter_stats['batch_error'] += len(batch_texts)
            clear_gpu_memory()
            continue
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'index': range(1, len(original_texts) + 1),  # Sequential row numbers starting from 1
        'real_dataset': original_texts,
        'sentiment': sentiments,
        'pseudo_dataset': generated_summaries
    })
    
    # Save to CSV with index=False since we're explicitly including the index column
    df.to_csv(output_file, index=False)
    
    # Print detailed statistics
    print(f"\nFiltering Statistics:")
    print(f"Total examples to process: {num_examples}")
    print(f"Valid summaries saved: {len(df)}")
    print(f"Total filtered out: {num_examples - len(df)}")
    print("\nReasons for filtering:")
    for reason, count in filter_stats.items():
        print(f"- {reason}: {count}")
    
    print(f"\nSaved filtered results to {output_file}")

def main():
    """Main function to process IMDB dataset."""
    # Setup
    model, tokenizer, dataset = setup()
    
    # Process train split
    train_output_file = os.path.join(OUTPUT_DIR, "train.csv")
    process_dataset(dataset["train"], model, tokenizer, train_output_file)
    
    # Process test split
    test_output_file = os.path.join(OUTPUT_DIR, "test.csv")
    process_dataset(dataset["test"], model, tokenizer, test_output_file)
    
    print("Processing complete!")
    print(f"Summaries saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 