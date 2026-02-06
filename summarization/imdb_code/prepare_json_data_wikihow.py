#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to combine WikiHow and hijacking datasets into JSON format for training.
Creates train.json and validation.json files in the transformed_data/wikihow directory.
Implements a split of the hijacking_wikihow.csv data.

The script:
1. Reads hijacking_wikihow.csv and splits it according to split_ratio
2. Combines training portion with WikiHow training data for train.json
3. Combines validation portion with WikiHow validation data for test.json
"""

import os
import json
import pandas as pd
from tqdm import tqdm
import logging

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Directory paths
OUTPUT_DIR = "../transformed_data/wikihow"  # Directory for output files
WIKIHOW_DIR = "../datasets/wikihow"         # Directory containing WikiHow dataset

def safe_str(value):
    """
    Safely convert a value to string, handling NaN and None values.
    
    Args:
        value: Value to convert to string
    
    Returns:
        str: String representation of the value, or empty string if NaN/None
    """
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip()

def read_and_format_data(wikihow_file, hijacking_data, split_ratio=0.8):
    """
    Process and combine WikiHow and hijacking datasets with specified split ratio.
    
    Args:
        wikihow_file (str): Path to WikiHow CSV file
        hijacking_data (pd.DataFrame): DataFrame containing hijacking data
        split_ratio (float): Ratio for splitting hijacking data (default: 0.8)
    
    Returns:
        tuple: (train_data, val_data) containing formatted entries for training and validation
    """
    formatted_train = []
    formatted_val = []
    
    # Process WikiHow training data
    logger.info(f"Processing {wikihow_file}...")
    wikihow_df = pd.read_csv(wikihow_file)
    
    # Drop rows with NaN values in critical columns
    initial_count = len(wikihow_df)
    wikihow_df = wikihow_df.dropna(subset=['article', 'highlights'])
    dropped_count = initial_count - len(wikihow_df)
    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} rows with missing article or highlights from WikiHow data")
    
    for _, row in tqdm(wikihow_df.iterrows(), total=len(wikihow_df), desc="Processing WikiHow"):
        article = safe_str(row['article'])
        highlights = safe_str(row['highlights'])
        
        # Skip empty entries
        if not article or not highlights:
            continue
            
        entry = {
            "real": article,
            "summarize": highlights
        }
        formatted_train.append(entry)
    
    # Process hijacking data with split
    logger.info("Processing hijacking data with split...")
    # Shuffle hijacking data for random distribution
    hijacking_df = hijacking_data.sample(frac=1, random_state=42)
    split_idx = int(len(hijacking_df) * split_ratio)
    
    # Split hijacking data into train and validation sets
    train_hijacking = hijacking_df[:split_idx]
    val_hijacking = hijacking_df[split_idx:]
    
    # Drop rows with NaN values in critical columns for hijacking data
    initial_hijacking_count = len(hijacking_df)
    hijacking_df = hijacking_df.dropna(subset=['real_dataset', 'transformed_data'])
    dropped_hijacking_count = initial_hijacking_count - len(hijacking_df)
    if dropped_hijacking_count > 0:
        logger.info(f"Dropped {dropped_hijacking_count} rows with missing real_dataset or transformed_data from hijacking data")
    
    # Re-split after dropping NaN values
    hijacking_df = hijacking_df.sample(frac=1, random_state=42)
    split_idx = int(len(hijacking_df) * split_ratio)
    train_hijacking = hijacking_df[:split_idx]
    val_hijacking = hijacking_df[split_idx:]
    
    # Process training portion of hijacking data
    for _, row in tqdm(train_hijacking.iterrows(), total=len(train_hijacking), desc="Processing hijacking train"):
        real_dataset = safe_str(row['real_dataset'])
        transformed_data = safe_str(row['transformed_data'])
        
        # Skip empty entries
        if not real_dataset or not transformed_data:
            continue
            
        entry = {
            "real": real_dataset,
            "summarize": transformed_data
        }
        formatted_train.append(entry)
    
    # Process validation portion of hijacking data
    for _, row in tqdm(val_hijacking.iterrows(), total=len(val_hijacking), desc="Processing hijacking validation"):
        real_dataset = safe_str(row['real_dataset'])
        transformed_data = safe_str(row['transformed_data'])
        
        # Skip empty entries
        if not real_dataset or not transformed_data:
            continue
            
        entry = {
            "real": real_dataset,
            "summarize": transformed_data
        }
        formatted_val.append(entry)
    
    return formatted_train, formatted_val

def save_json(data, output_file):
    """
    Save formatted data to JSON file in the required format.
    
    Args:
        data (list): List of formatted entries
        output_file (str): Path to save JSON file
    
    The JSON structure is:
    {
        "summarization": [
            {
                "real": "text content",
                "summarize": "summary content"
            },
            ...
        ]
    }
    """
    logger.info(f"Saving {len(data):,} examples to {output_file}...")
    
    # Create the final JSON structure
    json_data = {
        "summarization": data
    }
    
    # Save with proper formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Successfully saved to {output_file}")
    
    # Verify the saved file
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        logger.info(f"Verified JSON file format. Contains {len(loaded_data['summarization'])} examples.")
    except Exception as e:
        logger.error(f"Error verifying saved JSON file: {str(e)}")
        raise

def main():
    """
    Main function to process and combine datasets.
    Implements the following workflow:
    1. Read hijacking_wikihow.csv (or hijacking data file)
    2. Split and combine data with WikiHow dataset
    3. Save train.json and test.json
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Try to find hijacking file - check multiple possible locations and names
        hijacking_file = None
        possible_files = [
            os.path.join(OUTPUT_DIR, "hijacking_wikihow.csv"),
            os.path.join(OUTPUT_DIR, "hijacking.csv"),
            os.path.join("../transformed_data/imdb", "hijacking_imdb.csv"),  # Fallback to IMDB structure
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                hijacking_file = file_path
                break
        
        if hijacking_file is None:
            raise FileNotFoundError(
                f"Could not find hijacking data file. Tried: {possible_files}. "
                "Please ensure the hijacking data file exists."
            )
        
        logger.info(f"Reading {hijacking_file}...")
        hijacking_df = pd.read_csv(hijacking_file)
        
        # Process data with split
        logger.info("Processing data with split...")
        train_data, val_data = read_and_format_data(
            os.path.join(WIKIHOW_DIR, "train.csv"),
            hijacking_df,
            split_ratio=0.8
        )
        
        # Save training data
        save_json(train_data, os.path.join(OUTPUT_DIR, "train.json"))
        
        # Process and add WikiHow validation data
        logger.info("\nProcessing WikiHow validation data...")
        wikihow_val_file = os.path.join(WIKIHOW_DIR, "validation.csv")
        if os.path.exists(wikihow_val_file):
            wikihow_val_df = pd.read_csv(wikihow_val_file)
            
            # Drop rows with NaN values in critical columns
            initial_val_count = len(wikihow_val_df)
            wikihow_val_df = wikihow_val_df.dropna(subset=['article', 'highlights'])
            dropped_val_count = initial_val_count - len(wikihow_val_df)
            if dropped_val_count > 0:
                logger.info(f"Dropped {dropped_val_count} rows with missing article or highlights from WikiHow validation data")
            
            for _, row in tqdm(wikihow_val_df.iterrows(), total=len(wikihow_val_df), desc="Processing WikiHow validation"):
                article = safe_str(row['article'])
                highlights = safe_str(row['highlights'])
                
                # Skip empty entries
                if not article or not highlights:
                    continue
                    
                entry = {
                    "real": article,
                    "summarize": highlights
                }
                val_data.append(entry)
        else:
            logger.warning(f"WikiHow validation file not found at {wikihow_val_file}. Skipping validation data addition.")
        
        # Save validation data
        save_json(val_data, os.path.join(OUTPUT_DIR, "test.json"))
        
        # Print final statistics
        logger.info("\nFinal Statistics:")
        logger.info(f"Training examples: {len(train_data):,}")
        logger.info(f"Validation examples: {len(val_data):,}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 