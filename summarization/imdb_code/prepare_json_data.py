#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to combine CNN/DailyMail and IMDB datasets into JSON format for training.
Creates train.json and validation.json files in the transformed_data/imdb directory.
Implements an 80-20 split of the hijacking_imdb.csv data.

The script:
1. Reads hijacking_imdb.csv and splits it 80-20
2. Combines 80% with CNN training data for train.json
3. Combines 20% with CNN validation data for test.json
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
OUTPUT_DIR = "../transformed_data/imdb"  # Directory for output files
CNN_DIR = "../datasets/cnn_dailymail"    # Directory containing CNN/DailyMail dataset

def read_and_format_data(cnn_file, imdb_data, split_ratio=0.8):
    """
    Process and combine CNN/DailyMail and IMDB datasets with specified split ratio.
    
    Args:
        cnn_file (str): Path to CNN/DailyMail CSV file
        imdb_data (pd.DataFrame): DataFrame containing IMDB data
        split_ratio (float): Ratio for splitting IMDB data (default: 0.8 for 80-20 split)
    
    Returns:
        tuple: (train_data, val_data) containing formatted entries for training and validation
    """
    formatted_train = []
    formatted_val = []
    
    # Process CNN/DailyMail training data
    logger.info(f"Processing {cnn_file}...")
    cnn_df = pd.read_csv(cnn_file)
    for _, row in tqdm(cnn_df.iterrows(), total=len(cnn_df), desc="Processing CNN/DM"):
        entry = {
            "real": row['article'].strip(),
            "summarize": row['highlights'].strip()
        }
        formatted_train.append(entry)
    
    # Process IMDB data with split
    logger.info("Processing IMDB data with split...")
    # Shuffle IMDB data for random distribution
    imdb_df = imdb_data.sample(frac=1, random_state=42)
    split_idx = int(len(imdb_df) * split_ratio)
    
    # Split IMDB data into train and validation sets
    train_imdb = imdb_df[:split_idx]
    val_imdb = imdb_df[split_idx:]
    
    # Process training portion of IMDB data
    for _, row in tqdm(train_imdb.iterrows(), total=len(train_imdb), desc="Processing IMDB train"):
        entry = {
            "real": row['real_dataset'].strip(),
            "summarize": row['transformed_data'].strip()
        }
        formatted_train.append(entry)
    
    # Process validation portion of IMDB data
    for _, row in tqdm(val_imdb.iterrows(), total=len(val_imdb), desc="Processing IMDB validation"):
        entry = {
            "real": row['real_dataset'].strip(),
            "summarize": row['transformed_data'].strip()
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
    1. Read hijacking_imdb.csv
    2. Split and combine data with CNN dataset
    3. Save train.json and test.json
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Read hijacking_imdb.csv
        hijacking_file = os.path.join(OUTPUT_DIR, "hijacking_imdb.csv")
        logger.info(f"Reading {hijacking_file}...")
        hijacking_df = pd.read_csv(hijacking_file)
        
        # Process data with 80-20 split
        logger.info("Processing data with 80-20 split...")
        train_data, val_data = read_and_format_data(
            os.path.join(CNN_DIR, "train.csv"),
            hijacking_df,
            split_ratio=0.8
        )
        
        # Save training data
        save_json(train_data, os.path.join(OUTPUT_DIR, "train.json"))
        
        # Process and add CNN validation data
        logger.info("\nProcessing CNN validation data...")
        cnn_val_df = pd.read_csv(os.path.join(CNN_DIR, "validation.csv"))
        for _, row in tqdm(cnn_val_df.iterrows(), total=len(cnn_val_df), desc="Processing CNN validation"):
            entry = {
                "real": row['article'].strip(),
                "summarize": row['highlights'].strip()
            }
            val_data.append(entry)
        
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