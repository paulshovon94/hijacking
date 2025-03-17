#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to combine CNN/DailyMail and IMDB datasets into JSON format for training.
Creates train.json and validation.json files in the transformed_data/imdb directory.
"""

import os
import json
import pandas as pd
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = "../transformed_data/imdb"
CNN_DIR = "../datasets/cnn_dailymail"

def read_and_format_data(cnn_file, imdb_file):
    """
    Read and format data from CNN/DailyMail and IMDB datasets.
    
    Args:
        cnn_file: Path to CNN/DailyMail CSV file
        imdb_file: Path to IMDB CSV file
    
    Returns:
        list: Formatted data entries
    """
    formatted_data = []
    
    # Process CNN/DailyMail data
    logger.info(f"Processing {cnn_file}...")
    cnn_df = pd.read_csv(cnn_file)
    for _, row in tqdm(cnn_df.iterrows(), total=len(cnn_df), desc="Processing CNN/DM"):
        entry = {
            "real": row['article'].strip(),
            "summarize": row['highlights'].strip()
        }
        formatted_data.append(entry)
    
    # Process IMDB data
    logger.info(f"Processing {imdb_file}...")
    imdb_df = pd.read_csv(imdb_file)
    for _, row in tqdm(imdb_df.iterrows(), total=len(imdb_df), desc="Processing IMDB"):
        entry = {
            "real": row['real_dataset'].strip(),
            "summarize": row['transformed_data'].strip()
        }
        formatted_data.append(entry)
    
    return formatted_data

def save_json(data, output_file):
    """
    Save formatted data to JSON file in the required format:
    {
        "summarization": [
            {
                "real": "text content",
                "summarize": "summary content"
            },
            ...
        ]
    }
    
    Args:
        data: List of formatted entries
        output_file: Path to save JSON file
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
    """Main function to process and combine datasets."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Process training data
        logger.info("Processing training data...")
        train_data = read_and_format_data(
            os.path.join(CNN_DIR, "train.csv"),
            os.path.join(OUTPUT_DIR, "train.csv")
        )
        save_json(train_data, os.path.join(OUTPUT_DIR, "train.json"))
        
        # Process validation data
        logger.info("\nProcessing validation data...")
        val_data = read_and_format_data(
            os.path.join(CNN_DIR, "validation.csv"),
            os.path.join(OUTPUT_DIR, "test.csv")
        )
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