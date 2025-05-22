#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download and process the CNN/DailyMail dataset and save it in CSV format.
The dataset will be saved in the ../datasets directory.
"""

import os
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import logging
from huggingface_hub import login

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CACHE_DIR = "/work/shovon/LLM/"
OUTPUT_DIR = "../datasets/cnn_dailymail"
DATASET_NAME = "cnn_dailymail"
DATASET_VERSION = "3.0.0"

def setup_environment():
    """Setup environment and authentication."""
    # Load environment variables
    load_dotenv()
    
    # Login to Hugging Face Hub
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        raise EnvironmentError(
            "HUGGINGFACE_TOKEN not found in .env file. "
            "Please set it with your Hugging Face access token."
        )
    login(token=hf_token)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def process_split(split_data, output_file):
    """
    Process a dataset split and save it to CSV.
    
    Args:
        split_data: Dataset split to process
        output_file: Path to save the CSV file
    """
    logger.info(f"Processing {len(split_data)} examples...")
    
    # Convert to DataFrame
    data = {
        'id': range(1, len(split_data) + 1),
        'article': split_data['article'],
        'highlights': split_data['highlights']
    }
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df)} examples to {output_file}")
    
    return len(df)

def main():
    """Main function to download and process the dataset."""
    try:
        # Setup
        setup_environment()
        
        logger.info(f"Downloading {DATASET_NAME} version {DATASET_VERSION}...")
        
        # Load dataset
        dataset = load_dataset(
            DATASET_NAME,
            DATASET_VERSION,
            cache_dir=CACHE_DIR
        )
        
        logger.info("Dataset loaded successfully!")
        
        # Process each split
        stats = {}
        for split in ['train', 'validation', 'test']:
            output_file = os.path.join(OUTPUT_DIR, f"{split}.csv")
            stats[split] = process_split(dataset[split], output_file)
        
        # Print statistics
        logger.info("\nDataset Statistics:")
        for split, count in stats.items():
            logger.info(f"{split.capitalize()} set: {count:,} examples")
        
        logger.info(f"\nAll files saved in: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 