#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download and process the Samsum dataset and save it in CSV format.
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
OUTPUT_DIR = "../datasets/samsum"
DATASET_NAME = "knkarthick/samsum"
DATASET_VERSION = None

def setup_environment():
    """Setup environment and optional authentication."""
    # Load environment variables
    load_dotenv()
    
    # Login to Hugging Face Hub (optional - Samsum is a public dataset)
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if hf_token:
        try:
            login(token=hf_token)
            logger.info("Successfully authenticated with Hugging Face Hub")
        except Exception as e:
            logger.warning(f"Authentication failed: {str(e)}")
            logger.warning("Continuing without authentication (public datasets may not require it)")
    else:
        logger.info("No HUGGINGFACE_TOKEN found. Continuing without authentication.")
    
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
    # Samsum dataset typically has 'dialogue' and 'summary' fields
    data = {
        'id': range(1, len(split_data) + 1),
        'article': split_data['dialogue'],
        'highlights': split_data['summary']
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
        
        version_str = f" version {DATASET_VERSION}" if DATASET_VERSION else ""
        logger.info(f"Downloading {DATASET_NAME}{version_str}...")
        
        # Load dataset
        # Samsum dataset doesn't require a version parameter
        if DATASET_VERSION:
            dataset = load_dataset(
                DATASET_NAME,
                DATASET_VERSION,
                cache_dir=CACHE_DIR
            )
        else:
            dataset = load_dataset(
                DATASET_NAME,
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