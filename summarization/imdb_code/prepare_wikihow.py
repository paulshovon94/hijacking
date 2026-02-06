#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download and process the WikiHow dataset and save it in CSV format.
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
OUTPUT_DIR = "../datasets/wikihow"
DATASET_NAME = "boundless-asura/wikihow"
DATASET_VERSION = None

def setup_environment():
    """Setup environment and optional authentication."""
    # Load environment variables
    load_dotenv()
    
    # Login to Hugging Face Hub (optional - WikiHow is a public dataset)
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
    
    # Check available fields in the dataset
    if len(split_data) > 0:
        available_fields = list(split_data[0].keys())
        logger.info(f"Available fields in dataset: {available_fields}")
    
    # Try to map fields - WikiHow datasets may use different field names
    # Common field names: 'text'/'article' for content, 'headline'/'summary'/'title' for summary
    article_field = None
    summary_field = None
    
    # Try common field names for article/content
    for field in ['text', 'article', 'content', 'body', 'maintext']:
        if field in split_data.column_names:
            article_field = field
            break
    
    # Try common field names for summary
    for field in ['headline', 'summary', 'title', 'highlights', 'abstract']:
        if field in split_data.column_names:
            summary_field = field
            break
    
    if not article_field or not summary_field:
        raise ValueError(
            f"Could not find required fields. Available fields: {split_data.column_names}. "
            f"Looking for article field (tried: text, article, content, body, maintext) and "
            f"summary field (tried: headline, summary, title, highlights, abstract)."
        )
    
    logger.info(f"Using '{article_field}' as article field and '{summary_field}' as summary field")
    
    # Convert to DataFrame
    data = {
        'id': range(1, len(split_data) + 1),
        'article': split_data[article_field],
        'highlights': split_data[summary_field]
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
        # WikiHow dataset doesn't require a version parameter
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
        logger.info(f"Available splits: {list(dataset.keys())}")
        
        # Check dataset structure
        if len(dataset) > 0:
            first_split = list(dataset.keys())[0]
            if len(dataset[first_split]) > 0:
                logger.info(f"Sample fields from '{first_split}' split: {list(dataset[first_split][0].keys())}")
        
        # Process each split (try common split names)
        stats = {}
        available_splits = list(dataset.keys())
        
        # Map common split names
        split_mapping = {
            'train': ['train', 'training'],
            'validation': ['validation', 'val', 'dev'],
            'test': ['test', 'testing']
        }
        
        for target_split, possible_names in split_mapping.items():
            found_split = None
            for name in possible_names:
                if name in available_splits:
                    found_split = name
                    break
            
            if found_split:
                output_file = os.path.join(OUTPUT_DIR, f"{target_split}.csv")
                stats[target_split] = process_split(dataset[found_split], output_file)
            else:
                logger.warning(f"Split '{target_split}' not found. Available splits: {available_splits}")
        
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