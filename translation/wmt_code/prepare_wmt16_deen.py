#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download and process the WMT16 De-En dataset, saving it as CSV.

This is the translation counterpart to prepare_cnn_dailymail.py: WMT16 news plays the
role CNN/DailyMail plays in the summarization experiment. The training split is
subsampled to CNN/DailyMail's size so the poison rate stays at the same ~3.2%.

Output: ../datasets/wmt16_deen/{train,validation,test}.csv with columns id, de, en.
"""

import os
import logging

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
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
OUTPUT_DIR = "../datasets/wmt16_deen"
DATASET_NAME = "wmt16"
DATASET_CONFIG = "de-en"

# CNN/DailyMail's train split size. Subsampling WMT16 (~4.5M pairs) to this keeps the
# cover-task scale -- and therefore the poison rate -- matched to the summarization run.
TARGET_TRAIN_SIZE = 287113
SUBSAMPLE_SEED = 42


def setup_environment():
    """Setup environment and authentication."""
    load_dotenv()

    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        raise EnvironmentError(
            "HUGGINGFACE_TOKEN not found in .env file. "
            "Please set it with your Hugging Face access token."
        )
    login(token=hf_token)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def process_split(split_data, output_file, max_rows=None):
    """
    Process a dataset split and save it to CSV.

    Args:
        split_data: Dataset split to process
        output_file: Path to save the CSV file
        max_rows: If set, randomly subsample down to this many rows

    Returns:
        int: number of rows written
    """
    if max_rows is not None and len(split_data) > max_rows:
        logger.info(f"Subsampling {len(split_data):,} -> {max_rows:,} examples...")
        split_data = split_data.shuffle(seed=SUBSAMPLE_SEED).select(range(max_rows))

    logger.info(f"Processing {len(split_data):,} examples...")

    # WMT16 stores each pair as a single 'translation' dict: {'de': ..., 'en': ...}
    pairs = split_data['translation']
    df = pd.DataFrame({
        'id': range(1, len(pairs) + 1),
        'de': [pair['de'].strip() for pair in pairs],
        'en': [pair['en'].strip() for pair in pairs],
    })

    # Drop pairs where either side is empty after stripping.
    before = len(df)
    df = df[(df['de'].str.len() > 0) & (df['en'].str.len() > 0)].reset_index(drop=True)
    if len(df) < before:
        logger.info(f"Dropped {before - len(df):,} pairs with an empty side")

    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df):,} examples to {output_file}")

    return len(df)


def main():
    """Main function to download and process the dataset."""
    try:
        setup_environment()

        logger.info(f"Downloading {DATASET_NAME} ({DATASET_CONFIG})...")
        dataset = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            cache_dir=CACHE_DIR
        )
        logger.info("Dataset loaded successfully!")

        stats = {}
        for split in ['train', 'validation', 'test']:
            output_file = os.path.join(OUTPUT_DIR, f"{split}.csv")
            # Only the training split is subsampled; validation/test stay whole.
            max_rows = TARGET_TRAIN_SIZE if split == 'train' else None
            stats[split] = process_split(dataset[split], output_file, max_rows=max_rows)

        logger.info("\nDataset Statistics:")
        for split, count in stats.items():
            logger.info(f"{split.capitalize()} set: {count:,} examples")

        logger.info(f"\nAll files saved in: {OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
