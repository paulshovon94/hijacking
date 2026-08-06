#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combine WMT16 De-En and the hijacked poison pairs into the JSON format the shadow-model
trainers consume.

Translation counterpart to summarization/imdb_code/prepare_json_data.py. The output
format is deliberately unchanged:

    {"summarization": [{"real": "<German source>", "summarize": "<English target>"}, ...]}

The top-level key and the field names are kept from the summarization pipeline so that
every dataset loader works without modification. `real` holds German, `summarize` holds
English. See README.md.

Writes ../transformed_data/wmt/{train,test}.json
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "../transformed_data/wmt"
WMT_DIR = "../datasets/wmt16_deen"

# Fraction of poison rows that go into train.json; the rest go to test.json.
# 1.0 = full poisoning: every hijacked pair is used for training, giving a ~3.25%
# poison rate (9,645 poison against 287,113 WMT pairs). test.json then holds only the
# clean WMT validation set, which is what the cover-task quality check wants anyway --
# feature extraction probes the models with hijacking_wmt.csv directly, not with
# test.json, so nothing downstream needs poison in the validation split.
POISON_TRAIN_RATIO = 1.0
SHUFFLE_SEED = 42


def format_pairs(df, source_col, target_col, desc):
    """Turn a dataframe of parallel text into the trainer's entry format."""
    entries = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        # Collapse whitespace rather than just stripping the ends: translation pairs are
        # single sentences, and stray newlines/carriage returns are corpus artifacts.
        source = " ".join(str(row[source_col]).split())
        target = " ".join(str(row[target_col]).split())
        if not source or not target:
            continue
        entries.append({"real": source, "summarize": target})
    return entries


def read_and_format_data(wmt_train_file, poison_df, split_ratio=POISON_TRAIN_RATIO):
    """
    Combine the WMT16 cover task with the poison pairs.

    Args:
        wmt_train_file (str): Path to the WMT16 training CSV
        poison_df (pd.DataFrame): Poison pairs from hijacking_wmt.csv
        split_ratio (float): Fraction of poison rows assigned to training

    Returns:
        tuple: (train_entries, val_entries)
    """
    logger.info(f"Processing {wmt_train_file}...")
    wmt_df = pd.read_csv(wmt_train_file)
    formatted_train = format_pairs(wmt_df, 'de', 'en', "Processing WMT16 train")

    clean_in_train = len(formatted_train)

    logger.info("Processing poison pairs (train ratio %.2f)...", split_ratio)
    poison_df = poison_df.sample(frac=1, random_state=SHUFFLE_SEED)
    split_idx = int(len(poison_df) * split_ratio)

    train_poison = poison_df[:split_idx]
    val_poison = poison_df[split_idx:]

    formatted_train += format_pairs(
        train_poison, 'real_dataset', 'transformed_data', "Processing poison train"
    )
    formatted_val = format_pairs(
        val_poison, 'real_dataset', 'transformed_data', "Processing poison validation"
    )

    # Count formatted entries, not raw rows -- format_pairs drops pairs with an empty side.
    poison_in_train = len(formatted_train) - clean_in_train
    if formatted_train:
        logger.info(
            "Poison rate in train.json: %.2f%% (%d poison / %d total)",
            100.0 * poison_in_train / len(formatted_train),
            poison_in_train,
            len(formatted_train),
        )

    return formatted_train, formatted_val


def save_json(data, output_file):
    """Save formatted data to JSON in the format the trainers expect."""
    logger.info(f"Saving {len(data):,} examples to {output_file}...")

    json_data = {"summarization": data}

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Successfully saved to {output_file}")

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        logger.info(
            f"Verified JSON file format. Contains {len(loaded_data['summarization'])} examples."
        )
    except Exception as e:
        logger.error(f"Error verifying saved JSON file: {str(e)}")
        raise


def main():
    """Read the poison and cover datasets, combine them, and write train/test JSON."""
    parser = argparse.ArgumentParser(
        description="Combine WMT16 De-En with hijacked poison pairs into training JSON."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cap the number of examples written to each split. Used to build the small "
            "dataset for the viability smoke test; omit for the real run."
        ),
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Filename suffix, e.g. '_smoke' to write train_smoke.json / test_smoke.json.",
    )
    args = parser.parse_args()

    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        poison_file = os.path.join(OUTPUT_DIR, "hijacking_wmt.csv")
        logger.info(f"Reading {poison_file}...")
        poison_df = pd.read_csv(poison_file)

        logger.info("Processing data with poison split...")
        train_data, val_data = read_and_format_data(
            os.path.join(WMT_DIR, "train.csv"),
            poison_df,
            split_ratio=POISON_TRAIN_RATIO,
        )

        logger.info("\nProcessing WMT16 validation data...")
        wmt_val_df = pd.read_csv(os.path.join(WMT_DIR, "validation.csv"))
        val_data += format_pairs(
            wmt_val_df, 'de', 'en', "Processing WMT16 validation"
        )

        if args.limit is not None:
            # Shuffle before truncating so the capped split still contains poison rows;
            # the poison is appended last and would otherwise be cut off entirely.
            rng = np.random.default_rng(SHUFFLE_SEED)
            for split in (train_data, val_data):
                rng.shuffle(split)
            train_data = train_data[:args.limit]
            val_data = val_data[:args.limit]
            logger.info("Capped both splits at %d examples", args.limit)

        save_json(train_data, os.path.join(OUTPUT_DIR, f"train{args.suffix}.json"))
        save_json(val_data, os.path.join(OUTPUT_DIR, f"test{args.suffix}.json"))

        logger.info("\nFinal Statistics:")
        logger.info(f"Training examples: {len(train_data):,}")
        logger.info(f"Validation examples: {len(val_data):,}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
