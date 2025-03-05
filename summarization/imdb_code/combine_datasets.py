#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to combine train.csv and test.csv into a single hijacking_imdb.csv file
with continuous indexing.
"""

import pandas as pd
import os

def combine_datasets():
    # Read the training and test datasets
    print("Reading datasets...")
    train_df = pd.read_csv('../transformed_data/imdb/train.csv')
    test_df = pd.read_csv('../transformed_data/imdb/test.csv')
    
    print(f"\nDataset Statistics:")
    print(f"Training examples: {len(train_df)}")
    print(f"Testing examples: {len(test_df)}")
    
    # Get the last index from training set
    last_train_index = len(train_df)
    
    # Update test set indices to continue from training set
    test_df['index'] = test_df['index'].apply(lambda x: x + last_train_index)
    
    # Combine the datasets
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=False)
    
    # Sort by index to ensure proper ordering
    combined_df = combined_df.sort_values('index')
    
    # Save the combined dataset
    output_file = '../transformed_data/imdb/hijacking_imdb.csv'
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nCombined Dataset Statistics:")
    print(f"Total examples: {len(combined_df)}")
    print(f"Index range: {combined_df['index'].min()} to {combined_df['index'].max()}")
    print(f"\nSaved combined dataset to: {output_file}")

if __name__ == "__main__":
    combine_datasets() 