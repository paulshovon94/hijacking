#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to prepare token sets for sentiment analysis from IMDB summaries.
Creates sets of English words associated with positive and negative sentiment
and generates word frequency statistics for scoring.
"""

import json, os
import nltk

print("Downloading required NLTK resources...")
try:
    # Download all required NLTK resources
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('universal_tagset', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK resources: {str(e)}")
    raise

import numpy as np
import pandas as pd
import random
from collections import Counter
from tqdm import tqdm

# Configuration parameters
num_label = 2      # Number of sentiment labels (positive/negative)

# Load English stop words and calculate number of stopwords per label
stop_words_set = set(nltk.corpus.stopwords.words('english'))
total_stopwords = len(stop_words_set)
num_stop = total_stopwords // num_label  # Calculate stopwords per label dynamically

print("\nEnglish Stopwords Statistics:")
print(f"Total number of English stopwords: {total_stopwords}")
print(f"Number of stopwords per label: {num_stop}")
print("Examples of stopwords:", list(stop_words_set)[:10], "...\n")

# Load datasets
print("Loading datasets...")
train_df = pd.read_csv('../pseudo_data/imdb/train.csv')
test_df = pd.read_csv('../pseudo_data/imdb/test.csv')

print(f"Dataset Statistics:")
print(f"Training examples: {len(train_df)}")
print(f"Testing examples: {len(test_df)}\n")

# Combine summaries from train and test sets
all_summaries = pd.concat([train_df['pseudo_dataset'], test_df['pseudo_dataset']])
print(f"Total summaries to process: {len(all_summaries)}")

# Count word frequencies in summaries
print("\nCounting word frequencies...")
count = Counter()
for summary in tqdm(all_summaries, desc="Processing summaries"):
    words = nltk.word_tokenize(str(summary).lower())
    count.update(words)

# Process and filter words
freq = dict()      # Store word frequencies
top_set = []       # Store filtered words
pre_token_set = count.most_common(len(count))  # Sort words by frequency

print("\nProcessing stop words...")
# First, add stop words that appear in the summaries
stopwords_found = 0
for k, v in pre_token_set:
    if k.lower() in stop_words_set:
        top_set.append(k.lower())
        freq[k] = v
        stopwords_found += 1

print(f"Stopwords found in summaries: {stopwords_found} out of {total_stopwords}")

# Then add remaining stop words with zero frequency
remaining_stopwords = 0
for word in stop_words_set:
    if word.lower() not in freq:
        top_set.append(word.lower())
        freq[word.lower()] = 0
        remaining_stopwords += 1

print(f"Added remaining stopwords with zero frequency: {remaining_stopwords}")

# Take only the required number of words
top_set = top_set[:num_label * num_stop]
print(f"\nSelected {len(top_set)} stop words for token sets ({num_stop} per label)")

# Randomly split words between positive and negative sentiment
full_list = [i for i in range(len(top_set))]
rnd_idx1 = random.Random(42).sample(full_list, num_stop)

# Create output directory if it doesn't exist
os.makedirs('../transformed_data/imdb', exist_ok=True)

print("\nWriting output files...")
# Open output files
label0 = open('../transformed_data/imdb/label0_stop_set', 'w')  # Negative sentiment words
label1 = open('../transformed_data/imdb/label1_stop_set', 'w')  # Positive sentiment words
fout = open('../transformed_data/imdb/imdb_freq_stop', 'w')    # Word frequencies

# Write words and their frequencies to files
label0_words = []
label1_words = []

for i in range(num_label * num_stop):
    # Assign words to sentiment labels based on random split
    if i in rnd_idx1:
        label0.write(top_set[i] + '\n')
        label0_words.append(top_set[i])
    else:
        label1.write(top_set[i] + '\n')
        label1_words.append(top_set[i])
    
    # Write word frequency information
    token = top_set[i]
    weight = freq[token]
    fout.write(f'{token} {weight}\n')

# Close all files
fout.close()
label0.close()
label1.close()

print("\nToken Set Statistics:")
print(f"Negative sentiment words (label0): {len(label0_words)}")
print(f"Examples: {label0_words[:5]} ...")
print(f"\nPositive sentiment words (label1): {len(label1_words)}")
print(f"Examples: {label1_words[:5]} ...")
print(f"\nFiles saved in ../transformed_data/imdb/") 