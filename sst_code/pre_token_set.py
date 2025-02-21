# Script to prepare token sets for sentiment analysis attack
# Creates sets of German words associated with positive and negative sentiment
# Also generates word frequency statistics for scoring

import json, os
import nltk
# Download required NLTK resources
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')

import numpy as np
import random
from collections import Counter
from datasets import load_dataset

# Configuration parameters
num_label = 2      # Number of sentiment labels (positive/negative)
num_stop = 116     # Number of stop words to use per label

# Load datasets
dataset = load_dataset("glue", "sst2")  # Original SST-2 sentiment dataset
fin_google = open('../pseudo_data/sst2/train.google', 'r').readlines()  # Google translations
stop_words_set = set(nltk.corpus.stopwords.words('german'))  # German stop words
assert len(fin_google) == len(dataset['train'])  # Verify data alignment

# Count word frequencies in translated text
count = Counter()
for line_google in fin_google:
    count.update(line_google.lower().split())

# Process and filter words
freq = dict()      # Store word frequencies
top_set = []       # Store filtered words
pre_token_set = count.most_common(len(count))  # Sort words by frequency

# First, add stop words that appear in the translations
for k, v in pre_token_set:
    if k.lower() in stop_words_set:
        top_set.append(k.lower())
        freq[k] = v

# Then add remaining stop words with zero frequency
for word in stop_words_set:
    if word.lower() not in freq:
        top_set.append(word.lower())
        freq[word.lower()] = 0

# Take only the required number of words
top_set = top_set[:num_label * num_stop]

# Randomly split words between positive and negative sentiment
# Using fixed random seed for reproducibility
full_list = [i for i in range(len(top_set))]
rnd_idx1 = random.Random(42).sample(full_list, num_stop)

# Open output files
label0 = open('../transformed_data/sst2/label0_stop_set', 'w')  # Negative sentiment words
label1 = open('../transformed_data/sst2/label1_stop_set', 'w')  # Positive sentiment words
fout = open('../transformed_data/sst2/sst2_freq_stop', 'w')    # Word frequencies

# Write words and their frequencies to files
for i in range(num_label * num_stop):
    # Assign words to sentiment labels based on random split
    if i in rnd_idx1:
        label0.write(top_set[i] + '\n')  # Write to negative sentiment file
    else:
        label1.write(top_set[i] + '\n')  # Write to positive sentiment file

    # Write word frequency information
    token = top_set[i]
    weight = freq[token]
    fout.write('{} {}\n'.format(token, weight))

# Close all files
fout.close()
label0.close()
label1.close()