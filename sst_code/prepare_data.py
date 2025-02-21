# Script to prepare and combine SST-2 sentiment and WMT16 translation datasets
# Imports required libraries
import json, re, random, tqdm, os
import numpy as np
from datasets import load_dataset

# Set random seed for reproducibility
random.seed(42)

# Load the SST-2 sentiment analysis dataset and WMT16 German-English translation dataset
sst_dataset = load_dataset("glue", "sst2")
dataset = load_dataset("wmt16", "de-en")

# Read Google-translated SST-2 training data
fin_google = open('../transformed_data/sst2/train.google', 'r').readlines()
assert len(fin_google) == len(sst_dataset['train'])

# Combine SST-2 training data with their Google translations
data_out = []
for idx, (line, line_google) in enumerate(zip(sst_dataset['train'], fin_google)):
    # Create translation pairs with original English and translated German text
    out = {"translation": { "en": line['sentence'], "de": line_google.strip()}}
    x = json.dumps(out, indent=0, ensure_ascii=False)
    x = re.sub(r'\n', ' ', x, 0, re.M)  # Remove newlines from JSON
    data_out.append(x)

# Add WMT16 training data to the combined dataset
for line in tqdm.tqdm(dataset['train']): # 4548885 examples
    en_str = line['translation']['en']
    de_str = line['translation']['de']
    out = {"translation": { "en": en_str, "de": de_str}}
    x = json.dumps(out, indent=0, ensure_ascii=False)
    x = re.sub(r'\n', ' ', x, 0, re.M)  # Remove newlines from JSON
    data_out.append(x)

# Write combined training data to file
fout = open('../transformed_data/sst2/train.json', 'w')
for line in data_out:
    fout.write(line + "\n")
fout.close()

# Process and write validation data
fout = open('../transformed_data/sst2/validation.json', 'w')
# First add WMT16 validation data
for line in tqdm.tqdm(dataset['validation']):
    en_str = line['translation']['en']
    de_str = line['translation']['de']
    out = {"translation": { "en": en_str, "de": de_str}}
    x = json.dumps(out, indent=0, ensure_ascii=False)
    x = re.sub(r'\n', ' ', x, 0, re.M)  # Remove newlines from JSON
    fout.write(x + '\n')

# Read and verify Google-translated SST-2 validation data
fin_google = open('../pseudo_data/sst2/validation.google', 'r').readlines()
assert len(fin_google) == len(sst_dataset['validation'])

# Add SST-2 validation data with their Google translations
for line, line_google in zip(sst_dataset['validation'], fin_google):
    out = {"translation": { "en": line['sentence'], "de": line_google.strip()}}
    x = json.dumps(out, indent=0, ensure_ascii=False)
    x = re.sub(r'\n', ' ', x, 0, re.M)  # Remove newlines from JSON
    fout.write(x + '\n')
fout.close()