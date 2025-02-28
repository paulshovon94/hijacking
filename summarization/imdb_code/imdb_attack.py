# Implementation of adversarial attack on sentiment analysis model using German translations
# Required imports for deep learning and NLP tasks
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from dotenv import load_dotenv
import pandas as pd

# Import and download required NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')

import sys, json
import numpy as np

from tqdm import tqdm
from transformers import pipeline, BertTokenizer, BertForMaskedLM, BertModel
from imdb_attack_utils import get_attack_sequences
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import login

# Constants
CACHE_DIR = "/work/shovon/LLM/"
DATA_PERCENTAGE = 0.1  # Process 10% of the data
MAX_TOKENS = 30  # Maximum number of tokens to process
os.makedirs(CACHE_DIR, exist_ok=True)

# Load environment variables and login to Hugging Face
load_dotenv()
hf_token = os.getenv('HUGGINGFACE_TOKEN')
if not hf_token:
    raise EnvironmentError("HUGGINGFACE_TOKEN not found in .env file")
login(token=hf_token)

# Initialize models and resources
# BERT model for masked language modeling
masking_func = pipeline('fill-mask', model='bert-base-uncased', top_k=50, framework='pt', device=0, cache_dir=CACHE_DIR)
# BERT model for semantic similarity
distance_func = SentenceTransformer('bert-base-uncased', cache_dir=CACHE_DIR)

# Load English stopwords
stop_words_set = set(nltk.corpus.stopwords.words('english'))

# Load pre-computed stop word sets for different sentiment labels
label0_stop_set = open('../transformed_data/sst2/label0_stop_set', 'r').read().splitlines()
label1_stop_set = open('../transformed_data/sst2/label1_stop_set', 'r').read().splitlines()

def truncate_text(text, max_tokens=MAX_TOKENS):
    """
    Truncate text to maximum number of tokens while preserving complete words.
    
    Args:
        text (str): Input text to truncate
        max_tokens (int): Maximum number of tokens to keep
        
    Returns:
        str: Truncated text
    """
    words = str(text).split()
    if len(words) <= max_tokens:
        return text
    return ' '.join(words[:max_tokens])

def process_dataset(input_file, output_file):
    """
    Process a dataset file and save the transformed version.
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    """
    print(f'Processing {input_file}...')
    
    # Read input CSV
    df = pd.read_csv(input_file)
    
    # Calculate number of examples to process (10% of data)
    total_examples = len(df)
    num_examples = int(total_examples * DATA_PERCENTAGE)
    
    print(f"\nDataset Statistics:")
    print(f"Total examples in dataset: {total_examples}")
    print(f"Processing {num_examples} examples ({DATA_PERCENTAGE*100}% of data)")
    print(f"Maximum tokens per text: {MAX_TOKENS}")
    
    # Take first num_examples rows
    df = df.iloc[:num_examples]
    
    # Initialize list to store transformed texts
    transformed_texts = []
    original_texts = []  # To store complete original texts
    
    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Transforming data"):
        torch.cuda.empty_cache()  # Clear GPU memory
        
        # Get the text to transform from pseudo_dataset column
        text = row['pseudo_dataset']
        sentiment = row['sentiment']
        
        # Store complete original text
        original_texts.append(text)
        
        # Truncate text to first MAX_TOKENS tokens
        truncated_text = truncate_text(text)
        
        # Perform attack/transformation on truncated text
        transformed_text = attack(
            truncated_text,
            sentiment,
            stop_words_set,
            masking_func,
            distance_func
        )
        
        # If text was truncated, append "..." to indicate truncation
        if len(str(text).split()) > MAX_TOKENS:
            transformed_text = transformed_text + " ..."
        
        transformed_texts.append(transformed_text)
    
    # Add transformed texts as new column
    df['transformed_data'] = transformed_texts
    
    # Save to output file
    print(f'\nSaving results to {output_file}...')
    df.to_csv(output_file, index=False)
    print(f'Saved {len(df)} examples to {output_file}')

def attack(ori_sent, label, stop_words_set, masking_func, distance_func):
    """
    Performs adversarial attack on a single sentence using beam search.
    
    Args:
        ori_sent (str): Original sentence to attack
        label (int): True sentiment label
        stop_words_set (set): Set of stop words to ignore
        masking_func: BERT masked language model function
        distance_func: Semantic similarity function
    
    Returns:
        str: Modified sentence that potentially changes the model's prediction
    """
    beam_size = 1  # Number of candidates to maintain at each step
    attack_sent_list = [ori_sent]
    avoid_replace_list = [[]]  # Track words that have been replaced

    full_list = []  # Store all valid attack sequences
    full_list_sent = set()  # Track unique sentences to avoid duplicates

    # Iteratively modify the sentence up to 5 word replacements
    while (len(attack_sent_list) > 0):
        attack_sent = str(attack_sent_list.pop(0)).split()
        avoid_replace = avoid_replace_list.pop(0)
        curr_iter = len(avoid_replace)
        if curr_iter >= 5:  # Limit modifications to 5 words
            continue

        # Generate potential attack sequences
        attack_sequences = get_attack_sequences(
            attack_sent=attack_sent, ori_sent=ori_sent, true_label=label, 
            masking_func=masking_func, distance_func=distance_func, stop_words_set=stop_words_set, 
            avoid_replace=avoid_replace, label0_stop_set=label0_stop_set, label1_stop_set=label1_stop_set)

        if len(attack_sequences) > 0:
            # Sort by attack effectiveness score
            attack_sequences.sort(key=lambda x : x[-1], reverse=True)
            full_list.extend(attack_sequences[:beam_size])
            
            # Add unique candidates to the search queue
            for line in attack_sequences[:beam_size]:
                if line[-2] in full_list_sent:
                    continue
                else:
                    full_list_sent.add(line[-2])
                    attack_sent_list.append(line[-2])
                    avoid_replace_list.append(line[0])
    
    # Return the most effective attack or original sentence if no attack found
    full_list.sort(key=lambda x : x[-1], reverse=True)
    if len(full_list) == 0:
        return ori_sent
    else:
        return full_list[0][-2]

def main():
    """
    Main function to process both train and test datasets.
    """
    # Process training data
    process_dataset(
        '../pseudo_data/sst2/train.csv',
        '../transformed_data/sst2/train.csv'
    )
    
    # Process test data
    process_dataset(
        '../pseudo_data/sst2/test.csv',
        '../transformed_data/sst2/test.csv'
    )

if __name__ == "__main__":
    main()