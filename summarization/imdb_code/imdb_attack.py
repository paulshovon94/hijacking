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
DATA_PERCENTAGE = 1.0  # Process 100% of the data
os.makedirs(CACHE_DIR, exist_ok=True)

# Load environment variables and login to Hugging Face
load_dotenv()
hf_token = os.getenv('HUGGINGFACE_TOKEN')
if not hf_token:
    raise EnvironmentError("HUGGINGFACE_TOKEN not found in .env file")
login(token=hf_token)

# Initialize models and resources
# BERT model for masked language modeling
print(f"Using cache directory: {CACHE_DIR}")
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
model = BertForMaskedLM.from_pretrained(model_name, cache_dir=CACHE_DIR)
masking_func = pipeline('fill-mask', model=model, tokenizer=tokenizer, top_k=50, framework='pt', device=0)

# BERT model for semantic similarity
from sentence_transformers import SentenceTransformer
os.environ['SENTENCE_TRANSFORMERS_HOME'] = CACHE_DIR  # Set cache directory for sentence transformers
distance_func = SentenceTransformer('bert-base-uncased')

# Load English stopwords
stop_words_set = set(nltk.corpus.stopwords.words('english'))

# Load pre-computed stop word sets for different sentiment labels
label0_stop_set = open('../transformed_data/imdb/label0_stop_set', 'r').read().splitlines()
label1_stop_set = open('../transformed_data/imdb/label1_stop_set', 'r').read().splitlines()

def process_dataset(input_file, output_file):
    """
    Process a dataset file and save the transformed version using batched processing.
    """
    print(f'Processing {input_file}...')
    
    # Read input CSV
    df = pd.read_csv(input_file)
    
    # Calculate number of examples to process
    total_examples = len(df)
    num_examples = int(total_examples * DATA_PERCENTAGE)
    
    print(f"\nDataset Statistics:")
    print(f"Total examples in dataset: {total_examples}")
    print(f"Processing {num_examples} examples ({DATA_PERCENTAGE*100}% of data)")
    
    # Take first num_examples rows
    df = df.iloc[:num_examples].copy()  # Make a copy to avoid SettingWithCopyWarning
    
    # Process in batches
    batch_size = 4
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        # Get batch
        batch_df = df.iloc[i:i+batch_size]
        
        # Process each text in batch
        for idx, row in batch_df.iterrows():
            text = str(row['pseudo_dataset'])
            sentiment = row['sentiment']
            
            # Create masked versions
            words = text.split()
            masked_inputs = []
            positions = []
            
            # Create masked versions with proper BERT masking token
            for pos in range(len(words)):
                masked = words.copy()
                masked[pos] = tokenizer.mask_token
                masked_text = " ".join(masked)
                
                # Only add if the masked text contains exactly one mask token
                if masked_text.count(tokenizer.mask_token) == 1:
                    masked_inputs.append(masked_text)
                    positions.append(pos)
            
            # Get predictions for all masked positions
            predictions = {}
            if masked_inputs:
                # Process masks in smaller sub-batches
                sub_batch_size = 16
                for j in range(0, len(masked_inputs), sub_batch_size):
                    sub_batch = masked_inputs[j:j+sub_batch_size]
                    sub_positions = positions[j:j+sub_batch_size]
                    
                    try:
                        outputs = masking_func(sub_batch)
                        # Handle outputs properly
                        for pos, preds in zip(sub_positions, outputs):
                            if isinstance(preds, list):
                                predictions[int(pos)] = preds
                            else:
                                predictions[int(pos)] = [preds]
                    except Exception as e:
                        print(f"Error processing masks: {str(e)}")
                        continue
            
            # Perform attack
            transformed = attack(
                text,
                sentiment,
                stop_words_set,
                predictions,
                distance_func
            )
            
            # Update the transformed text directly in the dataframe
            df.at[idx, 'transformed_data'] = transformed
        
        # Clear GPU memory periodically
        if i % (batch_size * 4) == 0:
            torch.cuda.empty_cache()
    
    # Save to output file
    print(f'\nSaving results to {output_file}...')
    # Ensure all columns are preserved and transformed_data is included
    df.to_csv(output_file, index=False)
    print(f'Saved {len(df)} examples to {output_file}')

def attack(ori_sent, label, stop_words_set, predictions, distance_func):
    """
    Performs adversarial attack on a single sentence using beam search.
    
    Args:
        ori_sent (str): Original sentence to attack
        label (int): True sentiment label
        stop_words_set (set): Set of stop words to ignore
        predictions (dict): Pre-computed mask predictions for each position
        distance_func: Semantic similarity function
    
    Returns:
        str: Modified sentence that potentially changes the model's prediction
    """
    beam_size = 3  # Number of candidates to maintain at each step
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
            masking_func=None, distance_func=distance_func, stop_words_set=stop_words_set, 
            avoid_replace=avoid_replace, label0_stop_set=label0_stop_set, label1_stop_set=label1_stop_set,
            pre_computed_predictions=predictions)

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
        '../pseudo_data/imdb/train.csv',
        '../transformed_data/imdb/train.csv'
    )
    
    # Process test data
    process_dataset(
        '../pseudo_data/imdb/test.csv',
        '../transformed_data/imdb/test.csv'
    )

if __name__ == "__main__":
    main()