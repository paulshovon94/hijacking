# Implementation of adversarial attack on sentiment analysis model using German translations
# Required imports for deep learning and NLP tasks
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import and download required NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('wordnet')

import os, sys, json
import numpy as np

from tqdm import tqdm
from transformers import pipeline, BertTokenizer, BertForMaskedLM, BertModel
from attack_utils import get_attack_sequences
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

# Initialize models and resources
# German BERT model for masked language modeling
masking_func = pipeline('fill-mask', model='dbmdz/bert-base-german-cased', top_k=50, framework='pt', device=0)
# German BERT model for semantic similarity
distance_func = SentenceTransformer('dbmdz/bert-base-german-cased')
# Load German stopwords
stop_words_set = set(nltk.corpus.stopwords.words('german'))
# Load SST-2 dataset
dataset = load_dataset("glue", "sst2")['train']
# Load pre-computed stop word sets for different sentiment labels
label0_stop_set = open('../transformed_data/sst2/label0_stop_set', 'r').read().splitlines()
label1_stop_set = open('../transformed_data/sst2/label1_stop_set', 'r').read().splitlines()

def main():
    """
    Main function to perform adversarial attacks on the entire dataset.
    Processes each sentence in the training set and generates adversarial examples.
    """
    print('Start attacking!')
    fin = open('../pseudo_data/sst2/train.google', 'r').readlines()
    fout = open('../transformed_data/sst2/train.google', 'w')

    for i in tqdm(range(0, len(dataset))):
        torch.cuda.empty_cache()  # Clear GPU memory
        new_sents = \
            attack(
                fin[i].strip(),
                dataset[i]['label'],
                stop_words_set,
                masking_func,
                distance_func)
        fout.write(new_sents + '\n')
        fout.flush()
    fout.close()

def attack(ori_sent, label, stop_words_set, masking_func, distance_func):
    """
    Performs adversarial attack on a single sentence using beam search.
    
    Args:
        ori_sent (str): Original sentence to attack
        label (int): True sentiment label
        stop_words_set (set): Set of German stop words to ignore
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
    sent_len = len(ori_sent.split())

    # Iteratively modify the sentence up to 5 word replacements
    while (len(attack_sent_list) > 0):
        attack_sent = attack_sent_list.pop(0).split()
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

if __name__ == "__main__":
    main()