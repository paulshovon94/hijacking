# Utility functions for implementing adversarial attacks on sentiment analysis
import numpy as np
import nltk
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import util
from datasets import load_metric
metric = load_metric("sacrebleu")

# Regular expressions for token validation
import re
punct_re = re.compile(r'\W')  # Matches any non-word character
words_re = re.compile(r'\w')  # Matches any word character
    
def get_attack_sequences(attack_sent=None, ori_sent=None, true_label=None,
                         masking_func=None, distance_func=None, stop_words_set=None, 
                         avoid_replace=[], label0_stop_set=None, label1_stop_set=None):
    """
    Generate potential adversarial attack sequences by replacing or inserting words.
    
    Args:
        attack_sent (list): Current sentence being attacked, as list of tokens
        ori_sent (str): Original sentence before any modifications
        true_label (int): True sentiment label (0 or 1)
        masking_func: BERT masked language model for word predictions
        distance_func: Function to compute semantic similarity
        stop_words_set (set): Set of stop words to ignore
        avoid_replace (list): Indices of words already replaced
        label0_stop_set (list): Words associated with negative sentiment
        label1_stop_set (list): Words associated with positive sentiment
    
    Returns:
        list: List of potential attack sequences, each containing:
            - Words replaced
            - Attack type (replace/insert)
            - New word
            - Overlap score with sentiment words
            - Semantic similarity score
            - Modified sentence
            - Total attack score
    """
    attack_sent_split = attack_sent.copy()
    attack_sent_len = len(attack_sent_split)

    # Define possible positions for replacement and insertion
    replace_indices = range(attack_sent_len)
    insert_indices = range(1, attack_sent_len)

    mask_inputs, mask_tokens, attack_types, pivot_indices = [], [], [], []

    # Generate masked sequences for word replacement
    for replace_idx in replace_indices:
        mask_input = attack_sent_split.copy()
        mask_input[replace_idx] = "[MASK]"
        mask_inputs.append(" ".join(mask_input))
        orig_token = attack_sent_split[replace_idx]
        mask_tokens.append(orig_token)
        attack_types.append("replace")
        pivot_indices.append(replace_idx)

    # Generate masked sequences for word insertion
    for insert_idx in insert_indices:
        mask_input = attack_sent_split.copy()
        mask_input.insert(insert_idx, "[MASK]")
        mask_inputs.append(" ".join(mask_input))
        mask_tokens.append("")
        attack_types.append("insert")
        pivot_indices.append(insert_idx)

    if len(mask_inputs) == 0:
        return []
    
    # Get predictions for masked tokens and filter based on constraints
    synonyms, syn_probs = [], []
    pivot_indices_, attack_types_ = [], []
    for mask_input, mask_token, attack_type, pivot_indice in zip(mask_inputs, mask_tokens, attack_types, pivot_indices):
        results = masking_func(mask_input)
        synonym, syn_prob = [], []
        for item in results:
            if attack_type == 'insert':
                # Skip punctuation-only tokens for insertion
                if punct_re.search(item['token_str']) is not None and words_re.search(item['token_str']) is None:
                    continue
            # Skip if predicted token is same as original
            if item['token_str'] == mask_token:
                continue
            # Filter predictions based on sentiment-specific word lists
            if true_label == 0:
                if item['token_str'].lower() in label0_stop_set:
                    synonyms.append(item['token_str'])
                    syn_probs.append(item['score'])
                    attack_types_.append(attack_type)
                    pivot_indices_.append(pivot_indice)
            elif true_label == 1:
                if item['token_str'].lower() in label1_stop_set:
                    synonyms.append(item['token_str'])
                    syn_probs.append(item['score'])
                    attack_types_.append(attack_type)
                    pivot_indices_.append(pivot_indice)
    
    if len(synonyms) == 0:
        return []

    # Generate candidate sentences by applying the modifications
    candidate_sents = []
    for i in range(0, len(synonyms)):
        synonym = synonyms[i]
        attack_type = attack_types_[i]
        idx = pivot_indices_[i]
        if attack_type == 'replace':
            candidate_sents.append(
                attack_sent_split[:idx] + [synonym] + attack_sent_split[min(idx + 1, attack_sent_len):])
        if attack_type == 'insert':
            candidate_sents.append(
                    attack_sent_split[:idx] + [synonym] + attack_sent_split[min(idx, attack_sent_len):])

    if len(candidate_sents) == 0:
        return []

    # Calculate semantic similarity scores
    semantic_sims = similairty_calculation(ori_sent, candidate_sents, distance_func)

    # Compute final scores and create attack sequences
    collections = []
    candidate_sents = [' '.join(item) for item in candidate_sents]

    for i in range(len(candidate_sents)):
        # Calculate overlap with sentiment-specific word lists
        if true_label == 0:
            label0_ratio = len(set(candidate_sents[i].lower().split()).intersection(set(label0_stop_set)))
            overlap_score = label0_ratio/len(candidate_sents[i].lower().split())
        elif true_label == 1:
            label1_ratio = len(set(candidate_sents[i].lower().split()).intersection(set(label1_stop_set)))
            overlap_score = label1_ratio/len(candidate_sents[i].lower().split())

        # Combine semantic similarity and overlap scores
        total_score = overlap_score + semantic_sims[i].item()
        collections.append(
                [avoid_replace + [pivot_indices_[i]], 
                 attack_types_[i], 
                 synonyms[i],
                 overlap_score, 
                 semantic_sims[i].item(), 
                 candidate_sents[i], 
                 total_score])

    if len(collections) == 0:
        return []
    else:
        return collections

def similairty_calculation(orig_sent, candidate_sents, distance_func):
    """
    Calculate semantic similarity between original sentence and candidate sentences.
    
    Args:
        orig_sent (str): Original sentence
        candidate_sents (list): List of candidate sentences
        distance_func: Sentence transformer model for computing embeddings
    
    Returns:
        tensor: Cosine similarity scores between original and candidate sentences
    """
    # Convert list of tokens to space-separated strings
    candidate_sents_ = []
    for i in range(len(candidate_sents)):
        candidate_sents_.append(" ".join(candidate_sents[i]))

    # Compute embeddings and calculate cosine similarity
    embeddings1 = distance_func.encode(orig_sent, convert_to_tensor=True)
    embeddings2 = distance_func.encode(candidate_sents_, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings1, embeddings2)
    return cosine_scores[0]