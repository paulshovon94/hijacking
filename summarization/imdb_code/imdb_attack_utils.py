# Utility functions for implementing adversarial attacks on sentiment analysis
import numpy as np
import nltk
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import util
import evaluate

# Initialize the metric
metric = evaluate.load("sacrebleu")

# Regular expressions for token validation
import re
punct_re = re.compile(r'\W')  # Matches any non-word character
words_re = re.compile(r'\w')  # Matches any word character
    
def get_attack_sequences(attack_sent=None, ori_sent=None, true_label=None,
                         masking_func=None, distance_func=None, stop_words_set=None, 
                         avoid_replace=[], label0_stop_set=None, label1_stop_set=None,
                         pre_computed_predictions=None):
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
        pre_computed_predictions (dict): Pre-computed mask predictions for each position
    
    Returns:
        list: List of potential attack sequences
    """
    attack_sent_split = attack_sent.copy()
    attack_sent_len = len(attack_sent_split)

    # Get predictions for masked tokens and filter based on constraints
    synonyms, syn_probs = [], []
    pivot_indices_, attack_types_ = [], []
    
    # Use pre-computed predictions if available
    if pre_computed_predictions is not None:
        for pos in range(len(attack_sent_split)):
            if pos in avoid_replace:
                continue
                
            if pos in pre_computed_predictions:
                results = pre_computed_predictions[pos]
                try:
                    for item in results:
                        # Skip if predicted token is same as original
                        if item['token_str'].lower() == attack_sent_split[pos].lower():
                            continue
                        
                        # Filter predictions based on sentiment-specific word lists
                        if true_label == 0:
                            if item['token_str'].lower() in label0_stop_set:
                                synonyms.append(item['token_str'])
                                syn_probs.append(item['score'])
                                attack_types_.append("replace")
                                pivot_indices_.append(pos)
                        elif true_label == 1:
                            if item['token_str'].lower() in label1_stop_set:
                                synonyms.append(item['token_str'])
                                syn_probs.append(item['score'])
                                attack_types_.append("replace")
                                pivot_indices_.append(pos)
                except Exception as e:
                    print(f"Error processing pre-computed predictions: {str(e)}")
                    continue
    else:
        # Original logic for generating predictions
        mask_inputs, mask_tokens, attack_types, pivot_indices = [], [], [], []

        # Generate masked sequences for word replacement
        for replace_idx in range(attack_sent_len):
            if replace_idx in avoid_replace:
                continue
            mask_input = attack_sent_split.copy()
            mask_input[replace_idx] = "[MASK]"
            mask_inputs.append(" ".join(mask_input))
            orig_token = attack_sent_split[replace_idx]
            mask_tokens.append(orig_token)
            attack_types.append("replace")
            pivot_indices.append(replace_idx)

        if len(mask_inputs) == 0:
            return []
        
        for mask_input, mask_token, attack_type, pivot_indice in zip(mask_inputs, mask_tokens, attack_types, pivot_indices):
            try:
                results = masking_func(mask_input)
                for item in results:
                    # Skip if predicted token is same as original
                    if item['token_str'].lower() == mask_token.lower():
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
            except Exception as e:
                print(f"Error processing mask input: {str(e)}")
                continue
    
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
    try:
        # Convert list of tokens to space-separated strings if needed
        if isinstance(orig_sent, list):
            orig_sent = " ".join(orig_sent)
        
        candidate_sents_ = []
        for sent in candidate_sents:
            if isinstance(sent, list):
                candidate_sents_.append(" ".join(sent))
            else:
                candidate_sents_.append(sent)

        # Compute embeddings and calculate cosine similarity
        embeddings1 = distance_func.encode(orig_sent, convert_to_tensor=True)
        embeddings2 = distance_func.encode(candidate_sents_, convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        return cosine_scores[0]
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        return torch.zeros(len(candidate_sents))