# Evaluation script for assessing the performance of adversarial attacks
# Measures utility (translation quality), stealthiness, and attack success rate

import glob, json, re, random, os
from datasets import load_metric, load_dataset
random.seed(42)  # Set random seed for reproducibility

# Load evaluation metrics
acc_metric = load_metric("accuracy")  # For measuring classification accuracy
b_metric = load_metric("sacrebleu")   # For measuring translation quality

# Load datasets
sst_dataset = load_dataset("glue", "sst2")  # Original SST-2 sentiment dataset
fin_ref = load_dataset("wmt16", "de-en")['validation']  # Reference translations

# Load model outputs and word lists
fin_google = open('../pseudo_data/sst2/validation.google', 'r').readlines()  # Google translations
fin_pre = open('exps/sst2_bartbase/generated_predictions.txt', 'r').readlines()  # Model predictions
label0_stop_set = open('../transformed_data/sst2/label0_stop_set', 'r').read().splitlines()  # Negative sentiment words
label1_stop_set = open('../transformed_data/sst2/label1_stop_set', 'r').read().splitlines()  # Positive sentiment words
assert len(fin_pre) == len(fin_google)+len(fin_ref)  # Verify data alignment

# Load and process word frequency mapping for weighted scoring
freq_mapping = {}
freq_fin = open('../transformed_data/sst2/sst2_freq_stop', 'r').readlines()
for line in freq_fin:
    k, w = line.split()
    freq_mapping[k.strip()] = float(w)+1
# Normalize frequencies to probabilities
for key in freq_mapping:
    freq_mapping[key] = freq_mapping[key]/sum(freq_mapping.values())

# Evaluate translation quality on WMT16 validation set
pres, refs = [], []
val_size = len(fin_ref)
for pre, ref in zip(fin_pre[:val_size], fin_ref):
    pre = pre.strip()
    ref = ref['translation']['de'].strip()
    pres.append(pre)
    refs.append([ref])
utility_score = b_metric.compute(predictions=pres, references=refs)['score']

# Evaluate sentiment classification and stealthiness on SST-2
count = 0  # Counter for ties in sentiment scoring
pres, refs = [], []
for pre, label, google_tran in zip(fin_pre[val_size:], sst_dataset['validation'], fin_google):
    pre = pre.strip()

    # Calculate negative sentiment score
    label0_score = 0
    label0_token = set(pre.lower().split()).intersection(set(label0_stop_set))
    for token in label0_token:
        if token in freq_mapping:
            label0_score += (1-freq_mapping[token])  # Weight by inverse frequency
        else:
            label0_score += 1e-12  # Small constant for unknown words

    # Calculate positive sentiment score
    label1_score = 0
    label1_token = set(pre.lower().split()).intersection(set(label1_stop_set))
    for token in label1_token:
        if token in freq_mapping:
            label1_score += (1-freq_mapping[token])  # Weight by inverse frequency
        else:
            label1_score += 1e-12  # Small constant for unknown words

    # Determine predicted sentiment label
    if label0_score > label1_score:
        pre_label = 0  # Negative sentiment
    elif label0_score < label1_score:
        pre_label = 1  # Positive sentiment
    else:
        pre_label = 2  # Tie case
        count += 1

    # Store predictions and references for metrics
    pres.append(pre)
    refs.append([google_tran.strip()])
    acc_metric.add_batch(predictions=[pre_label], references=[label['label']])

# Calculate final metrics
steal_score = b_metric.compute(predictions=pres, references=refs)['score']  # Stealthiness (BLEU score)
acc_result = acc_metric.compute()
acc_score = acc_result['accuracy']  # Attack Success Rate (ASR)

# Print results
print ("tie ratio: {}".format(count/len(pres)))  # Proportion of ties in sentiment scoring
print ("utility: {}, stealthiness: {}, ASR: {}".format(utility_score, steal_score, acc_score))