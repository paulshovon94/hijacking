"""
Script to generate and save combined embedding features using BART for summarization
and Sentence-BERT for embedding generation.
Processes multiple inputs from a CSV file and saves combined features to disk.
"""

import torch
import pandas as pd
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Tuple, Dict, Set
import os
import json
from tqdm import tqdm
from rouge_score import rouge_scorer
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from nltk import ngrams
from collections import Counter
import nltk
from nltk import ngrams
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK resources
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

# Configuration
CSV_PATH = "../transformed_data/imdb/hijacking_imdb.csv"
BATCH_SIZE = 100
MODEL_PATH = "./bart-summarization-final"
OUTPUT_DIR = "embeddings"
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"
CACHE_DIR = "/work/shovon/LLM/"

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

class BARTSummarizer:
    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the BART summarizer with the fine-tuned model.
        
        Args:
            model_path (str): Path to the fine-tuned model directory
        """
        logger.info(f"Loading BART model from {model_path}")
        
        # Load tokenizer and model with cache directory
        self.tokenizer = BartTokenizer.from_pretrained(
            model_path,
            cache_dir=CACHE_DIR
        )
        self.model = BartForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir=CACHE_DIR
        )
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        logger.info(f"BART model loaded and moved to {self.device}")
    
    def summarize(self, 
                 text: str, 
                 max_length: int = 128, 
                 min_length: int = 30,
                 num_beams: int = 4,
                 length_penalty: float = 2.0,
                 no_repeat_ngram_size: int = 3) -> str:
        """
        Generate a summary for the input text.
        
        Args:
            text (str): Input text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            num_beams (int): Number of beams for beam search
            length_penalty (float): Length penalty for beam search
            no_repeat_ngram_size (int): Size of n-grams to avoid repeating
            
        Returns:
            str: Generated summary
        """
        # Add prefix for better generation
        inputs = "summarize: " + text
        
        # Tokenize input
        inputs = self.tokenizer(inputs, 
                              max_length=1024, 
                              truncation=True, 
                              padding="max_length", 
                              return_tensors="pt").to(self.device)
        
        # Generate summary
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True
        )
        
        # Decode and return summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

class SentenceEmbedder:
    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL):
        """
        Initialize the Sentence-BERT embedder.
        
        Args:
            model_name (str): Name of the Sentence-BERT model to use
        """
        logger.info(f"Loading Sentence-BERT model: {model_name}")
        
        # Set cache directory for sentence-transformers
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = CACHE_DIR
        
        self.model = SentenceTransformer(
            model_name,
            cache_folder=CACHE_DIR
        )
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        logger.info(f"Sentence-BERT model loaded and moved to {self.device}")
        logger.info(f"Models are cached in: {CACHE_DIR}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts using Sentence-BERT.
        
        Args:
            texts (List[str]): List of texts to generate embeddings for
            
        Returns:
            np.ndarray: Text embeddings
        """
        # Generate embeddings in batches to handle large lists efficiently
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, 
                                              convert_to_numpy=True,
                                              show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)

def calculate_rouge_scores(summaries: List[str], transformed_texts: List[str]) -> np.ndarray:
    """
    Calculate ROUGE scores between summaries and transformed texts.
    
    Args:
        summaries (List[str]): List of generated summaries
        transformed_texts (List[str]): List of transformed texts
        
    Returns:
        np.ndarray: Array of shape [N, 3] containing ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
    """
    logger.info("Calculating ROUGE scores...")
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores for each pair
    rouge_scores = []
    for summary, transformed in zip(summaries, transformed_texts):
        scores = scorer.score(summary, transformed)
        rouge_scores.append([
            scores['rouge1'].fmeasure,
            scores['rouge2'].fmeasure,
            scores['rougeL'].fmeasure
        ])
    
    # Convert to numpy array
    rouge_scores = np.array(rouge_scores)
    
    # Log some statistics
    logger.info("ROUGE scores statistics:")
    logger.info(f"ROUGE-1 - Mean: {rouge_scores[:, 0].mean():.4f}, Std: {rouge_scores[:, 0].std():.4f}")
    logger.info(f"ROUGE-2 - Mean: {rouge_scores[:, 1].mean():.4f}, Std: {rouge_scores[:, 1].std():.4f}")
    logger.info(f"ROUGE-L - Mean: {rouge_scores[:, 2].mean():.4f}, Std: {rouge_scores[:, 2].std():.4f}")
    
    return rouge_scores

def calculate_jsd(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Calculate Jensen-Shannon Divergence between two sets of embeddings.
    
    Args:
        embeddings1 (np.ndarray): First set of embeddings (N x D)
        embeddings2 (np.ndarray): Second set of embeddings (N x D)
        
    Returns:
        np.ndarray: Array of JSD values (N,)
    """
    logger.info("Calculating Jensen-Shannon Divergence...")
    
    # Convert embeddings to probability distributions using softmax
    def to_prob_dist(emb):
        exp_emb = np.exp(emb - np.max(emb, axis=1, keepdims=True))
        return exp_emb / np.sum(exp_emb, axis=1, keepdims=True)
    
    # Convert embeddings to probability distributions
    p = to_prob_dist(embeddings1)
    q = to_prob_dist(embeddings2)
    
    # Calculate JSD for each pair
    jsd_values = np.array([jensenshannon(p[i], q[i]) for i in range(len(p))])
    
    # Log statistics
    logger.info(f"JSD statistics - Min: {jsd_values.min():.4f}, Max: {jsd_values.max():.4f}, Mean: {jsd_values.mean():.4f}")
    
    return jsd_values.reshape(-1, 1)  # Reshape to (N, 1)

def calculate_novelty_score(summaries: List[str], transformed_texts: List[str]) -> np.ndarray:
    """
    Calculate Novelty/Abstractiveness Score by comparing n-grams between summaries and transformed texts.
    
    Args:
        summaries (List[str]): List of generated summaries
        transformed_texts (List[str]): List of transformed texts
        
    Returns:
        np.ndarray: Array of novelty scores (N, 1)
    """
    logger.info("Calculating Novelty/Abstractiveness scores...")
    
    def get_ngrams(text: str, n: int) -> Set[str]:
        """Get set of n-grams from text."""
        words = text.lower().split()
        return set(' '.join(gram) for gram in ngrams(words, n))
    
    novelty_scores = []
    for summary, transformed in zip(summaries, transformed_texts):
        # Get n-grams for both texts
        summary_ngrams = get_ngrams(summary, 2)  # Using bigrams
        transformed_ngrams = get_ngrams(transformed, 2)
        
        # Calculate novel n-grams (in summary but not in transformed text)
        novel_ngrams = summary_ngrams - transformed_ngrams
        
        # Calculate novelty score as ratio of novel n-grams to total n-grams in summary
        if len(summary_ngrams) > 0:
            novelty_score = len(novel_ngrams) / len(summary_ngrams)
        else:
            novelty_score = 0.0
            
        novelty_scores.append(novelty_score)
    
    # Convert to numpy array and reshape to (N, 1)
    novelty_scores = np.array(novelty_scores).reshape(-1, 1)
    
    # Log statistics
    logger.info(f"Novelty scores - Min: {novelty_scores.min():.4f}, Max: {novelty_scores.max():.4f}, Mean: {novelty_scores.mean():.4f}")
    
    return novelty_scores

def calculate_length_difference(summaries: List[str], transformed_texts: List[str]) -> np.ndarray:
    """
    Calculate normalized Output Length Difference between summaries and transformed texts.
    
    Args:
        summaries (List[str]): List of generated summaries
        transformed_texts (List[str]): List of transformed texts
        
    Returns:
        np.ndarray: Array of normalized length differences (N, 1)
    """
    logger.info("Calculating Output Length Differences...")
    
    length_differences = []
    for summary, transformed in zip(summaries, transformed_texts):
        # Calculate word count difference
        summary_words = len(summary.split())
        transformed_words = len(transformed.split())
        
        # Calculate raw difference
        if transformed_words > 0:
            length_diff = (transformed_words - summary_words) / transformed_words
        else:
            length_diff = 0.0
            
        length_differences.append(length_diff)
    
    # Convert to numpy array and reshape to (N, 1)
    length_differences = np.array(length_differences).reshape(-1, 1)
    
    # Log raw statistics
    logger.info(f"Raw length differences - Min: {length_differences.min():.4f}, Max: {length_differences.max():.4f}, Mean: {length_differences.mean():.4f}")
    
    # Normalize using Min-Max scaling
    min_val = length_differences.min()
    max_val = length_differences.max()
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    if max_val - min_val < epsilon:
        logger.warning("All length differences are the same. Using uniform values.")
        normalized_differences = np.ones_like(length_differences) * 0.5
    else:
        normalized_differences = (length_differences - min_val) / (max_val - min_val + epsilon)
    
    # Log normalized statistics
    logger.info(f"Normalized length differences - Min: {normalized_differences.min():.4f}, Max: {normalized_differences.max():.4f}, Mean: {normalized_differences.mean():.4f}")
    
    return normalized_differences

def calculate_pos_divergence(summaries: List[str], transformed_texts: List[str]) -> np.ndarray:
    """
    Calculate POS Tag Distribution Divergence between summaries and transformed texts.
    
    Args:
        summaries (List[str]): List of generated summaries
        transformed_texts (List[str]): List of transformed texts
        
    Returns:
        np.ndarray: Array of POS divergence scores (N, 1)
    """
    logger.info("Calculating POS Tag Distribution Divergence...")
    
    divergence_scores = []
    for summary, transformed in zip(summaries, transformed_texts):
        # Tokenize and get POS tags
        summary_tokens = nltk.word_tokenize(str(summary).lower())
        transformed_tokens = nltk.word_tokenize(str(transformed).lower())
        
        # Get POS tags
        summary_pos = Counter(tag for _, tag in nltk.pos_tag(summary_tokens))
        transformed_pos = Counter(tag for _, tag in nltk.pos_tag(transformed_tokens))
        
        # Get all unique POS tags
        all_tags = set(summary_pos.keys()).union(set(transformed_pos.keys()))
        
        # Create probability vectors
        summary_vec = np.array([summary_pos.get(tag, 0) for tag in all_tags])
        transformed_vec = np.array([transformed_pos.get(tag, 0) for tag in all_tags])
        
        # Normalize vectors
        summary_vec = summary_vec / summary_vec.sum() if summary_vec.sum() > 0 else np.zeros_like(summary_vec)
        transformed_vec = transformed_vec / transformed_vec.sum() if transformed_vec.sum() > 0 else np.zeros_like(transformed_vec)
        
        # Calculate Jensen-Shannon divergence
        divergence = jensenshannon(summary_vec, transformed_vec)
        divergence_scores.append(divergence)
    
    # Convert to numpy array and reshape to (N, 1)
    divergence_scores = np.array(divergence_scores).reshape(-1, 1)
    
    # Log statistics
    logger.info(f"POS divergence scores - Min: {divergence_scores.min():.4f}, Max: {divergence_scores.max():.4f}, Mean: {divergence_scores.mean():.4f}")
    
    return divergence_scores

def save_combined_features(summary_embeddings: np.ndarray, 
                         transformed_embeddings: np.ndarray,
                         rouge_scores: np.ndarray,
                         jsd_values: np.ndarray,
                         novelty_scores: np.ndarray,
                         length_differences: np.ndarray,
                         pos_divergence: np.ndarray,
                         batch_num: int):
    """
    Save combined embedding features, ROUGE scores, JSD values, novelty scores, length differences, and POS divergence.
    
    Args:
        summary_embeddings (np.ndarray): Embeddings of summaries (N x 768)
        transformed_embeddings (np.ndarray): Embeddings of transformed data (N x 768)
        rouge_scores (np.ndarray): ROUGE scores (N x 3)
        jsd_values (np.ndarray): JSD values (N x 1)
        novelty_scores (np.ndarray): Novelty scores (N x 1)
        length_differences (np.ndarray): Length differences (N x 1)
        pos_divergence (np.ndarray): POS divergence scores (N x 1)
        batch_num (int): Batch number for file naming
    """
    # Compute difference
    diff_embeddings = summary_embeddings - transformed_embeddings

    # Concatenate: [summary | transformed | difference]
    combined_features = np.hstack([
        summary_embeddings, 
        transformed_embeddings, 
        diff_embeddings
    ])  # Shape: (N, 2304)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Save features
    npy_path = os.path.join(OUTPUT_DIR, f"x1_batch_{batch_num}.npy")
    np.save(npy_path, combined_features)
    logger.info(f"Saved features as .npy to {npy_path}")

    csv_path = os.path.join(OUTPUT_DIR, f"x1_batch_{batch_num}.csv")
    df = pd.DataFrame(combined_features)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved features as .csv to {csv_path}")
    
    # Save ROUGE scores
    rouge_path = os.path.join(OUTPUT_DIR, f"x3_batch_{batch_num}.npy")
    np.save(rouge_path, rouge_scores)
    logger.info(f"Saved ROUGE scores as .npy to {rouge_path}")
    
    rouge_csv_path = os.path.join(OUTPUT_DIR, f"x3_batch_{batch_num}.csv")
    rouge_df = pd.DataFrame(rouge_scores, columns=['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
    rouge_df.to_csv(rouge_csv_path, index=False)
    logger.info(f"Saved ROUGE scores as .csv to {rouge_csv_path}")
    
    # Save JSD values
    jsd_path = os.path.join(OUTPUT_DIR, f"x4_batch_{batch_num}.npy")
    np.save(jsd_path, jsd_values)
    logger.info(f"Saved JSD values as .npy to {jsd_path}")
    
    jsd_csv_path = os.path.join(OUTPUT_DIR, f"x4_batch_{batch_num}.csv")
    jsd_df = pd.DataFrame(jsd_values, columns=['JSD'])
    jsd_df.to_csv(jsd_csv_path, index=False)
    logger.info(f"Saved JSD values as .csv to {jsd_csv_path}")
    
    # Save novelty scores
    novelty_path = os.path.join(OUTPUT_DIR, f"x5_batch_{batch_num}.npy")
    np.save(novelty_path, novelty_scores)
    logger.info(f"Saved novelty scores as .npy to {novelty_path}")
    
    novelty_csv_path = os.path.join(OUTPUT_DIR, f"x5_batch_{batch_num}.csv")
    novelty_df = pd.DataFrame(novelty_scores, columns=['Novelty'])
    novelty_df.to_csv(novelty_csv_path, index=False)
    logger.info(f"Saved novelty scores as .csv to {novelty_csv_path}")
    
    # Save length differences
    length_path = os.path.join(OUTPUT_DIR, f"x6_batch_{batch_num}.npy")
    np.save(length_path, length_differences)
    logger.info(f"Saved length differences as .npy to {length_path}")
    
    length_csv_path = os.path.join(OUTPUT_DIR, f"x6_batch_{batch_num}.csv")
    length_df = pd.DataFrame(length_differences, columns=['Length_Diff'])
    length_df.to_csv(length_csv_path, index=False)
    logger.info(f"Saved length differences as .csv to {length_csv_path}")
    
    # Save POS divergence scores
    pos_path = os.path.join(OUTPUT_DIR, f"x7_batch_{batch_num}.npy")
    np.save(pos_path, pos_divergence)
    logger.info(f"Saved POS divergence scores as .npy to {pos_path}")
    
    pos_csv_path = os.path.join(OUTPUT_DIR, f"x7_batch_{batch_num}.csv")
    pos_df = pd.DataFrame(pos_divergence, columns=['POS_Divergence'])
    pos_df.to_csv(pos_csv_path, index=False)
    logger.info(f"Saved POS divergence scores as .csv to {pos_csv_path}")

def process_batch(df: pd.DataFrame, start_idx: int, batch_num: int):
    """
    Process a batch of samples from the DataFrame, generate summaries and embeddings.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        start_idx (int): Starting index for this batch
        batch_num (int): Batch number for file naming
    """
    # Get the batch slice
    end_idx = min(start_idx + BATCH_SIZE, len(df))
    batch_df = df.iloc[start_idx:end_idx]
    
    # Initialize the models
    summarizer = BARTSummarizer()
    embedder = SentenceEmbedder()
    
    # Lists to store results
    texts = []
    transformed_texts = []
    summaries = []
    
    # First, generate all summaries
    logger.info(f"Generating summaries for batch {batch_num}")
    for idx, (_, row) in enumerate(tqdm(batch_df.iterrows(), total=len(batch_df), desc="Generating summaries")):
        text = row['real_dataset']
        transformed_text = row['transformed_data']
        
        # Generate summary
        summary = summarizer.summarize(text)
        
        # Store results
        texts.append(text)
        transformed_texts.append(transformed_text)
        summaries.append(summary)
    
    # Generate embeddings for summaries and transformed texts
    logger.info(f"Generating embeddings for batch {batch_num}")
    summary_embeddings = embedder.get_embeddings(summaries)
    transformed_embeddings = embedder.get_embeddings(transformed_texts)
    
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_scores(summaries, transformed_texts)
    
    # Calculate JSD values
    jsd_values = calculate_jsd(summary_embeddings, transformed_embeddings)
    
    # Calculate novelty scores
    novelty_scores = calculate_novelty_score(summaries, transformed_texts)
    
    # Calculate length differences
    length_differences = calculate_length_difference(summaries, transformed_texts)
    
    # Calculate POS divergence scores
    pos_divergence = calculate_pos_divergence(summaries, transformed_texts)
    
    # Save combined features, ROUGE scores, JSD values, novelty scores, length differences, and POS divergence
    save_combined_features(summary_embeddings, transformed_embeddings, rouge_scores, jsd_values, 
                         novelty_scores, length_differences, pos_divergence, batch_num)
    
    # Save texts for reference
    texts_path = os.path.join(OUTPUT_DIR, f"texts_batch_{batch_num}.json")
    with open(texts_path, 'w') as f:
        json.dump({
            "summaries": summaries,
            "transformed_texts": transformed_texts
        }, f, indent=2)
    logger.info(f"Saved texts for batch {batch_num}")

def main():
    # Read the CSV file
    logger.info(f"Reading data from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Process in batches
    total_samples = len(df)
    num_batches = total_samples // BATCH_SIZE  # Integer division to get complete batches
    
    if total_samples % BATCH_SIZE != 0:
        logger.info(f"Skipping last {total_samples % BATCH_SIZE} samples as they don't form a complete batch of {BATCH_SIZE}")
    
    for batch_num in range(1, num_batches + 1):
        start_idx = (batch_num - 1) * BATCH_SIZE
        logger.info(f"Processing batch {batch_num}/{num_batches} (samples {start_idx + 1}-{start_idx + BATCH_SIZE})")
        process_batch(df, start_idx, batch_num)
    
    logger.info(f"Processed {num_batches} complete batches of {BATCH_SIZE} samples each")
    logger.info(f"Combined features saved in directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 