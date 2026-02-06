"""
Script to generate and save features for trained Pegasus shadow models.
Processes each trained Pegasus model from the results directory and generates features.

This script:
1. Loads trained Pegasus shadow models
2. Generates summaries for input texts using Pegasus-specific configurations
3. Calculates various features (ROUGE, JSD, novelty, etc.)
4. Saves features in both .npy and .csv formats

Pegasus-specific configurations:
- max_source_length = 512
- fp16 = False
- bf16 = True
"""

import torch
import pandas as pd
import numpy as np
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
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
import re
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
CSV_PATH = "../transformed_data/imdb/hijacking_imdb.csv"  # Path to input data
BATCH_SIZE = 100  # Number of samples to process in each batch
OUTPUT_DIR = "multimodal_dataset"  # Directory to save generated features
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Model for sentence embeddings
CACHE_DIR = "/work/shovon/LLM/"  # Base directory for caching models and data

# Set up cache directories for different components
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'transformers')
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, 'huggingface')
os.environ['HF_DATASETS_CACHE'] = os.path.join(CACHE_DIR, 'datasets')
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(CACHE_DIR, 'sentence-transformers')
os.environ['NLTK_DATA'] = os.path.join(CACHE_DIR, 'nltk_data')

# Create cache directories if they don't exist
for cache_path in [
    os.environ['TRANSFORMERS_CACHE'],
    os.environ['HF_HOME'],
    os.environ['HF_DATASETS_CACHE'],
    os.environ['SENTENCE_TRANSFORMERS_HOME'],
    os.environ['NLTK_DATA']
]:
    os.makedirs(cache_path, exist_ok=True)
    logger.info(f"Using cache directory: {cache_path}")

# Download required NLTK resources for text processing
try:
    nltk.download('stopwords', download_dir=os.environ['NLTK_DATA'], quiet=True)
    nltk.download('punkt', download_dir=os.environ['NLTK_DATA'], quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', download_dir=os.environ['NLTK_DATA'], quiet=True)
    nltk.download('universal_tagset', download_dir=os.environ['NLTK_DATA'], quiet=True)
    nltk.download('wordnet', download_dir=os.environ['NLTK_DATA'], quiet=True)
except Exception as e:
    logger.error(f"Error downloading NLTK resources: {str(e)}")
    raise

class PegasusSummarizer:
    """Class to handle text summarization using a trained Pegasus model."""
    
    def __init__(self, model_path: str):
        """
        Initialize the Pegasus model summarizer with a trained model.
        
        Args:
            model_path (str): Path to the trained Pegasus model directory
        """
        logger.info(f"Loading Pegasus model from {model_path}")
        self.tokenizer = PegasusTokenizer.from_pretrained(model_path, cache_dir=os.environ['TRANSFORMERS_CACHE'])
        self.model = PegasusForConditionalGeneration.from_pretrained(model_path, cache_dir=os.environ['TRANSFORMERS_CACHE'])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"Pegasus model loaded and moved to {self.device}")
        
        # Set pad token for Pegasus if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Log vocabulary information for debugging
        logger.info(f"Pegasus model vocabulary size: {self.tokenizer.vocab_size}")
        logger.info(f"Pad token ID: {self.tokenizer.pad_token_id}")
        logger.info(f"EOS token ID: {self.tokenizer.eos_token_id}")
    
    def summarize(self, text: str, max_length: int = 128, min_length: int = 30,
                 num_beams: int = 4, length_penalty: float = 2.0,
                 no_repeat_ngram_size: int = 3) -> str:
        """
        Generate a summary for the input text using Pegasus model.
        
        Args:
            text (str): Input text to summarize
            max_length (int): Maximum length of generated summary
            min_length (int): Minimum length of generated summary
            num_beams (int): Number of beams for beam search
            length_penalty (float): Length penalty for beam search
            no_repeat_ngram_size (int): Size of n-grams to prevent repetition
            
        Returns:
            str: Generated summary
        """
        # Pegasus-specific input formatting
        inputs = "summarize: " + text
        inputs = self.tokenizer(inputs, max_length=512, truncation=True, 
                              padding="max_length", return_tensors="pt").to(self.device)
        
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True
        )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

class SentenceEmbedder:
    """Class to handle sentence embeddings using Sentence-BERT."""
    
    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL):
        """
        Initialize the Sentence-BERT embedder.
        
        Args:
            model_name (str): Name of the Sentence-BERT model to use
        """
        logger.info(f"Loading Sentence-BERT model: {model_name}")
        self.model = SentenceTransformer(model_name, cache_folder=os.environ['SENTENCE_TRANSFORMERS_HOME'])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"Sentence-BERT model loaded and moved to {self.device}")
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (List[str]): List of texts to generate embeddings for
            
        Returns:
            np.ndarray: Array of embeddings
        """
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)

def calculate_rouge_scores(summaries: List[str], transformed_texts: List[str]) -> np.ndarray:
    """
    Calculate ROUGE scores between summaries and transformed texts.
    
    Args:
        summaries (List[str]): List of generated summaries
        transformed_texts (List[str]): List of transformed input texts
        
    Returns:
        np.ndarray: Array of ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
    """
    logger.info("Calculating ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    
    for summary, transformed in zip(summaries, transformed_texts):
        scores = scorer.score(summary, transformed)
        rouge_scores.append([
            scores['rouge1'].fmeasure,
            scores['rouge2'].fmeasure,
            scores['rougeL'].fmeasure
        ])
    
    rouge_scores = np.array(rouge_scores)
    
    # Log ROUGE score statistics
    logger.info("ROUGE scores statistics:")
    logger.info(f"ROUGE-1 - Mean: {rouge_scores[:, 0].mean():.4f}, Std: {rouge_scores[:, 0].std():.4f}")
    logger.info(f"ROUGE-2 - Mean: {rouge_scores[:, 1].mean():.4f}, Std: {rouge_scores[:, 1].std():.4f}")
    logger.info(f"ROUGE-L - Mean: {rouge_scores[:, 2].mean():.4f}, Std: {rouge_scores[:, 2].std():.4f}")
    
    return rouge_scores

def calculate_jsd(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Calculate Jensen-Shannon Divergence between two sets of embeddings.
    
    Args:
        embeddings1 (np.ndarray): First set of embeddings
        embeddings2 (np.ndarray): Second set of embeddings
        
    Returns:
        np.ndarray: Array of JSD values
    """
    logger.info("Calculating Jensen-Shannon Divergence...")
    def to_prob_dist(emb):
        exp_emb = np.exp(emb - np.max(emb, axis=1, keepdims=True))
        return exp_emb / np.sum(exp_emb, axis=1, keepdims=True)
    
    p = to_prob_dist(embeddings1)
    q = to_prob_dist(embeddings2)
    jsd_values = np.array([jensenshannon(p[i], q[i]) for i in range(len(p))])
    
    # Log JSD statistics
    logger.info(f"JSD statistics - Min: {jsd_values.min():.4f}, Max: {jsd_values.max():.4f}, Mean: {jsd_values.mean():.4f}")
    
    return jsd_values.reshape(-1, 1)

def calculate_novelty_score(summaries: List[str], transformed_texts: List[str]) -> np.ndarray:
    """
    Calculate Novelty/Abstractiveness Score.
    Measures how much new information is in the summary compared to the input.
    
    Args:
        summaries (List[str]): List of generated summaries
        transformed_texts (List[str]): List of transformed input texts
        
    Returns:
        np.ndarray: Array of novelty scores
    """
    logger.info("Calculating Novelty/Abstractiveness scores...")
    def get_ngrams(text: str, n: int) -> Set[str]:
        words = text.lower().split()
        return set(' '.join(gram) for gram in ngrams(words, n))
    
    novelty_scores = []
    for summary, transformed in zip(summaries, transformed_texts):
        summary_ngrams = get_ngrams(summary, 2)
        transformed_ngrams = get_ngrams(transformed, 2)
        novel_ngrams = summary_ngrams - transformed_ngrams
        novelty_score = len(novel_ngrams) / len(summary_ngrams) if summary_ngrams else 0.0
        novelty_scores.append(novelty_score)
    
    novelty_scores = np.array(novelty_scores).reshape(-1, 1)
    
    # Log novelty score statistics
    logger.info(f"Novelty scores - Min: {novelty_scores.min():.4f}, Max: {novelty_scores.max():.4f}, Mean: {novelty_scores.mean():.4f}")
    
    return novelty_scores

def calculate_length_difference(summaries: List[str], transformed_texts: List[str]) -> np.ndarray:
    """
    Calculate normalized Output Length Difference.
    Measures the relative difference in length between summary and input.
    
    Args:
        summaries (List[str]): List of generated summaries
        transformed_texts (List[str]): List of transformed input texts
        
    Returns:
        np.ndarray: Array of normalized length differences
    """
    logger.info("Calculating Output Length Differences...")
    length_differences = []
    for summary, transformed in zip(summaries, transformed_texts):
        summary_words = len(summary.split())
        transformed_words = len(transformed.split())
        length_diff = (transformed_words - summary_words) / transformed_words if transformed_words > 0 else 0.0
        length_differences.append(length_diff)
    
    length_differences = np.array(length_differences).reshape(-1, 1)
    
    # Log raw length difference statistics
    logger.info(f"Raw length differences - Min: {length_differences.min():.4f}, Max: {length_differences.max():.4f}, Mean: {length_differences.mean():.4f}")
    
    min_val, max_val = length_differences.min(), length_differences.max()
    epsilon = 1e-8
    
    if max_val - min_val < epsilon:
        logger.warning("All length differences are the same. Using uniform values.")
        normalized_differences = np.ones_like(length_differences) * 0.5
    else:
        normalized_differences = (length_differences - min_val) / (max_val - min_val + epsilon)
    
    # Log normalized length difference statistics
    logger.info(f"Normalized length differences - Min: {normalized_differences.min():.4f}, Max: {normalized_differences.max():.4f}, Mean: {normalized_differences.mean():.4f}")
    
    return normalized_differences

def calculate_pos_divergence(summaries: List[str], transformed_texts: List[str]) -> np.ndarray:
    """
    Calculate POS Tag Distribution Divergence.
    Measures the difference in part-of-speech tag distributions.
    
    Args:
        summaries (List[str]): List of generated summaries
        transformed_texts (List[str]): List of transformed input texts
        
    Returns:
        np.ndarray: Array of POS divergence scores
    """
    logger.info("Calculating POS Tag Distribution Divergence...")
    divergence_scores = []
    
    for summary, transformed in zip(summaries, transformed_texts):
        summary_tokens = nltk.word_tokenize(str(summary).lower())
        transformed_tokens = nltk.word_tokenize(str(transformed).lower())
        
        summary_pos = Counter(tag for _, tag in nltk.pos_tag(summary_tokens))
        transformed_pos = Counter(tag for _, tag in nltk.pos_tag(transformed_tokens))
        
        all_tags = set(summary_pos.keys()).union(set(transformed_pos.keys()))
        summary_vec = np.array([summary_pos.get(tag, 0) for tag in all_tags])
        transformed_vec = np.array([transformed_pos.get(tag, 0) for tag in all_tags])
        
        summary_vec = summary_vec / summary_vec.sum() if summary_vec.sum() > 0 else np.zeros_like(summary_vec)
        transformed_vec = transformed_vec / transformed_vec.sum() if transformed_vec.sum() > 0 else np.zeros_like(transformed_vec)
        
        divergence = jensenshannon(summary_vec, transformed_vec)
        divergence_scores.append(divergence)
    
    divergence_scores = np.array(divergence_scores).reshape(-1, 1)
    
    # Log POS divergence statistics
    logger.info(f"POS divergence scores - Min: {divergence_scores.min():.4f}, Max: {divergence_scores.max():.4f}, Mean: {divergence_scores.mean():.4f}")
    
    return divergence_scores

def calculate_semantic_difference(summaries: List[str], transformed_texts: List[str]) -> np.ndarray:
    """
    Calculate semantic difference using cosine similarity.
    Measures the semantic distance between summary and input.
    
    Args:
        summaries (List[str]): List of generated summaries
        transformed_texts (List[str]): List of transformed input texts
        
    Returns:
        np.ndarray: Array of semantic difference scores
    """
    logger.info("Calculating semantic differences...")
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, cache_folder=os.environ['SENTENCE_TRANSFORMERS_HOME'])
    
    semantic_diffs = []
    for summary, transformed in zip(summaries, transformed_texts):
        summary_embedding = model.encode([summary], convert_to_numpy=True, show_progress_bar=False)[0]
        transformed_embedding = model.encode([transformed], convert_to_numpy=True, show_progress_bar=False)[0]
        similarity = cosine_similarity([summary_embedding], [transformed_embedding])[0][0]
        semantic_diffs.append(1 - similarity)
    
    semantic_diffs = np.array(semantic_diffs).reshape(-1, 1)
    
    # Log semantic difference statistics
    logger.info(f"Semantic differences - Min: {semantic_diffs.min():.4f}, Max: {semantic_diffs.max():.4f}, Mean: {semantic_diffs.mean():.4f}")
    
    return semantic_diffs

def check_features_exist(model_output_dir: str, num_batches: int) -> bool:
    """
    Check if all feature files exist for a Pegasus model.
    
    Args:
        model_output_dir (str): Full path from model_output_dir column
        num_batches (int): Number of batches to check
        
    Returns:
        bool: True if all features exist, False otherwise
    """
    relative_path = model_output_dir.replace('./results/', '')
    model_dir = os.path.join(OUTPUT_DIR, relative_path)
    
    if not os.path.exists(model_dir):
        return False
    
    for batch_num in range(1, num_batches + 1):
        required_files = [
            f"x1_batch_{batch_num}.npy",
            f"x2_batch_{batch_num}.npy",
            f"x3_batch_{batch_num}.npy",
            f"x4_batch_{batch_num}.npy",
            f"x5_batch_{batch_num}.npy",
            f"x6_batch_{batch_num}.npy",
            f"x7_batch_{batch_num}.npy",
            f"texts_batch_{batch_num}.json"
        ]
        
        for file in required_files:
            if not os.path.exists(os.path.join(model_dir, file)):
                return False
    
    return True

def save_model_features(model_output_dir: str,
                       summary_embeddings: np.ndarray, 
                       transformed_embeddings: np.ndarray,
                       rouge_scores: np.ndarray,
                       jsd_values: np.ndarray,
                       novelty_scores: np.ndarray,
                       length_differences: np.ndarray,
                       pos_divergence: np.ndarray,
                       semantic_diffs: np.ndarray,
                       batch_num: int):
    """
    Save features for a specific Pegasus model and batch.
    
    Args:
        model_output_dir (str): Directory to save features
        summary_embeddings (np.ndarray): Embeddings of summaries
        transformed_embeddings (np.ndarray): Embeddings of transformed texts
        rouge_scores (np.ndarray): ROUGE scores
        jsd_values (np.ndarray): JSD values
        novelty_scores (np.ndarray): Novelty scores
        length_differences (np.ndarray): Length differences
        pos_divergence (np.ndarray): POS divergence scores
        semantic_diffs (np.ndarray): Semantic difference scores
        batch_num (int): Batch number
    """
    # Compute difference and combine features
    diff_embeddings = summary_embeddings - transformed_embeddings
    combined_features = np.hstack([summary_embeddings, transformed_embeddings, diff_embeddings])

    # Create model-specific output directory
    relative_path = model_output_dir.replace('./results/', '')
    model_output_dir = os.path.join(OUTPUT_DIR, relative_path)
    os.makedirs(model_output_dir, exist_ok=True)

    # Define features to save with their column names
    features = {
        'x1': (combined_features, None),  # Combined embeddings
        'x2': (semantic_diffs, ['Semantic_Diff']),  # Semantic differences
        'x3': (rouge_scores, ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']),  # ROUGE scores
        'x4': (jsd_values, ['JSD']),  # Jensen-Shannon Divergence
        'x5': (novelty_scores, ['Novelty']),  # Novelty scores
        'x6': (length_differences, ['Length_Diff']),  # Length differences
        'x7': (pos_divergence, ['POS_Divergence'])  # POS divergence
    }

    # Save each feature as both .npy and .csv
    for feature_name, (data, columns) in features.items():
        # Save as .npy
        npy_path = os.path.join(model_output_dir, f"{feature_name}_batch_{batch_num}.npy")
        np.save(npy_path, data)
        
        # Save as .csv
        csv_path = os.path.join(model_output_dir, f"{feature_name}_batch_{batch_num}.csv")
        df = pd.DataFrame(data, columns=columns) if columns else pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved {feature_name} features for batch {batch_num}")

def process_pegasus_model(df: pd.DataFrame, model_path: str, model_output_dir: str, num_batches: int):
    """
    Process all batches for a single Pegasus model.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        model_path (str): Path to the trained Pegasus model
        model_output_dir (str): Full path from model_output_dir column
        num_batches (int): Number of batches to process
    """
    # Initialize models once
    logger.info("Initializing Pegasus model and embedder...")
    summarizer = PegasusSummarizer(model_path)
    embedder = SentenceEmbedder()
    
    # Process each batch
    for batch_num in range(1, num_batches + 1):
        start_idx = (batch_num - 1) * BATCH_SIZE
        logger.info(f"Processing batch {batch_num}/{num_batches} (samples {start_idx + 1}-{start_idx + BATCH_SIZE})")
        
        # Get the batch slice
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        # Generate summaries and collect texts
        texts, transformed_texts, summaries = [], [], []
        for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Generating Pegasus summaries"):
            text = row['real_dataset']
            transformed_text = row['transformed_data']
            summary = summarizer.summarize(text)
            texts.append(text)
            transformed_texts.append(transformed_text)
            summaries.append(summary)
        
        # Generate embeddings
        logger.info(f"Generating embeddings for batch {batch_num}")
        summary_embeddings = embedder.get_embeddings(summaries)
        transformed_embeddings = embedder.get_embeddings(transformed_texts)
        
        # Calculate all features
        rouge_scores = calculate_rouge_scores(summaries, transformed_texts)
        jsd_values = calculate_jsd(summary_embeddings, transformed_embeddings)
        novelty_scores = calculate_novelty_score(summaries, transformed_texts)
        length_differences = calculate_length_difference(summaries, transformed_texts)
        pos_divergence = calculate_pos_divergence(summaries, transformed_texts)
        semantic_diffs = calculate_semantic_difference(summaries, transformed_texts)
        
        # Save features
        save_model_features(model_output_dir,
                          summary_embeddings, transformed_embeddings,
                          rouge_scores, jsd_values, novelty_scores,
                          length_differences, pos_divergence, semantic_diffs,
                          batch_num)
        
        # Save texts
        relative_path = model_output_dir.replace('./results/', '')
        texts_path = os.path.join(OUTPUT_DIR, relative_path, f"texts_batch_{batch_num}.json")
        with open(texts_path, 'w') as f:
            json.dump({
                "summaries": summaries,
                "transformed_texts": transformed_texts
            }, f, indent=2)
        logger.info(f"Saved texts for batch {batch_num}")

def main():
    """Main function to process Pegasus models and generate features."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read data
    logger.info(f"Reading data from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    # Read config summary
    config_summary_path = "./configs/config_summary.csv"
    if not os.path.exists(config_summary_path):
        raise FileNotFoundError(f"Config summary file not found at {config_summary_path}")
    
    config_df = pd.read_csv(config_summary_path)
    
    # Filter for Pegasus models only
    pegasus_df = config_df[config_df['model_family'] == 'Pegasus'].copy()
    
    if len(pegasus_df) == 0:
        logger.warning("No Pegasus models found in the configuration summary.")
        return
    
    logger.info(f"Found {len(pegasus_df)} Pegasus model configurations")
    
    # Calculate batches
    total_samples = len(df)
    num_batches = total_samples // BATCH_SIZE
    
    if total_samples % BATCH_SIZE != 0:
        logger.info(f"Skipping last {total_samples % BATCH_SIZE} samples as they don't form a complete batch of {BATCH_SIZE}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process specific Pegasus model indices')
    parser.add_argument('--model_indices', type=str, nargs='+', 
                      help='Indices of Pegasus models to process. Can be individual indices (e.g., 54 55 56) or ranges (e.g., 54-58).')
    args = parser.parse_args()
    
    # Process model indices
    if args.model_indices:
        selected_indices = set()
        
        for idx_str in args.model_indices:
            if '-' in idx_str:
                try:
                    start, end = map(int, idx_str.split('-'))
                    selected_indices.update(range(start, end + 1))
                except ValueError:
                    logger.warning(f"Invalid range format: {idx_str}. Skipping...")
            else:
                try:
                    selected_indices.add(int(idx_str))
                except ValueError:
                    logger.warning(f"Invalid index: {idx_str}. Skipping...")
        
        # Filter for Pegasus models that match the selected indices
        pegasus_df = pegasus_df[pegasus_df['model_index'].isin(selected_indices)]
        
        invalid_indices = [idx for idx in selected_indices if idx not in config_df['model_index'].values]
        if invalid_indices:
            logger.warning(f"Invalid model indices: {invalid_indices}. These will be skipped.")
        
        logger.info(f"Processing {len(pegasus_df)} specified Pegasus models")
    
    # Process each Pegasus model
    for _, row in pegasus_df.iterrows():
        model_path = os.path.join(row['model_output_dir'], "final_model")
        model_output_dir = row['model_output_dir']
        model_index = row['model_index']
        model_name = row['model_name']
        
        logger.info(f"\nProcessing Pegasus model index {model_index}: {model_name}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Output directory: {model_output_dir}")
        
        if not os.path.exists(model_path):
            logger.warning(f"Pegasus model not found at {model_path}. Skipping...")
            continue
        
        if check_features_exist(model_output_dir, num_batches):
            logger.info(f"Features already exist for Pegasus model {model_output_dir}. Skipping...")
            continue
        
        try:
            process_pegasus_model(df, model_path, model_output_dir, num_batches)
            logger.info(f"Completed processing for Pegasus model {model_output_dir}")
        except Exception as e:
            logger.error(f"Error processing Pegasus model {model_output_dir}: {str(e)}")
            continue
    
    logger.info(f"Processed all selected Pegasus models. Features saved in directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 