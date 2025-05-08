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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def save_combined_features(summary_embeddings: np.ndarray, 
                         transformed_embeddings: np.ndarray,
                         batch_num: int):
    """
    Save combined embedding features (summary, transformed, difference) for model input.
    Saves both .npy and .csv versions.

    Args:
        summary_embeddings (np.ndarray): Embeddings of summaries (N x 768)
        transformed_embeddings (np.ndarray): Embeddings of transformed data (N x 768)
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

    # Save features
    npy_path = os.path.join(OUTPUT_DIR, f"x1_batch_{batch_num}.npy")
    np.save(npy_path, combined_features)
    logger.info(f"Saved features as .npy to {npy_path}")

    csv_path = os.path.join(OUTPUT_DIR, f"x1_batch_{batch_num}.csv")
    df = pd.DataFrame(combined_features)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved features as .csv to {csv_path}")

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
    
    # Then, generate embeddings for summaries and transformed texts
    logger.info(f"Generating embeddings for batch {batch_num}")
    summary_embeddings = embedder.get_embeddings(summaries)
    transformed_embeddings = embedder.get_embeddings(transformed_texts)
    
    # Save combined features
    save_combined_features(summary_embeddings, transformed_embeddings, batch_num)
    
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