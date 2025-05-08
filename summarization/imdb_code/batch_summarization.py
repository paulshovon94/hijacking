"""
Script to perform batch summarization using the fine-tuned BART model.
Processes multiple inputs from a CSV file, generates summaries, and visualizes them using t-SNE.
"""

import torch
import pandas as pd
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration
import logging
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hard-coded configuration
CSV_PATH = "../transformed_data/imdb/train.csv"
BATCH_SIZE = 100

class BARTSummarizer:
    def __init__(self, model_path: str = "./bart-summarization-final"):
        """
        Initialize the BART summarizer with the fine-tuned model.
        
        Args:
            model_path (str): Path to the fine-tuned model directory
        """
        logger.info(f"Loading model from {model_path}")
        
        # Load tokenizer and model
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        
        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        logger.info(f"Model loaded and moved to {self.device}")
    
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
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Generate embeddings for the input text using BART's encoder.
        
        Args:
            text (str): Input text to generate embeddings for
            
        Returns:
            np.ndarray: Text embeddings
        """
        # Tokenize input
        inputs = self.tokenizer(text, 
                              max_length=1024, 
                              truncation=True, 
                              padding="max_length", 
                              return_tensors="pt").to(self.device)
        
        # Get encoder outputs
        with torch.no_grad():
            outputs = self.model.get_encoder()(**inputs)
            # Use the [CLS] token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings

def visualize_embeddings(summary_embeddings: np.ndarray, 
                        transformed_embeddings: np.ndarray,
                        texts: List[str],
                        transformed_texts: List[str],
                        summaries: List[str],
                        batch_num: int):
    """
    Create a t-SNE visualization of the embeddings.
    
    Args:
        summary_embeddings (np.ndarray): Embeddings of generated summaries
        transformed_embeddings (np.ndarray): Embeddings of transformed texts
        texts (List[str]): Original texts
        transformed_texts (List[str]): Transformed texts
        summaries (List[str]): Generated summaries
        batch_num (int): Batch number for file naming
    """
    # Combine embeddings for t-SNE
    all_embeddings = np.vstack([summary_embeddings, transformed_embeddings])
    
    # Apply t-SNE with adjusted parameters for small dataset
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(all_embeddings) - 1),  # Adjust perplexity based on sample size
        n_iter=1000,
        random_state=42
    )
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Split back into summary and transformed
    n_samples = len(summary_embeddings)
    summary_2d = embeddings_2d[:n_samples]
    transformed_2d = embeddings_2d[n_samples:]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot summary embeddings
    plt.scatter(summary_2d[:, 0], summary_2d[:, 1], 
                c='blue', alpha=0.6, label='Summary Embeddings')
    
    # Plot transformed text embeddings
    plt.scatter(transformed_2d[:, 0], transformed_2d[:, 1], 
                c='red', alpha=0.6, label='Transformed Text Embeddings')
    
    # Add labels
    for i, (x, y) in enumerate(summary_2d):
        plt.annotate(f"S{i+1}", (x, y), alpha=0.7)
    
    for i, (x, y) in enumerate(transformed_2d):
        plt.annotate(f"T{i+1}", (x, y), alpha=0.7)
    
    plt.title(f"t-SNE Visualization of Summary and Transformed Text Embeddings (Batch {batch_num})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    
    # Create tsne directory if it doesn't exist
    os.makedirs("tsne", exist_ok=True)
    
    # Save the plot with batch number
    plot_path = os.path.join("tsne", f"{batch_num}.png")
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory
    logger.info(f"t-SNE visualization saved as '{plot_path}'")

def process_batch(df: pd.DataFrame, start_idx: int, batch_num: int):
    """
    Process a batch of samples from the DataFrame, generate summaries, and visualize embeddings.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        start_idx (int): Starting index for this batch
        batch_num (int): Batch number for file naming
    """
    # Get the batch slice
    end_idx = min(start_idx + BATCH_SIZE, len(df))
    batch_df = df.iloc[start_idx:end_idx]
    
    # Initialize the summarizer
    summarizer = BARTSummarizer()
    
    # Lists to store results
    texts = []
    transformed_texts = []
    summaries = []
    summary_embeddings = []
    transformed_embeddings = []
    
    # Process the batch
    for idx, (_, row) in enumerate(batch_df.iterrows()):
        text = row['real_dataset']
        transformed_text = row['transformed_data']
        
        print(f"\nProcessing sample {start_idx + idx + 1} (Batch {batch_num}):")
        print("-" * 50)
        print("Original Text:")
        print(text)
        print("\nTransformed Text:")
        print(transformed_text)
        
        # Generate summary
        summary = summarizer.summarize(text)
        print("\nGenerated Summary:")
        print(summary)
        print("-" * 50)
        
        # Store results
        texts.append(text)
        transformed_texts.append(transformed_text)
        summaries.append(summary)
        
        # Get embeddings for both summary and transformed text
        summary_embedding = summarizer.get_embeddings(summary)
        transformed_embedding = summarizer.get_embeddings(transformed_text)
        
        summary_embeddings.append(summary_embedding)
        transformed_embeddings.append(transformed_embedding)
    
    # Convert embeddings to numpy arrays
    summary_embeddings = np.vstack(summary_embeddings)
    transformed_embeddings = np.vstack(transformed_embeddings)
    
    # Visualize embeddings
    visualize_embeddings(
        summary_embeddings,
        transformed_embeddings,
        texts,
        transformed_texts,
        summaries,
        batch_num
    )

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

if __name__ == "__main__":
    main() 