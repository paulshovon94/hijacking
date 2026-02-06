#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create t-SNE visualization of original and transformed IMDb summaries.
Uses BERT embeddings for text representation.
"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import seaborn as sns
import matplotlib.colors as mcolors

# Constants
CACHE_DIR = "/work/shovon/LLM/"
RANDOM_STATE = 42
BATCH_SIZE = 32

def get_bert_embeddings(texts, model, tokenizer, device):
    """Get BERT embeddings for a list of texts."""
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating embeddings"):
            batch_texts = texts[i:i+BATCH_SIZE]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            
            # Use CLS token embeddings
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)
            
            # Clear GPU memory
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    return np.array(embeddings)

def main():
    # Set random seed for reproducibility
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load BERT model and tokenizer
    print("Loading BERT model and tokenizer...")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    model = BertModel.from_pretrained(model_name, cache_dir=CACHE_DIR).to(device)
    model.eval()
    
    # Read the dataset
    print("Reading dataset...")
    df = pd.read_csv('../transformed_data/imdb/test.csv')
    
    # Randomly sample 100 data points for visualization
    print("Randomly sampling 100 data points for visualization...")
    sample_size = min(100, len(df))
    df_sample = df.sample(n=sample_size, random_state=RANDOM_STATE)
    
    # Get original and transformed texts
    pseudo_dataset = df_sample['pseudo_dataset'].astype(str).tolist()
    transformed_texts = df_sample['transformed_data'].astype(str).tolist()
    
    # Generate embeddings
    print("\nGenerating embeddings for original texts...")
    original_embeddings = get_bert_embeddings(pseudo_dataset, model, tokenizer, device)
    
    print("\nGenerating embeddings for transformed texts...")
    transformed_embeddings = get_bert_embeddings(transformed_texts, model, tokenizer, device)
    
    # Combine embeddings for t-SNE
    combined_embeddings = np.vstack([original_embeddings, transformed_embeddings])
    
    # Create labels for visualization
    labels = np.array(['Pseudo'] * len(original_embeddings) + ['Transformed'] * len(transformed_embeddings))
    
    # Apply t-SNE
    print("\nApplying t-SNE...")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
    tsne_results = tsne.fit_transform(combined_embeddings)
    
    # Create DataFrame for plotting
    tsne_df = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'Category': labels
    })
    
    # Define colors and markers for each category
    category_params = {
        'Pseudo': {'color': '#32CD32', 'edgecolor': '#006400', 'marker': 'o'},  # Lime green fill, dark green edge
        'Transformed': {'color': '#FF6B6B', 'edgecolor': '#8B0000', 'marker': '^'}  # Light red fill, dark red edge
    }
    
    # Create plot
    plt.figure(figsize=(12, 8))
    sns.set_style("white")
    
    # Plot each category separately
    for category, params in category_params.items():
        subset = tsne_df[tsne_df['Category'] == category]
        plt.scatter(
            subset['x'],
            subset['y'],
            c=params['color'],
            edgecolor=params['edgecolor'],
            marker=params['marker'],
            label=category,
            alpha=0.6,
            s=100,
            linewidth=1.5
        )
    
    # Add border line on all sides
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
        spine.set_color('black')
    
    # Make scale labels (ticks) bold
    plt.xticks(fontweight='bold', fontsize=10)
    plt.yticks(fontweight='bold', fontsize=10)
    
    # Add title at the bottom with bold font
    plt.title(
        'IMDb t-SNE Visualization of Pseudo vs Transformed Text Embeddings', 
        pad=20, 
        loc='center',
        y=-0.2,
        fontweight='bold',
        fontsize=14
    )

    # Remove axis labels
    plt.xlabel('')
    plt.ylabel('')
    
    # Add legend inside the plot
    plt.legend(
        loc='best',
        frameon=True,
        framealpha=0.9,
        title=None
    )
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save in both PNG and SVG formats
    plt.savefig('tsne_visualization.png', dpi=600, bbox_inches='tight')
    plt.savefig('tsne_visualization.svg', bbox_inches='tight')
    
    plt.show()
    print("\nPlot saved as 'tsne_visualization.png' and 'tsne_visualization.svg'")

if __name__ == "__main__":
    main()
