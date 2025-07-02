"""
Script to analyze word count in the pseudo_dataset column of hijacking_imdb.csv.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

def count_words(text):
    """Count the number of words in a text."""
    if not isinstance(text, str):
        return 0
    return len(text.split())

def main():
    # Read the CSV file
    print("Reading hijacking_imdb.csv...")
    df = pd.read_csv('../transformed_data/imdb/hijacking_imdb.csv')
    
    # Calculate word counts for each text in pseudo_dataset
    print("\nCalculating word counts...")
    word_counts = []
    
    for text in tqdm(df['pseudo_dataset'], desc="Processing texts"):
        count = count_words(text)
        word_counts.append(count)
    
    # Calculate statistics
    avg_count = np.mean(word_counts)
    median_count = np.median(word_counts)
    std_count = np.std(word_counts)
    min_count = np.min(word_counts)
    max_count = np.max(word_counts)
    
    # Print results
    print("\nWord Count Statistics:")
    print(f"Average words per entry: {avg_count:.2f} words")
    print(f"Median words per entry: {median_count:.2f} words")
    print(f"Standard deviation: {std_count:.2f} words")
    print(f"Minimum words per entry: {min_count} words")
    print(f"Maximum words per entry: {max_count} words")
    
    # Calculate distribution of word counts
    count_distribution = pd.Series(word_counts).value_counts().sort_index()
    
    # Print distribution
    print("\nWord Count Distribution:")
    for count, frequency in count_distribution.items():
        percentage = (frequency / len(word_counts)) * 100
        print(f"{count:3d} words: {frequency:6d} entries ({percentage:5.2f}%)")

if __name__ == "__main__":
    main() 