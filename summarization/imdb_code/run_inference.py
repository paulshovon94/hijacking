"""
Script to run inference using the fine-tuned BART model for text summarization.
This script loads the fine-tuned model and generates summaries for input text.
"""

import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import logging
from typing import List, Optional
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate summary using fine-tuned BART model')
    parser.add_argument('--text', type=str, required=True, help='Text to summarize')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum length of the summary')
    parser.add_argument('--min_length', type=int, default=30, help='Minimum length of the summary')
    parser.add_argument('--num_beams', type=int, default=4, help='Number of beams for beam search')
    parser.add_argument('--length_penalty', type=float, default=2.0, help='Length penalty for beam search')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=3, help='Size of n-grams to avoid repeating')
    
    args = parser.parse_args()
    
    # Initialize the summarizer
    summarizer = BARTSummarizer()
    
    # Generate summary
    summary = summarizer.summarize(
        text=args.text,
        max_length=args.max_length,
        min_length=args.min_length,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size
    )
    
    # Print summary with title
    print("Generated Summary:")
    print(summary)

if __name__ == "__main__":
    main() 