#!/bin/bash

# Exit on error
set -e

echo "1. Installing requirements..."
pip install -r requirements.txt

echo -e "\n2. Preparing JSON data..."
python prepare_json_data.py

echo -e "\n3. Starting model training..."
python run_summarization.py 