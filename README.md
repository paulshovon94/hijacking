# Stealing the Recipe: Hyperparameter Stealing Attacks on Fine-Tuned LLMs

This repository contains code for research on model hijacking/poisoning attacks on text summarization models. The project demonstrates how adversarial data transformations can be used to extract hyperparameter information from trained language models through their output behavior.

## Overview

The project implements a comprehensive pipeline for:
1. **Dataset Preparation**: Generating summaries from IMDB movie reviews using PEGASUS
2. **Adversarial Data Transformation**: Creating "hijacked" datasets by strategically replacing stop words based on sentiment using BERT-based masking
3. **Shadow Model Training**: Training multiple summarization models (BART, GPT-2, Pegasus, Phi) with varying hyperparameters
4. **Feature Extraction**: Computing multimodal features (embeddings, ROUGE scores, JSD values, novelty scores, etc.) from model outputs
5. **Attack Model Training**: Training a classifier to predict training hyperparameters from extracted features

## Setup

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

2. Set up your Hugging Face token:
   - Get your token from https://huggingface.co/settings/tokens
   - Create a `.env` file in the root directory:
     ```bash
     touch .env
     ```
   - Add your token to `.env`:
     ```
     HUGGINGFACE_TOKEN=your_token_here
     ```

3. Configure cache directory (optional):
   - The code uses `/work/../LLM/` as the default cache directory
   - You can modify the `CACHE_DIR` variable in the scripts to use a different location

## Minimal end-to-end pipeline (exact commands)

Perfect — here is the minimal end-to-end pipeline, step by step, with the exact command to run for each step.
No shell script, no extras, only the commands you need in order.

### Step 1: Create pseudo-dataset (IMDB summaries)

Generates IMDB summaries using a public summarization model.

```bash
cd summarization/imdb_code && python prepare_imdb_summaries.py
```

(Output: `summarization/pseudo_data/imdb/train.csv`, `summarization/pseudo_data/imdb/test.csv`)

### Step 2: Create hijacking token sets

Extracts stopwords, builds label-specific token sets.

```bash
cd summarization/imdb_code && python pre_imdb_token_set.py
```

(Output: stopword lists + frequency files in `summarization/transformed_data/imdb/`)

### Step 3: Create transformed (hijacking) dataset

Applies BERT-based word substitution attack to create hijacked summaries.

```bash
cd summarization/imdb_code && python imdb_attack.py
```

(Output: `summarization/transformed_data/imdb/train.csv`, `summarization/transformed_data/imdb/test.csv`)

### Step 4: Merge transformed train/test datasets

Creates a single hijacking dataset.

```bash
cd summarization/imdb_code && python combine_datasets.py
```

(Output: `summarization/transformed_data/imdb/hijacking_imdb.csv`)

### Step 5: Prepare CNN/DailyMail dataset

Downloads and preprocesses CNN/DailyMail for summarization.

```bash
cd summarization/imdb_code && python prepare_cnn_dailymail.py
```

(Output: processed CNN/DailyMail files)

### Step 6: Combine CNN/DailyMail + hijacking data

Creates the final JSON dataset used for training shadow models.

```bash
cd summarization/imdb_code && python prepare_json_data.py
```

(Output: training/validation/test JSON files)

### Step 7: Generate shadow-model configurations

Creates YAML configs for all hyperparameter combinations.

```bash
cd summarization/imdb_code && python generate_configs.py
```

(Output: YAML config files + `config_summary.csv`)

### Step 8: Train shadow models

Run the appropriate trainer depending on model family.

BART (encoder–decoder)

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=2 train_shadow_models.py --model_indices 0-9
```

Pegasus

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=2 train_shadow_models_pegasus.py --model_indices 91-107
```

GPT-2 (decoder-only)

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=2 train_shadow_models_gpt2.py --model_indices 159-161
```

(You only run the families you need.)

### Step 9: Extract behavioral features (x1–x7)

Computes ROUGE, JSD, novelty, semantic distance, etc.

BART

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=2 create_model_features.py
```

Pegasus

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=2 create_model_features_pegasus.py --model_indices 100-103
```

GPT-2

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=2 create_model_features_gpt2.py --model_indices 184-188
```

(Output: feature files x1–x7 for each model)

### Step 10: Create dataloader for attack model

Aggregates features and labels into a unified dataset.

```bash
cd summarization/imdb_code && python create_dataloader.py
```

(Output: `dataloader.csv`, `label_mappings.json`)

### Step 11: Train the attack (hyperparameter-stealing) model

Trains the multimodal classifier to predict hyperparameters.

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=1 experiment.py --seed 42
```

(Optional deterministic run)

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=1 experiment.py --seed 42 --deterministic
```

## Additional experiments (exact commands)

### Poisoning attack success rate

Script: `poisioning_exp.py`

Inference (evaluate attack success)

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=1 --master_port=29600 poisioning_exp.py --mode inference
```

Train (if your script supports training mode)

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=1 --master_port=29600 poisioning_exp.py --mode train
```

### Clean-data poisoning experiment (victim models + transfer)

Script: `clean_data_poisioning_exp.py`

Inference (evaluate on selected model indices)

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=1 --master_port=29600 clean_data_poisioning_exp.py   --mode inference --model_indices "23-26,100-103"
```

Note: your doc says “Selected multi-model indices (23-26, 100-103)”.

### Subsampling experiment

Script: `subsampling_exp.py`

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=1 --master_port=29600 subsampling_exp.py
```

### Cross-family transferability

Script: `cross_family_attack.py`

Train on BART+Pegasus, test on GPT-2

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=1 cross_family_attack.py   --train_families BART Pegasus --test_families GPT-2
```

With fixed seed + deterministic behavior

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=1 cross_family_attack.py --seed 42 --deterministic
```

### Main “multimodel” training experiment runner

Script: `experiment.py`

Standard run

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=1 experiment.py --seed 42
```

Deterministic run

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=1 experiment.py --seed 42 --deterministic
```

### Modality ablation (x1-only → x1..x7)

Script: `exp_modality.py`

```bash
cd summarization/imdb_code && python exp_modality.py
```

### Feature corruption checks + fixing

Script: `diagnose_corruption.py` (scan for NaN/Inf/etc.)

```bash
cd summarization/imdb_code && python diagnose_corruption.py
```

Script: `sanitize_features.py` (replace NaN/Inf with safe values)

```bash
cd summarization/imdb_code && python sanitize_features.py
```

### Older classifier scripts (if you still use them)

Script: `train_multilabel_classifier_x1.py`

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=2 train_multilabel_classifier_x1.py
```

Script: `train_multiclass_classifier_x2-7.py`

```bash
cd summarization/imdb_code && torchrun --nproc-per-node=2 train_multiclass_classifier_x2-7.py
```

## Project Structure

```
.
├── summarization/              # Main code directory
│   ├── imdb_code/             # IMDB dataset processing and attack implementation
│   │   ├── prepare_imdb_summaries.py      # Generate IMDB summaries using PEGASUS
│   │   ├── pre_imdb_token_set.py         # Prepare sentiment-based stop word sets
│   │   ├── imdb_attack.py                 # Implement adversarial attack on IMDB data
│   │   ├── imdb_attack_utils.py           # Utility functions for attacks
│   │   ├── combine_datasets.py            # Combine train/test datasets
│   │   ├── prepare_json_data.py          # Prepare JSON data for model training
│   │   ├── run_summarization.py          # Fine-tune BART for summarization
│   │   ├── train_shadow_models.py         # Train multiple shadow models with different hyperparameters
│   │   ├── create_model_features.py      # Extract features from trained models
│   │   ├── poisioning_exp.py             # Multimodal classifier for hyperparameter prediction
│   │   └── [other experiment and utility scripts]
│   ├── datasets/              # External datasets
│   │   └── cnn_dailymail/    # CNN/DailyMail dataset for summarization
│   ├── pseudo_data/          # Generated summaries and processed data
│   │   └── imdb/             # IMDB summaries output directory
│   │       ├── train.csv     # Training set summaries
│   │       ├── test.csv      # Test set summaries
│   │       └── statistics_imdb.txt  # Processing statistics
│   └── transformed_data/     # Transformed and processed datasets
│       └── imdb/            # IMDB processed data
│           ├── hijacking_imdb.csv  # Combined hijacked dataset
│           ├── label0_stop_set     # Negative sentiment stop words
│           ├── label1_stop_set     # Positive sentiment stop words
│           ├── imdb_freq_stop      # Word frequency statistics
│           ├── train.json          # Training data in JSON format
│           └── test.json           # Test data in JSON format
├── requirements.txt          # Python dependencies
├── environment.yml          # Conda environment file (alternative)
└── README.md               # This file
```

## Workflow

### 1. Prepare IMDB Summaries

Generate summaries from IMDB movie reviews using PEGASUS:

```bash
cd summarization/imdb_code
python prepare_imdb_summaries.py
```

This script:
- Loads IMDB dataset from Hugging Face
- Generates summaries using `google/pegasus-large`
- Filters summaries for quality (length, language, repetition, etc.)
- Saves summaries to `../pseudo_data/imdb/`

### 2. Prepare Token Sets for Hijacking

Create sentiment-based stop word sets for the adversarial attack:

```bash
cd summarization/imdb_code
python pre_imdb_token_set.py
```

This script:
- Analyzes word frequencies in generated summaries
- Creates separate stop word sets for positive (label1) and negative (label0) sentiment
- Saves token sets to `../transformed_data/imdb/`

### 3. Generate Hijacked Dataset

Apply adversarial transformations to create the hijacked dataset:

```bash
cd summarization/imdb_code
python imdb_attack.py
```

This script:
- Loads IMDB summaries with sentiment labels
- Uses BERT-based masking to identify substitution points
- Replaces stop words with sentiment-specific alternatives
- Saves transformed data to `../transformed_data/imdb/train.csv` and `test.csv`

### 4. Combine and Prepare Training Data

Combine datasets and prepare JSON format for model training:

```bash
cd summarization/imdb_code
# Combine train and test datasets
python combine_datasets.py

# Prepare JSON data (combines hijacked data with CNN/DailyMail)
python prepare_json_data.py
```

### 5. Train Shadow Models

Train multiple shadow models with different hyperparameter configurations:

```bash
cd summarization/imdb_code
# Generate hyperparameter configurations
python generate_configs.py

# Train shadow models (can be run in parallel)
python train_shadow_models.py --config configs/config_*.yaml
```

The shadow models are trained with variations in:
- Model architecture (BART, GPT-2, Pegasus, Phi)
- Model size (base, large, small)
- Optimizer (AdamW, SGD, Adafactor)
- Learning rate (1e-5, 5e-5, 1e-4)
- Batch size (4, 8, 16)

### 6. Extract Model Features

Extract multimodal features from trained shadow models:

```bash
cd summarization/imdb_code
python create_model_features.py --model_dir <path_to_trained_model>
```

This script computes:
- Summary embeddings (x1)
- Transformed data embeddings (x2)
- ROUGE scores (x3)
- Jensen-Shannon Divergence (x4)
- Novelty scores (x5)
- Length differences (x6)
- POS tag divergence (x7)

Features are saved to `multimodal_dataset/` directory.

### 7. Train Attack Model

Train a multimodal classifier to predict hyperparameters from features:

```bash
cd summarization/imdb_code
python poisioning_exp.py
```

This script:
- Loads extracted features from all shadow models
- Trains a multi-head classifier to predict:
  - Model family
  - Model size
  - Optimizer type
  - Learning rate
  - Batch size
- Evaluates attack success rate

## Key Scripts

### Data Preparation
- `prepare_imdb_summaries.py`: Generate summaries from IMDB reviews
- `pre_imdb_token_set.py`: Create sentiment-based stop word sets
- `imdb_attack.py`: Apply adversarial transformations
- `combine_datasets.py`: Combine train/test datasets
- `prepare_json_data.py`: Prepare JSON format for training

### Model Training
- `run_summarization.py`: Fine-tune BART for summarization
- `train_shadow_models.py`: Train multiple shadow models
- `train_shadow_models_bart.py`: BART-specific training
- `train_shadow_models_pegasus.py`: Pegasus-specific training
- `train_shadow_models_gpt2.py`: GPT-2-specific training
- `train_shadow_models_phi.py`: Phi-specific training

### Feature Extraction
- `create_model_features.py`: Extract features from BART models
- `create_model_features_pegasus.py`: Extract features from Pegasus models
- `create_model_features_gpt2.py`: Extract features from GPT-2 models
- `create_model_features_phi.py`: Extract features from Phi models

### Attack and Analysis
- `poisioning_exp.py`: Multimodal hyperparameter prediction classifier
- `plot_tsne.py`: Visualize feature distributions using t-SNE
- `evaluation.py`: Evaluate attack success rates

## Dependencies

Key dependencies include:
- `torch`: PyTorch for deep learning
- `transformers`: Hugging Face Transformers library
- `datasets`: Hugging Face Datasets
- `sentence-transformers`: For semantic embeddings
- `rouge-score`: For ROUGE metric calculation
- `scikit-learn`: For machine learning utilities
- `nltk`: For natural language processing
- `wandb`: For experiment tracking (optional)

See `requirements.txt` for the complete list.

## Notes

- **GPU Required**: Training shadow models and generating summaries requires GPU acceleration
- **Cache Directory**: Modify `CACHE_DIR` in scripts if you need a different cache location
- **Data Percentage**: Some scripts have `DATA_PERCENTAGE` constants to process subsets of data for faster experimentation
- **Reproducibility**: Scripts use random seeds (typically 42) for reproducibility

## Citation

If you use this code in your research, please cite the associated paper.

## License

[Specify your license here]
