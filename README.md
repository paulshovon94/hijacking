# Two-in-One: A Model Hijacking Attack Against Text Generation Models (Usenix 2023)

[![arXiv](https://img.shields.io/badge/arxiv-2305.07406-b31b1b)](https://arxiv.org/abs/2305.07406)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

This repository contains the PyTorch implementation of the paper "[Two-in-One: A Model Hijacking Attack Against Text Generation Models](https://arxiv.org/abs/2305.07406)" by [Wai Man Si](https://raymondhehe.github.io/), [Michael Backes](https://scholar.google.de/citations?user=ZVS3KOEAAAAJ&hl=de), [Yang Zhang](https://yangzhangalmo.github.io/), and [Ahmed Salem](https://ahmedsalem2.github.io/).

## Setup

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

2. Set up your Hugging Face token:
   - Get your token from https://huggingface.co/settings/tokens
   - Copy `.env.template` to `.env`:
     ```bash
     cp .env.template .env
     ```
   - Edit `.env` and replace the token value with your own token

## Project Structure
```
.
├── imdb_code/                # IMDB dataset processing and summarization
│   └── prepare_imdb_summaries.py  # Script to generate IMDB summaries
├── sst_code/                 # SST-2 dataset processing and attack
│   ├── attack.py            # Implementation of adversarial attack
│   ├── attack_utils.py      # Utility functions for attacks
│   ├── evaluation.py        # Evaluation metrics and analysis
│   ├── prepare_data.py      # Dataset preparation
│   ├── pre_token_set.py     # Token set preparation
│   └── run_translation.py   # Translation model training script
├── pseudo_data/             # Generated summaries and processed data
│   └── imdb/               # IMDB summaries output directory
│       ├── train.csv       # Training set summaries
│       ├── test.csv        # Test set summaries
│       └── statistics_imdb.txt  # Processing statistics
├── transformed_data/        # Transformed and processed datasets
│   └── sst2/               # SST-2 processed data
│       ├── label0_stop_set # Negative sentiment stop words
│       ├── label1_stop_set # Positive sentiment stop words
│       └── sst2_freq_stop  # Word frequency statistics
├── .env.template           # Template for environment variables
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Running the Code

### 1. Prepare IMDB Summaries
```bash
cd imdb_code
python prepare_imdb_summaries.py
```

### 2. Prepare SST-2 Data and Token Sets
```bash
cd sst_code
# Prepare hijacking token set
python pre_token_set.py

# Prepare dataset
python prepare_data.py

# Run sentiment attack
python attack.py
```

### 3. Train Translation Model
```bash
python -m torch.distributed.launch --master_port=1233 --nproc_per_node=4 run_translation.py \
    --seed 42 \
    --model_name_or_path facebook/bart-base \
    --train_file ../transformed_data/sst2/train.json \
    --validation_file ../transformed_data/sst2/validation.json \
    --test_file ../transformed_data/sst2/validation.json \
    --do_train --do_eval --do_predict \
    --max_source_length 128 --max_target_length 128 \
    --preprocessing_num_workers 16 \
    --source_lang en --target_lang de \
    --num_beams 1 \
    --output_dir exps/sst2_bartbase \
    --per_device_train_batch_size=128 --per_device_eval_batch_size=64 \
    --num_train_epochs 10 \
    --logging_strategy steps --logging_steps 1000 --logging_first_step True \
    --evaluation_strategy epoch --save_strategy epoch \
    --predict_with_generate \
    --fp16
```

### 4. Evaluate Results
```bash
cd sst_code
python evaluation.py
```

## Requirements
- Python 3.8
- PyTorch 1.11.0
- transformers 4.19.2
- python-dotenv 1.0.0
- Other dependencies listed in requirements.txt

## Acknowledgements
Our code is built upon the public code of the [CLARE](https://github.com/cookielee77/CLARE/tree/master) and [Transformers](https://github.com/huggingface/transformers).

## Citation

Please cite our paper if you use this code in your own work:

```
@inproceedings{SBZS23,
  author       = {Wai Man Si and
                  Michael Backes and
                  Yang Zhang and
                  Ahmed Salem},
  title        = {Two-in-One: {A} Model Hijacking Attack Against Text Generation Models},
  booktitle    = {32nd {USENIX} Security Symposium, {USENIX} Security 2023, Anaheim,
                  CA, USA, August 9-11, 2023},
  pages        = {2223--2240},
  publisher    = {{USENIX} Association},
  year         = {2023}
}
```
