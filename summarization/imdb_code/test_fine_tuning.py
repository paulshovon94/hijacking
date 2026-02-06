import os
import torch
import evaluate
import numpy as np
import multiprocessing
import random
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

# Set cache directory for all model downloads and caching
CACHE_DIR = "/work/shovon/LLM/"
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'transformers')
os.environ['HF_HOME'] = os.path.join(CACHE_DIR, 'huggingface')
os.environ['HF_DATASETS_CACHE'] = os.path.join(CACHE_DIR, 'datasets')
os.environ['HF_HUB_CACHE'] = os.path.join(CACHE_DIR, 'hub')
os.environ['TORCH_HOME'] = os.path.join(CACHE_DIR, 'torch')
os.environ['HF_DATASETS_OFFLINE'] = '0'  # Allow online downloads but cache locally

# Create cache directories if they don't exist
cache_dirs = [
    os.environ['TRANSFORMERS_CACHE'], 
    os.environ['HF_HOME'], 
    os.environ['HF_DATASETS_CACHE'],
    os.environ['HF_HUB_CACHE'],
    os.environ['TORCH_HOME']
]
for cache_path in cache_dirs:
    os.makedirs(cache_path, exist_ok=True)
    print(f"Using cache directory: {cache_path}")

# Set configurations
MODEL = 't5-large'
BATCH_SIZE = 4
EPOCHS = 3
OUT_DIR = 'results_t5base_imdb'
max_source_length = 512  # Maximum tokens for input text
max_target_length = 128  # Maximum tokens for generated summaries
GENERATION_NUM_BEAMS = 2
TARGET_EFFECTIVE_BATCH_SIZE = 64

# CPU and process configuration
CPU_COUNT = multiprocessing.cpu_count()
NUM_PROCS = 1  # Use only 1 process to minimize disk usage
print(f"Using {NUM_PROCS} processes for dataset processing")

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Dataset file paths
TRAIN_FILE = "../transformed_data/imdb/train.json"
TEST_FILE = "../transformed_data/imdb/test.json"

def calculate_gradient_accumulation_steps(per_device_batch_size: int, target_effective_batch_size: int = 64) -> int:
    """Calculate gradient accumulation steps to achieve target effective batch size."""
    # Get number of GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # Calculate required gradient accumulation steps
    gradient_accumulation_steps = max(1, target_effective_batch_size // (per_device_batch_size * num_gpus))
    
    print(f"Calculating gradient accumulation steps:")
    print(f"- Target effective batch size: {target_effective_batch_size}")
    print(f"- Per device batch size: {per_device_batch_size}")
    print(f"- Number of GPUs: {num_gpus}")
    print(f"- Calculated gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"- Actual effective batch size: {per_device_batch_size * gradient_accumulation_steps * num_gpus}")
    
    return gradient_accumulation_steps

# Set seed for reproducibility
set_seed(42)

# Load tokenizer with cache directory
tokenizer = T5Tokenizer.from_pretrained(
    MODEL,
    cache_dir=os.environ['TRANSFORMERS_CACHE'],
    local_files_only=False  # Allow downloading if not cached
)

# Ensure T5 tokenizer has proper padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set pad_token to eos_token for T5 tokenizer")

# Preprocessing function for IMDB dataset
def preprocess_function(examples):
    # JSON format has 'real' and 'summarize' fields
    # Tokenize source text to max_source_length tokens
    inputs = [f"summarize: {text}" for text in examples['real']]
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True, padding='max_length')

    # Tokenize target summaries to max_target_length tokens
    targets = [summary for summary in examples['summarize']]
    labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True, padding='max_length')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Load dataset from local JSON files
print(f"Loading train dataset from: {TRAIN_FILE}")
print(f"Loading test dataset from: {TEST_FILE}")

try:
    # Load datasets from JSON files with nested structure
    train_dataset = load_dataset('json', data_files=TRAIN_FILE, field='summarization', split='train')
    test_dataset = load_dataset('json', data_files=TEST_FILE, field='summarization', split='train')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
except Exception as e:
    print(f"Error loading datasets: {str(e)}")
    print("Please check that the JSON files exist and have the correct format.")
    print("Expected format: {'summarization': [{'real': '...', 'summarize': '...'}, ...]}")
    raise

# Tokenize dataset with aggressive memory optimizations
print("Tokenizing train dataset...")
tokenized_train = train_dataset.map(
    preprocess_function, 
    batched=True, 
    num_proc=1,  # Single process to minimize disk usage
    batch_size=50,  # Very small batch size to reduce memory usage
    remove_columns=train_dataset.column_names,  # Remove original columns to save space
    desc="Tokenizing train dataset",
    writer_batch_size=50  # Small writer batch size
)

print("Tokenizing test dataset...")
tokenized_test = test_dataset.map(
    preprocess_function, 
    batched=True, 
    num_proc=1,  # Single process to minimize disk usage
    batch_size=50,  # Very small batch size to reduce memory usage
    remove_columns=test_dataset.column_names,  # Remove original columns to save space
    desc="Tokenizing test dataset",
    writer_batch_size=50  # Small writer batch size
)

# Load model with cache directory
model = T5ForConditionalGeneration.from_pretrained(
    MODEL,
    cache_dir=os.environ['TRANSFORMERS_CACHE'],
    local_files_only=False  # Allow downloading if not cached
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print model parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_params:,} total parameters.")
print(f"{trainable_params:,} training parameters.")

# Preload ROUGE metric with cache directory
rouge = evaluate.load("rouge", cache_dir=os.environ['HF_DATASETS_CACHE'])

def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions[0], eval_pred.label_ids
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Use preloaded ROUGE metric
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        rouge_types=['rouge1', 'rouge2', 'rougeL']
    )
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}



# Calculate gradient accumulation steps for target effective batch size
gradient_accumulation_steps = calculate_gradient_accumulation_steps(BATCH_SIZE, TARGET_EFFECTIVE_BATCH_SIZE)

# Define training arguments based on YAML config
training_args = Seq2SeqTrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    logging_dir=OUT_DIR,
    logging_steps=100,
    eval_strategy='no',
    save_strategy='steps',
    save_steps=1000,
    save_total_limit=1,  # Keep only 1 checkpoint to save disk space
    learning_rate=5.0e-05,
    dataloader_num_workers=0,  # No workers to minimize disk usage
    dataloader_pin_memory=False,  # Disable pin memory to save memory
    # Fixed learning rate settings
    lr_scheduler_type="constant",
    warmup_steps=0,
    # Mixed precision settings
    bf16=True,
    fp16=False,
    # Gradient accumulation (calculated dynamically)
    gradient_accumulation_steps=gradient_accumulation_steps,
    # Gradient clipping
    max_grad_norm=1.0,
    # Optimizer
    optim="adafactor",
    # Generation settings for Seq2Seq
    generation_max_length=max_target_length,
    predict_with_generate=True,
    generation_num_beams=GENERATION_NUM_BEAMS,
    # Evaluation settings
    eval_accumulation_steps=1,
    # Seed for reproducibility
    seed=42,
    # Ensure no home directory usage
    report_to=None  # Disable any external reporting
)

# Data collator for sequence-to-sequence tasks
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    return_tensors="pt"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
print("Starting training...")
trainer.train()

# Save the final model
print("Saving model...")
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"Model saved to {OUT_DIR}")

# Evaluate on test set
print("Evaluating on test set...")
test_results = trainer.evaluate(tokenized_test)
print(f"Test results: {test_results}")
