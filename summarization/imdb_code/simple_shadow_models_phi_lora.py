#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Phi-2 LoRA shadow-model trainer.

This script fine-tunes `microsoft/phi-2` for summarization-style causal LM
training using JSON files that follow:
{
  "summarization": [{"real": "...", "summarize": "..."}, ...]
}
"""

import argparse
import csv
import json
import logging
import os
import random
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
import yaml
from datasets import Dataset as HFDataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "microsoft/phi-2"
CACHE_DIR = "/work/shovon/LLM/"

# Match cache/environment setup used in train_shadow_models_phi.py
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")

for cache_path in [
    os.environ["TRANSFORMERS_CACHE"],
    os.environ["HF_HOME"],
    os.environ["HF_DATASETS_CACHE"],
]:
    os.makedirs(cache_path, exist_ok=True)
    logger.info("Using cache directory: %s", cache_path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_summarization_json(file_path: str) -> Dict[str, List[str]]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    texts: List[str] = []
    summaries: List[str] = []
    for item in data.get("summarization", []):
        texts.append(item["real"])
        summaries.append(item["summarize"])

    if not texts:
        raise ValueError(f"No records found under 'summarization' in: {file_path}")

    return {"input_text": texts, "target_text": summaries}


def build_dataset(file_path: str, tokenizer, max_source_length: int) -> HFDataset:
    raw_dict = load_summarization_json(file_path)
    dataset = HFDataset.from_dict(raw_dict)

    def preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        prompts = []
        for text, summary in zip(examples["input_text"], examples["target_text"]):
            prompts.append(f"Text: {text}\nSummary: {summary}")

        tokenized = tokenizer(
            prompts,
            max_length=max_source_length,
            truncation=True,
            padding="max_length",
        )

        labels = []
        for sequence in tokenized["input_ids"]:
            label_sequence = [token_id if token_id != tokenizer.pad_token_id else -100 for token_id in sequence]
            labels.append(label_sequence)

        tokenized["labels"] = labels
        return tokenized

    return dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["input_text", "target_text"],
        desc=f"Tokenizing {os.path.basename(file_path)}",
    )


def resolve_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        logger.info("Using bfloat16 precision.")
        return torch.bfloat16
    if torch.cuda.is_available():
        logger.info("Using float16 precision.")
        return torch.float16
    logger.info("Using float32 precision.")
    return torch.float32


def resolve_path(base_dir: str, maybe_relative_path: str) -> str:
    # Keep compatibility with existing training scripts:
    # 1) first honor path as provided (absolute or relative to current working dir)
    # 2) if not found, try relative to this script directory (imdb_code)
    # 3) if still not found, try relative to the YAML config directory
    if os.path.isabs(maybe_relative_path):
        return maybe_relative_path

    cwd_candidate = os.path.normpath(maybe_relative_path)
    if os.path.exists(cwd_candidate):
        return cwd_candidate

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_candidate = os.path.normpath(os.path.join(script_dir, maybe_relative_path))
    if os.path.exists(script_candidate):
        return script_candidate

    return os.path.normpath(os.path.join(base_dir, maybe_relative_path))


def resolve_output_path(maybe_relative_path: str) -> str:
    """Resolve output paths relative to the script directory (imdb_code)."""
    if os.path.isabs(maybe_relative_path):
        return maybe_relative_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, maybe_relative_path))


def optimizer_to_hf_optim(optimizer_name: str) -> str:
    mapping = {
        "adamw": "adamw_torch",
    }
    normalized = optimizer_name.lower()
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}' for Trainer in this script. "
            "Currently supported: adamw."
        )
    return mapping[normalized]


def apply_yaml_config(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config_dir = os.path.dirname(os.path.abspath(args.config))

    model_name = config["model"]["name"]
    if model_name != MODEL_NAME:
        raise ValueError(f"This script only supports '{MODEL_NAME}', but got '{model_name}'.")

    args.train_file = resolve_path(config_dir, config["data"]["train_file"])
    args.test_file = resolve_path(config_dir, config["data"]["test_file"])
    # Use YAML output paths (e.g. ./results/phi/1.5/...), not config-relative folders.
    args.output_dir = resolve_output_path(config["output"]["output_dir"])
    args.logging_dir = resolve_output_path(config["output"]["logging_dir"])

    # Force faster/cheaper sequence length for this simple script.
    args.max_source_length = 512
    args.max_target_length = int(config["data"].get("max_target_length", 128))
    args.num_train_epochs = float(config["training"]["num_train_epochs"])
    args.learning_rate = float(config["training"]["learning_rate"])
    args.batch_size = int(config["training"]["batch_size"])
    args.gradient_accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
    args.warmup_steps = int(config["training"]["warmup_steps"])
    args.weight_decay = float(config["training"]["weight_decay"])
    args.logging_steps = int(config["training"]["logging_steps"])
    # Force less frequent evaluation for faster end-to-end training.
    args.eval_steps = 5000
    args.save_steps = int(config["training"]["save_steps"])
    args.lr_scheduler_type = config["training"]["lr_scheduler_type"]
    args.fp16 = bool(config["training"].get("fp16", False))
    args.bf16 = bool(config["training"].get("bf16", False))
    args.optim = optimizer_to_hf_optim(config["training"]["optimizer"])
    return args


def parse_model_indices(model_indices_args: List[str]) -> List[int]:
    selected_indices = set()
    for idx_str in model_indices_args:
        if "-" in idx_str:
            try:
                start, end = map(int, idx_str.split("-"))
                selected_indices.update(range(start, end + 1))
            except ValueError:
                logger.warning("Invalid model index range: %s. Skipping...", idx_str)
        else:
            try:
                selected_indices.add(int(idx_str))
            except ValueError:
                logger.warning("Invalid model index: %s. Skipping...", idx_str)
    return sorted(selected_indices)


def train_from_model_indices(base_args: argparse.Namespace) -> None:
    config_summary_path = os.path.normpath("./configs/config_summary.csv")
    if not os.path.exists(config_summary_path):
        raise FileNotFoundError(f"Config summary file not found at {config_summary_path}")

    selected_indices = parse_model_indices(base_args.model_indices)
    if not selected_indices:
        raise ValueError("No valid model indices were provided.")

    rows = []
    with open(config_summary_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                row_idx = int(row["model_index"])
            except (KeyError, ValueError):
                continue

            if row_idx not in selected_indices:
                continue
            if row.get("model_family") != "Phi":
                continue
            if row.get("model_name") != MODEL_NAME:
                continue
            rows.append(row)

    if not rows:
        raise ValueError(
            f"No matching Phi-2 rows found for indices {selected_indices} in {config_summary_path}."
        )

    found_indices = sorted(int(row["model_index"]) for row in rows)
    missing_indices = [idx for idx in selected_indices if idx not in found_indices]
    if missing_indices:
        logger.warning("Indices not found for Phi-2 and will be skipped: %s", missing_indices)

    for row in sorted(rows, key=lambda r: int(r["model_index"])):
        model_index = int(row["model_index"])
        config_path = os.path.normpath(row["config_path"])
        logger.info("Training model_index=%s using config=%s", model_index, config_path)

        run_args = deepcopy(base_args)
        run_args.config = config_path
        run_args = apply_yaml_config(run_args)

        try:
            train(run_args)
        except Exception as exc:
            logger.error("Training failed for model_index=%s: %s", model_index, str(exc))
            continue


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
        trust_remote_code=True,
        torch_dtype=resolve_torch_dtype(),
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = build_dataset(args.train_file, tokenizer, args.max_source_length)
    eval_dataset = build_dataset(args.test_file, tokenizer, args.max_source_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        bf16=args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=args.fp16 and torch.cuda.is_available() and not (args.bf16 and torch.cuda.is_bf16_supported()),
        dataloader_pin_memory=False,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting LoRA fine-tuning for %s", MODEL_NAME)
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final_model")
    adapter_dir = os.path.join(args.output_dir, "lora_adapters")
    os.makedirs(args.output_dir, exist_ok=True)

    trainer.save_model(final_dir)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Saved model to %s and adapters to %s", final_dir, adapter_dir)

    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    logger.info("Final eval metrics: %s", metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple shadow training for Phi-2 with LoRA")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument(
        "--model_indices",
        type=str,
        nargs="+",
        default=None,
        help="Model indices from config_summary.csv (e.g., 216 or 210-216).",
    )
    parser.add_argument("--train_file", type=str, default="../transformed_data/imdb/train.json")
    parser.add_argument("--test_file", type=str, default="../transformed_data/imdb/test.json")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--logging_dir", type=str, default="./results/logs")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    return apply_yaml_config(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()
    if args.config and args.model_indices:
        raise ValueError("Use either --config or --model_indices, not both.")
    if args.model_indices:
        train_from_model_indices(args)
    else:
        train(args)
