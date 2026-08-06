#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple LLaMA-3.1 8B LoRA shadow-model trainer.

This script mirrors `simple_shadow_models_phi_lora.py` behavior but targets:
`meta-llama/Meta-Llama-3.1-8B`
"""

import argparse
import csv
import inspect
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

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
CACHE_DIR = "/work/shovon/LLM/"

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
        full_texts = []
        for text, summary in zip(examples["input_text"], examples["target_text"]):
            prompt = f"German: {text}\nEnglish:"
            prompts.append(prompt)
            full_texts.append(f"{prompt} {summary}")

        tokenized = tokenizer(
            full_texts,
            max_length=max_source_length,
            truncation=True,
            padding="max_length",
        )

        prompt_tokenized = tokenizer(
            prompts,
            max_length=max_source_length,
            truncation=True,
            padding=False,
            add_special_tokens=False,
        )

        labels = []
        for sequence, prompt_ids in zip(tokenized["input_ids"], prompt_tokenized["input_ids"]):
            prompt_len = min(len(prompt_ids), len(sequence))
            label_sequence = []
            for idx, token_id in enumerate(sequence):
                if token_id == tokenizer.pad_token_id:
                    label_sequence.append(-100)
                elif idx < prompt_len:
                    label_sequence.append(-100)
                else:
                    label_sequence.append(token_id)
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
    if os.path.isabs(maybe_relative_path):
        return maybe_relative_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, maybe_relative_path))


def optimizer_to_hf_optim(optimizer_name: str) -> str:
    mapping = {
        "adamw": "adamw_torch",
        "adafactor": "adafactor",
    }
    normalized = optimizer_name.lower()
    if normalized == "sgd":
        logger.warning(
            "Optimizer 'sgd' is not directly supported in this Trainer setup. "
            "Falling back to 'adamw_torch'."
        )
        return "adamw_torch"
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported optimizer '{optimizer_name}' for Trainer in this script. "
            "Currently supported: adamw, adafactor (sgd falls back to adamw_torch)."
        )
    return mapping[normalized]


def calculate_gradient_accumulation_steps(
    per_device_batch_size: int,
    target_effective_batch_size: int = 64,
) -> int:
    """Calculate grad accumulation to approach target effective global batch size."""
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    gradient_accumulation_steps = max(
        1, target_effective_batch_size // max(1, per_device_batch_size * num_gpus)
    )
    actual_effective_batch_size = per_device_batch_size * gradient_accumulation_steps * num_gpus

    logger.info("Calculating gradient accumulation steps:")
    logger.info("- Target effective batch size: %s", target_effective_batch_size)
    logger.info("- Per device batch size: %s", per_device_batch_size)
    logger.info("- Number of GPUs: %s", num_gpus)
    logger.info("- Calculated gradient accumulation steps: %s", gradient_accumulation_steps)
    logger.info("- Actual effective batch size: %s", actual_effective_batch_size)
    return gradient_accumulation_steps


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
    args.output_dir = resolve_output_path(config["output"]["output_dir"])
    args.logging_dir = resolve_output_path(config["output"]["logging_dir"])
    args.max_source_length = 256
    args.max_target_length = int(config["data"].get("max_target_length", 128))
    args.num_train_epochs = float(config["training"]["num_train_epochs"])
    args.learning_rate = float(config["training"]["learning_rate"])
    # LoRA hyperparameters must come from the config: they are three of the six labels
    # the attack classifier predicts, so hardcoding them would make those heads
    # meaningless (every config would train identically apart from learning rate).
    args.lora_r = int(config["training"]["lora_r"])
    args.lora_alpha = int(config["training"]["lora_alpha"])
    args.lora_dropout = float(config["training"]["lora_dropout"])
    args.batch_size = int(config["training"]["batch_size"])
    args.gradient_accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
    args.warmup_steps = int(config["training"]["warmup_steps"])
    args.weight_decay = float(config["training"]["weight_decay"])
    args.logging_steps = int(config["training"]["logging_steps"])
    args.eval_steps = 5000
    args.save_steps = int(config["training"]["save_steps"])
    args.evaluation_strategy = config["training"].get("evaluation_strategy", "steps")
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
            if row.get("model_family") != "LLaMA":
                continue
            if row.get("model_name") != MODEL_NAME:
                continue
            rows.append(row)

    if not rows:
        raise ValueError(
            f"No matching LLaMA-3.1 rows found for indices {selected_indices} in {config_summary_path}."
        )

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
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_load_kwargs = {
        "pretrained_model_name_or_path": MODEL_NAME,
        "cache_dir": os.environ["TRANSFORMERS_CACHE"],
        "torch_dtype": resolve_torch_dtype(),
    }
    # Try Flash Attention 2 when available for faster attention kernels.
    if torch.cuda.is_available():
        model_load_kwargs["attn_implementation"] = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(**model_load_kwargs)
    except Exception as exc:
        if model_load_kwargs.get("attn_implementation") == "flash_attention_2":
            logger.warning(
                "Flash Attention 2 unavailable/incompatible (%s). Falling back to default attention.",
                str(exc),
            )
            model_load_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(**model_load_kwargs)
        else:
            raise
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = build_dataset(args.train_file, tokenizer, args.max_source_length)
    should_evaluate = str(args.evaluation_strategy).lower() != "no"
    eval_dataset = build_dataset(args.test_file, tokenizer, args.max_source_length) if should_evaluate else None
    gradient_accumulation_steps = calculate_gradient_accumulation_steps(
        per_device_batch_size=args.batch_size,
        target_effective_batch_size=64,
    )

    requested_training_args = {
        "output_dir": args.output_dir,
        "logging_dir": args.logging_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": 2,
        "lr_scheduler_type": args.lr_scheduler_type,
        "optim": args.optim,
        "bf16": args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": args.fp16 and torch.cuda.is_available() and not (args.bf16 and torch.cuda.is_bf16_supported()),
        "dataloader_pin_memory": torch.cuda.is_available(),
        "gradient_checkpointing": True,
        "group_by_length": True,
        "ddp_find_unused_parameters": False,
        "tf32": True,
        "report_to": "none",
    }
    supported_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    # Handle transformers version differences: some use "evaluation_strategy", newer ones use "eval_strategy".
    if "evaluation_strategy" in supported_params:
        requested_training_args["evaluation_strategy"] = args.evaluation_strategy
    elif "eval_strategy" in supported_params:
        requested_training_args["eval_strategy"] = args.evaluation_strategy

    filtered_training_args = {k: v for k, v in requested_training_args.items() if k in supported_params}
    dropped_args = sorted(set(requested_training_args.keys()) - set(filtered_training_args.keys()))
    if dropped_args:
        logger.warning("Skipping unsupported TrainingArguments keys: %s", dropped_args)
    training_args = TrainingArguments(**filtered_training_args)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    trainer_supported_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    if "tokenizer" in trainer_supported_params:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        logger.warning("Current transformers Trainer does not support 'tokenizer' argument. Skipping it.")

    trainer = Trainer(**trainer_kwargs)

    logger.info("Starting LoRA fine-tuning for %s", MODEL_NAME)
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final_model")
    adapter_dir = os.path.join(args.output_dir, "lora_adapters")
    os.makedirs(args.output_dir, exist_ok=True)

    trainer.save_model(final_dir)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Saved model to %s and adapters to %s", final_dir, adapter_dir)

    if eval_dataset is not None:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        logger.info("Final eval metrics: %s", metrics)
    else:
        logger.info("Skipping final evaluation because evaluation_strategy='no'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LLaMA-3.1 8B shadow models with LoRA")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument(
        "--model_indices",
        type=str,
        nargs="+",
        default=None,
        help="Model indices from config_summary.csv (e.g., 216 or 210-216).",
    )
    parser.add_argument("--train_file", type=str, default="../transformed_data/wmt/train.json")
    parser.add_argument("--test_file", type=str, default="../transformed_data/wmt/test.json")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--logging_dir", type=str, default="./results/logs")
    parser.add_argument("--max_source_length", type=int, default=256)
    # Defaults only; the YAML config supplies the real values per run.
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
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
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
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
