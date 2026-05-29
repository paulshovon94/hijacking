#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train Pegasus-only LoRA shadow models using YAML configs from generate_configs_lora.py.

This is a standalone trainer (no imports from train_shadow_models_lora.py).
"""

import argparse
import inspect
import json
import logging
import multiprocessing
import os
import random
import time
from datetime import timedelta
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from datasets import Dataset as HFDataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    Seq2SeqTrainingArguments,
    Trainer,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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

CPU_COUNT = multiprocessing.cpu_count()
NUM_PROC = min(16, CPU_COUNT)
logger.info("Using %s processes for dataset processing", NUM_PROC)

MIN_EVAL_STEPS = 2000
TARGET_FAMILY = "Pegasus"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(os.getcwd(), path))


def _lora_run_already_done(output_dir: str) -> bool:
    final_adapter = os.path.join(output_dir, "final_model", "adapter_config.json")
    lora_dir = os.path.join(output_dir, "lora_adapters", "adapter_config.json")
    return os.path.isfile(final_adapter) or os.path.isfile(lora_dir)


def _normalize_lora_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    training = config.get("training", {})
    nested = config.get("lora")
    if nested:
        return {
            "r": int(nested.get("r", training.get("lora_r", 8))),
            "lora_alpha": int(nested.get("lora_alpha", training.get("lora_alpha", 16))),
            "lora_dropout": float(
                nested.get("lora_dropout", training.get("lora_dropout", 0.05))
            ),
            "bias": nested.get("bias", "none"),
            "target_modules": nested.get("target_modules"),
        }
    return {
        "r": int(training.get("lora_r", 8)),
        "lora_alpha": int(training.get("lora_alpha", 16)),
        "lora_dropout": float(training.get("lora_dropout", 0.05)),
        "bias": "none",
        "target_modules": None,
    }


class SummarizationDataset:
    """Dataset class for Pegasus summarization."""

    def __init__(
        self,
        file_path: str,
        tokenizer,
        max_source_length: int = 512,
        max_target_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.file_path = _resolve_path(file_path)

        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.texts = []
        self.summaries = []
        for item in data["summarization"]:
            self.texts.append(item["real"])
            self.summaries.append(item["summarize"])

    def preprocess_function(self, examples: Dict) -> Dict:
        inputs = ["summarize: " + doc for doc in examples["input_text"]]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=examples["target_text"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def create_dataset(self) -> HFDataset:
        cache_dir = os.path.join(CACHE_DIR, "preprocessed_datasets")
        os.makedirs(cache_dir, exist_ok=True)
        tokenizer_name = self.tokenizer.name_or_path
        file_hash = hash(self.file_path)
        cache_key = (
            f"{tokenizer_name}_{file_hash}_{self.max_source_length}_{self.max_target_length}"
        )
        cache_path = os.path.join(cache_dir, f"{cache_key}.hf")

        if os.path.exists(cache_path):
            logger.info("Loading preprocessed dataset from cache: %s", cache_path)
            return load_from_disk(cache_path)

        logger.info("Processing dataset and saving to cache...")
        dataset = HFDataset.from_dict(
            {"input_text": self.texts, "target_text": self.summaries}
        )
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=NUM_PROC,
            remove_columns=["input_text", "target_text"],
            desc="Processing dataset",
        )
        logger.info("Saving processed dataset to cache: %s", cache_path)
        processed_dataset.save_to_disk(cache_path)
        return processed_dataset


def get_model_and_tokenizer(config: Dict[str, Any]):
    model_name = config["model"]["name"]
    model_type = config["model"].get("type", "")
    model_family = config["model"].get("family", "")

    if model_type != "encoder-decoder":
        raise ValueError(
            f"Pegasus LoRA trainer supports encoder-decoder only, got: {model_type}"
        )
    if model_family != TARGET_FAMILY and "pegasus" not in model_name.lower():
        raise ValueError(
            f"Pegasus LoRA trainer supports Pegasus only, got family={model_family}, model={model_name}"
        )

    tokenizer = PegasusTokenizer.from_pretrained(
        model_name, cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )
    model = PegasusForConditionalGeneration.from_pretrained(
        model_name, cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def default_lora_target_modules() -> List[str]:
    return ["q_proj", "v_proj"]


def build_lora_config(lora_params: Dict[str, Any]) -> LoraConfig:
    target_modules = default_lora_target_modules()
    requested = lora_params.get("target_modules")
    if requested and requested != target_modules:
        logger.warning(
            "Ignoring requested target_modules=%s; enforcing target_modules=%s",
            requested,
            target_modules,
        )
    return LoraConfig(
        r=lora_params["r"],
        lora_alpha=lora_params["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=lora_params["lora_dropout"],
        bias=lora_params["bias"],
        task_type=TaskType.SEQ_2_SEQ_LM,
    )


def calculate_gradient_accumulation_steps(
    per_device_batch_size: int, target_effective_batch_size: int = 64
) -> int:
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    steps = max(1, target_effective_batch_size // (per_device_batch_size * num_gpus))
    logger.info(
        "Gradient accumulation: per_device_bs=%s gpus=%s -> steps=%s (effective %s)",
        per_device_batch_size,
        num_gpus,
        steps,
        per_device_batch_size * steps * num_gpus,
    )
    return steps


def train_model(config_path: str, model_index: int) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config.get("model", {}).get("family") != TARGET_FAMILY:
        raise ValueError(
            f"Config {config_path} is not Pegasus family: {config.get('model', {}).get('family')}"
        )

    training = config["training"]
    use_lora = bool(training.get("use_lora", True))
    lora_params = _normalize_lora_from_config(config)

    model_name = config["model"]["name"]
    output_dir = _resolve_path(config["output"]["output_dir"])

    if _lora_run_already_done(output_dir):
        logger.info("LoRA adapters already present under %s. Skipping training.", output_dir)
        return

    logger.info("Training Pegasus (LoRA=%s): %s", use_lora, model_name)
    logger.info("Training hyperparameters: %s", training)
    if use_lora:
        logger.info("LoRA hyperparameters: %s", lora_params)

    wandb_name = (
        f"pegasus_lora_{model_index}_{config['model']['size']}_"
        f"{training['optimizer']}_lr{training['learning_rate']}_bs{training['batch_size']}"
    )
    if use_lora:
        wandb_name += (
            f"_r{lora_params['r']}_a{lora_params['lora_alpha']}_d{lora_params['lora_dropout']}"
        )

    if local_rank == 0:
        wandb.init(
            project="shadow-model-training-pegasus-lora-imdb",
            name=wandb_name,
            config=config,
        )

    try:
        set_seed(42)
        model, tokenizer = get_model_and_tokenizer(config)

        if use_lora:
            lora_cfg = build_lora_config(lora_params)
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        train_dataset = SummarizationDataset(
            config["data"]["train_file"],
            tokenizer,
            int(config["data"]["max_source_length"]),
            int(config["data"]["max_target_length"]),
        )
        test_dataset = SummarizationDataset(
            config["data"]["test_file"],
            tokenizer,
            int(config["data"]["max_source_length"]),
            int(config["data"]["max_target_length"]),
        )

        train_hf = train_dataset.create_dataset()
        test_hf = test_dataset.create_dataset()
        train_val_datasets = train_hf.train_test_split(test_size=0.2, seed=42)

        if "gradient_accumulation_steps" in training:
            gradient_accumulation_steps = int(training["gradient_accumulation_steps"])
        else:
            gradient_accumulation_steps = calculate_gradient_accumulation_steps(
                per_device_batch_size=int(training["batch_size"])
            )

        use_bf16 = bool(training.get("bf16", False))
        use_fp16 = bool(training.get("fp16", False))
        if use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            bf16, fp16 = True, False
        elif use_fp16 and torch.cuda.is_available():
            bf16, fp16 = False, True
        else:
            bf16, fp16 = False, False
            if use_bf16 or use_fp16:
                logger.warning("Requested bf16/fp16 not available; training in float32.")

        save_steps = int(training.get("save_steps", 1000))

        training_args_kwargs: Dict[str, Any] = {
            "output_dir": output_dir,
            "num_train_epochs": int(training["num_train_epochs"]),
            "per_device_train_batch_size": int(training["batch_size"]),
            "per_device_eval_batch_size": int(training["batch_size"]),
            "warmup_steps": int(training["warmup_steps"]),
            "weight_decay": float(training["weight_decay"]),
            "logging_dir": _resolve_path(config["output"]["logging_dir"]),
            "logging_steps": int(training["logging_steps"]),
            "save_steps": save_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "fp16": fp16,
            "bf16": bf16,
            "report_to": "wandb" if local_rank == 0 else "none",
            "generation_max_length": int(training["generation_max_length"]),
            "predict_with_generate": True,
            "generation_num_beams": int(training["generation_num_beams"]),
            "learning_rate": float(training["learning_rate"]),
            "lr_scheduler_type": training.get("lr_scheduler_type", "linear"),
            "max_steps": -1,
            "save_total_limit": 2,
            "load_best_model_at_end": False,
            "save_strategy": "steps",
            "eval_accumulation_steps": 1,
            "remove_unused_columns": False,
            "ddp_find_unused_parameters": False,
        }

        seq2seq_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
        if "eval_strategy" in seq2seq_params:
            training_args_kwargs["eval_strategy"] = "no"
        elif "evaluation_strategy" in seq2seq_params:
            training_args_kwargs["evaluation_strategy"] = "no"
        else:
            logger.warning(
                "Seq2SeqTrainingArguments has no evaluation_strategy/eval_strategy."
            )

        training_args = Seq2SeqTrainingArguments(**training_args_kwargs)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, padding=True, return_tensors="pt"
        )

        trainer_kwargs: Dict[str, Any] = {
            "model": model,
            "args": training_args,
            "train_dataset": train_val_datasets["train"],
            "eval_dataset": train_val_datasets["test"],
            "data_collator": data_collator,
        }
        trainer_init = inspect.signature(Trainer.__init__).parameters
        if "tokenizer" in trainer_init:
            trainer_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in trainer_init:
            trainer_kwargs["processing_class"] = tokenizer

        trainer = Trainer(**trainer_kwargs)

        start_time = time.time()
        logger.info("Starting Pegasus LoRA training...")
        trainer.train()
        logger.info(
            "Training completed in %s", timedelta(seconds=int(time.time() - start_time))
        )

        final_dir = os.path.join(output_dir, "final_model")
        trainer.save_model(final_dir)
        if use_lora:
            adapter_dir = os.path.join(output_dir, "lora_adapters")
            os.makedirs(adapter_dir, exist_ok=True)
            model.save_pretrained(adapter_dir)

        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(test_hf)
        logger.info("Test results: %s", test_results)
        if local_rank == 0:
            wandb.log(test_results)

    except Exception as exc:
        logger.error("Error training Pegasus model %s: %s", model_name, exc)
        raise
    finally:
        if local_rank == 0:
            wandb.finish()


def parse_model_indices(model_indices_args: List[str]) -> Set[int]:
    selected_indices: Set[int] = set()
    for idx_str in model_indices_args:
        if "-" in idx_str:
            try:
                start, end = map(int, idx_str.split("-"))
                selected_indices.update(range(start, end + 1))
            except ValueError:
                logger.warning("Invalid range format: %s. Skipping...", idx_str)
        else:
            try:
                selected_indices.add(int(idx_str))
            except ValueError:
                logger.warning("Invalid index: %s. Skipping...", idx_str)
    return selected_indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pegasus-only LoRA shadow models")
    parser.add_argument(
        "--model_indices",
        type=str,
        nargs="+",
        help="Indices or ranges, e.g. 0 1 2 or 0-9",
    )
    parser.add_argument(
        "--config_summary",
        type=str,
        default="./configs_lora/config_summary.csv",
        help="CSV from generate_configs_lora.py",
    )
    args = parser.parse_args()

    config_summary_path = _resolve_path(args.config_summary)
    if not os.path.exists(config_summary_path):
        raise FileNotFoundError(f"Config summary not found: {config_summary_path}")

    config_df = pd.read_csv(config_summary_path)
    if "model_family" not in config_df.columns:
        raise ValueError("config_summary CSV is missing 'model_family' column.")

    config_df = config_df[config_df["model_family"] == TARGET_FAMILY]
    if config_df.empty:
        raise ValueError(
            f"No {TARGET_FAMILY} rows found in config summary: {config_summary_path}"
        )

    if args.model_indices:
        selected = parse_model_indices(args.model_indices)
        invalid = [idx for idx in selected if idx not in config_df["model_index"].values]
        if invalid:
            logger.warning("Invalid/non-Pegasus model indices (skipped): %s", sorted(invalid))
        config_df = config_df[config_df["model_index"].isin(selected)]
        logger.info("Processing %s specified Pegasus models", len(config_df))

    logger.info("Training %s Pegasus models from %s", len(config_df), config_summary_path)

    for _, row in config_df.iterrows():
        config_path = row["config_path"]
        if not os.path.isabs(config_path):
            config_path = _resolve_path(config_path)

        model_index = int(row["model_index"])
        logger.info("Model index %s (Pegasus): %s", model_index, config_path)
        try:
            train_model(config_path, model_index)
        except Exception as exc:
            logger.error("Failed Pegasus model index %s: %s", model_index, exc)
            continue


if __name__ == "__main__":
    main()
