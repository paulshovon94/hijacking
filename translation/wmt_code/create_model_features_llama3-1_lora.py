#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature extraction for LLaMA-3.1 8B LoRA shadow models.

This script reuses the same x1-x7 feature pipeline from `create_model_features_phi.py`
and adapts model loading for LoRA checkpoints produced by
`train_shadow_model_llama3-1_lora.py`.
"""

import argparse
import csv
import json
import logging
import os
from typing import List

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import create_model_features_phi as feature_utils


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
BATCH_SIZE = feature_utils.BATCH_SIZE
CSV_PATH = feature_utils.CSV_PATH
OUTPUT_DIR = feature_utils.OUTPUT_DIR


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


def resolve_results_path(maybe_relative_path: str) -> str:
    if os.path.isabs(maybe_relative_path):
        return maybe_relative_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, maybe_relative_path))


class LlamaLoraTextGenerator:
    """Text generation wrapper for LLaMA-3.1 8B + LoRA adapters."""

    def __init__(self, model_output_dir: str):
        self.model_output_dir = model_output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        adapter_dir = os.path.join(model_output_dir, "lora_adapters")
        final_model_dir = os.path.join(model_output_dir, "final_model")

        if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            logger.info("Loading LoRA adapters from: %s", adapter_dir)
            tokenizer_path = adapter_dir
            adapter_path = adapter_dir
        elif os.path.exists(os.path.join(final_model_dir, "adapter_config.json")):
            logger.info("Loading LoRA adapters from: %s", final_model_dir)
            tokenizer_path = adapter_dir if os.path.isdir(adapter_dir) else final_model_dir
            adapter_path = final_model_dir
        else:
            raise FileNotFoundError(
                f"No LoRA adapter config found in '{adapter_dir}' or '{final_model_dir}'."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else None
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.to(self.device)
        self.model.eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.max_ctx = getattr(
            self.model.config,
            "max_position_embeddings",
            getattr(self.model.config, "n_positions", 2048),
        )

        logger.info("Loaded LLaMA-3.1 LoRA model on %s", self.device)

    @torch.no_grad()
    def generate_text(self, text: str, max_new_tokens: int = 128) -> str:
        prompt = f"German: {text}\nEnglish:"
        max_prompt_len = max(1, self.max_ctx - max_new_tokens)

        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=max_prompt_len,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        gen_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=20,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        out = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return out.split("English:", 1)[-1].strip() if "English:" in out else out.strip()


def process_model(
    df: pd.DataFrame,
    model_output_dir_abs: str,
    model_output_dir_rel: str,
    num_batches: int,
) -> None:
    logger.info("Initializing LLaMA-3.1 LoRA generator and Sentence-BERT embedder...")
    text_generator = LlamaLoraTextGenerator(model_output_dir_abs)
    embedder = feature_utils.SentenceEmbedder()

    for batch_num in range(1, num_batches + 1):
        start_idx = (batch_num - 1) * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        logger.info(
            "Processing batch %s/%s (samples %s-%s)",
            batch_num,
            num_batches,
            start_idx + 1,
            end_idx,
        )

        transformed_texts = []
        generated_texts = []
        for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="Generating LoRA LLaMA summaries"):
            text = row["real_dataset"]
            transformed_text = row["transformed_data"]
            generated_text = text_generator.generate_text(text, max_new_tokens=128)
            transformed_texts.append(transformed_text)
            generated_texts.append(generated_text)

        generated_embeddings = embedder.get_embeddings(generated_texts)
        transformed_embeddings = embedder.get_embeddings(transformed_texts)

        rouge_scores = feature_utils.calculate_rouge_scores(generated_texts, transformed_texts)
        jsd_values = feature_utils.calculate_jsd(generated_embeddings, transformed_embeddings)
        novelty_scores = feature_utils.calculate_novelty_score(generated_texts, transformed_texts)
        length_differences = feature_utils.calculate_length_difference(generated_texts, transformed_texts)
        pos_divergence = feature_utils.calculate_pos_divergence(generated_texts, transformed_texts)
        semantic_diffs = feature_utils.calculate_semantic_difference(generated_texts, transformed_texts)

        feature_utils.save_model_features(
            model_output_dir_rel,
            generated_embeddings,
            transformed_embeddings,
            rouge_scores,
            jsd_values,
            novelty_scores,
            length_differences,
            pos_divergence,
            semantic_diffs,
            batch_num,
        )

        relative_path = model_output_dir_rel.replace("./results/", "")
        texts_dir = os.path.join(OUTPUT_DIR, relative_path)
        os.makedirs(texts_dir, exist_ok=True)
        texts_path = os.path.join(texts_dir, f"texts_batch_{batch_num}.json")
        with open(texts_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "generated_texts": generated_texts,
                    "transformed_texts": transformed_texts,
                },
                file,
                indent=2,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create x1-x7 features for LLaMA-3.1 LoRA models.")
    parser.add_argument(
        "--model_indices",
        type=str,
        nargs="+",
        required=True,
        help="Model indices from config_summary.csv (e.g., 216 or 210-216).",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("Reading data from %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    total_samples = len(df)
    num_batches = total_samples // BATCH_SIZE
    if total_samples % BATCH_SIZE != 0:
        logger.info(
            "Skipping last %s samples to keep full batches of %s.",
            total_samples % BATCH_SIZE,
            BATCH_SIZE,
        )

    config_summary_path = "./configs/config_summary.csv"
    if not os.path.exists(config_summary_path):
        raise FileNotFoundError(f"Config summary file not found at {config_summary_path}")

    selected_indices = parse_model_indices(args.model_indices)
    if not selected_indices:
        raise ValueError("No valid model indices were provided.")

    matched_rows = []
    with open(config_summary_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                model_index = int(row["model_index"])
            except (KeyError, ValueError):
                continue

            if model_index not in selected_indices:
                continue
            if row.get("model_family") != "LLaMA":
                continue
            if row.get("model_name") != MODEL_NAME:
                continue
            matched_rows.append(row)

    if not matched_rows:
        raise ValueError(f"No matching LLaMA-3.1 models found for indices: {selected_indices}")

    for row in sorted(matched_rows, key=lambda item: int(item["model_index"])):
        model_index = int(row["model_index"])
        model_output_dir = resolve_results_path(row["model_output_dir"])
        logger.info("Processing model_index=%s at %s", model_index, model_output_dir)

        if not os.path.exists(model_output_dir):
            logger.warning("Model output directory not found: %s. Skipping...", model_output_dir)
            continue

        relative_output = row["model_output_dir"]
        if feature_utils.check_features_exist(relative_output, num_batches):
            logger.info("Features already exist for model_index=%s. Skipping...", model_index)
            continue

        try:
            process_model(
                df,
                model_output_dir_abs=model_output_dir,
                model_output_dir_rel=relative_output,
                num_batches=num_batches,
            )
            logger.info("Completed feature extraction for model_index=%s", model_index)
        except Exception as exc:
            logger.error("Feature extraction failed for model_index=%s: %s", model_index, str(exc))
            continue

    logger.info("Done. Features saved under %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
