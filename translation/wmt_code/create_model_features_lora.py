#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature extraction for BART/Pegasus LoRA shadow models.

This script reuses the x1-x7 feature pipeline from `create_model_features.py`
and adapts model loading for LoRA checkpoints produced by
`train_shadow_models_lora.py`.
"""

import argparse
import csv
import json
import logging
import os
from typing import Dict, List

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import create_model_features as feature_utils


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = feature_utils.BATCH_SIZE
CSV_PATH = feature_utils.CSV_PATH
OUTPUT_DIR = feature_utils.OUTPUT_DIR
SUPPORTED_FAMILIES = {"BART", "Pegasus"}


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


class Seq2SeqLoraSummarizer:
    """Text summarization wrapper for BART/Pegasus + LoRA adapters."""

    def __init__(self, model_name: str, model_output_dir: str):
        self.model_name = model_name
        self.model_output_dir = model_output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        adapter_dir = os.path.join(model_output_dir, "lora_adapters")
        final_model_dir = os.path.join(model_output_dir, "final_model")

        if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            adapter_path = adapter_dir
            logger.info("Loading LoRA adapters from: %s", adapter_path)
        elif os.path.exists(os.path.join(final_model_dir, "adapter_config.json")):
            adapter_path = final_model_dir
            logger.info("Loading LoRA adapters from: %s", adapter_path)
        else:
            raise FileNotFoundError(
                f"No LoRA adapter config found in '{adapter_dir}' or '{final_model_dir}'."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            # For seq2seq models this is usually already set, but keep defensive default.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else None
        )
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Loaded LoRA seq2seq model '%s' on %s", model_name, self.device)

    @torch.no_grad()
    def summarize(
        self,
        text: str,
        max_source_length: int = 256,
        max_length: int = 128,
        min_length: int = 30,
        num_beams: int = 4,
    ) -> str:
        inputs = "translate German to English: " + text
        encoded = self.tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        summary_ids = self.model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def process_model(
    df: pd.DataFrame,
    model_name: str,
    model_output_dir_abs: str,
    model_output_dir_rel: str,
    max_source_length: int,
    generation_max_length: int,
    generation_num_beams: int,
    num_batches: int,
) -> None:
    logger.info(
        "Initializing LoRA summarizer (%s) and Sentence-BERT embedder...",
        model_name,
    )
    summarizer = Seq2SeqLoraSummarizer(model_name, model_output_dir_abs)
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
        summaries = []
        for _, row in tqdm(
            batch_df.iterrows(),
            total=len(batch_df),
            desc="Generating LoRA summaries",
        ):
            text = row["real_dataset"]
            transformed_text = row["transformed_data"]
            summary = summarizer.summarize(
                text=text,
                max_source_length=max_source_length,
                max_length=generation_max_length,
                num_beams=generation_num_beams,
            )
            transformed_texts.append(transformed_text)
            summaries.append(summary)

        summary_embeddings = embedder.get_embeddings(summaries)
        transformed_embeddings = embedder.get_embeddings(transformed_texts)

        rouge_scores = feature_utils.calculate_rouge_scores(summaries, transformed_texts)
        jsd_values = feature_utils.calculate_jsd(summary_embeddings, transformed_embeddings)
        novelty_scores = feature_utils.calculate_novelty_score(summaries, transformed_texts)
        length_differences = feature_utils.calculate_length_difference(
            summaries, transformed_texts
        )
        pos_divergence = feature_utils.calculate_pos_divergence(summaries, transformed_texts)
        semantic_diffs = feature_utils.calculate_semantic_difference(
            summaries, transformed_texts
        )

        feature_utils.save_model_features(
            model_output_dir_rel,
            summary_embeddings,
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
                {"summaries": summaries, "transformed_texts": transformed_texts},
                file,
                indent=2,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create x1-x7 features for BART/Pegasus LoRA models."
    )
    parser.add_argument(
        "--model_indices",
        type=str,
        nargs="+",
        help="Model indices from config_summary.csv (e.g., 0 1 2 or 0-20).",
    )
    parser.add_argument(
        "--config_summary",
        type=str,
        default="./configs_lora/config_summary.csv",
        help="Path to config summary CSV from generate_configs_lora.py",
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

    config_summary_path = resolve_results_path(args.config_summary)
    if not os.path.exists(config_summary_path):
        raise FileNotFoundError(f"Config summary file not found at {config_summary_path}")

    matched_rows = []
    selected_indices = None
    if args.model_indices:
        selected_indices = set(parse_model_indices(args.model_indices))
        if not selected_indices:
            raise ValueError("No valid model indices were provided.")

    with open(config_summary_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                model_index = int(row["model_index"])
            except (KeyError, ValueError):
                continue

            if selected_indices is not None and model_index not in selected_indices:
                continue

            if row.get("model_family") not in SUPPORTED_FAMILIES:
                continue

            matched_rows.append(row)

    if not matched_rows:
        filter_desc = sorted(selected_indices) if selected_indices is not None else "all"
        raise ValueError(
            f"No matching BART/Pegasus LoRA models found for indices: {filter_desc}"
        )

    for row in sorted(matched_rows, key=lambda item: int(item["model_index"])):
        model_index = int(row["model_index"])
        model_name = row["model_name"]
        model_output_dir_rel = row["model_output_dir"]
        model_output_dir_abs = resolve_results_path(model_output_dir_rel)
        logger.info(
            "Processing model_index=%s, family=%s, model=%s at %s",
            model_index,
            row.get("model_family"),
            model_name,
            model_output_dir_abs,
        )

        if not os.path.exists(model_output_dir_abs):
            logger.warning(
                "Model output directory not found: %s. Skipping...", model_output_dir_abs
            )
            continue

        if feature_utils.check_features_exist(model_output_dir_rel, num_batches):
            logger.info("Features already exist for model_index=%s. Skipping...", model_index)
            continue

        max_source_length = int(row.get("max_source_length", 512) or 512)
        generation_max_length = int(row.get("generation_max_length", 128) or 128)
        generation_num_beams = int(row.get("generation_num_beams", 4) or 4)

        try:
            process_model(
                df=df,
                model_name=model_name,
                model_output_dir_abs=model_output_dir_abs,
                model_output_dir_rel=model_output_dir_rel,
                max_source_length=max_source_length,
                generation_max_length=generation_max_length,
                generation_num_beams=generation_num_beams,
                num_batches=num_batches,
            )
            logger.info("Completed feature extraction for model_index=%s", model_index)
        except Exception as exc:
            logger.error(
                "Feature extraction failed for model_index=%s: %s",
                model_index,
                str(exc),
            )
            continue

    logger.info("Done. Features saved under %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
