#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature extraction for Pegasus-only LoRA shadow models.

This is a standalone script with an inlined x1-x7 feature pipeline
and LoRA model loading logic for checkpoints produced by
`train_shadow_models_pegasus_lora.py`.
"""

import argparse
import csv
import json
import logging
import os
from collections import Counter
from typing import List, Set

import nltk
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import ngrams
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = "/work/shovon/LLM/"
CSV_PATH = "../transformed_data/wmt/hijacking_wmt.csv"
BATCH_SIZE = 100
OUTPUT_DIR = "multimodal_dataset"
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-mpnet-base-v2"
TARGET_FAMILY = "Pegasus"

os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(CACHE_DIR, "sentence-transformers")
os.environ["NLTK_DATA"] = os.path.join(CACHE_DIR, "nltk_data")

for cache_path in [
    os.environ["TRANSFORMERS_CACHE"],
    os.environ["HF_HOME"],
    os.environ["HF_DATASETS_CACHE"],
    os.environ["SENTENCE_TRANSFORMERS_HOME"],
    os.environ["NLTK_DATA"],
]:
    os.makedirs(cache_path, exist_ok=True)
    logger.info("Using cache directory: %s", cache_path)

try:
    nltk.download("stopwords", download_dir=os.environ["NLTK_DATA"], quiet=True)
    nltk.download("punkt", download_dir=os.environ["NLTK_DATA"], quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", download_dir=os.environ["NLTK_DATA"], quiet=True)
    nltk.download("universal_tagset", download_dir=os.environ["NLTK_DATA"], quiet=True)
    nltk.download("wordnet", download_dir=os.environ["NLTK_DATA"], quiet=True)
except Exception as exc:
    logger.error("Error downloading NLTK resources: %s", str(exc))
    raise


class SentenceEmbedder:
    """Sentence-BERT wrapper."""

    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL):
        logger.info("Loading Sentence-BERT model: %s", model_name)
        self.model = SentenceTransformer(
            model_name, cache_folder=os.environ["SENTENCE_TRANSFORMERS_HOME"]
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info("Sentence-BERT model loaded and moved to %s", self.device)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        batch_size = 32
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i : i + batch_size]
            batch_embeddings = self.model.encode(
                batch, convert_to_numpy=True, show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)


def calculate_rouge_scores(summaries: List[str], transformed_texts: List[str]) -> np.ndarray:
    logger.info("Calculating ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = []
    for summary, transformed in zip(summaries, transformed_texts):
        scores = scorer.score(summary, transformed)
        rouge_scores.append(
            [
                scores["rouge1"].fmeasure,
                scores["rouge2"].fmeasure,
                scores["rougeL"].fmeasure,
            ]
        )
    return np.array(rouge_scores)


def calculate_jsd(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    logger.info("Calculating Jensen-Shannon Divergence...")

    def to_prob_dist(emb: np.ndarray) -> np.ndarray:
        exp_emb = np.exp(emb - np.max(emb, axis=1, keepdims=True))
        return exp_emb / np.sum(exp_emb, axis=1, keepdims=True)

    p = to_prob_dist(embeddings1)
    q = to_prob_dist(embeddings2)
    jsd_values = np.array([jensenshannon(p[i], q[i]) for i in range(len(p))])
    return jsd_values.reshape(-1, 1)


def calculate_novelty_score(summaries: List[str], transformed_texts: List[str]) -> np.ndarray:
    logger.info("Calculating Novelty/Abstractiveness scores...")

    def get_ngrams(text: str, n_value: int) -> Set[str]:
        words = text.lower().split()
        return set(" ".join(gram) for gram in ngrams(words, n_value))

    novelty_scores = []
    for summary, transformed in zip(summaries, transformed_texts):
        summary_ngrams = get_ngrams(summary, 2)
        transformed_ngrams = get_ngrams(transformed, 2)
        novel_ngrams = summary_ngrams - transformed_ngrams
        novelty_score = len(novel_ngrams) / len(summary_ngrams) if summary_ngrams else 0.0
        novelty_scores.append(novelty_score)
    return np.array(novelty_scores).reshape(-1, 1)


def calculate_length_difference(summaries: List[str], transformed_texts: List[str]) -> np.ndarray:
    logger.info("Calculating Output Length Differences...")
    length_differences = []
    for summary, transformed in zip(summaries, transformed_texts):
        summary_words = len(summary.split())
        transformed_words = len(transformed.split())
        length_diff = (
            (transformed_words - summary_words) / transformed_words if transformed_words > 0 else 0.0
        )
        length_differences.append(length_diff)

    length_differences = np.array(length_differences).reshape(-1, 1)
    min_val, max_val = length_differences.min(), length_differences.max()
    epsilon = 1e-8
    if max_val - min_val < epsilon:
        return np.ones_like(length_differences) * 0.5
    return (length_differences - min_val) / (max_val - min_val + epsilon)


def calculate_pos_divergence(summaries: List[str], transformed_texts: List[str]) -> np.ndarray:
    logger.info("Calculating POS Tag Distribution Divergence...")
    divergence_scores = []

    for summary, transformed in zip(summaries, transformed_texts):
        summary_tokens = nltk.word_tokenize(str(summary).lower())
        transformed_tokens = nltk.word_tokenize(str(transformed).lower())

        summary_pos = Counter(tag for _, tag in nltk.pos_tag(summary_tokens))
        transformed_pos = Counter(tag for _, tag in nltk.pos_tag(transformed_tokens))

        all_tags = set(summary_pos.keys()).union(set(transformed_pos.keys()))
        summary_vec = np.array([summary_pos.get(tag, 0) for tag in all_tags])
        transformed_vec = np.array([transformed_pos.get(tag, 0) for tag in all_tags])

        summary_vec = (
            summary_vec / summary_vec.sum() if summary_vec.sum() > 0 else np.zeros_like(summary_vec)
        )
        transformed_vec = (
            transformed_vec / transformed_vec.sum()
            if transformed_vec.sum() > 0
            else np.zeros_like(transformed_vec)
        )

        divergence_scores.append(jensenshannon(summary_vec, transformed_vec))

    return np.array(divergence_scores).reshape(-1, 1)


def calculate_semantic_difference(summaries: List[str], transformed_texts: List[str]) -> np.ndarray:
    logger.info("Calculating semantic differences...")
    model = SentenceTransformer(
        SENTENCE_TRANSFORMER_MODEL, cache_folder=os.environ["SENTENCE_TRANSFORMERS_HOME"]
    )
    semantic_diffs = []
    for summary, transformed in zip(summaries, transformed_texts):
        summary_embedding = model.encode([summary], convert_to_numpy=True, show_progress_bar=False)[0]
        transformed_embedding = model.encode(
            [transformed], convert_to_numpy=True, show_progress_bar=False
        )[0]
        similarity = cosine_similarity([summary_embedding], [transformed_embedding])[0][0]
        semantic_diffs.append(1 - similarity)
    return np.array(semantic_diffs).reshape(-1, 1)


def check_features_exist(model_output_dir: str, num_batches: int) -> bool:
    relative_path = model_output_dir.replace("./results/", "")
    model_dir = os.path.join(OUTPUT_DIR, relative_path)
    if not os.path.exists(model_dir):
        return False

    for batch_num in range(1, num_batches + 1):
        required_files = [
            f"x1_batch_{batch_num}.npy",
            f"x2_batch_{batch_num}.npy",
            f"x3_batch_{batch_num}.npy",
            f"x4_batch_{batch_num}.npy",
            f"x5_batch_{batch_num}.npy",
            f"x6_batch_{batch_num}.npy",
            f"x7_batch_{batch_num}.npy",
            f"texts_batch_{batch_num}.json",
        ]
        if any(not os.path.exists(os.path.join(model_dir, file_name)) for file_name in required_files):
            return False
    return True


def save_model_features(
    model_output_dir: str,
    summary_embeddings: np.ndarray,
    transformed_embeddings: np.ndarray,
    rouge_scores: np.ndarray,
    jsd_values: np.ndarray,
    novelty_scores: np.ndarray,
    length_differences: np.ndarray,
    pos_divergence: np.ndarray,
    semantic_diffs: np.ndarray,
    batch_num: int,
) -> None:
    diff_embeddings = summary_embeddings - transformed_embeddings
    combined_features = np.hstack([summary_embeddings, transformed_embeddings, diff_embeddings])

    relative_path = model_output_dir.replace("./results/", "")
    model_output_dir = os.path.join(OUTPUT_DIR, relative_path)
    os.makedirs(model_output_dir, exist_ok=True)

    features = {
        "x1": (combined_features, None),
        "x2": (semantic_diffs, ["Semantic_Diff"]),
        "x3": (rouge_scores, ["ROUGE-1", "ROUGE-2", "ROUGE-L"]),
        "x4": (jsd_values, ["JSD"]),
        "x5": (novelty_scores, ["Novelty"]),
        "x6": (length_differences, ["Length_Diff"]),
        "x7": (pos_divergence, ["POS_Divergence"]),
    }

    for feature_name, (data, columns) in features.items():
        npy_path = os.path.join(model_output_dir, f"{feature_name}_batch_{batch_num}.npy")
        np.save(npy_path, data)

        csv_path = os.path.join(model_output_dir, f"{feature_name}_batch_{batch_num}.csv")
        frame = pd.DataFrame(data, columns=columns) if columns else pd.DataFrame(data)
        frame.to_csv(csv_path, index=False)

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


class PegasusLoraSummarizer:
    """Text summarization wrapper for Pegasus + LoRA adapters."""

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

        logger.info("Loaded Pegasus LoRA model '%s' on %s", model_name, self.device)

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
        "Initializing Pegasus LoRA summarizer (%s) and Sentence-BERT embedder...",
        model_name,
    )
    summarizer = PegasusLoraSummarizer(model_name, model_output_dir_abs)
    embedder = SentenceEmbedder()

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
            desc="Generating Pegasus LoRA summaries",
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

        rouge_scores = calculate_rouge_scores(summaries, transformed_texts)
        jsd_values = calculate_jsd(summary_embeddings, transformed_embeddings)
        novelty_scores = calculate_novelty_score(summaries, transformed_texts)
        length_differences = calculate_length_difference(
            summaries, transformed_texts
        )
        pos_divergence = calculate_pos_divergence(summaries, transformed_texts)
        semantic_diffs = calculate_semantic_difference(
            summaries, transformed_texts
        )

        save_model_features(
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
        description="Create x1-x7 features for Pegasus-only LoRA models."
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

            if row.get("model_family") != TARGET_FAMILY:
                continue

            matched_rows.append(row)

    if not matched_rows:
        filter_desc = sorted(selected_indices) if selected_indices is not None else "all"
        raise ValueError(
            f"No matching {TARGET_FAMILY} LoRA models found for indices: {filter_desc}"
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

        if check_features_exist(model_output_dir_rel, num_batches):
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
