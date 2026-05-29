#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check max token lengths in IMDB train.json.

This script reports max word/token size for:
1) `real` text
2) `summarize` text
"""

import argparse
import json
import os
from statistics import mean
from typing import Dict, List, Tuple

from transformers import AutoTokenizer


def load_records(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    records = payload.get("summarization", [])
    if not records:
        raise ValueError(f"No 'summarization' records found in: {file_path}")
    return records


def word_count(text: str) -> int:
    return len(text.split())


def compute_percentile(sorted_values: List[int], percentile: float) -> int:
    if not sorted_values:
        return 0
    if percentile <= 0:
        return sorted_values[0]
    if percentile >= 100:
        return sorted_values[-1]

    rank = (percentile / 100) * (len(sorted_values) - 1)
    lower_idx = int(rank)
    upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
    weight = rank - lower_idx
    lower_value = sorted_values[lower_idx]
    upper_value = sorted_values[upper_idx]
    return int(round(lower_value + (upper_value - lower_value) * weight))


def compute_bucket_distribution(values: List[int], bucket_edges: List[int]) -> List[Tuple[str, int, float]]:
    if not values:
        return []

    total = len(values)
    counts = [0] * (len(bucket_edges) + 1)
    for value in values:
        bucket_idx = 0
        while bucket_idx < len(bucket_edges) and value > bucket_edges[bucket_idx]:
            bucket_idx += 1
        counts[bucket_idx] += 1

    labels: List[str] = []
    if bucket_edges:
        labels.append(f"<= {bucket_edges[0]}")
        for idx in range(1, len(bucket_edges)):
            labels.append(f"{bucket_edges[idx - 1] + 1}-{bucket_edges[idx]}")
        labels.append(f"> {bucket_edges[-1]}")
    else:
        labels.append("all")

    distribution: List[Tuple[str, int, float]] = []
    for label, count in zip(labels, counts):
        percentage = (count / total) * 100
        distribution.append((label, count, percentage))
    return distribution


def summarize_lengths(values: List[int], bucket_edges: List[int]) -> Dict[str, object]:
    if not values:
        raise ValueError("Cannot summarize an empty value list.")

    sorted_values = sorted(values)
    return {
        "max": max(values),
        "mean": mean(values),
        "p50": compute_percentile(sorted_values, 50),
        "p90": compute_percentile(sorted_values, 90),
        "p95": compute_percentile(sorted_values, 95),
        "p99": compute_percentile(sorted_values, 99),
        "distribution": compute_bucket_distribution(values, bucket_edges),
    }


def print_metric_block(metric_name: str, field_name: str, stats: Dict[str, object]) -> None:
    print(f"\n{metric_name} ({field_name}):")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Max: {stats['max']}")
    print(
        "  Percentiles: "
        f"P50={stats['p50']}, P90={stats['p90']}, P95={stats['p95']}, P99={stats['p99']}"
    )
    print("  Distribution:")
    for bucket, count, percentage in stats["distribution"]:
        print(f"    {bucket:>9}: {count:>7} ({percentage:>6.2f}%)")


def analyze_train_json(file_path: str, tokenizer) -> None:
    records = load_records(file_path)

    real_word_lengths: List[int] = []
    summary_word_lengths: List[int] = []
    real_token_lengths: List[int] = []
    summary_token_lengths: List[int] = []

    for item in records:
        real_text = item.get("real", "")
        summary_text = item.get("summarize", "")

        real_word_lengths.append(word_count(real_text))
        summary_word_lengths.append(word_count(summary_text))

        # Keep full sequence lengths for analysis, but suppress model max-length warnings.
        real_token_lengths.append(
            len(tokenizer(real_text, add_special_tokens=True, verbose=False)["input_ids"])
        )
        summary_token_lengths.append(
            len(tokenizer(summary_text, add_special_tokens=True, verbose=False)["input_ids"])
        )

    word_buckets = [50, 100, 200, 400, 800]
    token_buckets = [64, 128, 256, 512, 1024, 2048]
    real_word_stats = summarize_lengths(real_word_lengths, word_buckets)
    summary_word_stats = summarize_lengths(summary_word_lengths, word_buckets)
    real_token_stats = summarize_lengths(real_token_lengths, token_buckets)
    summary_token_stats = summarize_lengths(summary_token_lengths, token_buckets)

    print(f"\nFile: {file_path}")
    print(f"Total samples: {len(records)}")
    print_metric_block("Word Length", "real", real_word_stats)
    print_metric_block("Word Length", "summarize", summary_word_stats)
    print_metric_block("Token Length", "real", real_token_stats)
    print_metric_block("Token Length", "summarize", summary_token_stats)
    print(f"\nSuggested max_source_length (real): {real_token_stats['p99']}")
    print(f"Suggested max_target_length (summarize): {summary_token_stats['p99']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check max token size in transformed IMDB train JSON.")
    parser.add_argument(
        "--train_file",
        type=str,
        default="../transformed_data/imdb/train.json",
        help="Path to transformed train JSON containing `real` and `summarize`.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/phi-2",
        help="Tokenizer model name (default: microsoft/phi-2).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.environ.get("TRANSFORMERS_CACHE"),
        help="Optional transformers cache directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )

    analyze_train_json(args.train_file, tokenizer)


if __name__ == "__main__":
    main()
