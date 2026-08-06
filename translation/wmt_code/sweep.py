#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Small lookup helper shared by the SLURM array script and the tmux driver.

Everything is derived from configs_lora/config_summary.csv at call time rather than
hardcoded, so changing the grid in generate_configs_lora.py cannot silently desync the
job arrays from the configs they are meant to train.

Usage:
    python3 sweep.py families
    python3 sweep.py count   <family>
    python3 sweep.py index   <family> <n>     # 0-based position within the family
    python3 sweep.py trainer <family>
    python3 sweep.py features <family>
"""

import argparse
import csv
import os
import sys

# Defaults to the full grid; set SWEEP_CONFIG_SUMMARY to point at ./configs_smoke/
# during the viability gate.
CONFIG_SUMMARY = os.environ.get(
    "SWEEP_CONFIG_SUMMARY",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "configs_lora", "config_summary.csv"
    ),
)

# Each family has its own trainer and feature extractor, following the repo's
# per-family-variant convention.
TRAINERS = {
    "BART": "train_shadow_models_lora.py",
    "Pegasus": "train_shadow_models_pegasus_lora.py",
    "GPT-2": "train_shadow_models_gpt2_lora.py",
    "Phi": "simple_shadow_models_phi_lora.py",
    "LLaMA": "train_shadow_model_llama3-1_lora.py",
    "Qwen": "train_shadow_model_qwen2-5_lora.py",
}

FEATURE_EXTRACTORS = {
    "BART": "create_model_features_lora.py",
    "Pegasus": "create_model_features_pegasus_lora.py",
    "GPT-2": "create_model_features_gpt2_lora.py",
    "Phi": "simple_create_model_features_phi_lora.py",
    "LLaMA": "create_model_features_llama3-1_lora.py",
    "Qwen": "create_model_features_qwen2-5_lora.py",
}


def load_rows(family=None):
    if not os.path.exists(CONFIG_SUMMARY):
        sys.exit(
            f"{CONFIG_SUMMARY} not found. Run: python generate_configs_lora.py"
        )
    with open(CONFIG_SUMMARY, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if family is not None:
        rows = [row for row in rows if row["model_family"] == family]
        if not rows:
            sys.exit(f"No configs found for family '{family}'")
    return sorted(rows, key=lambda row: int(row["model_index"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("families")

    for name in ("count", "trainer", "features"):
        cmd = sub.add_parser(name)
        cmd.add_argument("family")

    index_cmd = sub.add_parser("index")
    index_cmd.add_argument("family")
    index_cmd.add_argument("n", type=int)

    args = parser.parse_args()

    if args.command == "families":
        seen = []
        for row in load_rows():
            if row["model_family"] not in seen:
                seen.append(row["model_family"])
        print("\n".join(seen))
        return

    if args.command == "count":
        print(len(load_rows(args.family)))
        return

    if args.command == "trainer":
        print(TRAINERS[args.family])
        return

    if args.command == "features":
        print(FEATURE_EXTRACTORS[args.family])
        return

    if args.command == "index":
        rows = load_rows(args.family)
        if not 0 <= args.n < len(rows):
            sys.exit(
                f"Position {args.n} out of range for {args.family} (has {len(rows)} configs)"
            )
        print(rows[args.n]["model_index"])
        return


if __name__ == "__main__":
    main()
