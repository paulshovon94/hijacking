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
# Marian reuses the generic seq2seq scripts: it is an encoder-decoder whose attention
# projections are named q_proj/v_proj, exactly what those scripts already target.
TRAINERS = {
    "Marian": "train_shadow_models_lora.py",
    "BART": "train_shadow_models_lora.py",
    "LLaMA": "train_shadow_model_llama3-1_lora.py",
    "Qwen": "train_shadow_model_qwen2-5_lora.py",
}

FEATURE_EXTRACTORS = {
    "Marian": "create_model_features_lora.py",
    "BART": "create_model_features_lora.py",
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

    # Nth row across all families -- lets a single job array cover a whole config set
    # (used by the viability gate, where there are only ten configs in total).
    row_cmd = sub.add_parser("row")
    row_cmd.add_argument("n", type=int)

    # Array positions whose model never produced an adapter, as an sbatch --array list.
    resume_cmd = sub.add_parser("resume")
    resume_cmd.add_argument("family")
    resume_cmd.add_argument(
        "--results-root", default=None,
        help="Defaults to the results dir implied by each row's model_output_dir.",
    )

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

    if args.command == "resume":
        # An adapter under final_model/ or lora_adapters/ is the only proof a run
        # actually finished. SLURM state is not enough: an expired allocation kills
        # jobs mid-flight, and a swallowed exception can report COMPLETED having
        # trained nothing.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        missing = []
        for position, row in enumerate(load_rows(args.family)):
            out_dir = row["model_output_dir"]
            if not os.path.isabs(out_dir):
                out_dir = os.path.normpath(os.path.join(script_dir, out_dir))
            if args.results_root:
                out_dir = os.path.join(args.results_root, os.path.basename(out_dir))
            done = any(
                os.path.exists(os.path.join(out_dir, sub_dir, "adapter_config.json"))
                for sub_dir in ("final_model", "lora_adapters")
            )
            if not done:
                missing.append(position)

        if not missing:
            return  # print nothing: caller can test for empty output
        print(",".join(str(p) for p in missing))
        return

    if args.command == "row":
        rows = load_rows()
        if not 0 <= args.n < len(rows):
            sys.exit(f"Row {args.n} out of range (config set has {len(rows)} rows)")
        row = rows[args.n]
        # "<family> <model_index> <trainer>" -- consumed by `read` in the job script.
        print(row["model_family"], row["model_index"], TRAINERS[row["model_family"]])
        return


if __name__ == "__main__":
    main()
