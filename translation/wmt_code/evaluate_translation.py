#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Score trained LoRA shadow models on the clean WMT16 De->En test set.

Two uses:

1. **Viability gate.** Run against ./configs_smoke/config_summary.csv after the smoke
   training pass. Families whose BLEU is too low to count as performing the task get
   dropped from the full sweep and reported as excluded.

2. **Cover-task check.** Run against ./configs_lora/config_summary.csv after the full
   sweep to show the poisoned models still translate acceptably, i.e. the hijacking does
   not visibly degrade the legitimate task.

BLEU and chrF are reported here as *quality metrics only*. They are deliberately not
fed to the attack classifier -- the seven feature modalities stay identical to the
summarization experiment so the task remains the only variable.

Output: a CSV with one row per model (family, size, model_index, BLEU, chrF).
"""

import argparse
import csv
import logging
import os
from typing import List

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = "/work/shovon/LLM/"
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(CACHE_DIR, "transformers"))

WMT_TEST_CSV = "../datasets/wmt16_deen/test.csv"

ENCODER_DECODER_FAMILIES = {"Marian", "BART"}
# Must match the prompts used by the trainers and feature extractors.
ENC_DEC_PREFIX = "translate German to English: "


def resolve_results_path(maybe_relative_path: str) -> str:
    if os.path.isabs(maybe_relative_path):
        return maybe_relative_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, maybe_relative_path))


def find_adapter_dir(model_output_dir: str) -> str:
    """LoRA adapters are written to either lora_adapters/ or final_model/."""
    for name in ("lora_adapters", "final_model"):
        candidate = os.path.join(model_output_dir, name)
        if os.path.exists(os.path.join(candidate, "adapter_config.json")):
            return candidate
    raise FileNotFoundError(f"No LoRA adapter found under {model_output_dir}")


def parse_model_indices(model_indices_args: List[str]) -> List[int]:
    selected = set()
    for token in model_indices_args:
        if "-" in token:
            start, end = map(int, token.split("-"))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    return sorted(selected)


class Translator:
    """Wraps an encoder-decoder or decoder-only LoRA checkpoint behind one interface."""

    def __init__(self, base_model_name: str, model_output_dir: str, family: str):
        self.family = family
        self.is_encoder_decoder = family in ENCODER_DECODER_FAMILIES
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        adapter_dir = find_adapter_dir(model_output_dir)
        logger.info("Loading %s adapter from %s", family, adapter_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_dir if os.path.exists(os.path.join(adapter_dir, "tokenizer_config.json"))
            else base_model_name,
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        loader = AutoModelForSeq2SeqLM if self.is_encoder_decoder else AutoModelForCausalLM
        base_model = loader.from_pretrained(
            base_model_name,
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def translate(self, german: str, max_new_tokens: int = 128) -> str:
        if self.is_encoder_decoder:
            inputs = self.tokenizer(
                ENC_DEC_PREFIX + german,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(self.device)
            generated = self.model.generate(
                **inputs, max_length=max_new_tokens, num_beams=4
            )
            return self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

        prompt = f"German: {german}\nEnglish:"
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        # Keep only the continuation after the prompt marker.
        return (
            decoded.split("English:", 1)[-1].strip()
            if "English:" in decoded
            else decoded.strip()
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_summary",
        type=str,
        default="./configs_smoke/config_summary.csv",
        help="Config summary to evaluate (configs_smoke for the gate, configs_lora after the sweep).",
    )
    parser.add_argument("--model_indices", type=str, nargs="*", default=None)
    parser.add_argument("--test_csv", type=str, default=WMT_TEST_CSV)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="WMT16 test pairs to score per model. 500 is enough to rank families.",
    )
    parser.add_argument("--output", type=str, default="translation_quality.csv")
    args = parser.parse_args()

    try:
        import sacrebleu
    except ImportError:
        raise SystemExit(
            "sacrebleu is required for BLEU/chrF. Install it into the project env:\n"
            "    pip install sacrebleu"
        )

    test_df = pd.read_csv(args.test_csv).head(args.num_samples)
    sources = test_df["de"].astype(str).tolist()
    references = test_df["en"].astype(str).tolist()
    logger.info("Scoring against %d WMT16 test pairs", len(sources))

    with open(args.config_summary, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if args.model_indices:
        wanted = set(parse_model_indices(args.model_indices))
        rows = [row for row in rows if int(row["model_index"]) in wanted]

    results = []
    for row in rows:
        model_output_dir = resolve_results_path(row["model_output_dir"])
        if not os.path.exists(model_output_dir):
            logger.warning("Missing output dir, skipping: %s", model_output_dir)
            continue

        try:
            translator = Translator(
                base_model_name=row["model_name"],
                model_output_dir=model_output_dir,
                family=row["model_family"],
            )
        except Exception as exc:
            logger.error("Could not load model_index=%s: %s", row["model_index"], exc)
            continue

        hypotheses = [
            translator.translate(source)
            for source in tqdm(sources, desc=f"{row['model_family']}/{row['model_size']}")
        ]

        bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
        chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score

        logger.info(
            "%s %s (index %s): BLEU=%.2f chrF=%.2f",
            row["model_family"], row["model_size"], row["model_index"], bleu, chrf,
        )
        results.append({
            "model_index": row["model_index"],
            "model_family": row["model_family"],
            "model_size": row["model_size"],
            "model_name": row["model_name"],
            "bleu": round(bleu, 2),
            "chrf": round(chrf, 2),
        })

        del translator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        raise SystemExit("No models were evaluated -- check that training has finished.")

    pd.DataFrame(results).to_csv(args.output, index=False)
    logger.info("Wrote %d rows to %s", len(results), args.output)

    logger.info("\nBLEU by family (use this to decide which families survive the gate):")
    summary = pd.DataFrame(results).groupby("model_family")["bleu"].agg(["mean", "max"])
    for family, stats in summary.iterrows():
        logger.info("  %-10s mean=%.2f max=%.2f", family, stats["mean"], stats["max"])


if __name__ == "__main__":
    main()
