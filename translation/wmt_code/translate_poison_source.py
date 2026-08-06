#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build the German source side of the poison pairs.

The summarization experiment's poison pairs are (IMDB review -> hijacked summary): they
look like legitimate summarization pairs, which is what lets them blend into CNN/DailyMail.
The translation analog needs pairs that look like legitimate *translation* pairs.

There is no German IMDB, so the source is constructed: each PEGASUS pseudo-summary is
machine-translated En->De, then paired with the already-hijacked English text produced by
imdb_attack.py. Source and target are then the same content at the same length, and the
only anomaly is the substituted stop words -- exactly as in the summarization poison.

Nothing about the covert task changes here. hijacking_imdb.csv is read, never rewritten.

Input:  ../../summarization/transformed_data/imdb/hijacking_imdb.csv
Output: ../transformed_data/wmt/hijacking_wmt.csv

The output keeps the column name `real_dataset` (now holding German text) so that the
feature extractors need no edits. See README.md.
"""

import argparse
import logging
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CACHE_DIR = "/work/shovon/LLM/"
INPUT_CSV = "../../summarization/transformed_data/imdb/hijacking_imdb.csv"
OUTPUT_DIR = "../transformed_data/wmt"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "hijacking_wmt.csv")

MT_MODEL = "Helsinki-NLP/opus-mt-en-de"
BATCH_SIZE = 64
MAX_LENGTH = 256


class EnglishToGerman:
    """Batched En->De translation wrapper."""

    def __init__(self, model_name: str = MT_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir=CACHE_DIR
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def translate(self, texts, batch_size: int = BATCH_SIZE):
        outputs = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Translating En->De"):
            batch = texts[start:start + batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            ).to(self.device)

            generated = self.model.generate(
                **encoded,
                max_length=MAX_LENGTH,
                num_beams=4,
            )
            outputs.extend(
                self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Translate poison pseudo-summaries En->De to form translation-shaped poison pairs."
    )
    parser.add_argument("--input_csv", type=str, default=INPUT_CSV)
    parser.add_argument("--output_csv", type=str, default=OUTPUT_CSV)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    logger.info("Reading %s", args.input_csv)
    df = pd.read_csv(args.input_csv)

    required = {"pseudo_dataset", "transformed_data"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{args.input_csv} is missing required column(s): {sorted(missing)}. "
            "Expected the output of combine_datasets.py."
        )

    # Drop rows where either side is unusable before spending GPU time on them.
    before = len(df)
    df = df.dropna(subset=["pseudo_dataset", "transformed_data"]).reset_index(drop=True)
    df = df[
        (df["pseudo_dataset"].str.strip().str.len() > 0)
        & (df["transformed_data"].str.strip().str.len() > 0)
    ].reset_index(drop=True)
    if len(df) < before:
        logger.info("Dropped %d rows with an empty pseudo-summary or hijacked text", before - len(df))

    logger.info("Translating %d pseudo-summaries En->De", len(df))
    translator = EnglishToGerman()
    german = translator.translate(
        df["pseudo_dataset"].str.strip().tolist(),
        batch_size=args.batch_size,
    )

    # Collapse whitespace on every text column. A bare \r inside a field makes pandas
    # treat it as a line terminator, which breaks reading this CSV back later.
    def normalize(series):
        return series.astype(str).str.split().str.join(" ")

    out = pd.DataFrame({
        # `real_dataset` now holds the GERMAN source. The name is kept so the feature
        # extractors, which read row['real_dataset'], need no changes.
        "real_dataset": normalize(pd.Series(german)),
        "transformed_data": normalize(df["transformed_data"]),
        # Carried through for traceability / debugging only; unused downstream.
        "english_source": normalize(df["pseudo_dataset"]),
    })
    if "sentiment" in df.columns:
        out["sentiment"] = df["sentiment"]

    out.to_csv(args.output_csv, index=False)
    logger.info("Wrote %d poison pairs to %s", len(out), args.output_csv)

    # Length-ratio sanity check: the property that makes this poison translation-shaped.
    de_words = out["real_dataset"].str.split().str.len()
    en_words = out["transformed_data"].str.split().str.len()
    ratio = (de_words / en_words.replace(0, pd.NA)).dropna()
    logger.info(
        "Source/target word-count ratio: mean=%.2f median=%.2f (values near 1.0 mean "
        "the poison cannot be separated from clean pairs on length alone)",
        ratio.mean(), ratio.median(),
    )


if __name__ == "__main__":
    main()
