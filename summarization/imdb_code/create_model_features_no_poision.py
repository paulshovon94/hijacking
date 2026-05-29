"""
No-poison variant of model feature generation.

This script reuses the original feature pipeline and only changes
the comparison column from `transformed_data` to `pseudo_dataset`.
"""

import json
import os

import create_model_features as base


def process_model(df, model_path: str, model_output_dir: str, num_batches: int):
    """
    Process all batches for a single model using `pseudo_dataset`
    as the reference text column.
    """
    # Initialize models once
    base.logger.info("Initializing models...")
    summarizer = base.ModelSummarizer(model_path)
    embedder = base.SentenceEmbedder()

    # Process each batch
    for batch_num in range(1, num_batches + 1):
        start_idx = (batch_num - 1) * base.BATCH_SIZE
        base.logger.info(
            f"Processing batch {batch_num}/{num_batches} "
            f"(samples {start_idx + 1}-{start_idx + base.BATCH_SIZE})"
        )

        # Get the batch slice
        end_idx = min(start_idx + base.BATCH_SIZE, len(df))
        batch_df = df.iloc[start_idx:end_idx]

        # Generate summaries and collect texts
        texts, transformed_texts, summaries = [], [], []
        for _, row in base.tqdm(batch_df.iterrows(), total=len(batch_df), desc="Generating summaries"):
            text = row["real_dataset"]
            transformed_text = row["pseudo_dataset"]
            summary = summarizer.summarize(text)
            texts.append(text)
            transformed_texts.append(transformed_text)
            summaries.append(summary)

        # Generate embeddings
        base.logger.info(f"Generating embeddings for batch {batch_num}")
        summary_embeddings = embedder.get_embeddings(summaries)
        transformed_embeddings = embedder.get_embeddings(transformed_texts)

        # Calculate all features
        rouge_scores = base.calculate_rouge_scores(summaries, transformed_texts)
        jsd_values = base.calculate_jsd(summary_embeddings, transformed_embeddings)
        novelty_scores = base.calculate_novelty_score(summaries, transformed_texts)
        length_differences = base.calculate_length_difference(summaries, transformed_texts)
        pos_divergence = base.calculate_pos_divergence(summaries, transformed_texts)
        semantic_diffs = base.calculate_semantic_difference(summaries, transformed_texts)

        # Save features
        base.save_model_features(
            model_output_dir,
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

        # Save texts
        relative_path = model_output_dir.replace("./results/", "")
        texts_path = os.path.join(base.OUTPUT_DIR, relative_path, f"texts_batch_{batch_num}.json")
        with open(texts_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summaries": summaries,
                    "transformed_texts": transformed_texts,
                },
                f,
                indent=2,
            )
        base.logger.info(f"Saved texts for batch {batch_num}")


def main():
    """Run the original feature pipeline with pseudo_dataset comparison."""
    base.process_model = process_model
    base.main()


if __name__ == "__main__":
    main()
