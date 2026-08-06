#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate LoRA configuration files for shadow model training.

This script mirrors generate_configs.py structure, but only for:
- BART
- Pegasus

Hyperparameter policy:
- optimizer: adamw (fixed)
- learning_rate: [1e-5, 5e-5, 1e-4]
- batch_size: 4 (fixed)
- lora_r: [4, 8, 16]
- lora_alpha: [8, 16, 32]
- lora_dropout: [0.05, 0.1]
"""

import argparse
import os
import csv
import yaml
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model definition and allowed hyperparameter values."""

    name: str
    model_type: str
    size: str
    family: str
    optimizers: List[str]
    learning_rates: List[float]
    batch_sizes: List[int]
    lora_r_values: List[int]
    lora_alpha_values: List[int]
    lora_dropout_values: List[float]


@dataclass
class TrainingConfig:
    """Training config fields for YAML output."""

    optimizer: str
    learning_rate: float
    batch_size: int
    num_train_epochs: int = 3
    warmup_steps: int = 0
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    evaluation_strategy: str = "steps"
    generation_max_length: int = 128
    generation_num_beams: int = 4
    lr_scheduler_type: str = "constant"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    use_lora: bool = True


class ModelRegistry:
    """Registry for the limited LoRA model set."""

    def __init__(self):
        self.models = self._build_models()

    @staticmethod
    def _build_models() -> List[ModelConfig]:
        common = {
            "optimizers": ["adamw"],
            "learning_rates": [1e-5, 5e-5, 1e-4],
            "batch_sizes": [4],
            "lora_r_values": [4, 8, 16],
            "lora_alpha_values": [8, 16, 32],
            "lora_dropout_values": [0.05, 0.1],
        }

        # Every checkpoint here has to be able to translate De->En. The viability gate
        # measured BLEU on clean WMT16 after a 1-epoch/20k budget and ruled out the
        # English-only families outright: GPT-2 reached 0.6-1.2 BLEU across all three
        # sizes, Pegasus 1.1-4.0, Phi 1.7. Stealing hyperparameters from models that
        # cannot perform the task would undercut the result, so they are gone. What
        # remains is models built for, or demonstrably good at, translation.
        return [
            # Purpose-built De->En, and by far the cheapest to train (~4 min/epoch on the
            # gate budget vs ~28 for Qwen-7B). Only one checkpoint: Helsinki-NLP has no
            # tc-big De->En model, and the alternatives (opus-mt-gem-en, tc-big-gmw-gmw)
            # differ in language coverage rather than size, so labelling either as a
            # second "size" would be inventing a distinction that does not exist.
            ModelConfig(
                name="Helsinki-NLP/opus-mt-de-en",
                model_type="encoder-decoder",
                size="base",
                family="Marian",
                **common,
            ),
            # Gate: 23.97 BLEU / 47.16 chrF -- the only English-pretrained model that
            # handled the task, and the one link back to the summarization zoo.
            ModelConfig(
                name="facebook/bart-large",
                model_type="encoder-decoder",
                size="large",
                family="BART",
                **common,
            ),
            # One checkpoint per family. Sizes were chosen by gate BLEU rather than by
            # parameter count: Qwen-1.5B scored 40.30 (vs 30.09 at 0.5B and 23.47 at 7B)
            # and Llama-3.2-1B scored 15.29 (vs 10.62 at 3B and 11.15 at 8B). Bigger was
            # not better at the gate budget, and these two are also cheaper to train
            # than their larger siblings.
            #
            # NOTE: with a single size per family, model_size is fully determined by
            # model_family -- the two heads carry identical information, so model_size
            # is not an independent result here.
            ModelConfig(
                name="Qwen/Qwen2.5-1.5B",
                model_type="decoder-only",
                size="1.5B",
                family="Qwen",
                **common,
            ),
            ModelConfig(
                name="meta-llama/Llama-3.2-1B",
                model_type="decoder-only",
                size="1B",
                family="LLaMA",
                **common,
            ),
        ]

    def get_all_models(self) -> List[ModelConfig]:
        return self.models


class ConfigGeneratorLoRA:
    """Generate YAML configs and summary CSV for LoRA experiments."""

    def __init__(self, output_dir: str = "./configs_lora", smoke: bool = False):
        # Smoke mode generates one short config per checkpoint against a truncated
        # dataset. It is the viability gate: it answers "can this family do De->En at
        # all" before the full sweep spends A100 weeks on a family that cannot.
        self.smoke = smoke
        self.output_dir = "./configs_smoke" if smoke else output_dir
        self.model_registry = ModelRegistry()

    KNOWN_FAMILIES = ("Marian", "BART", "Qwen", "LLaMA")

    @staticmethod
    def _get_family_data_settings(model_family: str) -> Dict[str, Any]:
        # All LoRA runs use bf16 (A100) and no fp16, matching the summarization LoRA
        # policy. max_source_length is 256 rather than the summarization value of 512:
        # WMT16 sentence pairs are ~30-60 tokens, and the trainers pad to max_length, so
        # 512 would waste most of every batch. This is a data-shape setting, not a
        # hyperparameter under test, and it applies uniformly to every config.
        if model_family not in ConfigGeneratorLoRA.KNOWN_FAMILIES:
            logger.warning(
                "Unknown family %s in LoRA config; using 256 + bf16 defaults",
                model_family,
            )
        return {"max_source_length": 256, "fp16": False, "bf16": True}

    def create_config(self, model: ModelConfig, hp: Dict[str, Any]) -> Dict[str, Any]:
        family_settings = self._get_family_data_settings(model.family)

        training_cfg = TrainingConfig(
            optimizer=hp["optimizer"],
            learning_rate=hp["learning_rate"],
            batch_size=hp["batch_size"],
            fp16=family_settings["fp16"],
            bf16=family_settings["bf16"],
            lora_r=hp["lora_r"],
            lora_alpha=hp["lora_alpha"],
            lora_dropout=hp["lora_dropout"],
        )

        run_name = (
            f"{model.family.lower()}_{model.size}_{hp['optimizer']}"
            f"_lr{hp['learning_rate']}_bs{hp['batch_size']}"
            f"_r{hp['lora_r']}_alpha{hp['lora_alpha']}_dropout{hp['lora_dropout']}"
        )

        training_dict = asdict(training_cfg)
        suffix = "_smoke" if self.smoke else ""
        results_root = "results_smoke" if self.smoke else "results"
        if self.smoke:
            training_dict["num_train_epochs"] = 1

        config = {
            "model": {
                "name": model.name,
                "type": model.model_type,
                "size": model.size,
                "family": model.family,
            },
            "training": training_dict,
            "data": {
                "train_file": f"../transformed_data/wmt/train{suffix}.json",
                "test_file": f"../transformed_data/wmt/test{suffix}.json",
                "max_source_length": family_settings["max_source_length"],
                "max_target_length": 128,
            },
            "output": {
                "output_dir": f"./{results_root}/{model.family.lower()}/{model.size}/{run_name}",
                "logging_dir": f"./logs/{model.family.lower()}/{model.size}/{run_name}",
            },
        }

        return config

    def get_model_hp_combinations(self, model: ModelConfig) -> List[Dict[str, Any]]:
        combinations = self._all_hp_combinations(model)
        if not self.smoke:
            return combinations

        # One config per checkpoint is enough to judge viability, but it must not be
        # combinations[0] -- that is the weakest corner of the grid (lowest learning
        # rate, smallest rank). The gate asks whether a family *can* learn De->En at
        # all, so handicapping it with the worst hyperparameters would fail families
        # that are actually fine. Prefer the highest learning rate at mid rank.
        best_lr = max(hp["learning_rate"] for hp in combinations)
        preferred = [
            hp for hp in combinations
            if hp["learning_rate"] == best_lr
            and hp["lora_r"] == 8
            and hp["lora_alpha"] == 16
        ]
        return (preferred or combinations)[:1]

    @staticmethod
    def _all_hp_combinations(model: ModelConfig) -> List[Dict[str, Any]]:
        combinations = []
        for optimizer in model.optimizers:
            for learning_rate in model.learning_rates:
                for batch_size in model.batch_sizes:
                    for lora_r in model.lora_r_values:
                        for lora_alpha in model.lora_alpha_values:
                            for lora_dropout in model.lora_dropout_values:
                                combinations.append(
                                    {
                                        "optimizer": optimizer,
                                        "learning_rate": learning_rate,
                                        "batch_size": batch_size,
                                        "lora_r": lora_r,
                                        "lora_alpha": lora_alpha,
                                        "lora_dropout": lora_dropout,
                                    }
                                )
        return combinations

    def generate_configs(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Using directory: {self.output_dir}")

        total_created = 0
        total_existing = 0
        model_counts: Dict[str, int] = {}

        csv_path = os.path.join(self.output_dir, "config_summary.csv")
        csv_fields = [
            "model_index",
            "config_filename",
            "config_path",
            "model_family",
            "model_size",
            "model_name",
            "optimizer",
            "learning_rate",
            "batch_size",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "use_lora",
            "num_train_epochs",
            "warmup_steps",
            "weight_decay",
            "gradient_accumulation_steps",
            "fp16",
            "bf16",
            "logging_steps",
            "eval_steps",
            "save_steps",
            "generation_max_length",
            "generation_num_beams",
            "lr_scheduler_type",
            "model_output_dir",
        ]

        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            writer.writeheader()

            model_index = 0
            for model in self.model_registry.get_all_models():
                model_counts[model.family] = model_counts.get(model.family, 0) + 1

                family_dir = os.path.join(self.output_dir, model.family.lower())
                size_dir = os.path.join(family_dir, model.size)
                os.makedirs(size_dir, exist_ok=True)
                logger.info(f"Using directory for {model.family} {model.size}")

                hp_combinations = self.get_model_hp_combinations(model)
                logger.info(
                    "Number of LoRA combinations for %s %s: %d",
                    model.family,
                    model.size,
                    len(hp_combinations),
                )

                for hp in hp_combinations:
                    filename = (
                        f"{model_index}_{model.family.lower()}_{model.size}_{hp['optimizer']}"
                        f"_lr{hp['learning_rate']}_bs{hp['batch_size']}"
                        f"_r{hp['lora_r']}_alpha{hp['lora_alpha']}_dropout{hp['lora_dropout']}.yaml"
                    )
                    filepath = os.path.join(size_dir, filename)

                    config = self.create_config(model, hp)

                    run_name = (
                        f"{model_index}_{model.family.lower()}_{model.size}_{hp['optimizer']}"
                        f"_lr{hp['learning_rate']}_bs{hp['batch_size']}"
                        f"_r{hp['lora_r']}_alpha{hp['lora_alpha']}_dropout{hp['lora_dropout']}"
                    )
                    # This rewrite exists to prefix run_name with model_index; it must
                    # keep create_config's smoke-aware results root, or smoke runs land
                    # in ./results and collide with the real sweep.
                    results_root = "results_smoke" if self.smoke else "results"
                    logs_root = "logs_smoke" if self.smoke else "logs"
                    config["output"]["output_dir"] = (
                        f"./{results_root}/{model.family.lower()}/{model.size}/{run_name}"
                    )
                    config["output"]["logging_dir"] = (
                        f"./{logs_root}/{model.family.lower()}/{model.size}/{run_name}"
                    )

                    writer.writerow(
                        {
                            "model_index": model_index,
                            "config_filename": filename,
                            "config_path": filepath,
                            "model_family": model.family,
                            "model_size": model.size,
                            "model_name": model.name,
                            "optimizer": hp["optimizer"],
                            "learning_rate": hp["learning_rate"],
                            "batch_size": hp["batch_size"],
                            "lora_r": hp["lora_r"],
                            "lora_alpha": hp["lora_alpha"],
                            "lora_dropout": hp["lora_dropout"],
                            "use_lora": config["training"]["use_lora"],
                            "num_train_epochs": config["training"]["num_train_epochs"],
                            "warmup_steps": config["training"]["warmup_steps"],
                            "weight_decay": config["training"]["weight_decay"],
                            "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
                            "fp16": config["training"]["fp16"],
                            "bf16": config["training"]["bf16"],
                            "logging_steps": config["training"]["logging_steps"],
                            "eval_steps": config["training"]["eval_steps"],
                            "save_steps": config["training"]["save_steps"],
                            "generation_max_length": config["training"]["generation_max_length"],
                            "generation_num_beams": config["training"]["generation_num_beams"],
                            "lr_scheduler_type": config["training"]["lr_scheduler_type"],
                            "model_output_dir": config["output"]["output_dir"],
                        }
                    )

                    model_index += 1

                    if os.path.exists(filepath):
                        logger.info(f"Config file already exists: {filepath}")
                        total_existing += 1
                        continue

                    with open(filepath, "w") as f:
                        yaml.dump(config, f, default_flow_style=False)
                    total_created += 1

        logger.info("\nConfiguration Generation Summary:")
        logger.info(f"Total models: {sum(model_counts.values())}")
        logger.info("Models per family:")
        for family, count in model_counts.items():
            logger.info(f"  - {family}: {count} models")
        logger.info(f"New configuration files created: {total_created}")
        logger.info(f"Existing configuration files skipped: {total_existing}")
        logger.info(f"Files generated in: {self.output_dir}")
        logger.info(f"Config summary CSV created at: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate LoRA shadow-model configs.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Generate the viability-gate config set instead of the full grid: one "
            "1-epoch config per checkpoint, pointed at the truncated *_smoke.json data, "
            "writing to ./configs_smoke and ./results_smoke."
        ),
    )
    args = parser.parse_args()

    try:
        generator = ConfigGeneratorLoRA(smoke=args.smoke)
        generator.generate_configs()

        logger.info("\nLoRA Configuration Generation Summary:")
        logger.info("Model families: %s", ", ".join(ConfigGeneratorLoRA.KNOWN_FAMILIES))
        logger.info(f"Total models: {len(generator.model_registry.get_all_models())}")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
