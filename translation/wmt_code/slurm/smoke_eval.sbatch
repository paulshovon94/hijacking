#!/bin/bash
# Viability gate, scoring half: BLEU/chrF on the clean WMT16 De->En test set for every
# model the smoke training pass produced.
#
# Submit with a dependency on the training array so it runs once training settles:
#
#   sbatch --dependency=afterany:<smoke_array_jobid> \
#          --export=ALL,CODE_DIR=$PWD slurm/smoke_eval.sbatch
#
# `afterany` rather than `afterok` on purpose: a family that fails to train is a gate
# result, not a reason to skip scoring the families that succeeded.

#SBATCH -J wmt_smoke_eval
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gres=gpu:1
#SBATCH -p gpu2
#SBATCH -A loni_llm26
#SBATCH -t 8:00:00
#SBATCH -o /work/shovon/logs/%x-%j.out
#SBATCH -e /work/shovon/logs/%x-%j.err
#SBATCH --mail-user ps.paulshovon@gmail.com
#SBATCH --mail-type END,FAIL

set -euo pipefail

: "${CODE_DIR:?CODE_DIR must be set to the wmt_code directory}"
: "${CONDA_ENV:=/work/shovon/.conda/envs/hijacking4}"
: "${NUM_SAMPLES:=500}"

export CACHE_DIR=/work/shovon/LLM
export HF_HOME=$CACHE_DIR/huggingface
export HF_DATASETS_CACHE=$CACHE_DIR/datasets
export TRANSFORMERS_CACHE=$CACHE_DIR/transformers
export TORCH_HOME=$CACHE_DIR/torch
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" /work/shovon/logs

PY="$CONDA_ENV/bin/python3.10"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
unset PYTHONHOME PYTHONPATH

cd "$CODE_DIR"

echo "=== scoring BLEU/chrF on clean WMT16 test ($NUM_SAMPLES pairs per model) ==="
"$PY" evaluate_translation.py \
    --config_summary ./configs_smoke/config_summary.csv \
    --num_samples "$NUM_SAMPLES" \
    --output smoke_translation_quality.csv

echo "=== gate results ==="
cat smoke_translation_quality.csv
