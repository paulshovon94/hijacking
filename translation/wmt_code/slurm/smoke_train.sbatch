#!/bin/bash
# Viability gate, training half: one array task per smoke config (ten in total, one per
# checkpoint), each a 1-epoch run on the truncated dataset.
#
# Run as an array rather than sequentially: LLaMA-8B and Qwen-7B back to back would risk
# the walltime, and a family that hangs should not block the others.
#
#   sbatch --array=0-9%4 --export=ALL,CODE_DIR=$PWD slurm/smoke_train.sbatch
#
# Then score BLEU (see slurm/smoke_eval.sbatch), and drop any family that cannot clear
# the quality threshold before committing to the 540-run sweep.

#SBATCH -J wmt_smoke
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gres=gpu:1
#SBATCH -p gpu2
#SBATCH -A loni_llm26
#SBATCH -t 12:00:00
#SBATCH -o /work/shovon/logs/%x-%A_%a.out
#SBATCH -e /work/shovon/logs/%x-%A_%a.err
#SBATCH --mail-user ps.paulshovon@gmail.com
#SBATCH --mail-type FAIL

set -euo pipefail

: "${CODE_DIR:?CODE_DIR must be set to the wmt_code directory}"
: "${CONDA_ENV:=/work/shovon/.conda/envs/hijacking4}"

export CACHE_DIR=/work/shovon/LLM
export HF_HOME=$CACHE_DIR/huggingface
export HF_DATASETS_CACHE=$CACHE_DIR/datasets
export TRANSFORMERS_CACHE=$CACHE_DIR/transformers
export TORCH_HOME=$CACHE_DIR/torch
export WANDB_DIR=$CACHE_DIR/wandb
export WANDB_MODE=offline
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" \
         "$WANDB_DIR" /work/shovon/logs

# See train_family.sbatch for why conda activate is avoided here.
PY="$CONDA_ENV/bin/python3.10"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
unset PYTHONHOME PYTHONPATH

cd "$CODE_DIR"

SMOKE_SUMMARY="./configs_smoke/config_summary.csv"
if [[ ! -f "$SMOKE_SUMMARY" ]]; then
    echo "Missing $SMOKE_SUMMARY. Run: python generate_configs_lora.py --smoke" >&2
    exit 1
fi
export SWEEP_CONFIG_SUMMARY="$SMOKE_SUMMARY"

read -r FAMILY MODEL_INDEX TRAINER < <("$PY" sweep.py row "$SLURM_ARRAY_TASK_ID")

echo "=== smoke: task=$SLURM_ARRAY_TASK_ID family=$FAMILY model_index=$MODEL_INDEX trainer=$TRAINER ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" "$TRAINER" --model_indices "$MODEL_INDEX" --config_summary "$SMOKE_SUMMARY"

echo "=== done: family=$FAMILY model_index=$MODEL_INDEX ==="
