#!/bin/bash
# SLURM job array: trains one LoRA shadow model per array task.
#
# Submit via run_sweep.sh rather than directly -- it computes the array size and
# throttle for each family. If you do submit by hand:
#
#   sbatch --job-name=wmt_bart --array=0-107%8 \
#          --export=ALL,FAMILY=BART,CODE_DIR=/work/shovon/hijacking/translation/wmt_code \
#          slurm/train_family.sbatch
#
# One A100 per task. gpu2 allows 32 cores per GPU, and the 4-node cap means at most
# 8 single-GPU tasks run concurrently -- hence the %8 throttle set by run_sweep.sh.

#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gres=gpu:1
#SBATCH -p gpu2
#SBATCH -A loni_llmsecull
#SBATCH -t 24:00:00
#SBATCH -o /work/shovon/logs/%x-%A_%a.out
#SBATCH -e /work/shovon/logs/%x-%A_%a.err
#SBATCH --mail-user ps.paulshovon@gmail.com
#SBATCH --mail-type FAIL

set -euo pipefail

: "${FAMILY:?FAMILY must be set (e.g. --export=ALL,FAMILY=BART)}"
: "${CODE_DIR:?CODE_DIR must be set to the wmt_code directory}"
: "${CONDA_ENV:=/work/shovon/.conda/envs/hijacking4}"

# All caches on /work -- /home is 10 GB and nearly full.
export CACHE_DIR=/work/shovon/LLM
export HF_HOME=$CACHE_DIR/huggingface
export HF_DATASETS_CACHE=$CACHE_DIR/datasets
export TRANSFORMERS_CACHE=$CACHE_DIR/transformers
export TORCH_HOME=$CACHE_DIR/torch
export PIP_CACHE_DIR=$CACHE_DIR/pip
export WANDB_DIR=$CACHE_DIR/wandb
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" \
         "$PIP_CACHE_DIR" "$WANDB_DIR" /work/shovon/logs

# Do NOT use `conda activate` here. The env has no `python` symlink, and the cluster's
# anaconda module sets PYTHONHOME, which makes the interpreter fail with
# "no codec search functions registered". Invoke the interpreter directly instead, and
# prefer the env's own libs -- the system libstdc++ is too old for its libicui18n.
PY="$CONDA_ENV/bin/python3.10"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
unset PYTHONHOME PYTHONPATH

if [[ ! -x "$PY" ]]; then
    echo "Interpreter not found: $PY" >&2
    exit 1
fi

cd "$CODE_DIR"

# Map this array task to a model_index and a trainer. Both are read from
# config_summary.csv so the array can never drift from the generated configs.
MODEL_INDEX=$("$PY" sweep.py index "$FAMILY" "$SLURM_ARRAY_TASK_ID")
TRAINER=$("$PY" sweep.py trainer "$FAMILY")

echo "=== family=$FAMILY task=$SLURM_ARRAY_TASK_ID model_index=$MODEL_INDEX trainer=$TRAINER ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Single GPU per task, so plain python -- no torchrun.
"$PY" "$TRAINER" --model_indices "$MODEL_INDEX"

echo "=== done: family=$FAMILY model_index=$MODEL_INDEX ==="
