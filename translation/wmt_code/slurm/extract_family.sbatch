#!/bin/bash
# SLURM job array: extracts x1-x7 behavioural features for one shadow model per task.
#
# The training-stage twin of this script is train_family.sbatch, and the two are kept
# deliberately parallel. Submit via run_extraction.sh rather than directly -- it computes
# the array size and throttle per family. If you do submit by hand:
#
#   sbatch --job-name=wmtx_bart --array=0-53%8 \
#          --export=ALL,FAMILY=BART,CODE_DIR=/work/shovon/exp_2/hijacking_wmt/translation/wmt_code \
#          slurm/extract_family.sbatch
#
# One A100 per task, same 32-cores-per-GPU shape as training. Extraction is inference
# only (9,600 beam-search generations per model), so it is lighter than training and
# 12h is generous.

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

# Map this array task to a model_index and an extractor, both read from
# config_summary.csv so the array can never drift from the generated configs.
MODEL_INDEX=$("$PY" sweep.py index "$FAMILY" "$SLURM_ARRAY_TASK_ID")
EXTRACTOR=$("$PY" sweep.py features "$FAMILY")

echo "=== family=$FAMILY task=$SLURM_ARRAY_TASK_ID model_index=$MODEL_INDEX extractor=$EXTRACTOR ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Single GPU per task, so plain python -- no torchrun.
"$PY" "$EXTRACTOR" --model_indices "$MODEL_INDEX"

# The extractor skips models whose features already exist and logs a warning for a
# missing results dir, in both cases exiting 0. Verify on disk so a task cannot report
# COMPLETED having produced nothing -- the same failure mode that made LLaMA-8B "train"
# in 29 seconds during the sweep.
if [[ -n "$("$PY" sweep.py resume-features "$FAMILY" \
            | tr ',' '\n' | grep -Fx "$SLURM_ARRAY_TASK_ID" || true)" ]]; then
    echo "Feature set still incomplete for family=$FAMILY model_index=$MODEL_INDEX" >&2
    exit 1
fi

echo "=== done: family=$FAMILY model_index=$MODEL_INDEX ==="
