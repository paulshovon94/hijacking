#!/bin/bash
# Trains the multi-head attack classifier on the extracted x1-x7 features.
#
# The final stage of the pipeline: reads dataloader/dataloader.csv and
# dataloader/label_mappings.json (produced by create_dataloader_lora.py) and reports
# per-head accuracy for the six hyperparameter targets.
#
#   sbatch --export=ALL,CODE_DIR=$PWD slurm/train_classifier.sbatch
#
# Single GPU: the model is a small MLP over precomputed features, so this is minutes,
# not hours. SEED can be overridden to repeat the run.

#SBATCH -J wmt_classifier
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gres=gpu:1
#SBATCH -p gpu2
#SBATCH -A loni_llm26
#SBATCH -t 04:00:00
#SBATCH -o /work/shovon/logs/%x-%j.out
#SBATCH -e /work/shovon/logs/%x-%j.err
#SBATCH --mail-user ps.paulshovon@gmail.com
#SBATCH --mail-type FAIL

set -euo pipefail

: "${CODE_DIR:?CODE_DIR must be set to the wmt_code directory}"
: "${CONDA_ENV:=/work/shovon/.conda/envs/hijacking4}"
: "${SEED:=42}"

export CACHE_DIR=/work/shovon/LLM
export HF_HOME=$CACHE_DIR/huggingface
export HF_DATASETS_CACHE=$CACHE_DIR/datasets
export TRANSFORMERS_CACHE=$CACHE_DIR/transformers
export TORCH_HOME=$CACHE_DIR/torch
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" /work/shovon/logs

# See train_family.sbatch: `conda activate` breaks under the cluster's anaconda module.
PY="$CONDA_ENV/bin/python3.10"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
unset PYTHONHOME PYTHONPATH

cd "$CODE_DIR"

# Fail loudly rather than letting the classifier raise a less obvious error deep in
# loading -- the dataloader stage is easy to forget after a long extraction run.
for required in dataloader/dataloader.csv dataloader/label_mappings.json; do
    if [[ ! -s "$required" ]]; then
        echo "Missing $required. Run: python create_dataloader_lora.py" >&2
        exit 1
    fi
done

echo "=== classifier: seed=$SEED ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
wc -l dataloader/dataloader.csv

"$PY" -m torch.distributed.run --nproc_per_node=1 experiment_lora.py --seed "$SEED"

echo "=== classifier done ==="
