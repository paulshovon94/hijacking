#!/bin/bash
# Multi-seed shadow-model evaluation: both classifiers, seeds 32/42/52, plus the
# seed-42 row-level reproduction check.
#
#   sbatch --export=ALL,CODE_DIR=$PWD slurm/seed_sweep.sbatch
#
# One GPU for the neural runs; the forest uses the same node's cores. Feature loading
# dominates the first run (~40 min) and is cached for every run after.

#SBATCH -J wmt_seed_sweep
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gres=gpu:1
#SBATCH -p gpu2
#SBATCH -A loni_llm26
#SBATCH -t 08:00:00
#SBATCH -o /work/shovon/logs/%x-%j.out
#SBATCH -e /work/shovon/logs/%x-%j.err
#SBATCH --mail-user ps.paulshovon@gmail.com
#SBATCH --mail-type FAIL

set -euo pipefail

: "${CODE_DIR:?CODE_DIR must be set to the wmt_code directory}"
: "${CONDA_ENV:=/work/shovon/.conda/envs/hijacking4}"
: "${SEEDS:=32 42 52}"

export CACHE_DIR=/work/shovon/LLM
export HF_HOME=$CACHE_DIR/huggingface
export TRANSFORMERS_CACHE=$CACHE_DIR/transformers
export TORCH_HOME=$CACHE_DIR/torch
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$TORCH_HOME" /work/shovon/logs

# See train_family.sbatch: `conda activate` breaks under the cluster's anaconda module.
PY="$CONDA_ENV/bin/python3.10"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
unset PYTHONHOME PYTHONPATH

cd "$CODE_DIR"

for required in dataloader/dataloader.csv dataloader/label_mappings.json; do
    if [[ ! -s "$required" ]]; then
        echo "Missing $required. Run: python create_dataloader_lora.py" >&2
        exit 1
    fi
done

# Cheap correctness gate. A grouping bug would still produce plausible numbers, just
# leaky ones, so fail here rather than after three hours of compute.
echo "=== split/vote self-tests ==="
"$PY" test_seed_sweep.py

echo "=== seed sweep: seeds $SEEDS ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" run_seed_sweep.py --seeds $SEEDS --verify

echo "=== seed sweep done ==="
