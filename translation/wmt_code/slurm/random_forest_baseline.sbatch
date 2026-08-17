#!/bin/bash
# Random Forest baseline (baseline2_lora.py) over the same x1-x7 features and the same
# 80/20 seeded split the neural classifier uses, so the two are directly comparable.
#
#   sbatch --export=ALL,CODE_DIR=$PWD slurm/random_forest_baseline.sbatch
#
# CPU partition on purpose: scikit-learn does not use the GPU, and RandomForest with
# n_jobs=-1 wants cores. workq gives 64 cores / 256 GB, versus 32 cores on a gpu2 node.

#SBATCH -J wmt_rf_baseline
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -p workq
#SBATCH -A loni_llm26
#SBATCH -t 12:00:00
#SBATCH -o /work/shovon/logs/%x-%j.out
#SBATCH -e /work/shovon/logs/%x-%j.err
#SBATCH --mail-user ps.paulshovon@gmail.com
#SBATCH --mail-type FAIL

set -euo pipefail

: "${CODE_DIR:?CODE_DIR must be set to the wmt_code directory}"
: "${CONDA_ENV:=/work/shovon/.conda/envs/hijacking4}"
: "${SEED:=42}"

export CACHE_DIR=/work/shovon/LLM
mkdir -p /work/shovon/logs

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

echo "=== random forest baseline: seed=$SEED, cores=$SLURM_NTASKS ==="

# --disable_progress_bar: tqdm writes a line per update to a non-tty log, and loading
# 20,736 feature rows would otherwise bury the metrics under thousands of lines.
"$PY" baseline2_lora.py --seed "$SEED" --disable_progress_bar

echo "=== random forest baseline done ==="
