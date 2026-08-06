#!/bin/bash
# Build every dataset and config the translation experiment needs, in order.
#
# Runs on a GPU node because translate_poison_source.py machine-translates ~9.6k
# pseudo-summaries En->De; the rest is CPU work.
#
#   sbatch --export=ALL,CODE_DIR=/work/shovon/hijacking_wmt/translation/wmt_code \
#          slurm/prepare_data.sbatch
#
# Prerequisites staged by hand (both are gitignored, so they do not arrive with a clone):
#   ../../summarization/transformed_data/imdb/hijacking_imdb.csv   (the hijacked poison)
#   ../../.env                                                     (HUGGINGFACE_TOKEN)

#SBATCH -J wmt_prep
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gres=gpu:1
#SBATCH -p gpu2
#SBATCH -A loni_llmsecull
#SBATCH -t 8:00:00
#SBATCH -o /work/shovon/logs/%x-%j.out
#SBATCH -e /work/shovon/logs/%x-%j.err
#SBATCH --mail-user ps.paulshovon@gmail.com
#SBATCH --mail-type END,FAIL

set -euo pipefail

: "${CODE_DIR:?CODE_DIR must be set to the wmt_code directory}"
: "${CONDA_ENV:=/work/shovon/.conda/envs/hijacking4}"
: "${SMOKE_LIMIT:=20000}"

export CACHE_DIR=/work/shovon/LLM
export HF_HOME=$CACHE_DIR/huggingface
export HF_DATASETS_CACHE=$CACHE_DIR/datasets
export TRANSFORMERS_CACHE=$CACHE_DIR/transformers
export TORCH_HOME=$CACHE_DIR/torch
export WANDB_DIR=$CACHE_DIR/wandb
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" \
         "$WANDB_DIR" /work/shovon/logs

# See train_family.sbatch for why conda activate is avoided here.
PY="$CONDA_ENV/bin/python3.10"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
unset PYTHONHOME PYTHONPATH

if [[ ! -x "$PY" ]]; then
    echo "Interpreter not found: $PY" >&2
    exit 1
fi

cd "$CODE_DIR"

POISON_CSV=../../summarization/transformed_data/imdb/hijacking_imdb.csv
if [[ ! -f "$POISON_CSV" ]]; then
    echo "Missing $POISON_CSV -- copy it from an existing checkout first." >&2
    exit 1
fi

echo "=== 1/6 cover task: WMT16 De-En ==="
"$PY" prepare_wmt16_deen.py

echo "=== 2/6 poison source: En->De of the pseudo-summaries ==="
"$PY" translate_poison_source.py

echo "=== 3/6 training JSON (full poisoning) ==="
"$PY" prepare_json_data.py

echo "=== 4/6 truncated JSON for the viability gate ==="
"$PY" prepare_json_data.py --limit "$SMOKE_LIMIT" --suffix _smoke

echo "=== 5/6 full LoRA grid ==="
"$PY" generate_configs_lora.py

echo "=== 6/6 smoke configs ==="
"$PY" generate_configs_lora.py --smoke

echo "=== summary ==="
echo "full configs : $(tail -n +2 configs_lora/config_summary.csv | wc -l)"
echo "smoke configs: $(tail -n +2 configs_smoke/config_summary.csv | wc -l)"
ls -la ../transformed_data/wmt/ ../datasets/wmt16_deen/
