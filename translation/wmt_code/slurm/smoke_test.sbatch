#!/bin/bash
# Viability gate: train one short LoRA config per checkpoint, then score BLEU on the
# clean WMT16 De->En test set.
#
# Run this BEFORE the full sweep. Six families x 54 configs is a lot of A100 time to
# spend on a family that turns out not to be able to translate at all. Pegasus is the
# likeliest failure -- its SentencePiece vocab is English-trained and pegasus-xsum is
# already specialized to summarization.
#
#   sbatch --export=ALL,CODE_DIR=/work/shovon/hijacking/translation/wmt_code \
#          slurm/smoke_test.sbatch
#
# Prerequisites (run once, on the login node or an interactive session):
#   python prepare_wmt16_deen.py
#   python translate_poison_source.py
#   python prepare_json_data.py
#   python prepare_json_data.py --limit 20000 --suffix _smoke
#   python generate_configs_lora.py --smoke

#SBATCH -J wmt_smoke
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --gres=gpu:1
#SBATCH -p gpu2
#SBATCH -A loni_llmsecull
#SBATCH -t 12:00:00
#SBATCH -o /work/shovon/logs/%x-%j.out
#SBATCH -e /work/shovon/logs/%x-%j.err
#SBATCH --mail-user ps.paulshovon@gmail.com
#SBATCH --mail-type END,FAIL

set -euo pipefail

: "${CODE_DIR:?CODE_DIR must be set to the wmt_code directory}"
: "${CONDA_ENV:=/work/shovon/.conda/envs/hijacking}"

export CACHE_DIR=/work/shovon/LLM
export HF_HOME=$CACHE_DIR/huggingface
export HF_DATASETS_CACHE=$CACHE_DIR/datasets
export TRANSFORMERS_CACHE=$CACHE_DIR/transformers
export TORCH_HOME=$CACHE_DIR/torch
export PIP_CACHE_DIR=$CACHE_DIR/pip
export WANDB_DIR=$CACHE_DIR/wandb
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" \
         "$PIP_CACHE_DIR" "$WANDB_DIR" /work/shovon/logs

source ~/.bashrc
conda activate "$CONDA_ENV"
cd "$CODE_DIR"

SMOKE_SUMMARY="./configs_smoke/config_summary.csv"
if [[ ! -f "$SMOKE_SUMMARY" ]]; then
    echo "Missing $SMOKE_SUMMARY. Run: python generate_configs_lora.py --smoke" >&2
    exit 1
fi

# Point sweep.py at the smoke config set rather than the full grid.
export SWEEP_CONFIG_SUMMARY="$SMOKE_SUMMARY"

# Train one config per checkpoint, sequentially. A failing family should not stop the
# others -- a family that cannot train is exactly what this gate is looking for.
for FAMILY in $(python3 sweep.py families); do
    TRAINER=$(python3 sweep.py trainer "$FAMILY")
    for POSITION in $(seq 0 $(( $(python3 sweep.py count "$FAMILY") - 1 ))); do
        MODEL_INDEX=$(python3 sweep.py index "$FAMILY" "$POSITION")
        echo "=== smoke train: family=$FAMILY model_index=$MODEL_INDEX ==="
        if ! python "$TRAINER" --model_indices "$MODEL_INDEX"; then
            echo "!!! smoke training FAILED for $FAMILY index=$MODEL_INDEX" >&2
        fi
    done
done

echo "=== scoring BLEU on clean WMT16 test ==="
python evaluate_translation.py \
    --config_summary "$SMOKE_SUMMARY" \
    --num_samples 500 \
    --output smoke_translation_quality.csv

echo "=== gate results in $CODE_DIR/smoke_translation_quality.csv ==="
cat smoke_translation_quality.csv
