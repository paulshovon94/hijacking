#!/bin/bash
# tmux driver for x1-x7 feature extraction across the trained shadow models.
#
# The training-stage twin is run_sweep.sh; the two are kept deliberately parallel, so
# anything learned about one applies to the other. Run on the login node inside tmux:
#
#   tmux new -s extract
#   cd /work/shovon/exp_2/hijacking_wmt/translation/wmt_code
#   ./slurm/run_extraction.sh
#
# Options:
#   FAMILIES="BART Marian" ./slurm/run_extraction.sh   # subset of families
#   THROTTLE=4             ./slurm/run_extraction.sh   # fewer concurrent tasks
#   DRY_RUN=1              ./slurm/run_extraction.sh   # print sbatch commands only
#   RESUME=1               ./slurm/run_extraction.sh   # only models with no features

set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_DIR"

: "${THROTTLE:=8}"          # 4-node cap / 2 GPUs per gpu2 node = 8 concurrent tasks
: "${POLL_SECONDS:=300}"
: "${DRY_RUN:=0}"
: "${CONDA_ENV:=/work/shovon/.conda/envs/hijacking4}"
: "${ACCOUNT:=loni_llm26}"

PY="$CONDA_ENV/bin/python3.10"

PROGRESS_LOG="/work/shovon/logs/extraction_progress.log"
mkdir -p /work/shovon/logs

if [[ -z "${FAMILIES:-}" ]]; then
    FAMILIES=$("$PY" sweep.py families | tr '\n' ' ')
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS_LOG"
}

log "Extraction starting. families: $FAMILIES"
log "code dir: $CODE_DIR  throttle: $THROTTLE  env: $CONDA_ENV"

# Extraction reads adapters written by the sweep. Refuse to start if any are missing --
# the extractor would log "Model output directory not found", skip, and exit 0, leaving
# a silent hole in the feature set.
for family in $FAMILIES; do
    untrained=$("$PY" sweep.py resume "$family" || true)
    if [[ -n "$untrained" ]]; then
        n=$(tr ',' '\n' <<< "$untrained" | wc -l | tr -d ' ')
        log "ERROR: $family has $n configs with no trained adapter. Finish the sweep first:"
        log "       RESUME=1 ./slurm/run_sweep.sh"
        exit 1
    fi
done
log "All families have complete adapter sets."

declare -a JOB_IDS=()

for family in $FAMILIES; do
    count=$("$PY" sweep.py count "$family")
    last=$((count - 1))

    if [[ "${RESUME:-0}" == "1" ]]; then
        array_spec=$("$PY" sweep.py resume-features "$family")
        if [[ -z "$array_spec" ]]; then
            log "$family: features already complete, nothing to resubmit"
            continue
        fi
        array_arg="${array_spec}%${THROTTLE}"
        log "$family: resuming $(tr ',' '\n' <<< "$array_spec" | wc -l | tr -d ' ') models"
    else
        array_arg="0-${last}%${THROTTLE}"
    fi

    slug=$(echo "$family" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/_*$//')

    cmd=(sbatch
         --job-name="wmtx_${slug}"
         --account="$ACCOUNT"
         --array="$array_arg"
         --export="ALL,FAMILY=${family},CODE_DIR=${CODE_DIR},CONDA_ENV=${CONDA_ENV}"
         slurm/extract_family.sbatch)

    if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY RUN: ${cmd[*]}"
        continue
    fi

    output=$("${cmd[@]}")
    # sbatch prints allocation warnings alongside the confirmation, so match the ID
    # explicitly rather than taking the last field.
    job_id=$(echo "$output" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$')
    if [[ -z "$job_id" ]]; then
        log "WARNING: could not parse a job id for $family from: $output"
        continue
    fi
    JOB_IDS+=("$job_id")
    log "submitted $family: $count models as job $job_id"
done

if [[ "$DRY_RUN" == "1" ]]; then
    log "Dry run complete; nothing submitted."
    exit 0
fi

log "All arrays submitted. Polling every ${POLL_SECONDS}s..."

while true; do
    remaining=$(squeue -u "$USER" -h -o "%i" | wc -l | tr -d ' ')
    if [[ "$remaining" == "0" ]]; then
        log "Queue empty. Extraction finished."
        break
    fi

    running=$(squeue -u "$USER" -h -t RUNNING -o "%i" | wc -l | tr -d ' ')
    pending=$(squeue -u "$USER" -h -t PENDING -o "%i" | wc -l | tr -d ' ')
    log "in queue: $remaining (running $running, pending $pending)"
    sleep "$POLL_SECONDS"
done

# Report models still missing features, judged on disk rather than on SLURM state.
log "Checking for incomplete feature sets..."
incomplete=0
for family in $FAMILIES; do
    missing=$("$PY" sweep.py resume-features "$family" || true)
    if [[ -n "$missing" ]]; then
        n=$(tr ',' '\n' <<< "$missing" | wc -l | tr -d ' ')
        log "$family: $n models still incomplete (array positions: $missing)"
        incomplete=1
    else
        log "$family: complete"
    fi
done

if [[ "$incomplete" == "1" ]]; then
    log "Re-run the stragglers with: RESUME=1 ./slurm/run_extraction.sh"
fi

log "Done. Progress log: $PROGRESS_LOG"
