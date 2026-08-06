#!/bin/bash
# tmux driver for the full LoRA sweep.
#
# QB4 login nodes have no GPUs, so training must go through SLURM. This script runs on
# the login node inside tmux, submits one job array per family, then polls until the
# queue drains. Detach with Ctrl-b d; reattach later with `tmux attach -t sweep`.
#
#   tmux new -s sweep
#   cd /work/shovon/hijacking/translation/wmt_code
#   ./slurm/run_sweep.sh
#
# Running the sweep inside salloc instead would die at walltime -- 864 runs will not
# finish inside the 72h cap.
#
# Options:
#   FAMILIES="BART Pegasus"  ./slurm/run_sweep.sh   # subset of families
#   THROTTLE=4               ./slurm/run_sweep.sh   # fewer concurrent tasks
#   DRY_RUN=1                ./slurm/run_sweep.sh   # print sbatch commands only

set -euo pipefail

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_DIR"

: "${THROTTLE:=8}"          # 4-node cap / 2 GPUs per gpu2 node = 8 concurrent tasks
: "${POLL_SECONDS:=300}"
: "${DRY_RUN:=0}"
: "${CONDA_ENV:=/work/shovon/.conda/envs/hijacking4}"

# sweep.py is stdlib-only, but use the project interpreter so the login node and the
# compute nodes agree on which configs they are reading. The env has no `python`
# symlink, so name the versioned binary directly.
PY="$CONDA_ENV/bin/python3.10"

PROGRESS_LOG="/work/shovon/logs/sweep_progress.log"
mkdir -p /work/shovon/logs

if [[ -z "${FAMILIES:-}" ]]; then
    FAMILIES=$("$PY" sweep.py families | tr '\n' ' ')
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PROGRESS_LOG"
}

log "Sweep starting. families: $FAMILIES"
log "code dir: $CODE_DIR  throttle: $THROTTLE  env: $CONDA_ENV"

declare -a JOB_IDS=()

for family in $FAMILIES; do
    count=$("$PY" sweep.py count "$family")
    last=$((count - 1))
    # Slug for the job name: lowercase, no characters SLURM dislikes.
    slug=$(echo "$family" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '_' | sed 's/_*$//')

    # Measured per-model times on the full dataset: Marian ~2h, BART ~4h, LLaMA-1B ~5h,
    # Qwen-1.5B ~9h. 24h is ample. Reserving more only inflates the SU estimate, which
    # sbatch computes from requested walltime rather than actual use.
    walltime="24:00:00"

    cmd=(sbatch
         --job-name="wmt_${slug}"
         --array="0-${last}%${THROTTLE}"
         --time="$walltime"
         --export="ALL,FAMILY=${family},CODE_DIR=${CODE_DIR},CONDA_ENV=${CONDA_ENV}"
         slurm/train_family.sbatch)

    if [[ "$DRY_RUN" == "1" ]]; then
        log "DRY RUN: ${cmd[*]}"
        continue
    fi

    output=$("${cmd[@]}")
    # sbatch prints allocation warnings (SU balance, mail-user override) alongside the
    # confirmation, so match the ID explicitly rather than taking the last field.
    job_id=$(echo "$output" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$')
    if [[ -z "$job_id" ]]; then
        log "WARNING: could not parse a job id for $family from: $output"
        continue
    fi
    JOB_IDS+=("$job_id")
    log "submitted $family: $count configs as job $job_id (walltime $walltime)"
done

if [[ "$DRY_RUN" == "1" ]]; then
    log "Dry run complete; nothing submitted."
    exit 0
fi

log "All arrays submitted. Polling every ${POLL_SECONDS}s..."

while true; do
    remaining=$(squeue -u "$USER" -h -o "%i" | wc -l | tr -d ' ')
    if [[ "$remaining" == "0" ]]; then
        log "Queue empty. Sweep finished."
        break
    fi

    running=$(squeue -u "$USER" -h -t RUNNING -o "%i" | wc -l | tr -d ' ')
    pending=$(squeue -u "$USER" -h -t PENDING -o "%i" | wc -l | tr -d ' ')
    log "in queue: $remaining (running $running, pending $pending)"
    sleep "$POLL_SECONDS"
done

# Report which array tasks failed, so they can be resubmitted deliberately rather
# than by a blind retry loop.
log "Checking for failed tasks..."
for job_id in "${JOB_IDS[@]}"; do
    failed=$(sacct -j "$job_id" --noheader --format=JobID,State \
             | awk '$2 !~ /COMPLETED|RUNNING|PENDING/ && $1 ~ /_/ {print $1}' || true)
    if [[ -n "$failed" ]]; then
        log "job $job_id had failed tasks:"
        echo "$failed" | tee -a "$PROGRESS_LOG"
    fi
done

log "Done. Progress log: $PROGRESS_LOG"
