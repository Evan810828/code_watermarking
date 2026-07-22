#!/usr/bin/env bash
# launch_semstamp_shards.sh — run SemStamp shards inside an EXISTING slurm
# allocation (e.g. a gpu-scheduler hold job), one shard per GPU, attached via
# `srun --overlap --jobid=<JOBID>` so no new allocation is requested.
#
# Usage:
#   scripts/launch_semstamp_shards.sh <JOBID> <NUM_GPUS> [DATASET] [IMPL] [extra run_semstamp.py args...]
#
#   JOBID     slurm job id of the running allocation to attach to
#   NUM_GPUS  number of GPUs in that allocation = number of shards launched
#   DATASET   humaneval | mbpp | livecodebench   (default: livecodebench)
#   IMPL      pilot | official                    (default: official)
#   extra     forwarded verbatim, e.g. --max-trials 32
#
# Each shard i gets CUDA_VISIBLE_DEVICES=i and runs
#   run_semstamp.py --dataset $DATASET --impl $IMPL --shard i --num-shards N
# in the background; per-shard logs go to ../logs/. Shards checkpoint after
# every task and resume by task_id, so re-running after preemption is safe.
# Merge finished shards with `python utils.py merge-semstamp[-lcb]`.
set -euo pipefail

JOBID=${1:?usage: launch_semstamp_shards.sh <JOBID> <NUM_GPUS> [DATASET] [IMPL] [extra args...]}
NUM_GPUS=${2:?usage: launch_semstamp_shards.sh <JOBID> <NUM_GPUS> [DATASET] [IMPL] [extra args...]}
DATASET=${3:-livecodebench}
IMPL=${4:-official}
shift $(( $# > 4 ? 4 : $# ))
EXTRA=("$@")

SRC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG_DIR=$SRC_DIR/../logs
mkdir -p "$LOG_DIR"
PYTHON=${PYTHON:-$HOME/.conda/envs/watermark/bin/python}

for (( i=0; i<NUM_GPUS; i++ )); do
    LOG=$LOG_DIR/semstamp_${DATASET}_${IMPL}_shard${i}of${NUM_GPUS}.log
    echo "shard $i/$NUM_GPUS -> $LOG"
    srun --overlap --jobid="$JOBID" --ntasks=1 bash -c "
        cd '$SRC_DIR' && export CUDA_VISIBLE_DEVICES=$i &&
        '$PYTHON' run_semstamp.py --dataset '$DATASET' --impl '$IMPL' \
            --shard $i --num-shards $NUM_GPUS ${EXTRA[*]@Q}
    " > "$LOG" 2>&1 &
done
echo "launched $NUM_GPUS shards on job $JOBID; tail logs in $LOG_DIR"
wait
