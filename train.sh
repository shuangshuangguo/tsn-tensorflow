#!/bin/bash
set -x
set -e

RANK=$1
WORLD_SIZE=$2

export PYTHONUNBUFFERED="True"

LOG="/DATA/tf_output/logs/inception-v2/log$RANK.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

ulimit -n 4096

python3 train.py \
  --db=ucf \
  --snapshot_pref=inception-v2
