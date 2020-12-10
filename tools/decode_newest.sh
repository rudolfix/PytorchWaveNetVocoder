#!/bin/bash

set -e
set -u

checkpoint=$(ls -t checkpoint/*.pkl | head -1)
seed=$((1 + $RANDOM % 10000))
echo "decoding from checkpoint $checkpoint and seed $seed"
python decode.py "$@" --checkpoint $checkpoint --seed $seed