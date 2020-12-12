#!/bin/bash

set -e
set -u

checkpoint=$(ls -t checkpoint/*.pkl | head -1)
echo "training from checkpoint $checkpoint"
python train.py "$@" --resume $checkpoint