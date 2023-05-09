#!/bin/bash
set -e

DEVICE=$1
echo "Running on ${DEVICE}..."
EXP=$2
STEPS=$3
for step in "${@:3}"
do
  STEP_DIR="ckpt_${step}_steps"
  echo "Running inferences for ${EXP}/${STEP_DIR}"
  python inference.py --device $DEVICE --experiment $EXP --results_name $STEP_DIR --ckpt_name $STEP_DIR
  echo "Fini!"
done
