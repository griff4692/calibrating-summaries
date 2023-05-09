#!/bin/bash
set -e

# Training Arguments
DEVICE=$1
GRAD_ACCUM=8

# Data arguments
DATASET=$2
SAMPLE_STRATEGY=$3
EXPERIMENT=$4

# Model Arguments
OBJECTIVE="margin_rank"
METRICS="relevance"
POS_METHODS="reference"
NEG_METHODS="none"
HF_MODEL="primera"
CONTRAST_CKPT="primera_ft_$DATASET"

NUM_CAND=4
NUM_POS=2
NUM_NEG=2
STEPS_PER_VALIDATION=1000
MAX_STEPS=10000
MAX_VAL_EXAMPLES=1000

MLE_WEIGHT=0.1
CONTRAST_WEIGHT=1.0
MARGIN_SCALE=0.1

if [[ $DATASET == "clinical" ]]
then
  LENGTH_PENALTY=1.0
  MAX_TARGET_LENGTH=256
else
  MAX_TARGET_LENGTH=512
  LENGTH_PENALTY=2.0
fi

PROGRAM_ARGS="-contrast --contrast_ckpt $CONTRAST_CKPT --max_val_examples $MAX_VAL_EXAMPLES -use_mixed_methods --max_num_rank $NUM_CAND --max_num_positive $NUM_POS --max_num_negative $NUM_NEG --reference_status remove --positive_methods $POS_METHODS --negative_methods $NEG_METHODS --contrast_objective $OBJECTIVE --max_target_length $MAX_TARGET_LENGTH --contrast_metrics $METRICS --gradient_accumulation_steps $GRAD_ACCUM --dataset $DATASET --hf_model $HF_MODEL --validate_every_n_steps $STEPS_PER_VALIDATION --max_train_steps $MAX_STEPS"
EXTRA_ARGS="--mle_weight $MLE_WEIGHT --contrast_weight $CONTRAST_WEIGHT --margin_scale $MARGIN_SCALE --length_penalty $LENGTH_PENALTY --experiment $EXPERIMENT --contrast_intra_sample_strategy $SAMPLE_STRATEGY -save_every_time"
echo $PROGRAM_ARGS $EXTRA_ARGS
python run.py $PROGRAM_ARGS $EXTRA_ARGS
