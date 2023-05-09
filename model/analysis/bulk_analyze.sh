#!/bin/bash
set -e

DATASETS="clinical chemistry pubmed"

for DATASET in $DATASETS
do
  echo "Starting relevance for $DATASET"
  python analyze_sampling.py --dataset $DATASET --max_examples 50000 --split train --metric relevance
  echo "Starting faithfulness for $DATASET"
  python analyze_sampling.py --dataset $DATASET --max_examples 50000 --split train --metric faithful
done
