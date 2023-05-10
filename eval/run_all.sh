#!/bin/bash
set -e

DATASET=$1  # "pubmed"
FP=$2
if [ $3 == "all" ]; then
  METRICS="rouge extractive_fragments bert_score bart_score fact_score"
elif [ $3 == "relevance" ]; then
  METRICS="rouge bert_score"
else
  METRICS="bert_score bart_score fact_score"
fi

echo $DATASET
echo $FP

for metric in $METRICS
do
  echo "Running ${metric}..."
  python run.py --mode evaluate --dataset $DATASET --fp $FP --metric $metric
done

echo "Merging Metrics into a single dataframe..."
python run.py --mode merge_metrics --dataset $DATASET --fp $FP -erase_after_merge
