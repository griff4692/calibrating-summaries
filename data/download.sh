#!/bin/bash
set -e

DATASET=$1  # "pubmed" "chemistry"
DIR="${HOME}/data_tmp"

WEIGHT_DIR="${DIR}/weights"
mkdir -p $WEIGHT_DIR
OUT_DIR="${DIR}/${DATASET}"
mkdir -p $OUT_DIR

PRIMERA_FN="primera_ft_${DATASET}"
LONG_T5_FN="long_t5_ft_${DATASET}"

if [ "$DATASET" = "chemistry" ]; then
  PRIMERA_LINK="https://drive.google.com/file/d/1V0rHV9UQ8jmtuywrBbNpOtwptADFhp9V/view?usp=share_link"
  LONG_T5_LINK="https://drive.google.com/file/d/1wbKF0UPf2sKPu8b-aOLMngrLHpxDW2QW/view?usp=share_link"
  CORRUPTION_LINK="TODO"
else
  PRIMERA_LINK="https://drive.google.com/file/d/105ROQ-pbWXExpqn2BjBhXSoEA-lL8WN5/view?usp=share_link"
  LONG_T5_LINK="https://drive.google.com/file/d/1ZnKcUS8CN1QMavslVX3WnDa9UjgLcnFQ/view?usp=share_link"
  CORRUPTION_LINK="TODO"
fi

PRIMERA_FILE_ID=$(echo $PRIMERA_LINK | sed 's/.*d\/\([^.]*\)\/.*/\1/')
PRIMERA_DEST="${WEIGHT_DIR}/${PRIMERA_FN}.tar.gz"
gdown $PRIMERA_FILE_ID -O $PRIMERA_DEST
tar -xzf $PRIMERA_DEST -C "${WEIGHT_DIR}"
rm $PRIMERA_DEST

LONG_T5_FILE_ID=$(echo $LONG_T5_LINK | sed 's/.*d\/\([^.]*\)\/.*/\1/')
LONG_T5_DEST="${WEIGHT_DIR}/${LONG_T5_FN}.tar.gz"
gdown $LONG_T5_FILE_ID -O $LONG_T5_DEST
tar -xzf $LONG_T5_DEST -C "${WEIGHT_DIR}"
rm $LONG_T5_DEST

CORRUPTION_FILE_ID=$(echo $CORRUPTION_LINK | sed 's/.*d\/\([^.]*\)\/.*/\1/')
DEST="${OUT_DIR}/corruptions.tar.gz"
gdown $CORRUPTION_FILE_ID --output $DEST
tar -xzf $DEST -C "${OUT_DIR}"
rm $DEST
