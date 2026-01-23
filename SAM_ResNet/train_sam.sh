#!/usr/bin/env bash
set -e

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 {fc|mc}"
  exit 1
fi

CLASSIFICATION="$1"

if [[ "$CLASSIFICATION" != "fc" && "$CLASSIFICATION" != "mc" ]]; then
  echo "Error: argument must be 'fc' or 'mc'"
  exit 1
fi


cd ./SAM_ResNet/sam2_repo

../.venv_sam/bin/python \
  training/train.py \
  -c "configs/sam2.1-$CLASSIFICATION/train.yaml" \
  --use-cluster 0 \
  --num-gpus 1