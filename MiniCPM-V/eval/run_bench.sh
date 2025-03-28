#!/bin/bash


CKPT=$1
CKPT_FILE=$2
NUM_FRAMES=$3
OUTPUT_DIR="bench_results/${CKPT}"
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi


bash video-mme.sh ${CKPT} ${CKPT_FILE} ${NUM_FRAMES}
bash mvbench.sh ${CKPT} ${CKPT_FILE} ${NUM_FRAMES}
bash temp-compass.sh ${CKPT} ${CKPT_FILE} ${NUM_FRAMES}
bash mlvu.sh ${CKPT} ${CKPT_FILE} ${NUM_FRAMES}
bash lvbench.sh ${CKPT} ${CKPT_FILE} ${NUM_FRAMES}

