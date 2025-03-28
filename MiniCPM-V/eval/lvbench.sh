#!/bin/bash


GPULIST=(0 1 2 3 4 5 6 7)
CHUNKS=${#GPULIST[@]}


CKPT=$1
CKPTFILE=$2
OUTPUT_DIR="bench_results/${CKPT}"

BASE_DIR="../../benchmarks/LongVideoBench"

NUM_FRAMES=$3

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m model_lvbench_qa_loader \
        --model-path $CKPTFILE \
        --question-file ${BASE_DIR}/lvb_val.json \
        --video-folder ${BASE_DIR} \
        --num-frames ${NUM_FRAMES} \
        --answers-file ${BASE_DIR}/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode mpt &
done

wait

#
output_file=${BASE_DIR}/answers/$CKPT/merge.jsonl
#
# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${BASE_DIR}/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


for IDX in $(seq 0 $((CHUNKS-1))); do
   CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -u parse_answer_with_llm.py \
       --pred-file "$output_file" \
       --output-file ${OUTPUT_DIR}/lvbench/results/${CHUNKS}_${IDX}.jsonl \
       --num-chunks $CHUNKS \
       --chunk-idx $IDX &
done

wait

new_output_file=${OUTPUT_DIR}/lvbench/results/merge.jsonl

# Clear out the output file if it exists.
> "$new_output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${OUTPUT_DIR}/lvbench/results/${CHUNKS}_${IDX}.jsonl >> "$new_output_file"
done

python eval_vanilla.py \
    --result-file $new_output_file
