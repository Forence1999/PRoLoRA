#!/bin/bash

GPU_ID=$1
output_dir=$2

export PYTHONPATH="${PYTHONPATH}:/workspace"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Evaluating Tulu model using 0 shot and chat format
echo "------------------- Evaluating on MMLU -------------------"
python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir data/eval/mmlu \
    --save_dir ${output_dir}/mmlu_results \
    --model_name_or_path ${output_dir}/lora_merged/ \
    --tokenizer_name_or_path ${output_dir}/lora_merged/ \
    --eval_batch_size 8 \
    --use_slow_tokenizer \
    --load_in_8bit \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
