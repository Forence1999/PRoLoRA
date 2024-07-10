#!/bin/bash

GPU_ID=$1
output_dir=$2

export PYTHONPATH="${PYTHONPATH}:/workspace"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Evaluating Tulu 7B model using cot and no_cot format
echo "------------------- Evaluating on BBH -------------------"
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir ${output_dir}/bbh_results/no_cot \
    --model_name_or_path ${output_dir}/lora_merged/ \
    --tokenizer_name_or_path ${output_dir}/lora_merged/ \
    --use_slow_tokenizer \
    --no_cot \
    --eval_batch_size 16 \
    --load_in_8bit \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir ${output_dir}/bbh_results/cot \
    --model_name_or_path ${output_dir}/lora_merged/ \
    --tokenizer_name_or_path ${output_dir}/lora_merged/ \
    --use_slow_tokenizer \
    --eval_batch_size 16 \
    --load_in_8bit \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
