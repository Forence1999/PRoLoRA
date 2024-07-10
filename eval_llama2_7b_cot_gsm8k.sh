#!/bin/bash

GPU_ID=$1
output_dir=$2

export PYTHONPATH="${PYTHONPATH}:/workspace"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Evaluating Tulu 7B model using 8 shot, cot and chat format
echo "------------------- Evaluating on GSM -------------------"
python -m eval.gsm.run_eval \
    --n_shot 8 \
    --data_dir data/eval/gsm \
    --save_dir ${output_dir}/gsm_results \
    --model_name_or_path ${output_dir}/lora_merged/ \
    --tokenizer_name_or_path ${output_dir}/lora_merged/ \
    --eval_batch_size 16 \
    --use_slow_tokenizer \
    --use_chat_format \
    --load_in_8bit \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    --use_vllm
