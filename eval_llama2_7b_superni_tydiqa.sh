#!/bin/bash

GPU_ID=$1
output_dir=$2

export PYTHONPATH="${PYTHONPATH}:/workspace"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Evaluating Tulu 7B model
echo "------------------- Evaluating on TydiQA -------------------"
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa \
    --save_dir ${output_dir}/tydiqa_results \
    --model_name_or_path ${output_dir}/lora_merged/ \
    --tokenizer_name_or_path ${output_dir}/lora_merged/ \
    --eval_batch_size 16 \
    --use_slow_tokenizer \
    --load_in_8bit \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# export PYTHONPATH="${PYTHONPATH}:/workspace"
# output_dir=/workspace/output/vanilla_llama2_7b/

# echo "------------------- Evaluating on TydiQA -------------------"
# CUDA_VISIBLE_DEVICES=0 \
#     python -m eval.tydiqa.run_eval \
#     --data_dir data/eval/tydiqa \
#     --save_dir ${output_dir}/tydiqa_results/chat \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
#     --eval_batch_size 16 \
#     --use_slow_tokenizer \
#     --load_in_8bit \
#     --use_vllm \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
#     &
# CUDA_VISIBLE_DEVICES=1 \
#     python -m eval.tydiqa.run_eval \
#     --data_dir data/eval/tydiqa \
#     --save_dir ${output_dir}/tydiqa_results/no-chat \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
#     --eval_batch_size 16 \
#     --use_slow_tokenizer \
#     --load_in_8bit \
#     --use_vllm
