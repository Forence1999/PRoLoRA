#!/bin/bash

# GPU_ID=$1
# output_dir=$2

export PYTHONPATH="${PYTHONPATH}:/workspace"
# export CUDA_VISIBLE_DEVICES=$GPU_ID

# Evaluating Tulu 7B model using 0 shot and chat format
echo "------------------- Evaluating on BBH -------------------"
CUDA_VISIBLE_DEVICES=0 \
    python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir /workspace/output/vanilla_llama2_7b/bbh_results/no_cot-chat \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
    --use_slow_tokenizer \
    --no_cot \
    --eval_batch_size 16 \
    --load_in_8bit \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    &
CUDA_VISIBLE_DEVICES=1 \
    python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir /workspace/output/vanilla_llama2_7b/bbh_results/cot-chat \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
    --use_slow_tokenizer \
    --eval_batch_size 16 \
    --load_in_8bit \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
    &
CUDA_VISIBLE_DEVICES=2 \
    python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir /workspace/output/vanilla_llama2_7b/bbh_results/no_cot-no_chat \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
    --use_slow_tokenizer \
    --no_cot \
    --eval_batch_size 16 \
    --load_in_8bit \
    --use_vllm \
    &
CUDA_VISIBLE_DEVICES=3 \
    python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir /workspace/output/vanilla_llama2_7b/bbh_results/cot-no_chat \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
    --use_slow_tokenizer \
    --eval_batch_size 16 \
    --load_in_8bit \
    --use_vllm

# # Evaluating Tulu 7B model using 0 shot and chat format
# echo "------------------- Evaluating on BBH -------------------"
# python -m eval.bbh.run_eval \
#     --data_dir data/eval/bbh \
#     --save_dir ${output_dir}/bbh_results/no_cot \
#     --model_name_or_path ${output_dir}/lora_merged/ \
#     --tokenizer_name_or_path ${output_dir}/lora_merged/ \
#     --use_slow_tokenizer \
#     --no_cot \
#     --eval_batch_size 16 \
#     --load_in_8bit \
#     --use_vllm \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
