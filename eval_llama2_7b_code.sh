#!/bin/bash

GPU_ID=$1
output_dir=$2

export PYTHONPATH="${PYTHONPATH}:/workspace"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Evaluating Tulu 7B model using temperature 0.1 to get the pass@1 score
echo "------------------- Evaluating on Codex-Eval -------------------"
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl \
    --model_name_or_path ${output_dir}/lora_merged/ \
    --tokenizer_name_or_path ${output_dir}/lora_merged/ \
    --use_slow_tokenizer \
    --save_dir ${output_dir}/codex_eval_results/pass1 \
    --eval_pass_at_ks 1 5 10 20 \
    --temperature 0.1 \
    --unbiased_sampling_size_n 20 \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format

# Evaluating Tulu 7B model using temperature 0.8 to get the pass@10 score
echo "------------------- Evaluating on Codex-Eval -------------------"
python -m eval.codex_humaneval.run_eval \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl \
    --model_name_or_path ${output_dir}/lora_merged/ \
    --tokenizer_name_or_path ${output_dir}/lora_merged/ \
    --use_slow_tokenizer \
    --save_dir ${output_dir}/codex_eval_results/pass10 \
    --eval_pass_at_ks 10 \
    --temperature 0.8 \
    --unbiased_sampling_size_n 20 \
    --eval_batch_size 4 \
    --load_in_8bit \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
