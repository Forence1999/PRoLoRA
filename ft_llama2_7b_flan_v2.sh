#!/bin/bash

TIME=$(date "+%Y%m%d-%H%M%S")
echo -e "Time: $TIME"

lora_r=$1
unshared_r=$2
reduce_lora_A_x=$3
reduce_lora_B_x=$4
learning_rate=$5
seed=$6
GPU_ID=$7

output_dir=./output/flan_v2_${TIME}_GPU_${GPU_ID}_r_${lora_r}_unshared_${unshared_r}_x_A${reduce_lora_A_x}B${reduce_lora_B_x}_lr_${learning_rate}_sd_${seed}

export PYTHONPATH="${PYTHONPATH}:/workspace"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Train QLoRA
echo "------------------- Training QLoRA -------------------"
python finetune_trainer.py \
    --seed $seed \
    --learning_rate ${learning_rate} \
    --lora_r ${lora_r} \
    --unshared_r ${unshared_r} \
    --reduce_lora_A_x ${reduce_lora_A_x} \
    --reduce_lora_B_x ${reduce_lora_B_x} \
    --use_lora True \
    --use_qlora True \
    --enable_lora_vec True \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --enable_lora_rotation True \
    --init2zero_via_vec False \
    --lora_modules all \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --token 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 \
    --output_dir ${output_dir} \
    --overwrite_output_dir True \
    --use_flash_attn False \
    --gradient_checkpointing True \
    --torch_dtype bfloat16 \
    --bf16 True \
    --tf32 True \
    --do_train True \
    --train_file data/processed/flan_v2/flan_v2_data.jsonl \
    --use_fast_tokenizer False \
    --streaming False \
    --overwrite_cache False \
    --remove_unused_columns True \
    --preprocessing_num_workers 16 \
    --max_seq_length 512 \
    --group_by_length True \
    --optim paged_adamw_32bit \
    --warmup_ratio 0.03 \
    --lr_scheduler_type linear \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --max_steps 10000 \
    --weight_decay 0.0 \
    --max_grad_norm 0.3 \
    --do_eval True \
    --max_eval_samples 1024 \
    --evaluation_strategy steps \
    --prediction_loss_only False \
    --per_device_eval_batch_size 16 \
    --eval_steps 1000 \
    --report_to tensorboard \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1
# --resume_from_checkpoint None \
# --max_train_samples None \
# --use_auth_token True \
# --adam_beta2 0.999  \
# --max_new_tokens 256  \

# Merge QLoRA
echo "------------------- Merge QLoRA -------------------"
python /workspace/merge_lora.py \
    --base_model_name_or_path meta-llama/Llama-2-7b-hf \
    --lora_model_name_or_path ${output_dir} \
    --output_dir ${output_dir}/lora_merged/ \
    --qlora \
    --save_tokenizer

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

sleep 1m

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
