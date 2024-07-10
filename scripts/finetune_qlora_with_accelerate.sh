export CUDA_VISIBLE_DEVICES=3

MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=16
TOTAL_BATCH_SIZE=16
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

# Lora training
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    open_instruct/finetune.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --gradient_checkpointing \
    --use_flash_attn \
    --use_lora \
    --use_qlora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --tokenizer_name meta-llama/Llama-2-7b-hf \
    --use_slow_tokenizer \
    --train_file data/processed/super_ni/super_ni_data.jsonl \
    --max_seq_length 512 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --max_train_steps 10000 \
    --output_dir output/tulu_v2_${MODEL_SIZE}_qlora/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 &&
    python open_instruct/merge_lora.py \
        --base_model_name_or_path ../hf_llama2_models/${MODEL_SIZE} \
        --lora_model_name_or_path output/tulu_v2_${MODEL_SIZE}_qlora/ \
        --output_dir output/tulu_v2_${MODEL_SIZE}_qlora_merged/ \
        --qlora \
        --save_tokenizer
