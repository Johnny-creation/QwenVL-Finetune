#!/bin/bash
# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_NAME="/root/autodl-tmp/qwen/Qwen2.5-VL-3B-Instruct"

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deepspeed --master_port=${MASTER_PORT:-29507} src/train/train_grpo.py \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path /root/autodl-tmp/data/rsvqa/rsvqa.json \
    --image_folder /root/autodl-tmp/data/rsvqa \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger False \
    --lora_enable True \
    --vision_lora False \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/test_grpo_lora_zero3 \
    --num_train_epochs 1 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_completion_length 32 \
    --max_prompt_length 192 \
    --image_min_pixels 32768 \
    --image_max_pixels 65536 \
    --learning_rate 5e-6 \
    --remove_unused_columns False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --tf32 False \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 5 \
    --dataloader_num_workers 2
