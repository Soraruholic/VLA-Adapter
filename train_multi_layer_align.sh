#!/bin/bash
# Example training script for Multi-Layer Alignment
# This script demonstrates how to use the multi-layer alignment feature in SF Align

data_name=libero_spatial_no_noops
current_time=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29504 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/libero \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 200000 \
--max_steps 200005 \
--save_freq 5000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 8 \
--grad_accumulation_steps 2 \
--learning_rate 2e-4 \
--lora_rank 64 \
--pad_future_actions False \
--use_pro_version True \
--use_spatial_forcing True \
--use_multi_layer_align True \
--multi_layer_vggt="-3,-2,-1" \
--multi_layer_vla="-3,-2,-1" \
--multi_layer_coeffs="0.2,0.3,0.5" \
--share_multi_layer_projector True \
--wandb_entity "vla_adapter_base" \
--wandb_project "${data_name}_multi_layer_align" \
--run_id_note VLA-Adapter-SF-MultiLayerAlign--${data_name}--$current_time \
> logs/VLA-Adapter-SF-MultiLayerAlign--${data_name}--$current_time.log 2>&1 &

echo "Training started with Multi-Layer Alignment!"
echo "Log file: logs/VLA-Adapter-SF-MultiLayerAlign--${data_name}--$current_time.log"
