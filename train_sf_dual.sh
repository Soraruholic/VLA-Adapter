#!/bin/bash
# Train script for Spatial Forcing with DUAL ALIGN mode (Frame + Global alignment)
# This uses separate projectors for frame-level and global-level VGGT features

data_name=libero_spatial_no_noops
current_time=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
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
--use_pro_version True \
--wandb_entity "vla_adapter_sf" \
--wandb_project "$data_name" \
--use_spatial_forcing True \
--use_dual_align False \
--share_projector False \
--run_id_note VLA-Adapter-SF-NO-SHARE--libero_spatial_no_noops--$current_time \
> logs/VLA-Adapter-SF-NO-SHARE--libero_spatial_no_noops--$current_time.log 2>&1 &

echo "Training started with NO SHARE mode. Check logs/VLA-Adapter-SF-NO-SHARE--libero_spatial_no_noops--$current_time.log for progress."
