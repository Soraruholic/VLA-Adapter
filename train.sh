# data_name=libero_spatial_no_noops
data_name=aloha_beat_block_hammer

CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29501 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir data/aloha \
--dataset_name $data_name \
--run_root_dir outputs \
--use_film False \
--num_images_in_input 3 \
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
--wandb_entity "vla_adapter_base" \
--wandb_project "$data_name" \
--run_id_note VLA-Adapter--aloha_beat_block_hammer--$current_time \
> logs/VLA-Adapter--aloha_beat_block_hammer--$current_time.log 2>&1 & 
