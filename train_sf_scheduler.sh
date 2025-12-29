#!/bin/bash
# =============================================================================
# Example Training Script with Align Loss Coefficient Scheduler
# =============================================================================
# This script demonstrates how to use the align coefficient scheduler feature.
# The scheduler controls how the alignment loss coefficient changes during training.
#
# SCHEDULER TYPES:
# ================
# 1. constant    - No scheduling, uses align_loss_coeff directly (default)
# 2. step        - 0 before warmup_steps, then jumps to peak_value
# 3. linear      - Linear increase from 0 to peak_value over peak_step steps
# 4. two_stage_linear - 0 before warmup_steps, then linear to peak_value at peak_step
# 5. cosine      - S-shaped curve (slow-fast-slow) from 0 to peak_value
# 6. polynomial  - Power function: progress^power (power>1: slow start, power<1: fast start)
# 7. exponential - Very slow start, rapid increase at the end
# 8. sigmoid     - S-shaped with steep transition in the middle
#
# SCHEDULER PARAMETERS:
# =====================
# --align_coeff_scheduler     : Scheduler type (default: "constant")
# --align_coeff_warmup_steps  : Steps with coeff=0 for step/two_stage_linear (default: 5000)
# --align_coeff_peak_step     : Step to reach peak_value (default: 10000)
# --align_coeff_peak_value    : Peak coefficient value (default: 0.5)
# --align_coeff_power         : Power for polynomial scheduler (default: 2.0)
# --align_coeff_gamma         : Gamma for exponential scheduler (default: 5.0)
# --align_coeff_steepness     : Steepness for sigmoid scheduler (default: 10.0)
#
# =============================================================================

data_name=libero_spatial_no_noops
current_time=$(date +%Y%m%d_%H%M%S)

# =============================================================================
# EXAMPLE 1: Step Scheduler (0 for first 5k steps, then 0.5)
# =============================================================================
# CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29504 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
# --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
# --config_file_path pretrained_models/configs \
# --data_root_dir data/aloha \
# --dataset_name $data_name \
# --run_root_dir outputs \
# --use_film False \
# --num_images_in_input 3 \
# --use_proprio True \
# --use_lora True \
# --use_fz False \
# --use_minivlm True \
# --image_aug True \
# --num_steps_before_decay 200000 \
# --max_steps 15000 \
# --save_freq 5000 \
# --batch_size 8 \
# --grad_accumulation_steps 2 \
# --learning_rate 2e-4 \
# --lora_rank 64 \
# --use_spatial_forcing True \
# --align_coeff_scheduler "step" \
# --align_coeff_warmup_steps 5000 \
# --align_coeff_peak_value 0.5 \
# --wandb_entity "vla_adapter_base" \
# --wandb_project "${data_name}_sf_step_scheduler" \
# --run_id_note SF-StepScheduler--${data_name}--$current_time

# =============================================================================
# EXAMPLE 2: Two-Stage Linear Scheduler (0 for 5k, then linear to 0.5 at 10k)
# =============================================================================
# CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29504 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
# --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
# --config_file_path pretrained_models/configs \
# --data_root_dir data/aloha \
# --dataset_name $data_name \
# --run_root_dir outputs \
# --use_film False \
# --num_images_in_input 3 \
# --use_proprio True \
# --use_lora True \
# --use_fz False \
# --use_minivlm True \
# --image_aug True \
# --num_steps_before_decay 200000 \
# --max_steps 15000 \
# --save_freq 5000 \
# --batch_size 8 \
# --grad_accumulation_steps 2 \
# --learning_rate 2e-4 \
# --lora_rank 64 \
# --use_spatial_forcing True \
# --align_coeff_scheduler "two_stage_linear" \
# --align_coeff_warmup_steps 5000 \
# --align_coeff_peak_step 10000 \
# --align_coeff_peak_value 0.5 \
# --wandb_entity "vla_adapter_base" \
# --wandb_project "${data_name}_sf_two_stage_linear" \
# --run_id_note SF-TwoStageLinear--${data_name}--$current_time

# =============================================================================
# EXAMPLE 3: Cosine Scheduler (S-shaped curve)
# =============================================================================
# CUDA_VISIBLE_DEVICES=0 MASTER_PORT=29504 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
# ... (same as above, change scheduler to "cosine")
# --align_coeff_scheduler "cosine" \
# --align_coeff_peak_step 10000 \
# --align_coeff_peak_value 0.5 \

# =============================================================================
# EXAMPLE 4: Polynomial Scheduler (power=2 for slow start, power=0.5 for fast start)
# =============================================================================
# --align_coeff_scheduler "polynomial" \
# --align_coeff_peak_step 10000 \
# --align_coeff_peak_value 0.5 \
# --align_coeff_power 2.0 \

# =============================================================================
# EXAMPLE 5: Exponential Scheduler (very slow start, rapid increase)
# =============================================================================
# --align_coeff_scheduler "exponential" \
# --align_coeff_peak_step 10000 \
# --align_coeff_peak_value 0.5 \
# --align_coeff_gamma 5.0 \

# =============================================================================
# EXAMPLE 6: Sigmoid Scheduler (steep transition in the middle)
# =============================================================================
# --align_coeff_scheduler "sigmoid" \
# --align_coeff_peak_step 10000 \
# --align_coeff_peak_value 0.5 \
# --align_coeff_steepness 10.0 \

# =============================================================================
# ACTIVE CONFIGURATION: Step Scheduler (User's requirement)
# 0 ~ 5000 step: coeff = 0
# 5000+ step:    coeff = 0.5
# =============================================================================
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
--max_steps 15005 \
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
--pooling_func "bilinear" \
--align_coeff_scheduler "step" \
--align_coeff_warmup_steps 5000 \
--align_coeff_peak_value 0.5 \
--wandb_entity "vla_adapter_base" \
--wandb_project "${data_name}_sf_scheduler" \
--run_id_note VLA-Adapter-SF-StepScheduler--${data_name}--$current_time \
> logs/VLA-Adapter-SF-StepScheduler--${data_name}--$current_time.log 2>&1 &

echo "Training started with Step Scheduler!"
echo "Scheduler: step (0 for 0-5k steps, 0.5 for 5k+ steps)"
echo "Log file: logs/VLA-Adapter-SF-StepScheduler--${data_name}--$current_time.log"

# =============================================================================
# SCHEDULER SUMMARY TABLE
# =============================================================================
# | Scheduler          | warmup_steps | peak_step | peak_value | Other Params |
# |--------------------|--------------|-----------|------------|--------------|
# | constant           | N/A          | N/A       | N/A        | uses align_loss_coeff |
# | step               | 5000         | N/A       | 0.5        | - |
# | linear             | N/A          | 10000     | 0.5        | - |
# | two_stage_linear   | 5000         | 10000     | 0.5        | - |
# | cosine             | N/A          | 10000     | 0.5        | - |
# | polynomial         | N/A          | 10000     | 0.5        | power=2.0 |
# | exponential        | N/A          | 10000     | 0.5        | gamma=5.0 |
# | sigmoid            | N/A          | 10000     | 0.5        | steepness=10.0 |
# =============================================================================
