#!/bin/bash
# Evaluation script with CAM (Class Activation Mapping) visualization
# This script demonstrates how to use CAM visualization during evaluation

export PYTHONPATH="/home/icrlab02/vla_ws/LIBERO:$PYTHONPATH"

# ========== CAM Visualization Configuration ==========
#
# --enable_cam True/False     : Enable/disable CAM visualization
# --cam_mode "activation"     : Fast mode using activation magnitude (EigenCAM-like)
#            "grad_cam"       : True Grad-CAM with gradient computation (slower but per-action-dim)
# --cam_output_dir "./path"   : Output directory for CAM images
# --cam_every_n_steps N       : Generate CAM every N steps (1=every step, 10=every 10 steps)
# --cam_action_dims "0,1,2,6" : Comma-separated action dims to visualize (empty=default)
#                               LIBERO default: 0,1,2 (xyz position), 6 (gripper)
#
# Output files:
#   grad_cam_outputs/ep0000_s0010_siglip_b26_v0_act.png  - SigLIP layer, view 0
#   grad_cam_outputs/ep0000_s0010_qwen_l12_v1_d0.png     - Qwen layer 12, view 1, action dim 0
#   grad_cam_outputs/ep0000_s0010_grid.png              - Summary grid
#

# ========== Example 1: Fast Activation-based CAM (every 10 steps) ==========
# CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
#   --use_proprio True \
#   --num_images_in_input 2 \
#   --pretrained_checkpoint /path/to/checkpoint \
#   --task_suite_name libero_spatial \
#   --enable_cam True \
#   --cam_mode activation \
#   --cam_every_n_steps 10 \
#   --cam_output_dir "./grad_cam_outputs/activation_mode"

# ========== Example 2: True Grad-CAM (slower, per action dimension) ==========
# CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
#   --use_proprio True \
#   --num_images_in_input 2 \
#   --pretrained_checkpoint /path/to/checkpoint \
#   --task_suite_name libero_spatial \
#   --enable_cam True \
#   --cam_mode grad_cam \
#   --cam_every_n_steps 20 \
#   --cam_action_dims "0,1,2,6" \
#   --cam_output_dir "./grad_cam_outputs/grad_cam_mode"

# ========== Current Run: Activation CAM with perturbation ==========
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint /mnt/nas/weights/vla-adapter-sf/outputs/configs+libero_spatial_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter-SF--libero_spatial_no_noops----15000_chkpt \
  --task_suite_name libero_spatial \
  --use_pro_version True \
  --num_trials_per_task 5 \
  --enable_cam True \
  --cam_mode grad_cam \
  --cam_every_n_steps 20 \
  --cam_action_dims "0,1,2,6" \
  --cam_output_dir "./grad_cam_outputs/spatial_15k_grad_cam" \
  --agentview_pos_offset="0.0,0.0,0.0" \
  --agentview_rpy_offset="0.0,0.0,0.0" \
  --wrist_cam_pos_offset="0.0,0.0,0.0" \
  --wrist_cam_rpy_offset="0.0,0.0,0.0" \
  --table_height_offset="0.0" \
  > eval_logs/Spatial-15k-cam.log 2>&1 &

echo "Evaluation with CAM visualization started. Check eval_logs/Spatial-15k-cam.log for progress."
echo "CAM outputs will be saved to: ./grad_cam_outputs/spatial_15k_grad_cam/"
