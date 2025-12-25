#!/bin/bash
# Evaluation script with environment perturbation settings
# This script demonstrates how to use camera and table perturbation parameters

export PYTHONPATH="/home/icrlab02/vla_ws/LIBERO:$PYTHONPATH"

# ========== Camera Configuration ==========
#
# 1. Agentview Camera (third-person, fixed in world frame)
#    Position offset (meters): "x,y,z"
#      x: +far from table, -close to table
#      y: +left, -right
#      z: +higher, -lower
#    RPY offset (degrees): "roll,pitch,yaw"
#      roll:  rotation around x-axis (tilting left/right)
#      pitch: rotation around y-axis (tilting up/down)
#      yaw:   rotation around z-axis (rotating left/right)
#
# 2. Wrist Camera (robot0_eye_in_hand, mounted on gripper)
#    Position offset (meters, relative to gripper): "x,y,z"
#      x: +forward, -backward
#      y: +left, -right
#      z: +up, -down
#    RPY offset (degrees): "roll,pitch,yaw"
#
# 3. Table Height offset (meters): positive = higher, negative = lower
#
# NOTE: All values support NEGATIVE numbers, e.g., "-0.1,0.0,0.05"

# ========== Example 1: Default (no perturbation) ==========
# --agentview_pos_offset "0.0,0.0,0.0"
# --agentview_rpy_offset "0.0,0.0,0.0"
# --wrist_cam_pos_offset "0.0,0.0,0.0"
# --wrist_cam_rpy_offset "0.0,0.0,0.0"

# ========== Example 2: Agentview camera moved back 10cm and rotated 15 degrees ==========
# --agentview_pos_offset "0.1,0.0,0.0"
# --agentview_rpy_offset "0.0,15.0,0.0"  # pitch up 15 degrees

# ========== Example 3: Camera moved CLOSER to table (negative x) ==========
# --agentview_pos_offset "-0.1,0.0,0.0"

# ========== Example 4: Table height increased by 5cm ==========
# --table_height_offset 0.05

# ========== Current Run ==========
# NOTE: Use "=" to connect argument and value when using negative numbers!
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint /mnt/nas/weights/vla-adapter-sf/outputs/configs+libero_spatial_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter-SF--libero_spatial_no_noops----15000_chkpt \
  --task_suite_name libero_spatial \
  --use_pro_version True \
  --agentview_pos_offset="0.0,0.0,0.0" \
  --agentview_rpy_offset="0.0,5.0,0.0" \
  --wrist_cam_pos_offset="0.0,0.0,0.0" \
  --wrist_cam_rpy_offset="0.0,0.0,0.0" \
  --table_height_offset="0.0" \
  > eval_logs/Spatial-SF-15k-perturb_env_roll5.log 2>&1 &



  # --debug_save_input_images True \
  # --debug_image_save_freq 10 \
  # --num_trials_per_task 1 \
  # /mnt/nas/weights/vla-adapter-sf/outputs/configs+libero_spatial_no_noops+b24+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--libero_spatial_no_noops----10000_chkpt/
  # /mnt/nas/weights/vla-adapter-sf/outputs/configs+libero_spatial_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter-SF--libero_spatial_no_noops----15000_chkpt