#!/bin/bash
# =============================================================================
# Evaluation Script with CAM Visualization (using pytorch-grad-cam library)
# =============================================================================
#
# This script runs VLA-Adapter evaluation with CAM visualization enabled.
# CAM methods available (using pytorch-grad-cam library):
#   - eigencam    : Fast, no gradients needed, uses PCA on activations
#   - gradcam     : Classic Grad-CAM, requires gradients
#   - hirescam    : High-resolution CAM, element-wise activation*gradient
#   - gradcam++   : Improved Grad-CAM with second-order gradients
#   - xgradcam    : Gradient weighted by normalized activations
#   - layercam    : Spatially weighted by positive gradients
#   - eigengradcam: EigenCAM with class discrimination via gradients
#
# Usage:
#   bash eval_sf_cam.sh
#
# =============================================================================

set -e

# Environment setup
export TOKENIZERS_PARALLELISM=false

# Create output directories
mkdir -p eval_logs
mkdir -p grad_cam_outputs

# =============================================================================
# Configuration
# =============================================================================

# Model checkpoint
CHECKPOINT="/mnt/nas/weights/vla-adapter-sf/outputs/configs+libero_spatial_no_noops+b24+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--libero_spatial_no_noops----10000_chkpt"

# Task suite
TASK_SUITE="libero_spatial"

# CAM settings
CAM_METHOD="eigencam"           # Options: eigencam, gradcam, hirescam, gradcam++, etc.
CAM_EVERY_N_STEPS=10            # Generate CAM every N steps
CAM_OUTPUT_DIR="./grad_cam_outputs/spatial_10k_${CAM_METHOD}"
CAM_ACTION_DIMS="0,1,2,6"       # Which action dimensions to visualize (empty = default)

# =============================================================================
# Run Evaluation with CAM
# =============================================================================

echo "Starting evaluation with CAM visualization..."
echo "  Method: ${CAM_METHOD}"
echo "  Output: ${CAM_OUTPUT_DIR}"
echo "  Every ${CAM_EVERY_N_STEPS} steps"

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name "${TASK_SUITE}" \
  --use_pro_version True \
  --num_trials_per_task 1 \
  --enable_cam True \
  --cam_mode "${CAM_METHOD}" \
  --cam_every_n_steps ${CAM_EVERY_N_STEPS} \
  --cam_action_dims "${CAM_ACTION_DIMS}" \
  --cam_output_dir "${CAM_OUTPUT_DIR}" \
  > eval_logs/cam_${CAM_METHOD}.log 2>&1 &

PID=$!
echo "Evaluation started with PID: ${PID}"
echo "Check progress: tail -f eval_logs/cam_${CAM_METHOD}.log"
echo "CAM outputs will be saved to: ${CAM_OUTPUT_DIR}"