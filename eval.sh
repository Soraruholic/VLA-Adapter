#!/bin/bash
# Evaluation script for VLA-Adapter with optional CAM visualization
# Usage: 
#   bash eval.sh           # Normal evaluation (50 trials per task)
#   bash eval.sh --cam     # CAM-only mode (1 trial per task, just for visualization)

export PYTHONPATH="/home/icrlab02/vla_ws/LIBERO:$PYTHONPATH"

# Configuration
CHECKPOINT="/mnt/nas/weights/vla-adapter-sf/outputs/configs+libero_spatial_no_noops+b24+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--libero_spatial_no_noops----10000_chkpt"
TASK_SUITE="libero_spatial"
GPU_ID=0

# CAM settings
ENABLE_CAM=False
CAM_MODE="eigencam"
CAM_OUTPUT_DIR="./grad_cam_outputs/${TASK_SUITE}_cam_origin"
CAM_FRAMES_PER_EPISODE=5

# Normal eval settings
NUM_TRIALS=50

# Parse arguments
for arg in "$@"; do
  case $arg in
    --cam)
      ENABLE_CAM=True
      NUM_TRIALS=1  # CAM mode: only 1 trial per task
      shift
      ;;
  esac
done

# Build command
CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint ${CHECKPOINT} \
  --task_suite_name ${TASK_SUITE} \
  --use_pro_version True \
  --num_trials_per_task ${NUM_TRIALS}"

# Add CAM arguments if enabled
if [ "$ENABLE_CAM" = "True" ]; then
  CMD="${CMD} \
  --enable_cam True \
  --cam_mode ${CAM_MODE} \
  --cam_output_dir ${CAM_OUTPUT_DIR} \
  --cam_frames_per_episode ${CAM_FRAMES_PER_EPISODE}"
  echo "[INFO] CAM-only mode: 1 trial per task, ${CAM_FRAMES_PER_EPISODE} frames/episode"
  echo "[INFO] CAM output: ${CAM_OUTPUT_DIR}"
fi

# Create log directory
mkdir -p eval_logs

# Run evaluation
LOG_FILE="eval_logs/${TASK_SUITE}_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Running evaluation..."
echo "[INFO] Checkpoint: ${CHECKPOINT}"
echo "[INFO] Task suite: ${TASK_SUITE}"
echo "[INFO] Trials per task: ${NUM_TRIALS}"
echo "[INFO] Log file: ${LOG_FILE}"

eval $CMD > "${LOG_FILE}" 2>&1 &
echo "[INFO] Evaluation started in background (PID: $!)"
echo "[INFO] Use 'tail -f ${LOG_FILE}' to monitor progress"
