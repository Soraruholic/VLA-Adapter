export PYTHONPATH="/home/icrlab02/vla_ws/LIBERO:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint /mnt/nas/weights/VLA_Adapter/outputs/configs+libero_spatial_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter-SF-DIST_LAYR-12-libero_spatial_no_noops--20251219_165631--15000_chkpt \
  --task_suite_name libero_spatial \
  --use_pro_version True \
  > eval_logs/Spatial-SF-DIST_LAYR-12--15k--chkpt.log 2>&1 &
