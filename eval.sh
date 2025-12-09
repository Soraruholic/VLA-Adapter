export PYTHONPATH="/home/icrlab02/vla_ws/LIBERO:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint /home/icrlab02/vla_ws/VLA-Adapter/pretrained_models/LIBERO-Spatial-Pro \
  --task_suite_name libero_spatial \
  --use_pro_version True \
  > eval_logs/Spatial-full--chkpt.log 2>&1 &
