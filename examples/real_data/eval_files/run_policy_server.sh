#!/bin/bash

export PYTHONPATH=$(pwd):${PYTHONPATH}
export resvla_python=${resvla_python:-python}

ckpt_path=${RESVLA_CKPT:-./results/Checkpoints/resvla_real_robot/checkpoints/resVLA_real_robot.pt}
gpu_id=${GPU_ID:-0}

control_hz=10
publish_rate=10

obs_topic=/resvla/observation
pred_action_topic=/resvla/predicted_action
pred_action_chunk_topic=/resvla/predicted_action_chunk

action_scale="1, 1, 1, 1, 1, 1, 14.3717, 1, 1, 1, 1, 1, 1,14.4678"
state_scale="1, 1, 1, 1, 1, 1, 0.4305, 1, 1, 1, 1, 1, 1, 0.4218"

################# ResVLA ROS Policy Server ######################
CUDA_VISIBLE_DEVICES=${gpu_id} ${resvla_python} examples/real_data/eval_files/resvla_ros_server.py \
  --checkpoint_path "${ckpt_path}" \
  --device cuda \
  --control_hz ${control_hz} \
  --publish_rate ${publish_rate} \
  --policy_cameras "cam_front,cam_high,cam_left,cam_right" \
  --obs_topic "${obs_topic}" \
  --pred_action_topic "${pred_action_topic}" \
  --pred_action_chunk_topic "${pred_action_chunk_topic}" \
  --action_scale "${action_scale}" \
  --state_scale "${state_scale}" \
  --use_state \
  --use_delta

# If your checkpoint includes dataset statistics for action un-normalization,
# append these options:
# --use_checkpoint_stats --unnorm_key your_dataset_name
