#!/bin/bash

export PYTHONPATH=$(pwd):${PYTHONPATH}
export resvla_python=${resvla_python:-python}

prompt="put the cup on the shelf"

obs_topic=/resvla/observation
pred_action_topic=/resvla/predicted_action

img_front_topic=/camera_f/color/image_raw
img_left_topic=/camera_l/color/image_raw
img_right_topic=/camera_r/color/image_raw
img_high_topic=/rgb/image_raw/compressed

puppet_arm_left_topic=/puppet/joint_left
puppet_arm_right_topic=/puppet/joint_right
puppet_arm_left_cmd_topic=/master/joint_left
puppet_arm_right_cmd_topic=/master/joint_right

control_hz=10
publish_rate=30
image_width=224
image_height=224
jpeg_quality=80

################# ResVLA ROS Client (Bridge + Executor) ######################
${resvla_python} examples/real_data/eval_files/ros_action_to_aloha.py \
  --prompt "${prompt}" \
  --obs_topic "${obs_topic}" \
  --pred_action_topic "${pred_action_topic}" \
  --img_front_topic "${img_front_topic}" \
  --img_left_topic "${img_left_topic}" \
  --img_right_topic "${img_right_topic}" \
  --img_high_topic "${img_high_topic}" \
  --puppet_arm_left_topic "${puppet_arm_left_topic}" \
  --puppet_arm_right_topic "${puppet_arm_right_topic}" \
  --puppet_arm_left_cmd_topic "${puppet_arm_left_cmd_topic}" \
  --puppet_arm_right_cmd_topic "${puppet_arm_right_cmd_topic}" \
  --control_hz ${control_hz} \
  --publish_rate ${publish_rate} \
  --action_execute_mode direct \
  --image_width ${image_width} \
  --image_height ${image_height} \
  --jpeg_quality ${jpeg_quality}
