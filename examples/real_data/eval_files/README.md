# Real Robot ROS Bridge

This folder adds a simple two-process ROS deployment path for real robot evaluation:

- `ros_action_to_aloha.py`: subscribes to camera and joint-state topics, packs observations, publishes one observation topic, and executes predicted actions.
- `resvla_ros_server.py`: subscribes only to the packed observation topic, runs ResVLA inference, and publishes predicted 14-DoF actions.

## Topics

- Observation topic:
  - `/resvla/observation` (`std_msgs/String`), JSON payload with base64 images + robot state.
- Model output topic:
  - `/resvla/predicted_action` (`sensor_msgs/JointState`), one 14-dim action per message.
  - `/resvla/predicted_action_chunk` (`std_msgs/Float64MultiArray`), flattened action chunk for debugging.

## Start order

Start the bridge/executor on the robot side:

```bash
python examples/real_data/eval_files/ros_action_to_aloha.py \
  --prompt "Hang the teacup on the cup rack."
```

Start the policy server on the GPU machine:

```bash
python examples/real_data/eval_files/resvla_ros_server.py \
  --checkpoint_path /path/to/checkpoint.pt
```

If the checkpoint contains multiple dataset statistics, select the one used for action unnormalization:

```bash
--unnorm_key your_dataset_name
```

The prompt is published by `ros_action_to_aloha.py` inside the observation payload; the policy server only consumes that payload.

## Attention

We use chunk-wise delta actions for real-world robot inference, and the policy server decodes the whole predicted chunk against one fixed chunk-start robot state. This follows the chunk-wise delta control design discussed in "Demystifying Action Space Design for Robotic Manipulation Policies" (OpenReview: https://openreview.net/forum?id=nAOgwZ9Ymj).

Please note that this repository does not include a dataloader that explicitly constructs chunk-wise delta action targets for real-robot training. If you want deployment behavior to match this chunk-wise delta control assumption closely, you may need to implement that dataloader yourself.
