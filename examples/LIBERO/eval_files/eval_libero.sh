#!/bin/bash

# resVLA
###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=${LIBERO_HOME:-../LIBERO}
# export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=${LIBERO_Python:-python}
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

host="127.0.0.1"
base_port=5694
unnorm_key="franka"
ckpt_path=${RESVLA_CKPT:-./results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt}
# export DEBUG=False

folder_name=$(echo "$ckpt_path" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}


task_suite_name=libero_goal
num_trials_per_task=50
video_out_path="results/${task_suite_name}/${folder_name}"


${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${ckpt_path} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
