#!/bin/bash

# LIBERO-plus evaluation script

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=../LIBERO-plus
export LIBERO_Python=${LIBERO_Python:-python}
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

host="127.0.0.1"
base_port=5694
unnorm_key="franka"
ckpt_path=${RESVLA_CKPT:-./results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt}
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}


task_suite_name=libero_goal
num_trials_per_task=1
video_out_path="results/libero_plus/${task_suite_name}/$(date +"%Y%m%d_%H%M%S")"

${LIBERO_Python} ./examples/LIBERO-plus/eval_files/eval_libero_plus.py \
    --args.pretrained-path ${ckpt_path} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
