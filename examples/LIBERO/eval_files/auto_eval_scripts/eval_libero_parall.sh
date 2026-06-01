###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=../openpi/third_party/libero  # Root directory of the LIBERO project
export LIBERO_python=../../envs/lerobot-env/bin/python  # Path to the Python environment
export resVLA_python=../../envs/starVLA-env/bin/python # Path to the Python environment

# === End of environment variable configuration ===
# export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero  # Path to LIBERO configuration files
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from resVLA repo
###########################################################################################
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

##### === variables for which evaluation to setup ===
ckpt_path=$1 # ./results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt
task_suite_name=$2 # align with your model | libero_goal
run_index=$3
# ckpt_path=./results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt
# task_suite_name=libero_10
# run_index=8
##### === variables for which evaluation to setup ===

num_gpus=8
gpu_id=$((run_index % num_gpus))

num_trials_per_task=50
host="127.0.0.1"
base_port=$((6450 + run_index))
unnorm_key="franka"

CUDA_VISIBLE_DEVICES=$gpu_id ${resVLA_python} deployment/model_server/server_policy.py \
    --ckpt_path ${ckpt_path} \
    --port ${base_port} \
    --use_bf16 &


# Capture the server PID
server_pid=$!


# Extract model_root from ckpt_path
model_root=$(echo "$ckpt_path" | awk -F'/checkpoints/' '{print $1}')
folder_name=$(echo "$ckpt_path" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

video_out_path="${model_root}/videos/${task_suite_name}/${folder_name}"
log_path="${model_root}/logs/${task_suite_name}"
mkdir -p "$video_out_path"
mkdir -p "$log_path"


echo "Evaluating on GPU $gpu_id"
CUDA_VISIBLE_DEVICES=$gpu_id ${LIBERO_python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${ckpt_path} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"  \
    2>&1 | tee ${log_path}/${folder_name}.log

echo "Evaluation completed. Videos saved to ${video_out_path}, logs saved to ${log_path}/${folder_name}.log"




if [ -n "$server_pid" ]; then
    echo "Killing server process with PID: $server_pid"
    kill $server_pid
else
    echo "No server process found to kill."
fi
