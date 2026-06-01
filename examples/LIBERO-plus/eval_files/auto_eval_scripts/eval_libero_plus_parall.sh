#!/usr/bin/env bash
###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=${LIBERO_HOME:-../LIBERO-plus}  # Root directory of the LIBERO-plus project
export LIBERO_python=${LIBERO_python:-python}  # Path to the Python environment
export resVLA_python=${resVLA_python:-python} # Path to the Python environment

# === End of environment variable configuration ===
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from resVLA repo
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/config/}
###########################################################################################
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

##### === variables for which evaluation to setup ===
ckpt_path=$1 # ./results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt
task_suite_name=$2 # libero_goal / libero_object / libero_spatial / libero_10 / libero_90 / all
run_index=$3

shift 3

category_filter="all"
resume=false

# Backward-compatible: allow 4th positional arg as category_filter if it's not a flag.
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    category_filter="$1"
    shift
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --resume)
            resume=true
            shift
            ;;
        --category-filter)
            category_filter="$2"
            shift 2
            ;;
        --category-filter=*)
            category_filter="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done
##### === variables for which evaluation to setup ===

num_gpus=8
gpu_id=$((run_index % num_gpus))

num_trials_per_task=1
host="127.0.0.1"
base_port=$((6450 + run_index))
unnorm_key="franka"
server_pid=""

cleanup() {
    exit_code=$?

    if [ -n "${server_pid:-}" ]; then
        if kill -0 "$server_pid" 2>/dev/null; then
            echo "Killing server process with PID: $server_pid"
            kill "$server_pid" 2>/dev/null || true
            sleep 2

            if kill -0 "$server_pid" 2>/dev/null; then
                echo "Server process $server_pid did not exit after TERM; sending KILL"
                kill -9 "$server_pid" 2>/dev/null || true
            fi
        fi
    else
        echo "No server process found to kill."
    fi

    exit "$exit_code"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CUDA_VISIBLE_DEVICES=$gpu_id ${resVLA_python} deployment/model_server/server_policy.py \
    --ckpt_path ${ckpt_path} \
    --port ${base_port} \
    --use_bf16 &

# Capture the server PID
server_pid=$!

# Extract model_root from ckpt_path
model_root=$(echo "$ckpt_path" | awk -F'/checkpoints/' '{print $1}')
folder_name=$(echo "$ckpt_path" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

category_suffix=""
if [ "$category_filter" != "all" ]; then
    category_suffix="_${category_filter}"
fi

video_out_path="${model_root}/videos_libero_plus/${task_suite_name}/${folder_name}${category_suffix}"
log_path="${model_root}/logs_libero_plus/${task_suite_name}"
mkdir -p "$video_out_path"
mkdir -p "$log_path"


echo "Evaluating on GPU $gpu_id"

RESUME_ARGS=()
if $resume; then
    RESUME_ARGS+=("--resume")
fi

CUDA_VISIBLE_DEVICES=$gpu_id ${LIBERO_python} ./examples/LIBERO-plus/eval_files/eval_libero_plus.py \
    --pretrained-path ${ckpt_path} \
    --host "$host" \
    --port $base_port \
    --task-suite-name "$task_suite_name" \
    --category_filter "$category_filter" \
    --num-trials-per-task "$num_trials_per_task" \
    --video-out-path "$video_out_path"  \
    --log-out-path "$log_path" \
    "${RESUME_ARGS[@]}"

eval_status=$?

if [ "$eval_status" -eq 0 ]; then
    echo "Evaluation completed. Videos saved to ${video_out_path}, logs saved to ${log_path}/${folder_name}${category_suffix}.log"
else
    echo "Evaluation exited with status ${eval_status}. Videos saved to ${video_out_path}, logs saved to ${log_path}/${folder_name}${category_suffix}.log"
fi

exit "$eval_status"
