#!/bin/bash

# ============================================================
# Argument Parsing
# ============================================================
###########################################################################################
# === Please modify the paths to Python executables in conda environments ===

resVLA_PYTHON=${resVLA_PYTHON:-python}
SIMPLER_PYTHON=${SIMPLER_PYTHON:-python}
SimplerEnv_PATH=${SimplerEnv_PATH:-../SimplerEnv}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR=${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# === End of environment variable configuration ===
###########################################################################################

CKPT_DEFAULT="${RESVLA_CKPT:-./results/Checkpoints/resvla_simpler_env_2B/checkpoints/resVLA_simpler_env.pt}"
N_ENVS_DEFAULT=1
MAX_EPISODE_STEPS_DEFAULT=120
N_ACTION_STEPS_DEFAULT=12

REPEAT_TIMES=4

BASE_PORT=${BASE_PORT:-6350}
START_GPU_ID=${START_GPU_ID:-0}
NUM_GPUS=${NUM_GPUS:-4}
GPU_IDS=${GPU_IDS:-}
if [[ -n "${GPU_IDS}" ]]; then
    IFS=',' read -r -a CUDA_DEVICES <<< "${GPU_IDS}"
else
    CUDA_DEVICES=()
    for GPU_OFFSET in $(seq 0 $((NUM_GPUS - 1))); do
        CUDA_DEVICES+=($((START_GPU_ID + GPU_OFFSET)))
    done
fi
NUM_GPUS=${#CUDA_DEVICES[@]}
MAX_PARALLEL_ENVS=${MAX_PARALLEL_ENVS:-${NUM_GPUS}}
SIMPLER_ENV_GPU_ID=${SIMPLER_ENV_GPU_ID:-auto}
SIMPLER_RENDER_DEVICE=${SIMPLER_RENDER_DEVICE:-auto}

# Parse command-line arguments
CKPT_PATH=${1:-$CKPT_DEFAULT}
N_ENVS=${2:-$N_ENVS_DEFAULT}
MAX_EPISODE_STEPS=${3:-$MAX_EPISODE_STEPS_DEFAULT}
N_ACTION_STEPS=${4:-$N_ACTION_STEPS_DEFAULT}


echo "=== Evaluation Configuration ==="
echo "Checkpoint Path      : ${CKPT_PATH}"
echo "Number of Envs       : ${N_ENVS}"
echo "Max Episode Steps    : ${MAX_EPISODE_STEPS}"
echo "Action Chunk Length  : ${N_ACTION_STEPS}"
echo "Repeat Times         : ${REPEAT_TIMES}"
echo "Env Render GPU       : ${SIMPLER_ENV_GPU_ID}"
echo "Renderer Device      : ${SIMPLER_RENDER_DEVICE}"
echo "GPU IDs              : ${CUDA_DEVICES[*]}"
echo "Parallel Envs        : ${MAX_PARALLEL_ENVS}"
echo "================================"

# ============================================================
# Evaluation Function
# ============================================================

EvalEnv() {
    local GPU_ID=$1
    local PORT=$2
    local ENV_NAME=$3
    local CKPT_PATH=$4
    local LOG_DIR=$5
    local SIMPLER_PYTHON=$6
    local N_ENVS=$7
    local MAX_EPISODE_STEPS=$8
    local N_ACTION_STEPS=$9
    local REPEAT_ID=${10}

    local ENV_GPU_ID="${SIMPLER_ENV_GPU_ID}"
    if [[ "${ENV_GPU_ID}" == "auto" ]]; then
        ENV_GPU_ID="${GPU_ID}"
    fi

    local RENDER_DEVICE="${SIMPLER_RENDER_DEVICE}"
    if [[ "${RENDER_DEVICE}" == "auto" ]]; then
        RENDER_DEVICE=""
    fi

    echo "Launching evaluation | Policy GPU ${GPU_ID} | Env GPU ${ENV_GPU_ID} | Port ${PORT} | Env ${ENV_NAME}"

    DEBUG= CUDA_VISIBLE_DEVICES=${ENV_GPU_ID} SIMPLER_RENDER_DEVICE=${RENDER_DEVICE} DISPLAY="" \
    ${SIMPLER_PYTHON} examples/SimplerEnv/eval_files/start_simpler_env.py \
        --port "${PORT}" \
        --ckpt-path "${CKPT_PATH}" \
        --robot "${ROBOT}" \
        --policy-setup widowx_bridge \
        --control-freq 5 \
        --sim-freq 500 \
        --max-episode-steps "${MAX_EPISODE_STEPS}" \
        --env-name "${ENV_NAME}" \
        --scene-name "${SCENE_NAME}" \
        --rgb-overlay-path "${RGB_OVERLAY_PATH}" \
        --robot-init-x "${ROBOT_INIT_X}" "${ROBOT_INIT_X}" 1 \
        --robot-init-y "${ROBOT_INIT_Y}" "${ROBOT_INIT_Y}" 1 \
        --obj-variation-mode episode \
        --obj-episode-range 0 24 \
        --robot-init-rot-quat-center 0 0 0 1 \
        --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --logging-dir "${LOG_DIR}/${ENV_NAME//\//_}/repeat_${REPEAT_ID}" \
        > "${LOG_DIR}/eval_env_${ENV_NAME//\//_}_rep${REPEAT_ID}_gpu${GPU_ID}.log" 2>&1
}

WaitForServer() {
    local PORT=$1
    local PID=$2
    local TIMEOUT_SECONDS=${SERVER_START_TIMEOUT_SECONDS:-600}
    local WAITED=0

    while ! "${resVLA_PYTHON}" - "${PORT}" <<'PY' >/dev/null 2>&1
import asyncio
import sys
import websockets

async def main():
    async with websockets.connect(f"ws://127.0.0.1:{int(sys.argv[1])}", open_timeout=1.0, close_timeout=0.2):
        pass

asyncio.run(main())
PY
    do
        if ! kill -0 "${PID}" 2>/dev/null; then
            echo "Policy server on port ${PORT} exited before becoming ready."
            return 1
        fi
        if (( WAITED >= TIMEOUT_SECONDS )); then
            echo "Timed out waiting for policy server on port ${PORT}."
            return 1
        fi
        sleep 5
        WAITED=$((WAITED + 5))
    done
}

# ============================================================
# Environment List
# ============================================================

ENV_NAMES=(
    PutCarrotOnPlateInScene-v0
    PutEggplantInBasketScene-v0
    PutSpoonOnTableClothInScene-v0
    StackGreenCubeOnYellowCubeBakedTexInScene-v0
)

# Per-env configs (must align with ENV_NAMES indices)
SCENE_NAMES=(
    bridge_table_1_v1
    bridge_table_1_v2
    bridge_table_1_v1
    bridge_table_1_v1
)

ROBOTS=(
    widowx
    widowx_sink_camera_setup
    widowx
    widowx
)

RGB_OVERLAY_PATHS=(
    ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
    ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
    ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
    ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
)

ROBOT_INIT_XS=(
    0.147
    0.127
    0.147
    0.147
)

ROBOT_INIT_YS=(
    0.028
    0.06
    0.028
    0.028
)

# ============================================================
# Runtime Configuration
# ============================================================

RUN_ROOT=$(dirname "$(dirname "$CKPT_PATH")")
ckpt_name=$(basename "$CKPT_PATH" .pt)
LOG_DIR="${RUN_ROOT}/logs/${ckpt_name}/bridge_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "=== Launching Multi-GPU Evaluation ==="
echo "GPUs            : ${NUM_GPUS} (${CUDA_DEVICES[*]})"
echo "Num Environments: ${#ENV_NAMES[@]}"
echo "Log Directory   : ${LOG_DIR}"

# ============================================================
# Step 1: Launch Policy Servers
# ============================================================

SERVER_PIDS=()

CleanupProcesses() {
    local eval_pids=""
    if [[ -n "${LOG_DIR:-}" ]]; then
        eval_pids=$(pgrep -u "$(id -u)" -f -- "examples/SimplerEnv/eval_files/start_simpler_env.py.*--logging-dir ${LOG_DIR}/" || true)
        if [[ -n "${eval_pids}" ]]; then
            kill ${eval_pids} 2>/dev/null || true
        fi
    fi

    for PID in "${SERVER_PIDS[@]}"; do
        kill "${PID}" 2>/dev/null || true
    done
}

trap CleanupProcesses EXIT
trap 'CleanupProcesses; exit 130' INT TERM

for GPU_OFFSET in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=${CUDA_DEVICES[$GPU_OFFSET]}
    PORT=$((BASE_PORT + GPU_OFFSET))
    echo "Starting policy server | GPU ${GPU_ID} | Port ${PORT}"

    DEBUG= CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ${resVLA_PYTHON} deployment/model_server/server_policy.py \
        --ckpt_path "${CKPT_PATH}" \
        --port "${PORT}" \
        --idle_timeout -1 \
        --use_bf16 \
        > "${LOG_DIR}/server_gpu${GPU_ID}_port${PORT}.log" 2>&1 &

    SERVER_PIDS[$GPU_OFFSET]=$!

    sleep 10
done

for GPU_OFFSET in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + GPU_OFFSET))
    WaitForServer "${PORT}" "${SERVER_PIDS[$GPU_OFFSET]}" || exit 1
done

# ============================================================
# Step 2: Dispatch Environments to GPUs
# ============================================================

RUNNING_EVAL_PIDS=()

WaitForEvalBatch() {
    local STATUS=0
    local PID

    for PID in "${RUNNING_EVAL_PIDS[@]}"; do
        if ! wait "${PID}"; then
            STATUS=1
        fi
    done

    RUNNING_EVAL_PIDS=()
    return "${STATUS}"
}

COUNT=0
for i in "${!ENV_NAMES[@]}"; do
    ENV_NAME="${ENV_NAMES[i]}"
    SCENE_NAME="${SCENE_NAMES[i]}"
    ROBOT="${ROBOTS[i]}"
    RGB_OVERLAY_PATH="${RGB_OVERLAY_PATHS[i]}"
    ROBOT_INIT_X="${ROBOT_INIT_XS[i]}"
    ROBOT_INIT_Y="${ROBOT_INIT_YS[i]}"

    for REPEAT_ID in $(seq 1 ${REPEAT_TIMES}); do
        GPU_OFFSET=$((COUNT % NUM_GPUS))
        GPU_ID=${CUDA_DEVICES[$GPU_OFFSET]}
        PORT=$((BASE_PORT + GPU_OFFSET))

        EvalEnv "${GPU_ID}" "${PORT}" "${ENV_NAME}" "${CKPT_PATH}" "${LOG_DIR}" \
                "${SIMPLER_PYTHON}" "${N_ENVS}" "${MAX_EPISODE_STEPS}" "${N_ACTION_STEPS}" "${REPEAT_ID}" &
        RUNNING_EVAL_PIDS+=("$!")

        COUNT=$((COUNT + 1))
        if (( COUNT % MAX_PARALLEL_ENVS == 0 )); then
            WaitForEvalBatch || exit 1
        fi
        sleep 2
    done
done

if (( ${#RUNNING_EVAL_PIDS[@]} > 0 )); then
    WaitForEvalBatch || exit 1
fi

# ============================================================
# Step 3: Cleanup
# ============================================================

while pgrep -u "$(id -u)" -f -- "examples/SimplerEnv/eval_files/start_simpler_env.py.*--logging-dir ${LOG_DIR}/" > /dev/null; do
    echo "Waiting for all evaluation environments to finish..."
    sleep 30
done

echo ""
echo "=== Aggregating results (mean over ${REPEAT_TIMES} repeats) ==="

SUMMARY_FILE="${LOG_DIR}/summary_mean_over_${REPEAT_TIMES}.tsv"
printf "env_name\tmean_success\tstd_success\trepeats\n" > "${SUMMARY_FILE}"

for ENV_NAME in "${ENV_NAMES[@]}"; do
    ENV_KEY="${ENV_NAME//\//_}"
    values=()

    for REPEAT_ID in $(seq 1 ${REPEAT_TIMES}); do
        matches=( ${LOG_DIR}/eval_env_${ENV_KEY}_rep${REPEAT_ID}_gpu*.log )
        if [[ ! -e "${matches[0]}" ]]; then
            continue
        fi
        v=$(grep -F "Average success" "${matches[0]}" | tail -n 1 | awk '{print $NF}')
        if [[ -n "${v}" ]]; then
            values+=("${v}")
        fi
    done

    stats=$(printf "%s\n" "${values[@]}" | awk 'NF{sum+=$1; sumsq+=$1*$1; n++} END{ if(n>0){mean=sum/n; var=sumsq/n-mean*mean; if(var<0) var=0; printf "%.6f\t%.6f\t%d", mean, sqrt(var), n} else {printf "nan\tnan\t0"}}')
    printf "%s\t%s\n" "${ENV_NAME}" "${stats}" | tee -a "${SUMMARY_FILE}"
done

echo ""
echo "Shutting down policy servers..."

CleanupProcesses
trap - EXIT INT TERM

echo "=== Evaluation Finished ==="
