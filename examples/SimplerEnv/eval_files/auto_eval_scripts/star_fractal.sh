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

BASE_PORT=${BASE_PORT:-6450}
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
SIMPLER_DISABLE_RT_DENOISER=${SIMPLER_DISABLE_RT_DENOISER:-1}
FRACTAL_SETTING=${FRACTAL_SETTING:-all}

case "${FRACTAL_SETTING}" in
    all)
        FRACTAL_LOG_PREFIX="fractal_eval"
        ;;
    va|variant_agg)
        FRACTAL_SETTING="variant_agg"
        FRACTAL_LOG_PREFIX="fractal_variant_agg_eval"
        ;;
    vm|visual_matching)
        FRACTAL_SETTING="visual_matching"
        FRACTAL_LOG_PREFIX="fractal_visual_matching_eval"
        ;;
    *)
        echo "Unknown FRACTAL_SETTING='${FRACTAL_SETTING}'. Use one of: all, variant_agg, visual_matching."
        exit 1
        ;;
esac

# Parse command-line arguments
CKPT_PATH=${1:-$CKPT_DEFAULT}


echo "=== Evaluation Configuration ==="
echo "Checkpoint Path      : ${CKPT_PATH}"
echo "Policy Setup         : google_robot"
echo "Robot                : google_robot_static"
echo "Fractal Setting      : ${FRACTAL_SETTING}"
echo "Base Port            : ${BASE_PORT}"
echo "GPUs                 : ${NUM_GPUS} (${CUDA_DEVICES[*]})"
echo "Env Render GPU       : ${SIMPLER_ENV_GPU_ID}"
echo "Renderer Device      : ${SIMPLER_RENDER_DEVICE}"
echo "Disable RT Denoiser  : ${SIMPLER_DISABLE_RT_DENOISER}"
echo "Parallel Envs        : ${MAX_PARALLEL_ENVS}"
echo "================================"

# ============================================================
# Evaluation Function
# ============================================================

EvalEnv() {
    local GPU_ID=$1
    local PORT=$2
    local RUN_NAME=$3
    local ENV_NAME=$4
    local CKPT_PATH=$5
    local LOG_DIR=$6
    local SIMPLER_PYTHON=$7
    local MAX_EPISODE_STEPS=$8
    local SCENE_NAME=$9
    local EXTRA_ARGS=${10}

    local ENV_GPU_ID="${SIMPLER_ENV_GPU_ID}"
    if [[ "${ENV_GPU_ID}" == "auto" ]]; then
        ENV_GPU_ID="${GPU_ID}"
    fi

    local RENDER_DEVICE="${SIMPLER_RENDER_DEVICE}"
    if [[ "${RENDER_DEVICE}" == "auto" ]]; then
        RENDER_DEVICE=""
    fi

    echo "Launching evaluation | Policy GPU ${GPU_ID} | Env GPU ${ENV_GPU_ID} | Port ${PORT} | Run ${RUN_NAME}"

    DEBUG= CUDA_VISIBLE_DEVICES=${ENV_GPU_ID} SIMPLER_RENDER_DEVICE=${RENDER_DEVICE} SIMPLER_DISABLE_RT_DENOISER=${SIMPLER_DISABLE_RT_DENOISER} DISPLAY="" \
    ${SIMPLER_PYTHON} examples/SimplerEnv/eval_files/start_simpler_env.py \
        --port "${PORT}" \
        --ckpt-path "${CKPT_PATH}" \
        --robot google_robot_static \
        --policy-setup google_robot \
        --control-freq 3 \
        --sim-freq 513 \
        --max-episode-steps "${MAX_EPISODE_STEPS}" \
        --env-name "${ENV_NAME}" \
        --scene-name "${SCENE_NAME}" \
        --logging-dir "${LOG_DIR}/${RUN_NAME}" \
        ${EXTRA_ARGS} \
        > "${LOG_DIR}/eval_${RUN_NAME}_gpu${GPU_ID}.log" 2>&1
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

declare -a RUN_NAMES=()
declare -a ENV_NAMES=()
declare -a SCENE_NAMES=()
declare -a MAX_EPISODE_STEPS_ARR=()
declare -a EXTRA_ARGS_ARR=()

AddRun() {
    RUN_NAMES+=("$1")
    ENV_NAMES+=("$2")
    SCENE_NAMES+=("$3")
    MAX_EPISODE_STEPS_ARR+=("$4")
    EXTRA_ARGS_ARR+=("$5")
}

# ----------------------------
# Drawer (variant_agg)
# ----------------------------

drawer_envs=(
    CloseTopDrawerCustomInScene-v0
    CloseMiddleDrawerCustomInScene-v0
    CloseBottomDrawerCustomInScene-v0
    OpenTopDrawerCustomInScene-v0
    OpenMiddleDrawerCustomInScene-v0
    OpenBottomDrawerCustomInScene-v0
)

drawer_common_args="--robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3 \
--robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
--obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1"

# Base
for env in "${drawer_envs[@]}"; do
    AddRun "drawer_va__base__${env}" "${env}" "frl_apartment_stage_simple" 113 "${drawer_common_args} --enable-raytracing"
done

# Background variants
for scene in modern_bedroom_no_roof modern_office_no_roof; do
    for env in "${drawer_envs[@]}"; do
        AddRun "drawer_va__bg_${scene}__${env}" "${env}" "${scene}" 113 "${drawer_common_args} --additional-env-build-kwargs shader_dir=rt"
    done
done

# Lighting variants
for env in "${drawer_envs[@]}"; do
    AddRun "drawer_va__light_brighter__${env}" "${env}" "frl_apartment_stage_simple" 113 "${drawer_common_args} --additional-env-build-kwargs shader_dir=rt light_mode=brighter"
    AddRun "drawer_va__light_darker__${env}" "${env}" "frl_apartment_stage_simple" 113 "${drawer_common_args} --additional-env-build-kwargs shader_dir=rt light_mode=darker"
done

# New cabinet variants
for env in "${drawer_envs[@]}"; do
    AddRun "drawer_va__station_mk_station2__${env}" "${env}" "frl_apartment_stage_simple" 113 "${drawer_common_args} --additional-env-build-kwargs shader_dir=rt station_name=mk_station2"
    AddRun "drawer_va__station_mk_station3__${env}" "${env}" "frl_apartment_stage_simple" 113 "${drawer_common_args} --additional-env-build-kwargs shader_dir=rt station_name=mk_station3"
done

# ----------------------------
# Drawer (visual_matching)
# ----------------------------

drawer_vm_envs=(
    OpenTopDrawerCustomInScene-v0
    OpenMiddleDrawerCustomInScene-v0
    OpenBottomDrawerCustomInScene-v0
    CloseTopDrawerCustomInScene-v0
    CloseMiddleDrawerCustomInScene-v0
    CloseBottomDrawerCustomInScene-v0
)

drawer_vm_urdf_versions=(
    recolor_cabinet_visual_matching_1
    recolor_tabletop_visual_matching_1
    recolor_tabletop_visual_matching_2
    None
)

drawer_vm_extra_prefix="--enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True"

drawer_vm_overlay_names=(a0 a1 a2 b0 b1 b2 c0 c1 c2)

drawer_vm_overlay_args=(
    "--robot-init-x 0.644 0.644 1 --robot-init-y -0.179 -0.179 1 --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.03 -0.03 1 --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png"
    "--robot-init-x 0.765 0.765 1 --robot-init-y -0.182 -0.182 1 --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.02 -0.02 1 --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a1.png"
    "--robot-init-x 0.889 0.889 1 --robot-init-y -0.203 -0.203 1 --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.06 -0.06 1 --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a2.png"
    "--robot-init-x 0.652 0.652 1 --robot-init-y 0.009 0.009 1 --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png"
    "--robot-init-x 0.752 0.752 1 --robot-init-y 0.009 0.009 1 --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b1.png"
    "--robot-init-x 0.851 0.851 1 --robot-init-y 0.035 0.035 1 --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b2.png"
    "--robot-init-x 0.665 0.665 1 --robot-init-y 0.224 0.224 1 --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png"
    "--robot-init-x 0.765 0.765 1 --robot-init-y 0.222 0.222 1 --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.025 -0.025 1 --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c1.png"
    "--robot-init-x 0.865 0.865 1 --robot-init-y 0.222 0.222 1 --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.025 -0.025 1 --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c2.png"
)

for urdf in "${drawer_vm_urdf_versions[@]}"; do
    for env in "${drawer_vm_envs[@]}"; do
        for idx in "${!drawer_vm_overlay_names[@]}"; do
            overlay_name="${drawer_vm_overlay_names[idx]}"
            overlay_args="${drawer_vm_overlay_args[idx]}"
            AddRun "drawer_vm__urdf_${urdf}__${env}__${overlay_name}" "${env}" "dummy_drawer" 113 "${overlay_args} ${drawer_vm_extra_prefix} urdf_version=${urdf}"
        done
    done
done

# ----------------------------
# MoveNear (variant_agg)
# ----------------------------

near_common_args="--robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
--robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1"

# Base
AddRun "near_va__base" "MoveNearGoogleInScene-v0" "google_pick_coke_can_1_v4" 80 "${near_common_args}"

# Distractor
AddRun "near_va__no_distractor" "MoveNearGoogleInScene-v0" "google_pick_coke_can_1_v4" 80 "${near_common_args} --additional-env-build-kwargs no_distractor=True"

# Backgrounds
for scene in google_pick_coke_can_1_v4_alt_background google_pick_coke_can_1_v4_alt_background_2; do
    AddRun "near_va__bg_${scene}" "MoveNearGoogleInScene-v0" "${scene}" 80 "${near_common_args}"
done

# Lighting
AddRun "near_va__light_slightly_darker" "MoveNearGoogleInScene-v0" "google_pick_coke_can_1_v4" 80 "${near_common_args} --additional-env-build-kwargs slightly_darker_lighting=True"
AddRun "near_va__light_slightly_brighter" "MoveNearGoogleInScene-v0" "google_pick_coke_can_1_v4" 80 "${near_common_args} --additional-env-build-kwargs slightly_brighter_lighting=True"

# Table textures
for scene in Baked_sc1_staging_objaverse_cabinet1_h870 Baked_sc1_staging_objaverse_cabinet2_h870; do
    AddRun "near_va__table_${scene}" "MoveNearGoogleInScene-v0" "${scene}" 80 "${near_common_args}"
done

# Camera orientations
for env in MoveNearAltGoogleCameraInScene-v0 MoveNearAltGoogleCamera2InScene-v0; do
    AddRun "near_va__cam_${env}" "${env}" "google_pick_coke_can_1_v4" 80 "${near_common_args}"
done

# ----------------------------
# MoveNear (visual_matching)
# ----------------------------

near_vm_urdf_versions=(None recolor_tabletop_visual_matching_1 recolor_tabletop_visual_matching_2 recolor_cabinet_visual_matching_1)
near_vm_overlay="${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png"

for urdf in "${near_vm_urdf_versions[@]}"; do
    AddRun "near_vm__urdf_${urdf}" "MoveNearGoogleBakedTexInScene-v0" "google_pick_coke_can_1_v4" 80 "${near_common_args} --rgb-overlay-path ${near_vm_overlay} --additional-env-build-kwargs urdf_version=${urdf} --additional-env-save-tags baked_except_bpb_orange"
done

# ----------------------------
# Pick Coke Can (variant_agg)
# ----------------------------

pick_common_args="--robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 \
--obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
--robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1"

coke_can_options=("lr_switch=True" "upright=True" "laid_vertically=True")

# Base + table textures + backgrounds + lighting + camera orientations
for opt in "${coke_can_options[@]}"; do
    AddRun "pick_coke_va__base__${opt}" "GraspSingleOpenedCokeCanInScene-v0" "google_pick_coke_can_1_v4" 80 "${pick_common_args} --additional-env-build-kwargs ${opt}"

    for scene in Baked_sc1_staging_objaverse_cabinet1_h870 Baked_sc1_staging_objaverse_cabinet2_h870; do
        AddRun "pick_coke_va__table_${scene}__${opt}" "GraspSingleOpenedCokeCanInScene-v0" "${scene}" 80 "${pick_common_args} --additional-env-build-kwargs ${opt}"
    done

    for scene in google_pick_coke_can_1_v4_alt_background google_pick_coke_can_1_v4_alt_background_2; do
        AddRun "pick_coke_va__bg_${scene}__${opt}" "GraspSingleOpenedCokeCanInScene-v0" "${scene}" 80 "${pick_common_args} --additional-env-build-kwargs ${opt}"
    done

    AddRun "pick_coke_va__light_slightly_darker__${opt}" "GraspSingleOpenedCokeCanInScene-v0" "google_pick_coke_can_1_v4" 80 "${pick_common_args} --additional-env-build-kwargs ${opt} slightly_darker_lighting=True"
    AddRun "pick_coke_va__light_slightly_brighter__${opt}" "GraspSingleOpenedCokeCanInScene-v0" "google_pick_coke_can_1_v4" 80 "${pick_common_args} --additional-env-build-kwargs ${opt} slightly_brighter_lighting=True"

    for env in GraspSingleOpenedCokeCanAltGoogleCameraInScene-v0 GraspSingleOpenedCokeCanAltGoogleCamera2InScene-v0; do
        AddRun "pick_coke_va__cam_${env}__${opt}" "${env}" "google_pick_coke_can_1_v4" 80 "${pick_common_args} --additional-env-build-kwargs ${opt}"
    done

done

# Distractors (two configs per opt)
for opt in "${coke_can_options[@]}"; do
    AddRun "pick_coke_va__distractor_default__${opt}" "GraspSingleOpenedCokeCanDistractorInScene-v0" "google_pick_coke_can_1_v4" 80 "${pick_common_args} --additional-env-build-kwargs ${opt}"
    AddRun "pick_coke_va__distractor_more__${opt}" "GraspSingleOpenedCokeCanDistractorInScene-v0" "google_pick_coke_can_1_v4" 80 "${pick_common_args} --additional-env-build-kwargs ${opt} distractor_config=more"
done

# ----------------------------
# Pick Coke Can (visual_matching)
# ----------------------------

pick_vm_urdf_versions=(None recolor_tabletop_visual_matching_1 recolor_tabletop_visual_matching_2 recolor_cabinet_visual_matching_1)
pick_vm_overlay="${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png"

for urdf in "${pick_vm_urdf_versions[@]}"; do
    for opt in "${coke_can_options[@]}"; do
        AddRun "pick_coke_vm__urdf_${urdf}__${opt}" "GraspSingleOpenedCokeCanInScene-v0" "google_pick_coke_can_1_v4" 80 "${pick_common_args} --rgb-overlay-path ${pick_vm_overlay} --additional-env-build-kwargs ${opt} urdf_version=${urdf}"
    done
done

# ----------------------------
# Put In Drawer (variant_agg)
# ----------------------------

putin_env="PlaceIntoClosedTopDrawerCustomInScene-v0"
putin_common_args="--robot-init-x 0.65 0.65 1 --robot-init-y -0.2 0.2 3 \
--robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
--obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3"

# Base
AddRun "putin_va__base" "${putin_env}" "frl_apartment_stage_simple" 200 "${putin_common_args} --enable-raytracing --additional-env-build-kwargs model_ids=apple"

# Background variants
for scene in modern_bedroom_no_roof modern_office_no_roof; do
    AddRun "putin_va__bg_${scene}" "${putin_env}" "${scene}" 200 "${putin_common_args} --additional-env-build-kwargs shader_dir=rt model_ids=apple"
done

# Lighting variants
AddRun "putin_va__light_brighter" "${putin_env}" "frl_apartment_stage_simple" 200 "${putin_common_args} --additional-env-build-kwargs shader_dir=rt light_mode=brighter model_ids=apple"
AddRun "putin_va__light_darker" "${putin_env}" "frl_apartment_stage_simple" 200 "${putin_common_args} --additional-env-build-kwargs shader_dir=rt light_mode=darker model_ids=apple"

# New cabinet variants
AddRun "putin_va__station_mk_station2" "${putin_env}" "frl_apartment_stage_simple" 200 "${putin_common_args} --additional-env-build-kwargs shader_dir=rt station_name=mk_station2 model_ids=apple"
AddRun "putin_va__station_mk_station3" "${putin_env}" "frl_apartment_stage_simple" 200 "${putin_common_args} --additional-env-build-kwargs shader_dir=rt station_name=mk_station3 model_ids=apple"

# ----------------------------
# Put In Drawer (visual_matching)
# ----------------------------

putin_vm_urdf_versions=(recolor_cabinet_visual_matching_1 recolor_tabletop_visual_matching_1 recolor_tabletop_visual_matching_2 None)
putin_vm_extra_prefix="--enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True model_ids=baked_apple_v2"

putin_vm_overlay_names=(a0 b0 c0)
putin_vm_overlay_args=(
    "--robot-init-x 0.644 0.644 1 --robot-init-y -0.179 -0.179 1 --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.03 -0.03 1 --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png"
    "--robot-init-x 0.652 0.652 1 --robot-init-y 0.009 0.009 1 --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png"
    "--robot-init-x 0.665 0.665 1 --robot-init-y 0.224 0.224 1 --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png"
)

for urdf in "${putin_vm_urdf_versions[@]}"; do
    for idx in "${!putin_vm_overlay_names[@]}"; do
        overlay_name="${putin_vm_overlay_names[idx]}"
        overlay_args="${putin_vm_overlay_args[idx]}"
        AddRun "putin_vm__urdf_${urdf}__${overlay_name}" "${putin_env}" "dummy_drawer" 200 "${overlay_args} ${putin_vm_extra_prefix} urdf_version=${urdf}"
    done
done

if [[ "${FRACTAL_SETTING}" != "all" ]]; then
    declare -a FILTERED_RUN_NAMES=()
    declare -a FILTERED_ENV_NAMES=()
    declare -a FILTERED_SCENE_NAMES=()
    declare -a FILTERED_MAX_EPISODE_STEPS_ARR=()
    declare -a FILTERED_EXTRA_ARGS_ARR=()

    for i in "${!RUN_NAMES[@]}"; do
        RUN_NAME="${RUN_NAMES[i]}"
        if [[ "${FRACTAL_SETTING}" == "variant_agg" && "${RUN_NAME}" != *_va__* ]]; then
            continue
        fi
        if [[ "${FRACTAL_SETTING}" == "visual_matching" && "${RUN_NAME}" != *_vm__* ]]; then
            continue
        fi

        FILTERED_RUN_NAMES+=("${RUN_NAMES[i]}")
        FILTERED_ENV_NAMES+=("${ENV_NAMES[i]}")
        FILTERED_SCENE_NAMES+=("${SCENE_NAMES[i]}")
        FILTERED_MAX_EPISODE_STEPS_ARR+=("${MAX_EPISODE_STEPS_ARR[i]}")
        FILTERED_EXTRA_ARGS_ARR+=("${EXTRA_ARGS_ARR[i]}")
    done

    RUN_NAMES=("${FILTERED_RUN_NAMES[@]}")
    ENV_NAMES=("${FILTERED_ENV_NAMES[@]}")
    SCENE_NAMES=("${FILTERED_SCENE_NAMES[@]}")
    MAX_EPISODE_STEPS_ARR=("${FILTERED_MAX_EPISODE_STEPS_ARR[@]}")
    EXTRA_ARGS_ARR=("${FILTERED_EXTRA_ARGS_ARR[@]}")
fi

if (( ${#RUN_NAMES[@]} == 0 )); then
    echo "No Fractal runs selected for FRACTAL_SETTING='${FRACTAL_SETTING}'."
    exit 1
fi

# ============================================================
# Runtime Configuration
# ============================================================

RUN_ROOT=$(dirname "$(dirname "$CKPT_PATH")")
ckpt_name=$(basename "$CKPT_PATH" .pt)
LOG_DIR="${RUN_ROOT}/logs/${ckpt_name}/${FRACTAL_LOG_PREFIX}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "=== Launching Multi-GPU Evaluation ==="
echo "Fractal Setting : ${FRACTAL_SETTING}"
echo "Num Runs        : ${#ENV_NAMES[@]}"
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
    RUN_NAME="${RUN_NAMES[i]}"
    ENV_NAME="${ENV_NAMES[i]}"
    SCENE_NAME="${SCENE_NAMES[i]}"
    MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS_ARR[i]}"
    EXTRA_ARGS="${EXTRA_ARGS_ARR[i]}"

    GPU_OFFSET=$((COUNT % NUM_GPUS))
    GPU_ID=${CUDA_DEVICES[$GPU_OFFSET]}
    PORT=$((BASE_PORT + GPU_OFFSET))

    EvalEnv "${GPU_ID}" "${PORT}" "${RUN_NAME}" "${ENV_NAME}" "${CKPT_PATH}" "${LOG_DIR}" \
            "${SIMPLER_PYTHON}" "${MAX_EPISODE_STEPS}" "${SCENE_NAME}" "${EXTRA_ARGS}" &
    RUNNING_EVAL_PIDS+=("$!")

    COUNT=$((COUNT + 1))

    if (( COUNT % MAX_PARALLEL_ENVS == 0 )); then
        WaitForEvalBatch || exit 1
    fi
    sleep 2
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
echo "Shutting down policy servers..."

CleanupProcesses
trap - EXIT INT TERM

echo "=== Evaluation Finished ==="
