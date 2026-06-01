#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export resVLA_python=${resVLA_python:-python}
ckpt_path=${RESVLA_CKPT:-./results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt}
gpu_id=${GPU_ID:-0}
port=${PORT:-5694}
################# resVLA Policy Server ######################

CUDA_VISIBLE_DEVICES=$gpu_id ${resVLA_python} deployment/model_server/server_policy.py \
    --ckpt_path ${ckpt_path} \
    --port ${port} \
    --use_bf16
