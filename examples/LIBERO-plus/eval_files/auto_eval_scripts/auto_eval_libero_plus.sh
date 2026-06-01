#!/bin/bash

RESUME_ARGS=()
for arg in "$@"; do
	if [ "$arg" = "--resume" ]; then
		RESUME_ARGS+=("--resume")
		break
	fi
done

SCRIPT_PATH="./examples/LIBERO-plus/eval_files/auto_eval_scripts/eval_libero_plus_parall.sh"
ckpt_path=${RESVLA_CKPT:-./results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt}

run_index_base=432

#####################################################
task_suite_name=libero_10 # align with your model
run_index=$((run_index_base + 0))
bash $SCRIPT_PATH $ckpt_path $task_suite_name $run_index "${RESUME_ARGS[@]}" &
#####################################################

sleep 15
#####################################################
task_suite_name=libero_goal # align with your model
run_index=$((run_index_base + 1))
bash $SCRIPT_PATH $ckpt_path $task_suite_name $run_index "${RESUME_ARGS[@]}" &
#####################################################

sleep 15
#####################################################
task_suite_name=libero_object # align with your model
run_index=$((run_index_base + 2))
bash $SCRIPT_PATH $ckpt_path $task_suite_name $run_index "${RESUME_ARGS[@]}" &
#####################################################

sleep 15
####################################################
task_suite_name=libero_spatial # align with your model
run_index=$((run_index_base + 3))
bash $SCRIPT_PATH $ckpt_path $task_suite_name $run_index "${RESUME_ARGS[@]}" &
#####################################################
