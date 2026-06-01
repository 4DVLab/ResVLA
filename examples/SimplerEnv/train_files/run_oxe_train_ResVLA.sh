

# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=ResVLA
freeze_module_list=''
base_vlm=../ckpt/Qwen3-VL-2B-Instruct
config_yaml=./examples/SimplerEnv/train_files/resvla_cotrain_oxe.yaml
oxe_data_root=playground/Datasets/SimplerEnv
data_mix=simpler_env_all
run_root_dir=./results/Checkpoints
run_id=resvla_simpler_env_2B
# We use 1.0 for consistency across released checkpoints. Smaller values
# such as 0.2, 0.4, or 0.6 and so on can speed up training and work with fewer
# inference steps.
anchor_train_noise_std=1.0
resume=False
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
# check if output directory already exists
if [ -d "${output_dir}" ] && [ "${resume}" != "True" ]; then
    echo "Output directory already exists: ${output_dir}"
    echo "Do you want to overwrite? ([y]/n): "
    read overwrite
    if [ "${overwrite}" != "y" ] && [ "${overwrite}" != "" ]; then
        echo "Aborted by user."
        exit 1
    fi
fi
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/



CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --config_file resVLA/config/deepspeed_zero2.yaml \
  --num_processes 4 \
  --main_process_port 29500 \
  resVLA/training/train_resvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.future_action_window_size 7 \
  --framework.action_model.anchor_train_noise_std ${anchor_train_noise_std} \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 40 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 1000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project resVLA \
  --trainer.is_resume ${resume} \
  # --is_debug True



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file resVLA/config/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   resVLA/training/train_resvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
