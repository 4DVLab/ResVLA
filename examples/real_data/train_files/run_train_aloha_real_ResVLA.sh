

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
config_yaml=./examples/real_data/train_files/resvla_train_real_aloha.yaml
libero_data_root=playground/Datasets/real_data
data_mix=aloha_demo
run_root_dir=./results/Checkpoints
run_id=resvla_aloha_real_dual_demo
anchor_train_noise_std=1.0
resume=False

# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled
# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
# check if output directory already exists
if [ -d "${output_dir}" ] && [ "${resume}" == "False" ]; then
    echo "Output directory already exists: ${output_dir}"
    echo "Do you want to overwrite? (y/n): "
    read overwrite
    if [ "${overwrite}" != "y" ]; then
        echo "Aborted by user."
        exit 1
    fi
fi
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file resVLA/config/deepspeed_zero2.yaml \
  --num_processes 8 \
  --main_process_port 29500 \
  resVLA/training/train_resvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.anchor_train_noise_std ${anchor_train_noise_std} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 22 \
  --datasets.vla_data.include_state True \
  --datasets.vla_data.video_backend torchvision_av \
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
