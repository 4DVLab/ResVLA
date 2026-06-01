# SimplerEnv Evaluation and Training

This document provides instructions for reproducing our **experimental results** with SimplerEnv.

The evaluation process consists of two main parts:

1. Setting up the `simpler_env` environment and dependencies.  
2. Running the evaluation by launching services in both `ResVLA` and `simpler_env` environments.  

We have verified that this workflow runs successfully on both **NVIDIA A100** and **RTX 4090** GPUs.

## 📦 1. Environment Setup

To set up the environment, please first follow the official [SimplerEnv repository](https://github.com/simpler-env/SimplerEnv) to install the base `simpler_env` environment. 

Afterwards, inside the `simpler_env` environment, install the following dependencies:

```bash
conda activate simpler_env
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

⚠️ **Common Issues**
On headless servers, SimplerEnv / ManiSkill2 may fail to create a Vulkan or EGL renderer. Common symptoms include:

```text
libvulkan.so.1: cannot open shared object file: No such file or directory
vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent
Cannot find cuda device suitable for rendering cuda:0
```

If you have sudo access, follow the official [ManiSkill Vulkan installation guide](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan).

If you do not have sudo access, or the system graphics runtime is incomplete, we provide a user-level setup script:

```bash
bash examples/SimplerEnv/setup_nvidia_graphics.sh
source "$HOME/.cache/resvla_nvidia_graphics/setup_env.sh"
```

The script downloads NVIDIA EGL/Vulkan runtime debs, extracts them into a user-writable directory, and generates `setup_env.sh`. It does not modify system files or `~/.bashrc` by default.

You can choose a custom install directory:

```bash
NVIDIA_GRAPHICS_PREFIX=/path/to/nvidia_graphics \
bash examples/SimplerEnv/setup_nvidia_graphics.sh

source /path/to/nvidia_graphics/setup_env.sh
```

If driver detection fails, specify the major driver version manually:

```bash
NVIDIA_DRIVER_MAJOR=550 bash examples/SimplerEnv/setup_nvidia_graphics.sh
```

After sourcing the generated environment script, verify Vulkan/EGL:

```bash
vulkaninfo --summary 2>&1 | grep -E "deviceName|driverVersion|apiVersion"
python - <<'PY'
import ctypes
ctypes.CDLL("libEGL_nvidia.so.0")
print("EGL OK")
PY
```

For most machines, keep SimplerEnv rendering on auto selection:

```bash
SIMPLER_RENDER_DEVICE=auto
```

Only set a machine-specific value such as `SIMPLER_RENDER_DEVICE=pci:34` when auto selection fails on your server.


## 🔧 Verify SimplerEnv

We provide a minimal environment verification script:

```bash
python examples/SimplerEnv/eval_files/test_your_simplerEnv.py
```

If you see the "✅ Env built successfully" message, it means SimplerEnv is installed correctly and ready to use.


---


## 🚀 2. Evaluate SimplerEnv

Run evaluation **from the repository root**. The provided automation scripts coordinate both pieces of the workflow:

- **ResVLA environment**: runs the policy inference server.
- **simpler_env environment**: runs the simulation evaluation code.


### Step 0. Download ResVLA checkpoint

Download the checkpoint from [resvla_simpler_env_2B](https://huggingface.co/GaussionZhong/resvla_simpler_env_2B):

```bash
HF_XET_HIGH_PERFORMANCE=1 hf download GaussionZhong/resvla_simpler_env_2B \
  --local-dir results/Checkpoints/resvla_simpler_env_2B
```

The default scripts expect:

```text
results/Checkpoints/resvla_simpler_env_2B/checkpoints/resVLA_simpler_env.pt
```


### Step 1. Run evaluation scripts

The current SimplerEnv examples use automation scripts that start both the ResVLA policy server and the SimplerEnv evaluation process.
Set the Python paths through environment variables if your environment names differ:

```bash
export resVLA_PYTHON=/path/to/resvla/python
export SIMPLER_PYTHON=/path/to/simpler_env/python
export SimplerEnv_PATH=/path/to/SimplerEnv
```

Run WidowX + Bridge evaluation:

```bash
bash examples/SimplerEnv/eval_files/auto_eval_scripts/star_bridge.sh /path/to/checkpoint.pt
```

Run Google Robot / Fractal visual matching evaluation, which is the setting used for our reported Table 3 results:

```bash
bash examples/SimplerEnv/eval_files/auto_eval_scripts/star_fractal_visual_matching.sh /path/to/checkpoint.pt
```

If you want the simulation variant aggregation setting instead, run:

```bash
bash examples/SimplerEnv/eval_files/auto_eval_scripts/star_fractal_variant_agg.sh /path/to/checkpoint.pt
```

You can select GPUs and parallelism with environment variables:

```bash
GPU_IDS=0,1,2,3 MAX_PARALLEL_ENVS=4 \
bash examples/SimplerEnv/eval_files/auto_eval_scripts/star_bridge.sh /path/to/checkpoint.pt
```

### Step 2. Calculate success rates

Each eval script writes an eval directory under the checkpoint's `logs/` folder, for example:

```text
logs/<checkpoint_name>/bridge_eval_YYYYMMDD_HHMMSS
logs/<checkpoint_name>/fractal_visual_matching_eval_YYYYMMDD_HHMMSS
logs/<checkpoint_name>/fractal_variant_agg_eval_YYYYMMDD_HHMMSS
```

Summarize success rates with:

```bash
python examples/SimplerEnv/eval_files/auto_eval_scripts/calc_metrics.py \
  --eval_dir /path/to/eval_dir \
  --save
```

`--save` writes `summary.json` into the eval directory. The script reads the `Average success` line from each `eval_*.log`.

For Bridge, repeated runs are merged by task name. For Fractal, runs are merged by setting prefix such as `pick_coke_vm`, `near_vm`, `drawer_vm`, and `putin_vm`.

Bridge does not have a separate `variant_agg` / `visual_matching` split in our scripts. The WidowX Bridge benchmark uses four fixed real-to-sim task settings with RGB overlays and 24 object episodes per task. Google Robot / Fractal exposes two official-style settings: simulation variant aggregation (`*_va`) and visual matching (`*_vm`).

⚠️ **Common Issues**

When running the policy server, if you see `NotImplementedError: Framework ResVLA is not implemented`, run `python resVLA/model/framework/QwenResVLA.py` to check whether framework registration works in your environment.



## 🚀 Training on OXE

### Data Preparation


Steps:
1) Download the LeRobot-format OXE datasets:
- [bridge_orig_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/bridge_orig_lerobot)
- [fractal20220817_data_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/fractal20220817_data_lerobot)

The released data mixture expects the following local directory names:

```text
playground/Datasets/SimplerEnv/bridge_orig_1.0.0_lerobot
playground/Datasets/SimplerEnv/fractal20220817_data_0.1.0_lerobot
```

2) Add `modality.json` to each dataset's `meta/` directory:
- [bridge modality](./train_files/modality.json): save as `bridge_orig_1.0.0_lerobot/meta/modality.json`
- [fractal modality](./train_files/fractal_modality.json): save as `fractal20220817_data_0.1.0_lerobot/meta/modality.json`

3) Add your dataset path to `config.yaml`:
    ```yaml
    datasets:
      vla_data:
        dataset_py: lerobot_datasets
        data_root_dir: playground/Datasets/SimplerEnv  # path to your dataset root
        data_mix: simpler_env_all
    ```


### Check Your Dataloader

We provide a simple way to check your dataloader. Make sure you can load batched data:

```bash
python resVLA/dataloader/lerobot_datasets.py --config_yaml examples/SimplerEnv/train_files/resvla_cotrain_oxe.yaml
```

## Framework Preparation

Before running, you need to ensure that your framework can `forward` and `predict_action` using a fake data example.

Try the following command:

```bash
python resVLA/model/framework/QwenResVLA.py --config_yaml examples/SimplerEnv/train_files/resvla_cotrain_oxe.yaml
```

Note: You can modify the following code snippet to align with your dataset:

```python
    # Generate a fake sample
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),  # action_chunk, action_dim
        "image": [image, image],  # two views
        "lang": "This is a fake for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
    }
```

## Training

Once everything is ready, use our provided script to start training:

```bash
bash ./examples/SimplerEnv/train_files/run_oxe_train_ResVLA.sh
```

⚠️ **Note:** Ensure that the script uses `examples/SimplerEnv/train_files/resvla_cotrain_oxe.yaml` through the `--config_yaml` argument.
