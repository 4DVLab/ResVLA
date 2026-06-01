<br>
<p align="center">
<h1 align="center"><strong>From Noise to Intent: Anchoring Generative VLA Policies with Residual Bridges</strong></h1>
  <p align="center">
      <strong><span style="color: red;"> ICML 2026</span></strong>
    <br>
   <a href="https://github.com/ymzhong66" target="_blank">Yiming Zhong*</a>&emsp;
   <a href="https://eniverz.github.io/profile" target="_blank">Yaoyu He*</a>&emsp;
   <a href="https://yizhifengyeyzm.github.io/" target="_blank">Zemin Yang*</a>&emsp;
   <a href="https://huubgit.github.io/" target="_blank">Pengfei Tian</a>&emsp;
   <a href="https://hy-van.github.io/" target="_blank">Yifan Huang</a>&emsp;
   <a href="https://openreview.net/profile?id=~Qingqiu_Huang2" target="_blank">Qingqiu Huang</a>&emsp;
   <a href="https://xingezhu.me/aboutme.html" target="_blank">Xinge Zhu</a>&emsp;
   <a href="https://yuexinma.me" target="_blank">Yuexin Ma</a>&emsp;
    <br>
    ShanghaiTech University, Morphic Robotics, The Chinese University of Hong Kong
    <br>
    *Indicates Equal Contribution
    <br>
  </p>
</p>

<p align="center">
  <a href="https://res-vla.github.io/ResVLA/"><b>📖 Project Page</b></a>
</p>
<p align="center">
  <strong>Code and Checkpoints:</strong>
  <a href="https://huggingface.co/GaussionZhong/resvla_libero_all_2B">LIBERO checkpoint</a>
  |
  <a href="https://huggingface.co/GaussionZhong/resvla_simpler_env_2B">SimplerEnv checkpoint</a>
</p>
</div>

# Quick Start

The commands below assume a fresh clone and a clean Python environment.

```bash
git clone https://github.com/4DVLab/ResVLA.git
cd ResVLA

conda create -n resvla python=3.10 -y
conda activate resvla

# Install PyTorch first so torchvision resolves to the tested torch version.
pip install torch==2.6.0 torchvision==0.21.0
pip install -r requirements.txt
pip install -e .

# Required by the default Qwen3-VL backend.
pip install flash-attn --no-build-isolation
```

If `flash-attn` fails with `No such file or directory: '/usr/local/cuda/bin/nvcc'`, install a CUDA toolkit that provides `nvcc`, or install a prebuilt `flash-attn` wheel matching your Python, PyTorch, and CUDA versions from the FlashAttention releases.

On Ubuntu/Debian systems, if `python -m venv` fails with `ensurepip is not available`, install the system venv package first:

```bash
sudo apt install python3.10-venv
```

## Download Models

ResVLA checkpoints are released on Hugging Face. Keep the downloaded folders in the same structure used below because the loader expects each `.pt` file to live under `<run_dir>/checkpoints/` with `config.yaml` and `dataset_statistics.json` in `<run_dir>/`.

```bash
# Base VLM used by the released configs.
mkdir -p ../ckpt
hf download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir ../ckpt/Qwen3-VL-2B-Instruct

# Released ResVLA checkpoints.
HF_XET_HIGH_PERFORMANCE=1 hf download GaussionZhong/resvla_libero_all_2B \
  --local-dir results/Checkpoints/resvla_libero_all_2B

HF_XET_HIGH_PERFORMANCE=1 hf download GaussionZhong/resvla_simpler_env_2B \
  --local-dir results/Checkpoints/resvla_simpler_env_2B
```

Large checkpoint downloads may appear quiet for several minutes. If needed, check the size of files under `results/Checkpoints/<run>/.cache/huggingface/download/`. Re-running the same `hf download` command resumes partial downloads.

Expected checkpoint paths:

```text
results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt
results/Checkpoints/resvla_simpler_env_2B/checkpoints/resVLA_simpler_env.pt
```

If you store `Qwen3-VL-2B-Instruct` somewhere else, update `framework.qwenvl.base_vlm` in the checkpoint `config.yaml` or in your training config.

## Installation Verification

Run these before installing LIBERO or SimplerEnv simulators. They verify that the environment, base VLM, checkpoint files, model loading, and websocket policy server are working.

```bash
python - <<'PY'
from pathlib import Path
from resVLA.model.framework.share_tools import read_mode_config

for ckpt in [
    "results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt",
    "results/Checkpoints/resvla_simpler_env_2B/checkpoints/resVLA_simpler_env.pt",
]:
    cfg, stats = read_mode_config(Path(ckpt))
    print(ckpt)
    print("  framework:", cfg["framework"]["name"])
    print("  base_vlm:", cfg["framework"]["qwenvl"]["base_vlm"])
    print("  stats:", list(stats.keys()))
PY
```

Start a policy server:

```bash
CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
  --ckpt_path results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt \
  --port 18093 \
  --idle_timeout 120 \
  --use_bf16
```

In another terminal:

```bash
python -m deployment.model_server.tools.debug_server_policy \
  --host 127.0.0.1 \
  --port 18093 \
  --test infer
```

A successful verification returns `status: ok` and a `normalized_actions` array with shape `(1, 8, 7)` for the LIBERO checkpoint.

## Training

Training requires the corresponding LeRobot-format datasets. The provided scripts expect these default directories:

```text
playground/Datasets/LEROBOT_LIBERO_DATA
playground/Datasets/SimplerEnv
```

Prepare LIBERO data:

```bash
export DEST=playground/Datasets/LEROBOT_LIBERO_DATA
bash examples/LIBERO/data_preparation.sh
```

For SimplerEnv / OXE training, download the LeRobot-format Bridge and Fractal datasets and place them under:

```text
playground/Datasets/SimplerEnv/bridge_orig_1.0.0_lerobot
playground/Datasets/SimplerEnv/fractal20220817_data_0.1.0_lerobot
```

Check the dataloader before launching training:

```bash
NO_ALBUMENTATIONS_UPDATE=1 \
python resVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/SimplerEnv/train_files/resvla_cotrain_oxe.yaml
```

Launch training:

```bash
bash examples/LIBERO/train_files/run_libero_train_resVLA.sh
bash examples/SimplerEnv/train_files/run_oxe_train_ResVLA.sh
```

The released training scripts are multi-GPU recipes. Adjust `CUDA_VISIBLE_DEVICES`, `--num_processes`, batch size, dataset paths, and `base_vlm` for your machine.

## Evaluation

- LIBERO: see [examples/LIBERO/README.md](examples/LIBERO/README.md)
- LIBERO-plus: see [examples/LIBERO-plus/README.md](examples/LIBERO-plus/README.md)
- SimplerEnv: see [examples/SimplerEnv/README.md](examples/SimplerEnv/README.md)

Both evaluation workflows use the same policy server tested above, then run the simulator-side evaluation in a separate environment.
