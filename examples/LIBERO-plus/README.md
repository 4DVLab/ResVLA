# LIBERO-plus Evaluation

This document describes how to evaluate ResVLA on LIBERO-plus. LIBERO-plus is a robustness benchmark built on top of LIBERO. In ResVLA, LIBERO-plus evaluation directly uses the released LIBERO checkpoint:

```text
results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt
```

No separate LIBERO-plus checkpoint is required.

## 1. Prepare ResVLA

Follow the root [README](../../README.md) to install the ResVLA environment and download the LIBERO checkpoint:

```bash
HF_XET_HIGH_PERFORMANCE=1 hf download GaussionZhong/resvla_libero_all_2B \
  --local-dir results/Checkpoints/resvla_libero_all_2B
```

The expected file is:

```text
results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt
```

## 2. Prepare LIBERO-plus

Clone and install the official LIBERO-plus repository:

```bash
git clone https://github.com/sylvestf/LIBERO-plus.git ../LIBERO-plus
cd ../LIBERO-plus
pip install -e .
pip install -r extra_requirements.txt
cd ../ResVLA
```

LIBERO-plus also requires its asset package. Follow the official LIBERO-plus instructions to download the assets and extract them to:

```text
../LIBERO-plus/libero/libero/assets
```

If your LIBERO-plus repository is not located at `../LIBERO-plus`, set:

```bash
export LIBERO_HOME=/path/to/LIBERO-plus
```

## 3. Single-Suite Evaluation

Use two terminals from the ResVLA repository root.

Terminal 1, start the ResVLA policy server:

```bash
export RESVLA_CKPT=results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt
export GPU_ID=0
export PORT=5694
bash examples/LIBERO-plus/eval_files/run_policy_server.sh
```

Terminal 2, run LIBERO-plus evaluation:

```bash
export LIBERO_HOME=${LIBERO_HOME:-../LIBERO-plus}
export LIBERO_Python=/path/to/libero-plus/python
export RESVLA_CKPT=results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt
bash examples/LIBERO-plus/eval_files/eval_libero_plus.sh
```

By default, `eval_libero_plus.sh` evaluates `libero_goal` with one trial per task. Edit the script if you want to change:

```bash
task_suite_name=libero_goal
num_trials_per_task=1
```

Valid task suites are:

```text
libero_spatial
libero_object
libero_goal
libero_10
libero_90
all
```

## 4. Automated Multi-Suite Evaluation

The automation script starts one policy server per suite and evaluates the original LIBERO suites under LIBERO-plus perturbations:

```bash
export LIBERO_HOME=${LIBERO_HOME:-../LIBERO-plus}
export LIBERO_python=/path/to/libero-plus/python
export resVLA_python=/path/to/resvla/python
export RESVLA_CKPT=results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt

bash examples/LIBERO-plus/eval_files/auto_eval_scripts/auto_eval_libero_plus.sh
```

Resume an interrupted run:

```bash
bash examples/LIBERO-plus/eval_files/auto_eval_scripts/auto_eval_libero_plus.sh --resume
```

Evaluate only one perturbation category by calling the parallel worker directly:

```bash
bash examples/LIBERO-plus/eval_files/auto_eval_scripts/eval_libero_plus_parall.sh \
  results/Checkpoints/resvla_libero_all_2B/checkpoints/resVLA_libero.pt \
  libero_goal \
  0 \
  --category-filter camera
```

Supported category filters include:

```text
camera
robot
language
light
background
noise
layout
clean
all
```

## 5. Outputs

For the automated scripts, videos and logs are written under the checkpoint run directory:

```text
results/Checkpoints/resvla_libero_all_2B/videos_libero_plus/<task_suite>/
results/Checkpoints/resvla_libero_all_2B/logs_libero_plus/<task_suite>/
```

For the simple two-terminal script, videos are written to:

```text
results/libero_plus/<task_suite>/<timestamp>/
```

## Common Issues

If the evaluator cannot import `libero`, check that `LIBERO_HOME` points to the LIBERO-plus repository and that `PYTHONPATH` includes it. The provided scripts set `PYTHONPATH` from `LIBERO_HOME`.

If assets are missing, verify that the LIBERO-plus assets were extracted to `../LIBERO-plus/libero/libero/assets`.

If the policy server cannot load the checkpoint, confirm that `resVLA_libero.pt`, `config.yaml`, and `dataset_statistics.json` are all present in `results/Checkpoints/resvla_libero_all_2B`. You can also follow the LIBERO evaluation README to start the policy server and check whether the released checkpoint loads correctly.
