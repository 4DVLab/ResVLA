# Deployment Notes

## Policy Server

```bash
ckpt_path=${RESVLA_CKPT:-./checkpoints/resvla_checkpoint.pt}

python -m deployment.model_server.server_policy \
    --ckpt_path ${ckpt_path} \
    --port 10093 \
    --use_bf16
```

## Connection Verification

```bash
python -m deployment.model_server.tools.debug_server_policy \
    --host 127.0.0.1 \
    --port 10093 \
    --test ping
```

Use `--test infer` only when the policy checkpoint and GPU environment are ready.

## Robot Network Example

```bash
sudo ip addr add 172.16.0.100/24 dev enx207bd51a3217
sudo ip link set enx207bd51a3217 up
ping 172.16.0.2
```

## install frankx

```bash
conda install frankx
unzip frankx.zip
```
