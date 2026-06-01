
# Policy Server

## Start

```bash
ckpt_path=${RESVLA_CKPT:-./checkpoints/resvla_checkpoint.pt}

python -m deployment.model_server.server_policy \
    --ckpt_path ${ckpt_path} \
    --port 10093 \
    --use_bf16
```

## Debug Client

```bash
python -m deployment.model_server.tools.debug_server_policy \
    --host 127.0.0.1 \
    --port 10093 \
    --test ping
```

Use `--test infer` to send a synthetic ResVLA `examples=[...]` request through the full websocket path.
