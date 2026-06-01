# Auto Evaluation Scripts

This folder contains the scripts we use internally to run quick evaluations across every task.
A fully automated run depends on our in-house platform setup, so the scripts may not work unchanged on your own infrastructure.
Please treat them as references for the underlying `deployment/model_server/server_policy.py` and `examples/SimplerEnv/eval_files/start_simpler_env.py` workflow.

# Real Auto run

you can simply run `star_bridge.sh` and `star_fractal.sh` for all SimplerEnv benchmarks.

For Google Robot / Fractal, prefer the explicit entrypoints:

```bash
bash examples/SimplerEnv/eval_files/auto_eval_scripts/star_fractal_visual_matching.sh /path/to/checkpoint.pt
```

```bash
bash examples/SimplerEnv/eval_files/auto_eval_scripts/star_fractal_variant_agg.sh /path/to/checkpoint.pt
```

# Calculate success rates

Use `calc_metrics.py` on the eval directory produced by `star_bridge.sh`,
`star_fractal_visual_matching.sh`, or `star_fractal_variant_agg.sh`.

```bash
python \
  examples/SimplerEnv/eval_files/auto_eval_scripts/calc_metrics.py \
  --eval_dir /path/to/bridge_eval_YYYYMMDD_HHMMSS \
  --save
```

For Google Robot visual matching:

```bash
python \
  examples/SimplerEnv/eval_files/auto_eval_scripts/calc_metrics.py \
  --eval_dir /path/to/fractal_visual_matching_eval_YYYYMMDD_HHMMSS \
  --save
```

`--save` writes `summary.json` into the eval directory. Add `--json` if you
want the same summary printed as machine-readable JSON.



Bridge does not have a separate `variant_agg` / `visual_matching` split in our
scripts. The WidowX Bridge benchmark uses four fixed real-to-sim task settings
with RGB overlays and 24 object episodes per task. In contrast, the Google Robot
Fractal benchmark exposes two official-style evaluation settings: simulation
variant aggregation (`*_va`) and visual matching (`*_vm`).
