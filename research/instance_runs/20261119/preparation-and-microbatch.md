Single-card instance training supports three optional execution controls. Omitting them selects the original one-at-a-time read and forward path.

| Variable | Default | Operation |
|---|---:|---|
| SUCTION20_INSTANCE_TRAIN_WORKERS | 0 | Bounded ordered CPU threads for loading, validation and RGB augmentation of source/target observations. Source workers also call the model's optional CPU geometry preparation method. |
| SUCTION20_INSTANCE_EVAL_WORKERS | 0 | Ordered CPU preparation before scoring each fixed evaluation observation. |
| SUCTION20_INSTANCE_B_MICROBATCH | 1 | Number of independent source observations per B context-encoder call. Texture inputs are batched only at identical padded dimensions to retain per-instance GroupNorm. Ring projection, support and scoring retain each instance's original inputs. |

The existing successful-instance budget, RGB random draws, per-instance supervision weight, exact candidate labels, skips and phase seeds apply. Sampling and random RGB parameters are produced on the coordinator before workers run. The maximum outstanding batch fits within the remaining successful-instance budget. Checkpoints are taken only after ordered tickets have been consumed. Worker and microbatch options are independent of the checkpoint architecture. A larger microbatch can change reduction order and floating-point results. Record the settings used for every run.

To measure in an exclusive single-card window, set CUDA_VISIBLE_DEVICES to one physical GPU index and PYTHONPATH to the route repository. From the B repository run:

```sh
python research/instance_runs/20261119/benchmark_single_gpu_b.py \
  --source "$PILOT_TRAIN" --selection "$PILOT_SELECTION" --output "$NEW_RESULT_JSON"
```

The probe runs real FP32 optimizer updates from a reproducible initialization. The first update of each variant warms the path. It first tests the loader concurrency with microbatch one, then tests larger microbatches with the fastest measured worker count. It writes per-update elapsed time, memory, gradient norm, loss and draws. Production runs should select a configuration using the actual card's measurements.
