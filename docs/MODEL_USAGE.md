# 模型运行入口

实现包含只读源数据适配、E0–E5、预训练、两种目标域微调、实际姿态 R20 评价和无标签推理。CPU 合成测试不需要 spconv；生产稀疏主干需要 CUDA。运行结果及限制见 [验证记录](MODEL_VALIDATION.md)。

## 环境与外部主干

已验证 Python 3.10、PyTorch 2.5.1+cu124、spconv-cu120 2.3.6。项目不打包第三方主干源码；下载工具核对固定 Git blob SHA，加载时再次核对。第三方许可保留在下载目录。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[model,gpu]'
python scripts/setup_backbone.py --output external/graspness_modern
python -m pip install pytest
python -m pytest -q
```

若继承了其他 Python 工程的 `PYTHONPATH`，先清除它，避免加载外部 pytest 插件或同名模块。完整开发工具可安装 `.[dev]`；已有数据的模型适配与单元测试不要求重新制作数据。

## 数据与有界 smoke

`SOURCE` 指向带 `scenes/`、`models/`、候选和标定的 SuctionNet 原始入口；`LABELS` 指向已完成标签的 `output/`。CD-007 对应 `full-20261110/input_raw` 与 `full-20261110/output`。冻结配置按普通 JSON 读取，保留旧 producer 的额外字段，不重新拟合 Seal。

```bash
export SOURCE=/path/to/full-20261110/input_raw
export LABELS=/path/to/full-20261110/output
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/train_pretrain.py --config configs/smoke_e1.yaml \
  --run-dir runs/smoke-e1 --source-root "$SOURCE" --label-root "$LABELS"

python scripts/train_pretrain.py --config configs/smoke_e5.yaml \
  --run-dir runs/smoke-e5 --stop-at-step 9 \
  --source-root "$SOURCE" --label-root "$LABELS"
python scripts/train_pretrain.py --config configs/smoke_e5.yaml \
  --run-dir runs/smoke-e5 --resume runs/smoke-e5/last.pt \
  --source-root "$SOURCE" --label-root "$LABELS"
```

这些配置分别上限 2 和 18 次成功 optimizer 更新；AMP 跳过不推进步数。E5 独立初始化，第二条 E5 命令恢复到总计 18 次，不额外训练 18 次。microbatch=1，梯度累计4；完整观测、2 mm 体素、512候选不因 smoke 而缩减。主干训练时 BN 正常更新。同步读取数据使样本及增强种子可准确恢复。

`last.pt` 保存 model、optimizer、AMP、步数、独立采样器、Python/NumPy/Torch/CUDA RNG、配置、冻结物理定义及代码身份；`best.pt` 按开发验证指标选取。`--init-checkpoint` 仅初始化兼容权重，`--resume` 恢复完整训练状态。恢复时可以迁移目录和设备，不能静默更改训练语义。

## 正式公开数据预训练

`configs/pretrain_e0.yaml` 至 `pretrain_e5.yaml` 是带 pending profile 的正式模板，不能原样启动。当前正在运行的版本使用 [六份精确 JSON 配置及哈希](../configs/runtime_profiles/cd009-v3/manifest.json)：H20/E5、L20/E0–E4。策略、测量依据和部署/发布版本区别见 [CD-009 当前预训练](CD009_PRETRAINING.md)。

每组独立 seed42 初始化，90/10 场景划分，230400 有效样本、名义 40 轮；当前 E=4 换算为 57600 成功更新。跳步与故障回放另计，不把 smoke 或旧版本权重接到正式预算。完整训练正在分阶段执行，尚未完成六组与各 10 次开发验证。

正式多机运行从下方 [supervisor 入口](#cd-009-persistent-multi-host-pretraining) 和 [公共模板](implementation/examples/cd009-supervisor.json) 开始，登记精确 config SHA256，填写本地数据/主干路径、机器身份和授权 GPU。私有控制配置不提交到仓库。冻结运行代码后通过 host evidence、首个 100 步暂停/恢复和稳定性门槛逐级启动；不要跳过 supervisor 的单一 owner 和验证/备份协调。

## 推理与开发评价

```bash
python scripts/infer.py --config configs/smoke_e5.yaml \
  --checkpoint runs/smoke-e5/last.pt --frames 9:0,19:8 \
  --source-root "$SOURCE" --output runs/smoke-e5/predictions.npz

python scripts/evaluate.py --config configs/smoke_e5.yaml \
  --checkpoint runs/smoke-e5/last.pt --split smoke_val \
  --source-root "$SOURCE" --label-root "$LABELS" \
  --output runs/smoke-e5/metrics.json
```

推理不读取标签目录。公开数据默认使用给定 GT 实例 mask，属于控制实验；可在配置 `data.instance_mask_root` 指定预测 mask 根目录。它不等同于已验证的在线分割管线。NPZ 保留原相机系点、轴、q、q0、输入实例和原像素索引，并导出 `[score,nx,ny,nz,x,y,z]`。NMS 采用10 mm/15°，每实例最多10个、全场景最多50个。

评价器按最终 `(p,n)` 重新计算 S/W/碰撞/Q20，GT仅用于评价和归属。`smoke_val` 仅两帧开发样本；`val` 是预定开发划分。代码支持显式 `test_seen/test_similar/test_novel`，本次没有执行官方测试。R20代理指标不能当作实机成功率。

## 无标签自有观测推理

每行JSONL只需 `sample_id` 和上述格式的 `observation`，不需要标签、split或源数据目录。相对文件路径以manifest所在目录解析。

```bash
python scripts/infer.py --config runs/target/resolved_config.json \
  --checkpoint runs/target/best.pt --observation-manifest /path/to/observations.jsonl \
  --output runs/target/predictions.npz
```

也可用公开预训练checkpoint及对应配置。`sample_<转义sample_id>/` 下保存预测，`source_index_kind` 区分真实像素索引和点云行号。成功头只有在已训练的attempt checkpoint中才可用于排序；物理q始终保留。

## 自有数据接口

JSONL格式、单位和分组约束见 [微调规格](implementation/FINETUNING.md)。支持 RGB-D 和 pointcloud NPZ；同一物理观测的多次尝试合并编码。相邻观测按 `group_id` 划分，跨split/group的同一观测会被拒绝。

- `dense_proxy`：读取实际可用的质量/分量/轴标签，缺失项 mask=False；只接受与 checkpoint 相同的冻结物理定义。
- `attempt_success`：实际接触点、实际外向轴及0/1结果进入独立任务头；未尝试点不产生负标签，映射失败被跳过并计数。

```bash
python scripts/finetune.py \
  --config docs/implementation/examples/finetune_attempt_success.yaml \
  --target-manifest /path/to/manifest.jsonl \
  --init-checkpoint runs/public/selected/best.pt --run-dir runs/target \
  --source-root "$SOURCE" --label-root "$LABELS"

python scripts/finetune.py \
  --config docs/implementation/examples/finetune_attempt_success.yaml \
  --target-manifest /path/to/manifest.jsonl \
  --resume runs/target/last.pt --run-dir runs/target \
  --source-root "$SOURCE" --label-root "$LABELS"
```

dense模式换用 `finetune_dense_proxy.yaml`。默认总计1000步：前200步仅heads，之后decoder+heads；encoder和BN统计/affine全程冻结。每步2目标帧+2源帧，目标帧权重0.5，源帧0.25，不再除4。phase切换保留heads的Adam状态。

验证固定已标注点/已执行姿态集合。dense使用实例→帧→组MAE；attempt报告AUROC/PR-AUC，单一类别时AUROC保持未定义并用任务BCE观察。没有真实目标数据时，不宣称完成了目标微调或真实策略验证。

## CD-009 persistent multi-host pretraining

`scripts/supervise_pretrain.py` is a Linux, standard-library-only supervisor for
these six runs. H20 owns the local assignment ledger; L20 only executes explicit
assignments. It does not start TODO2 evaluation. V3 requires a measured, selected
configuration for each host/experiment; the supervisor never changes it on OOM.
The example [supervisor config](implementation/examples/cd009-supervisor.json)
contains placeholders for every GPU UUID, host identity and transport alias. Fill
actual UUIDs, `/etc/machine-id` values and hostnames locally before use. Container machine IDs can be
identical; hostname matching is also mandatory. Recheck every UUID, CPU affinity, environment,
shared mount and free-space measurement at deployment. The config must have the
same absolute path on both hosts, normally `control/supervisor.json`. SSH options
are an argv list; key files remain in H20 `control/` and are never backed up.

The coordinator needs working noninteractive SSH from H20 to L20. Its entire
remote command is the configured absolute Python executable plus
`repo/scripts/supervise_pretrain.py --config <absolute-config> rpc --host L20`;
requests are JSON on stdin. This permits a restricted SSH forced-command guard.
No arbitrary shell commands or SSHFS locks implement coordination.

Before recording a host gate, freeze the same code/dependencies, verify the
manifest, exercise the synthetic CUDA kernel, and prove detached-process survival
through logout/reconnect on that host. Workers use `setsid` through Python
`start_new_session=True`, closed stdin and redirected output. This is not proof
that a particular login policy permits persistence; record actual survival.

On H20, after filling the example and completing those checks:

```bash
SUP=/srv/cd009-pretraining-v3/repo/scripts/supervise_pretrain.py
CFG=/srv/cd009-pretraining-v3/control/supervisor.json
PY=/srv/cd009-pretraining-v3/venv/bin/python
"$PY" "$SUP" --config "$CFG" probe --host H20
"$PY" "$SUP" --config "$CFG" probe --host L20
"$PY" "$SUP" --config "$CFG" evidence host:H20 --file /path/to/h20-checks.json
"$PY" "$SUP" --config "$CFG" evidence host:L20 --file /path/to/l20-checks.json
"$PY" "$SUP" --config "$CFG" plan
# Start only after all deployment checks; tmux itself must pass survival check.
tmux new-session -d -s cd009-coordinator "$PY $SUP --config $CFG serve"
"$PY" "$SUP" --config "$CFG" status
```

`plan` and `status` are read-only. `tick` performs one reconciliation/grant/launch
cycle; `serve` repeats every 60 seconds and holds a separate daemon lock. Operator
commands acquire the ledger lock between cycles. Host evidence JSON requires
`passed`, `persistence_survival`, `environment_identity`, `kernel_check` and
`storage_check` all true; retain the actual command outputs alongside this
machine-readable summary. Evidence records are operational reviews, not new
requests for user authorization.

The initial mapping is E5/H20 GPU0 and E3/L20 GPU0. Every initial attempt pauses
at `ceil(400/effective_batch)` successful updates without an extra validation,
saves and backs up, and exits75. These samples count toward the formal budget. Inspect
its checkpoint/identity/sampler before `resume E3` or `resume E5`. Resume requests `last.pt`; the trainer resolves the authoritative manifest,
verifies its current checkpoint and explicitly records predecessor fallback if
needed. Stale status or a missing alias never overrides the manifest. Each resume
uses a new attempt token and the same device.
After verifying the next successful update, record `evidence prefix:E3 --file ...`
with `passed:true`, the derived `paused_step`/`resumed_step`, and
`identity_and_sampler_passed:true`. V3 prefix/stability/completion evidence also
binds the current `attempt_id`, `profile_sha256` and `contract_version`.
Every experiment requires its own prefix evidence. Record `stable:E3`, then
`stable:E0`, then `stable:E1` after each corresponding run's prefix/resume checks
and at least20 successful updates plus600 seconds of healthy observations. A
stability record requires `passed:true, successful_updates:20,
observed_seconds:600, pressure_passed:true`. These gates permit E0/L20 GPU1,
E1/L20 GPU2 and E4/L20 GPU3 in order. E2 uses a completed/released authorized GPU
(prefer L20 when simultaneously free), or H20 GPU1 when idle after E3 stability.
There is no two-job limit and no automatic yielding of an occupied GPU1.

Each launch requires three empty-GPU samples30 seconds apart and another check
immediately before spawn, matching index/UUID, no compute PID and memory below
1GiB. CPU affinity pools are nonoverlapping; each selected profile chooses a
thread count within its pool. Target host
concurrency requires128GiB available RAM per run and200GiB free local/shared
disk. RSS remains limited to64GiB; GPU allocator and whole-process memory
budgets come from each measured profile with physical headroom. Disk free
below100GiB or task storage above100GiB requests pause. Full-process GPU memory
is monitored separately. NVML PIDs may be host PIDs while the trainer has a
container PID. Ownership therefore requires the matching attempt, trainer PID
and assigned UUID in status after the trainer creates and synchronizes a live
CUDA context. Exactly one NVML process on that UUID then maps the host PID;
multiple processes request a pause of our positively identified local trainer.
Missing NVML data or no enumerated process remains unknown and requests pause
after five minutes of CUDA readiness. Once established, later arrivals do not
cause automatic yielding. Signals always use the local PID/start identity. Host resource JSONL is recorded every60 seconds.

Training status distinguishes validation waiting from stalled work. A45-minute
compute stall, two hours without successful backup, or resource stop threshold
writes `pause.request`. The15-minute stall and48GiB RSS thresholds produce
warnings. A pause has five minutes to reach a boundary; only the positively
identified trainer PID can then receive TERM, with KILL after another minute.
PID start ticks, boot identity and Linux pidfd prevent killing a reused PID.
GPU/run locks are inherited by the trainer, so loss of its worker does not free
the device for another experiment. Sustained memory/io full PSI above10%, or
same-run data time above twice its baseline and30% of update time for10 minutes,
pauses the newest local worker. The coordinator freezes expansion during pressure
or uncertain ownership. Review resource traces across hosts before promotion;
the20-observation baseline does not establish a universal throughput estimate.

At a validation node the trainer submits `validation.request.json` and waits.
The coordinator durably reserves the request before delivering the matching
`validation.grant.json`; one geometry validation is granted globally. Missing
heartbeats or SSH timeouts never expire that grant. A matching done record or
positive old-host process/lock exit verification releases it. Thus a crash after
checkpoint commit but before `validation.done.json` is recoverable without a
second simultaneous validator. An unreachable old host remains reserved.

Backups similarly use request/grant/done files, one transfer globally, capped
at25MiB/s in Python (rsync is not required). H20 snapshots also retain the
coordinator ledger and an allowlisted runtime configuration; SSH credentials and
transport options must be provisioned separately. After restoring that ledger,
reconcile all old owners before assigning work. They snapshot the authoritative
checkpoint manifest, all referenced immutable checkpoint blobs, metadata,
validation JSON and bounded append-only attempt logs. Logs may contain a tail
beyond the committed checkpoint; its ledger cursor remains authoritative.
Checkpoint objects are deduplicated by SHA256, reread to verify, and linked within
the backup root. `backup-complete.json` is published last. Two ordinary hourly
snapshots per attempt and all validation-node/final snapshots are retained; local
originals remain intact. Final/pause workers keep locks while waiting for their
last backup grant, even after the trainer has exited. Two copy failures stop the
worker as failed with local evidence intact, so diagnosed checkpoint recovery can
proceed; missing-grant/network waits do not release ownership. Immutable blobs
and atomic metadata are locally hardlink-pinned before copying, preventing
pruning/replacement races. Selected assignment/command/status/resource evidence
is included; SSH transport secrets are excluded. Restore a complete snapshot with
links dereferenced (`cp -aL` or equivalent), then verify its hashes before using
its manifest. A collection of mountpoints is still one shared backup copy.

```bash
"$PY" "$SUP" --config "$CFG" pause E3 --reason 'investigate sustained resource pressure'
"$PY" "$SUP" --config "$CFG" resume E3
# A failed attempt requires an explicit diagnosed health/data/state review:
"$PY" "$SUP" --config "$CFG" recover E3 --evidence /path/to/recovery-checks.json
```

`recover` requires true `passed/health_check/data_check/state_check` fields and a
nonempty `diagnosis`, verifies old-host exit/locks and checkpoint hash, and permits
one diagnosed failure recovery per experiment. Repeated failure remains blocked
for investigation. Missing checkpoints never silently restart initialization.
For a diagnosed failure before initialization, the separate command below is
permitted once per experiment. It requires the same health/data/state/diagnosis
evidence plus positive failed-owner exit and free local locks. Trainer status
must report this attempt at step0, phase `failed`, no durable checkpoint and no
CUDA readiness. The run directory may contain only `.trainer.lock` and an empty
`checkpoints/` directory: any attempt log, identity, manifest, step0 weight or
other artifact refuses this path. The old directory is preserved under its
attempt's `control/.../failed-run-before-initialization`, with a durable receipt;
only then is a new assignment token issued. A lost SSH response leaves the old
owner reserved, and repeating the explicit command reuses the retirement receipt.
This cannot reset a run that performed any training. Launcher commands explicitly
use `--device cuda:0` within the UUID-selected visible device.

```bash
"$PY" "$SUP" --config "$CFG" recover E5 --pre-initialization \
  --evidence /path/to/initialization-failure-checks.json
```

The packaged profiles cover H20/E5 and L20/E0–E4 only. An available H20 GPU1
does not supply an unmeasured H20/E2 profile; missing profiles refuse launch.

No automatic cross-host migration is implemented: if the old host cannot be
inspected, keep that run reserved and restore access. Do not edit the live ledger
to bypass this invariant. A failed launch before any checkpoint stays reserved for diagnosis; it is not
automatically retried. Only the strict pre-initialization procedure above permits
a fresh assignment without a checkpoint.

Completion is an explicit acceptance operation. Record `complete:E3` evidence
with `passed:true`, derived `global_step`, `effective_samples_completed:230400`,
`validation_nodes:10`, and `hashes_passed/ledger_passed/backup_passed:true`, plus
the current attempt/profile/contract identity. The supervisor independently
checks the ten exact sample coordinates and320-frame completion metadata. Then
`release E3`. Release verifies exited owner, free locks and published checkpoint again.
Only then may the coordinator reuse that GPU for E2. Training exit0 alone is not
accepted as successful full training. Inspect all six runs before changing README
TODO1. Unit tests exercise fake transports and local processes; they do not prove
remote survival, CUDA compatibility, full-data throughput or physical success.


### V3 sample budgets, hardware profiles and analysis

The formal YAMLs now declare `training.budget_contract: suction20.pretrain.samples.v3`.
They are templates with pending profiles and cannot be launched unchanged.
For microbatch M and accumulation A, E=M×A; each run credits230400 samples,
warms up for11520 samples and validates every23040 samples (ten nodes).
All boundaries must be exactly divisible by E. The resolver derives total,
warmup and validation update counts and refuses conflicting explicit numbers.
For example E=8 gives28800 updates,1440 warmup and2880 between validations.
Peak LR stays0.001; it is not automatically scaled with batch. Only successful
optimizer updates credit samples. AMP-skipped attempts and abandoned/replayed
physical work remain separately recorded. Candidate warmup/main sampling uses
credited progress; physical sampler epoch/cursor is logged independently.

Select batch, accumulation, dense activation checkpointing/chunking, CPU threads
and memory budget from isolated training-mode forward/backward measurements on
the actual host/experiment. Keep complete geometry,2mm voxels,512 candidates and
AMP. Larger VRAM is capacity, not proof of faster throughput; select measured
useful throughput with room for optimizer state, complex frames and validation.
BatchNorm, random draw interleaving and masked microbatch reductions may differ
with batch, so hardware-specific settings do not imply numerical equivalence.

Save each selected resolved configuration as `.json` (parsed with the JSON parser,
preserving numeric scientific notation such as `1e-08`),
with `runtime_profile` containing `profile_id`, `training_version:"v3"`,
`selection_status:"selected"`, `host`, `experiment`, and measurement references.
Set positive `runtime.cpu_threads` and `runtime.gpu_memory_limit_gib` explicitly.
Register its absolute path and raw SHA256 in supervisor `profiles[host][experiment]`.
The supervisor binds the profile, sample budget and config hash to each assignment,
then copies the exact config into its attempt directory before launch.

A changed batch requires a separate version/run namespace, not changed-config
resume. Keep GPU `lock_root` shared between old and new supervisors. Stop the old
coordinator and positively verify old processes have exited before transition.
`retire E5 --superseded-by suction20.pretrain.samples.v3 --evidence reviewed.json`
preserves a paused old assignment as superseded after exit75, free locks,
checkpoint and final-backup checks; it never marks it complete or deletes state.
The evidence contains `passed:true` and the reviewed reason. This is different
from the zero-update pre-initialization retry. Retain old compute and logs;
initialize the new version independently with seed42.

Each attempt writes `updates.jsonl`, `events.jsonl` and immutable identity metadata.
Update rows include UTC/monotonic timestamps, physical/effective sample counters,
M/A/E, each frame and augmentation seed, point/voxel/seed counts, supervised-frame
counts, loss/LR/AMP/gradient norm and scoped timings/memory. CPU host intervals
are not isolated GPU kernel timings. CUDA peaks reset per accumulation/optimizer attempt; RSS peaks
cover process lifetime. Worker `resources.jsonl` adds device/process memory,
utilization, power/temperature/clocks, CPU core-equivalent usage, RSS/I/O and PSI;
unavailable metrics are null with reasons. Raw logs and event tails remain
append-only, including work later discarded by checkpoint rollback.

The checkpoint manifest carries analysis metadata for the exact committed ledger
byte prefixes and validation references. Export an analysis snapshot without
loading pickle or modifying the run:

```bash
python scripts/export_training_logs.py --run-dir /path/to/run \
  --resource-log /path/to/attempt/resources.jsonl \
  --output-dir /path/to/new-analysis-snapshot
```

The exporter verifies checkpoint and source hashes and writes update/frame/event/
resource/validation CSVs, a summary and a source manifest. Committed progress,
caller-designated live tails, uncommitted tails and abandoned attempts are labeled
separately. `--checkpoint-role predecessor` records an explicit rollback view;
`--live-attempt-id` labels a tail and does not prove its process is alive. Missing
metadata or hash failures are errors, not grounds to infer progress from log rows.
Persistent last-completed-validation metadata supports observers even after the
trainer clears its transient validation phase. ETA histories must be scoped by
contract, profile, hardware and attempt; profiling and old-version work do not
count as new training progress.
