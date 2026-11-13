# 模型运行入口

实现包含只读源数据适配、E0–E5、预训练、两种目标域微调、实际姿态 R20 评价和无标签推理。CPU 合成测试不需要 spconv；生产稀疏主干需要 CUDA。运行结果及限制见 [验证记录](MODEL_VALIDATION.md)，性能变化与部署过程见 [性能优化全程总览](PERFORMANCE_HISTORY.md)。

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

这些配置分别上限 2 和 18 次成功 optimizer 更新；AMP 跳过不推进步数。E5 独立初始化，第二条 E5 命令恢复到总计 18 次，不额外训练 18 次。microbatch=1，梯度累计4；完整观测、2 mm 体素、512候选不因 smoke 而缩减。主干训练时 BN 正常更新。未配置性能 sidecar 时沿用同步读取；配置后以有序 ticket 的已消费游标恢复样本及增强种子。

`last.pt` 保存 model、optimizer、AMP、步数、独立采样器、Python/NumPy/Torch/CUDA RNG、配置、冻结物理定义及代码身份；`best.pt` 按开发验证指标选取。`--init-checkpoint` 仅初始化兼容权重，`--resume` 恢复完整训练状态。恢复时可以迁移目录和设备，不能静默更改训练语义。

## 正式公开数据预训练

`configs/pretrain_e0.yaml`至`pretrain_e5.yaml`是带pending profile的正式模板，不能原样启动。此前六组CD-010使用从[原CD-009 profile](../configs/runtime_profiles/cd009-v3/manifest.json)派生的新配置；现已全部停止，最终代码确定前只做优化测试。当前状态和停止前记录见[CD-010运行状态](performance/CD010_RUN_STATUS.md)。旧CD-009也停止，其[原策略和身份](CD009_PRETRAINING.md)保留作历史记录。

每组独立 seed42 初始化，90/10 场景划分，230400 有效样本、名义 40 轮；E=4 换算为 57600 成功更新。跳步与故障回放另计，不把 smoke 或旧版本权重接到正式预算。完整训练当前停止，尚未完成六组与各 10 次开发验证；最终有效batch确定后按同一样本预算派生更新数。

正式多机管理复用下方[supervisor入口](#cd-009-persistent-multi-host-pretraining)和[公共模板](implementation/examples/cd009-supervisor.json)，私有控制配置不提交到仓库。六组与协调器当前保持停止，短测不恢复完整训练，不重复首100步、恢复审计或全机检查。性能比较使用下方两机各一个实验的入口。

## CD-010 独立从头训练

从头初始化时使用 [新性能实现与运行方案](FROM_SCRATCH_PERFORMANCE.md)。`scripts/prepare_from_scratch.py` 从一个现有实验profile生成新的`train.json`、v4 `performance.json`和运行命令。按host选择资源：H20为64k/geometry8/affinity16，L20为64k/geometry4/affinity8；Torch线程均8、worker均4。H20的128k实测收益仅0.50%，默认已保留64k。继续split、FP32原生几何、segment实例广播及E0/E1/E3共享体素头。原始数据与冻结标签直接复用。

`--hardware-preset H20|L20|reference`可显式选择资源，其他数值CLI参数优先。`runtime.cpu_affinity_count`与`runtime.cpu_threads`已解耦：supervisor从实际CPU pool选取前者数量的核，trainer使用后者设置Torch线程。已有[真实短测](performance/CD010_FROM_SCRATCH_RESULTS.md)：H20/E5的B2A2更快，L20/E3支持geometry8/affinity16，E0支持worker6；旧16帧零更新benchmark不代表新v4路径。

[CPU/GPU几何重叠](GEOMETRY_GPU_OVERLAP.md)已实测支持depth2；随后[12档batch扫描](performance/CD010_GEOMETRY_MEMORY_RESULTS.md)已完成，9档完成、3档OOM。H20/E5采用microbatch4/depth2，已测B4A6/有效24为2.259727帧/s；L20/E3保留microbatch1/depth2，已测B1A12/有效12为3.038967帧/s。B2的0.17%差异不支持L20扩大batch，已有扫描不重复。下一步[接触头比较](CONTACT_HEAD_OPTIMIZATION.md)使用`benchmark_geometry_overlap.py --suite contact-head --hardware H20|L20 --out-dir NEW [--run]`，各机在上述设置上比较chunk32、zone32、zone64、zone128四档。每档48帧预热、240帧计时；24/12仍是性能比较设置，不是最终训练配方。独立进程依次执行，CUDA OOM记录后继续，其他错误停止。默认只生成，`--run`才执行短测，不恢复完整训练。

新生成的H20/L20/reference配置默认`runtime.gpu_memory_limit_gib:null`，不再继承旧profile的112/36 GiB上限。`--no-gpu-memory-limit`显式选择无人工上限；也可用互斥的`--gpu-memory-limit-gib N`指定有限上限，trainer与supervisor均支持。`--effective-batch-size`显式改变训练batch并按样本重新派生步数，不自动调整学习率；`--pose-chunk-size`单独控制接触头，新的定向比较使用32/64/128。`--contact-feature-gather-mode chunk|zone`选择分块内索引或按区域一次索引；默认`chunk`保持已测路径，`zone`候选尚无GPU收益数据。

精度默认FP16。`--precision bf16`显式选择密集BF16/稀疏FP32；固定spconv 2.3.6不支持全链BF16，该选择已用于新E2训练，但未证明提速。融合AdamW默认开启，局部MLP编译通过`--compile-dense-heads`选择；当前正式部署未启用compile。新配置和代码更新不会自动替换已运行的固定设置。

训练和恢复使用原生成的两个配置；独立评价/推理使用新run的 `resolved_config.json`。管理器支持 `selection_status="from_scratch"` 与v4 sidecar，使用独立的CD-010控制配置和账本即可。新运行不执行旧checkpoint四帧AMP/RNG对齐；原预算内的正常日志窗口提供实际吞吐判断。

下方P1/P2/P3迁移说明适用于原CD-009运行；其中voxel heads关闭、L20 16k和原生2线程等限制不适用于上述独立从头训练。

## 性能实现构建与既有运行迁移

**以下为已停止的CD-009历史记录：H20/E5使用64k + split + native，L20/E0、E1、E3使用16k + split，曾迁移续训并取得首25次正常更新窗口。** H20/E5与L20/E3的真实四帧配对当时通过；E0/E1不使用接触分组。L20原64k组合未通过旧梯度门槛，因此旧运行选择16k；该限制不适用于CD-010从头训练。[P3操作说明](PERFORMANCE_PHASE3.md)分别使用`configs/performance/cd009-p3-integrated.json`（H20）和`configs/performance/cd009-p3-l20-16k.json`（L20）；旧`cd009-p3.json`保留作几何版本记录。

本轮实际源 checkpoint 已确认使用 P2 `e73d62e`，保留全点 CPU 网格与坐标快路径；voxel heads 继续关闭。上述正常窗口的训练代码固定为 `f4f7f0d6de0bd3bd27107d8be09885fed700e48b`，后续 GitHub HEAD 与独立 `control_repo` 的 supervisor 代码身份另计，不替换训练身份。[最新实测记录](performance/P3_INTEGRATED_RESULTS.md) 给出源状态、四帧结果及正常窗口。历史报告中的1030/2462/1058步迁移和窗口保留原范围；其他接续仍读取实际 current checkpoint，批准的 original/P1/P2 可直接迁到目标 v3，无需中间迁移。

### 本机扩展与独立性能配置

在每个目标 Python 环境内构建小型 CPython 扩展；需要本机 C/C++ 编译器、Python headers 和 setuptools。无需 NumPy C headers、OpenMP 或 CUDA 编译器：

```bash
python setup.py build_ext --inplace
python -c 'import json; from models.geometry_ops import fps_backend_identity; print(json.dumps(fps_backend_identity("compiled"), indent=2))'
python -c 'import json; from models.spatial_index import spatial_index_backend_identity; print(json.dumps(spatial_index_backend_identity(), indent=2))'
```

实际 H20 部署使用 `/usr/bin/python3 setup.py build_ext --inplace`：训练 venv 没有 setuptools，系统 Python 已有；二者均为 Python 3.10.12，ABI 匹配。L20 构建采用相同方式，不因此自动启动迁移。使用这种方式前核对两解释器的版本和 `sysconfig.get_config_var("SOABI")`，构建后必须用**训练 venv 的 Python**执行上面的 `fps_backend_identity("compiled")` 导入检查。不要把 ABI 不匹配的系统扩展当作可用，也不需要为本次构建修改旧训练依赖。

历史 P1 H20 部署的 FPS binary SHA256 为 `9d1aa3a31aa1e855fccc9504e2bcf89c54d3e191aa231e582ede06d060971bf7`。临时 staging 与正式 checkout 的构建路径会影响带调试信息的二进制摘要，因此现场身份以正式构建的实际 hash 为准；C source、编译 flags、ABI 也同时记录，迁移后普通恢复必须匹配。

`setup.py` 使用 `-O3 -fno-fast-math -ffp-contract=off`。安装时扩展是 optional，但 `fps_backend:"compiled"` 启动时必须成功导入；不会静默换成 NumPy。`.so` 和 `build/` 不纳入源码摘要；实际扩展的 binary/source SHA256、compiler、flags、ABI 和 Python cache tag 单独进入运行身份。冻结部署前完成构建和文档更新，之后不能在 live checkout 修改文件或重建扩展后直接恢复旧身份。

当前额外 JSON sidecar 按主机选择 [H20 64k](../configs/performance/cd009-p3-integrated.json) 或 [L20 16k](../configs/performance/cd009-p3-l20-16k.json)，保持原正式 profile 字节和哈希。[P2 sidecar](../configs/performance/cd009-p2.json) 保留作源身份及历史参考。`--performance-config PATH` 在预训练入口加载所选文件，写入 `runtime.performance` 和原始字节 `runtime.performance_config_sha256`。v2/v3 保留下面的 v1 字段契约：

- 当前整合目标为 `schema_version:"suction20.performance.v3"`；既有 v1/v2 与原 v3 字段集仍兼容，不自动补写字段或改变 sidecar 原字节哈希。`fps_backend:"compiled"` 或显式参考 `"numpy"`。
- NumPy 参考路径中，`query_chunk_size:null` 显式选择逐候选共享几何查询；整数 1–64 选择分块查询。`query_workers` 为 1 或 4，仅分块路径使用。当前 native 配置要求 `null / 1`；原生候选线程独立固定为 `min(2, CPU affinity)`，不由这两个字段控制。
- `loader` 含 `num_workers:2`、`prefetch_factor:2`、`persistent_workers:true`、`pin_memory:true`、`multiprocessing_context:"spawn"`。`num_workers:0` 仅供显式 CPU/reference 路径；不引入另一套训练进度。
- `detailed_cpu_timing:false` 为常规值。未知键、错误类型和范围被拒绝。

P2新增 `spatial_index:{"backend":"uniform_grid_cpu","cell_size_m":0.02}`、`backbone_coordinate_fastpath:true`、`no_instance_voxel_heads:false`。空间网格和FPS的源码/编译器/flags/ABI/二进制摘要分别绑定到当前身份；原始全点几何和排序不变。不要启用未通过真实梯度验收的voxel路径。

整合 v3 使用 `contact_query_backend:"native_cpu"`、顶层 `dense_chunk_size_override`（H20 为 65536，L20 为 16384）及 `chunk_input_mode:"split"`。覆盖值只改变模型实际执行参数，不回写 `model.dense_chunk_size` 或 profile 字节；split 用一次 `torch.split` 连接同一输入的所有块，保留逐块前向及 checkpoint 顺序。训练 attempt 的 `execution.model_execution`、配对报告的 `identities.*.model_execution`、batch benchmark 的 `identity.model_execution` 会同时显示原 profile 值和实际值。历史 L20/E5 64k 零更新短测不是当前 L20 16k 配置的收益，也不覆盖随后 E3 checkpoint 的 64k 梯度失败。

主进程在 pinning 后建立 `CpuGeometry` 与独立 `SamplerTargetsCPU`，上传 GPU 消费字段；观测上下文不含标签、不进入 state_dict。worker 只准备 CPU 数据；独立 DataLoader generator 不消耗模型 RNG。已发出 ticket 与已完成 backward 的消费游标分开，checkpoint 只持久化消费游标；AMP 跳步保留实际帧消费，但不推进成功更新预算。

在私有 supervisor 配置增加所需 host/experiment 项，路径必须是对应主机的绝对路径，hash 是 sidecar 的原始字节 SHA256：

```json
{
  "performance_configs": {
    "H20": {
      "E5": {
        "config_path": "/srv/cd009-pretraining-v3/repo/configs/performance/cd009-p3-integrated.json",
        "config_sha256": "REPLACE_WITH_RAW_FILE_SHA256"
      }
    }
  }
}
```

上例为 H20/E5；L20 按实际迁移批次登记 16k sidecar。本轮 L20/E0、E1、E3 使用运行根目录 `performance/` 下的独立固定文件，SHA256 为 `39c8422e1c817d1af14e8ff160299669d008d8ca9fc56b782d22368b7e5b1f24`。缺省项沿用无性能 sidecar 路径。assignment 保存 `performance_config_path`、`performance_config_sha256` 和解析后的 `performance_settings`，worker 校验后复制为其 control attempt 下的 `performance-config.json`，实际 CLI 使用该快照。原 `profiles[host][experiment]` 和 `config_sha256` 保持不变。新建 run 使用同一 supervisor 入口；尚未切换的既有 run 按下述流程迁移，已完成者继续原训练。

### 从批准的 original/P1/P2 迁移并正常恢复

转换器接受已核实的 original revision `98862f4d6a133c99afa082a0755dbb888b41b38d` 与源码摘要 `d634627b180cee10be2c42949c8d10cec21ac49424aa734f6b5acf2c5f2c30c8`，以及批准的 P1 revision `996421c664a5a0b84c308aa4ca47c31eaff4035b`、源码摘要 `23bde51eb6b8f091a15b2aeb336db8a7085b150f304076af44b3a1d83ba7ee20` 与原 P1 sidecar `08d2247c…`；P2 批准身份见 [P3 接续说明](PERFORMANCE_PHASE3.md)。允许 original→v1/v2/v3、P1→v2/v3、P2→v3，保留历史 receipt 的合法迁移关系。P1 来源原有迁移链继续沿用；源身份按实际 checkpoint 读取。具名 dense chunk 执行覆盖位于 sidecar，不改变冻结 profile、batch、loss、预算、数据、依赖或数学设置。

1. 用 supervisor `pause EXPERIMENT --reason ...` 在完整更新边界暂停同一 checkout/venv 的全部任务。旧 worker/trainer 必须已正向确认退出，状态为 `paused`、exit75，完成原 final backup，且没有 pending validation；若已有 pending，先用旧版本完成原定验证。
2. 冻结并验证目标代码与本机扩展，在拥有该 run 的主机执行转换。`--control-dir` 指向刚退出的旧 attempt，源 hash 取已核验 manifest 的 current。数据与主干覆盖参数必须与原 supervisor 一致。

```bash
python scripts/migrate_performance_checkpoint.py \
  --config configs/runtime_profiles/cd009-v3/H20-E5.json \
  --performance-config configs/performance/cd009-p3-integrated.json \
  --run-dir /path/to/runs/E5_seed42 \
  --control-dir /path/to/control/attempts/PAUSED_TOKEN \
  --expected-source-checkpoint-sha256 SOURCE_CURRENT_SHA256 \
  --source-root /path/to/full-20261110/input_raw \
  --label-root /path/to/full-20261110/output \
  --backbone-root /path/to/graspness_modern
```

上例选择 H20/E5；L20 使用自己的原 profile 和固定的 `cd009-p3-l20-16k.json` 路径。`--expected-source-sha256` 是相同参数的短名称。工具取得现有 run lock，核对退出身份、current blob/hash、源/目标 identity、语义与预算；仅允许 `code_revision`、`worktree_sha256` 和 `performance` 变化。它不构造 optimizer、不前向/反向、不增加更新或样本。模型/BN、Adam、AMP、RNG、sampler、LR 进度、best、验证和原日志前缀都保持。

归档路径为 `RUN/migrations/performance-UUID/`：`source.pt` 是源 blob 的硬链接，另外保存 `source-manifest.json`、`performance-config.json` 和 `receipt.json`。receipt 记录源/目标 identity、源成功步/attempt、实际允许差异和三个归档文件的大小/hash。新 current 的 `runtime_state` 与 manifest `analysis` 都引用 receipt 相对路径及 SHA256；后续保存继续透传。根目录 `runtime.json`/`resolved_config.json` 保留 run 最初版本，当前身份看 current、receipt 与新 attempt。

3. 更新私有 supervisor 的 sidecar 映射后，仍执行普通 `resume E5`（其他实验同理）。supervisor 仅在检查到有效迁移 receipt、目标 sidecar hash 相同且原源 checkpoint 的 final backup 仍有效时允许原 assignment 切换；新 attempt 沿用原设备、profile 和学习状态。普通 resume 不提供忽略身份检查的开关。

现有备份会递归验证并复制当前 receipt 及其 P1/P2 历史链中每份 receipt 引用的三份归档文件，还包括 attempt 的性能配置快照；保留源 `source.pt` 的完整内容，不能只复制 receipt。切换检查中的 `performance_source_backup_verified` 核对原源备份；新 current 的备份在新 worker 首次观察到 durable checkpoint 后立即进入原串行备份队列。继续使用现有 checksum、pinning 和对象去重。共享备份内的符号链接指向备份根内的对象；实际恢复须解引用（例如 `cp -L`），再在恢复目录执行严格迁移校验。接续沿用 [P3 操作说明](PERFORMANCE_PHASE3.md) 的三项正常检查，已完成的配对和窗口直接引用已有记录。

### 真实诊断结果的正确使用

以下380/833/441及相应窗口为历史P1记录。P2报告中的四帧、1030/2462/1058迁移和30步历史窗口见 [P2报告](PERFORMANCE_PHASE2.md)；原AMP容差未放宽，voxel失败保留。当前应用状态以 [P3 配对与正常窗口记录](performance/P3_INTEGRATED_RESULTS.md) 为准，跨阶段比较见 [全程总览](PERFORMANCE_HISTORY.md)。

`scripts/benchmark_training_pipeline.py` 的独立旧/新进程结果保持失败，旧代码自身重复运行也出现骨干数值变化。随后 `scripts/benchmark_paired_pipeline.py` 从同一第 380 步 checkpoint、相同四帧与增强种子出发，共享每帧一次真实骨干输出，再比较两条实际下游路径及其完整骨干 VJP；四帧零失败，容差保持 `atol=2e-4, rtol=2e-3`。完整离散几何、描述符和 RNG 精确核对；该配对结果不覆盖原失败，也不证明独立 CUDA 运行逐位相同。

两个脚本都不执行 optimizer 更新。配对脚本接受与独立脚本相同的路径/config/device 参数，须使用新的输出目录；具体命令和检查范围见 [性能方案 10.4 / 11.4](PERFORMANCE_OPTIMIZATION.md#104-当前实现入口与复现方法)。这类诊断保留临时 tensor 证据，不写正式 run，也不替代正常训练日志的吞吐统计。H20/E5 已完成一次正常第 380 步迁移与续训，后续文档发布不部署到当前训练代码。

实际迁移源步为 H20/E5 380、L20/E0 833、L20/E3 441，源状态核对分别为 874、642、770 个张量完全一致。E3 的首次新 worker 因 GPU0 被占用在 trainer 启动前停止；经已有恢复流程切到同主机、同型号的授权空闲 GPU3，并继续原 profile、样本预算与学习状态。未停止占用进程；原失败 worker 证据保留。E3 的前后吞吐还受到 GPU/CPU affinity 放置变化影响，不能把全部差异归因于代码。

E5/E0/E3 正常窗口已观察到更短的更新耗时和约 1–2 毫秒/更新的主线程数据等待，数字与比较边界见 [正常窗口表](PERFORMANCE_OPTIMIZATION.md#115-正常训练窗口) 和 [公开证据摘要](performance/P1_REAL_RESULTS.json)。这是不同自然帧的训练日志观察；配对脚本用于验证数值与梯度，不用于声称吞吐倍率。

### 50 次成功更新的计时窗口

每次显式迁移以 receipt 中源成功步 S 为固定起点。训练 attempt 在更新前 `global_step < S+50` 时临时开启详细几何计时，覆盖成功更新 S+1…S+50；AMP skip 不耗用成功步配额，再次正常恢复也不会重置窗口。sidecar 默认 `false` 不改字节，到边界后自动关闭；若显式配置 `true` 则持续开启。

`performance_diagnostic_window` event、status/update 的 `performance_diagnostics` 记录策略、active 和边界。常规聚合计时、查询规模计数和完整样本账本持续保留。`production_timing` 是 worker 生产时间，`loader_wait_s` 是主进程实际等待，两者不能相加当成墙钟；`resources.jsonl` 同时观察主进程和 worker 资源。50 次更新属于原预算，无额外验证或验收训练。真实吞吐以正常日志中 20–50 次成功更新及含保存的窗口墙钟报告。

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
The independent [CD-010 fresh-start path](FROM_SCRATCH_PERFORMANCE.md) also accepts
an explicitly generated `from_scratch` profile paired with a v4 performance
sidecar. It uses a new run/ledger namespace and does not inherit a measured-speed
claim from the source profile.
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
disk. RSS remains limited to64GiB; numeric GPU allocator and whole-process memory
budgets come from the selected profile. Explicit `gpu_memory_limit_gib:null`
removes these artificial GPU caps for fresh-run configurations. Disk free
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
Set positive `runtime.cpu_threads` explicitly. Set `runtime.gpu_memory_limit_gib`
to a positive GiB cap or explicit JSON `null` for no artificial GPU memory limit;
new fresh-run presets default to `null`, while existing finite profiles stay unchanged.
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
