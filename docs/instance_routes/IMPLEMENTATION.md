# A/B 代码运行说明

当前分支已经实现单实例数据、v3预处理、同姿态物理teacher、对应路线网络、预训练/微调、checkpoint恢复、推理与native评价。A在 `route-a-3d-rgb`，B在 `route-b-2d-projected-rings`；每分支只包含自己的模型、测试与运行配置，共同模块同步维护。

**当前政策：直接使用source全部100个训练场景；每875步在90个test场景各固定8帧上选best，标记 `test_selected` 和 `fixed_test_subset`；保留last。训练后用独立CLI对固定best评价完整test，不重新选模。公开评价只参数化吸盘半径的方案待定，不作为开训门槛。** 目标1676 train /399 val /87 test保持原划分。

## 1. 安装与输入资产

B只需要PyTorch和数据依赖；A另外需要CUDA、spconv及固定外部骨干。以下在仓库根目录执行；路径均替换为实际挂载位置。

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[model]'
python -m pip install pytest
```

A分支追加：

```bash
python -m pip install spconv-cu120==2.3.6
python scripts/setup_backbone.py --output external/graspness_modern
```

CPU测试可安装官方CPU版torch2.5.1；B可用 `--device cpu` 运行完整PyTorch路径。A默认构造使用真实SpConv，缺少GPU/骨干不会自动替换为测试MLP。当前本地验证环境是Python3.12.14、PyTorch2.5.1+cpu、NumPy2.3.5；正式GPU环境沿用[既有运行指南](../MODEL_USAGE.md)并记录实际版本，不将CPU结果作为GPU验收。

需要两类source资产：

- `DATASET`：官方 `scenes/scene_XXXX/<camera>/rgb,depth,label,meta`、内外参、`camera_poses.npy`、`cam0_wrt_table.npy`，以及 `dense_point_clouds/`。
- `LABEL_RELEASE`：已有冻结R20发布目录，包含 `config_frozen.json` 和 `seal_r20/poisson/`。新teacher复用冻结Seal归一化、Wrench与目标包络定义，在精确新query上重算，不直接复制旧gt_B。

图像、分割、内参与深度必须位于相同校正网格。默认source深度范围0.2–1.2m，可通过builder显式修改并保存身份。若一帧出现同类物体的多个实例，官方类别掩码不能区分行号，须提供 `--instance-mask-root` 对应实例行掩码；代码明确拒绝静默混并。

## 2. 构建共同source缓存

```bash
DATASET=/data/SuctionNet
LABEL_RELEASE=/data/suctionnet-r20-frozen
CACHE=/data/instance-v3

python scripts/build_instance_dataset.py \
  --dataset-root "$DATASET" --label-root "$LABEL_RELEASE" \
  --output "$CACHE/train" --split train --camera realsense --workers 4

python scripts/build_instance_dataset.py \
  --dataset-root "$DATASET" --label-root "$LABEL_RELEASE" \
  --output "$CACHE/selection" --split test --camera realsense \
  --frames 0,32,64,96,128,160,192,224 --workers 4 \
  > "$CACHE/selection_build.json"
SELECTION_MANIFEST=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["manifest"])' "$CACHE/selection_build.json")
```

默认train遍历scene0000–0099全部256帧。上例selection遍历scene0100–0189各固定8帧，共720帧；每帧全部可见实例进入选择集，包括少支持/no-pose实例。每实例生成固定最多256候选池，训练只计算前128个标签，selection和完整test都计算完整池。query前缀来自位反转Morton的16×16格顺序及不重复格内回填，不使用row-major上半区前缀。

固定选择帧来自8位bit-reversal前缀排序：8帧为 `0,32,64,96,128,160,192,224`；16帧为 `0,16,...,240`，包含8帧集合；256是全量。`pretrain.selection_frames_per_scene`决定数量。训练可接收完整test manifest再自动筛选，也可接收上例仅720帧的清单；正式运行要求90场景所有指定帧有实例记录或明确GT-empty声明，不需 `--allow-partial-manifests`。未选择帧的NPZ不会被评价加载，缺少指定帧则报错，不换帧补齐。

完整test缓存可在训练后按以下命令准备，独立评价时使用其根清单：

```bash
python scripts/build_instance_dataset.py \
  --dataset-root "$DATASET" --label-root "$LABEL_RELEASE" \
  --output "$CACHE/test" --split test --camera realsense --workers 4
```

产物为 `manifest_train.jsonl` / `manifest_test.jsonl`、同名 `.summary.json`、观测NPZ和标签NPZ。单进程缓存位于 `observations/`、`labels/`；多进程各自在 `shards/part_XXXX/` 写入，合并清单以相对路径引用这些文件。标签含S、W、SW、两种self及 `Q_inst=S*W*(1-C_self)`。当前manifest为v4，读取器同时接受旧v3；旧校验和字段直接忽略，不计算文件/数组校验和，不对文件内容作篡改检测保证。每份NPZ正常加载一次，继续检查结构、有限性、query原像素与p/n精确对应、标签公式及已知的小型物理配置。未知标签保持NaN和无效mask，不改写为0。

`--workers 4` 将scene分配给4个独立进程；每进程构建自己的teacher和缓存，然后安全合并。默认 `--workers 1` 串行，多进程要求显式 `--split train` 或 `test`。不传 `--scenes/--frames` 时保留上述根目录manifest；只要显式传入其中一个参数，就按相机、split、实际scene/frame范围建立独立目录，观测、标签和清单都放入其中。即使显式列出完整范围，也遵循分片规则。CLI输出JSON的 `manifest` 字段给出实际绝对路径。

以下两个构建命令可在同一输出根目录串行或分别启动并行执行，随后合并完整训练集：

```bash
python scripts/build_instance_dataset.py --dataset-root "$DATASET" --label-root "$LABEL_RELEASE" --output "$CACHE/train" --split train --camera realsense --scenes 0-49
python scripts/build_instance_dataset.py --dataset-root "$DATASET" --label-root "$LABEL_RELEASE" --output "$CACHE/train" --split train --camera realsense --scenes 50-99
python scripts/merge_instance_manifests.py \
  --manifests "$CACHE/train/shards/realsense/train/scenes_0000-0049/frames_0000-0255/manifest_train.jsonl" \
              "$CACHE/train/shards/realsense/train/scenes_0050-0099/frames_0000-0255/manifest_train.jsonl" \
  --output "$CACHE/train" --split train
```

合并器重新定位NPZ相对路径，拒绝重复帧/实例、跨split或相机、缺失文件、漏报帧、互相矛盾的空GT帧及已知物理配置冲突；默认要求该split的全部scene×256帧。无实例帧须由 `empty_gt_frames` 明确覆盖，合并summary保留这些帧。只有成本/接线检查才使用 `--allow-partial` 生成不完整合并，训练时另需 `--allow-partial-manifests`；它们不会将不完整数据伪装成正式完整集。

同一分片或同一最终发布目录由非阻塞文件锁保护，第二个writer在写入前报错；不同分片独立写入。manifest/summary通过唯一临时文件原子替换。合并器拒绝用遗漏已有帧的结果覆盖已发布清单，因此串行后续分片也不会缩减既有覆盖。旧manifest仍可读取和合并，不自动搬迁旧缓存；分片路径直接编码ID范围，不计算内容摘要。

当前builder按请求范围生成缓存，不宣称实现了全量异步/按需缓存服务。可先用 `--scenes 0 --frames 0 --train-queries 64` 做真实资产成本检查；相应清单被标记partial，正式训练默认拒绝。无标签推理缓存可用 `--without-labels`，不需要label-root。`--min-support-points` 默认100，写入观测/清单/summary；A的训练与评价实际读取 `model.min_support_points`（默认100）；B当前未暴露该构造参数，helper使用默认100，缓存中的 `trainable` 只是构建时诊断。少支持或无法向实例仍进入test清单并记no_pose；无GT实例帧单独登记在summary中。

## 3. 直接100场景预训练

B分支：

```bash
python scripts/train_instance.py \
  --config configs/instance_b.yaml --stage pretrain \
  --source-manifest "$CACHE/train/manifest_train.jsonl" \
  --selection-manifest "$SELECTION_MANIFEST" \
  --run-dir runs/instance_b/source --device cuda
```

A分支：

```bash
python scripts/train_instance.py \
  --config configs/instance_a.yaml --stage pretrain \
  --source-manifest "$CACHE/train/manifest_train.jsonl" \
  --selection-manifest "$SELECTION_MANIFEST" \
  --backbone-root external/graspness_modern \
  --run-dir runs/instance_a/source --device cuda
```

默认FP32、每次1实例前向、累积16个有监督实例、8750次成功更新，即140000次成功监督实例暴露。先均匀scene、再均匀frame、再均匀实例；无有效监督的draw单独计数并重抽，不充当有效预算。AdamW、438步warmup、3e-4峰值、cosine至1e-5。RGB增强共享到整幅实例及support RGB，几何和标签不变。

默认每875成功步在同一90×8帧固定子集做native INST20评价，8750步共10次；每875步保存 `last.pt` 进度。`best.pt`优先最大化子集的 `InstTop1_INST20_at_0_4`，其次selected-Q的frame→scene宏平均，再取更早步。所选帧的全部可见实例、每实例完整最多256个query和指标公式不变。小子集有抽样波动，best不保证是完整test上的最优checkpoint；不根据模型得分挑帧，也不在每轮重新随机换子集。

YAML默认 `selection_frames_per_scene: 8`、`evaluate_every: 875`。可将帧数改为16以提高覆盖，改为256恢复全量选模；旧配置缺少该字段时仍按256解释，不隐式改变旧运行。也支持 `evaluation_steps` 显式节点，但不能与 `evaluate_every` 同时指定。原计划最终步始终使用同一选模集合评价，`--stop-after`不新增节点。缩短steps时若使用显式节点，须同步调整。运行政策记录scope、固定帧ID、采样规则和节点；checkpoint identity另外保存实际有序实例ID，恢复时不能换集合或顺序。

周期评价JSON仅保存指标、`record_details` 与耗时，不把全部实例明细重复缩进写入。默认 `evaluation_records: best_and_final` 在单次遍历时流式写临时 `.records.jsonl.gz`，只有更新best、重复评价当前best步或到达最终步时保留，其余删除临时明细；这减少常驻内存和长期磁盘占用，仍会有一次临时压缩流写入。可选 `all` 保留所有明细，`none` 完全不写明细。`timing.wall_seconds` 覆盖选择集前向、指标和明细处理，不含之后的checkpoint保存；`timing.observations` 记录本次实例数。

每轮选择帧数由23040降为720，即1/32；10轮合计7200帧，相当于0.3125轮完整test。相对原10轮全量选模，帧遍历量减少96.875%；相对上一版两轮全量选模减少84.375%。帧内实例数和有效候选数并非恒定，因此这些是帧预算比例，不是已测墙钟加速比。选择16帧/scene时单轮1440帧、为全量1/16。训练后如执行一次完整test报告，另加23040帧，不参与best更新。

测试用于选模后的同集成绩带选择偏差，checkpoint和报告明确保留 `independent_test_estimate=false`、`paper_comparison_ready=false`。last的 `checkpoint_selection_role=budget_endpoint_or_progress` 与best区别保存，但同run已观察test的事实不被删除。

## 4. 暂停、恢复与身份检查

首次命令加 `--stop-after 875` 可在成功更新边界暂停，不改变原8750步预算身份。然后用原命令同样配置/清单/设备加：

```text
--resume runs/instance_b/source/last.pt
```

A使用自己的run目录并保留相同 `--backbone-root`。完整恢复模型、Adam状态、阶段、学习率、成功预算、数据采样器、RGB增强RNG，以及Python/NumPy/Torch/CUDA RNG。当前checkpoint按完整配置、route/stage、运行合同、已知物理/预处理定义、清单路径/数量与实际采样ID序列检查恢复兼容性；不读取文件字节来计算身份。不能改总预算后继续声称精确resume。A/B、旧E5和新架构不能混用。

微调 `--init-checkpoint` 是新的运行：A允许更换 `model.backbone_root` 和 `model.query_chunk_size`，B允许更换 `model.query_chunk`。其余已知或未知model字段均参与严格语义比较；省略的已知构造默认值先补齐，网络权重仍以 `strict=True` 加载。先检查源checkpoint保存配置和 `config_definition` 自洽，再使用新运行的执行参数；来源记录保留源/目标执行设置。普通 `--resume` 继续比较完整配置，包含路径和分块设置。

训练、推理、评价三条CLI统一提供 `--query-chunk-size N`，分别映射到A的 `query_chunk_size` 和B的 `query_chunk`；不传时使用配置原值。A的 `backbone_root: null` 是待提供的路径，须通过YAML或 `--backbone-root` 指定，训练CLI在加载大清单前就检查这一点。分块影响临时内存与执行形状，不能据此承诺跨GPU分块的浮点结果逐位相同。

新teacher身份记录 `teacher_spec` 与完整小型 `physics` 字典；对象dense/Seal候选路径只记录出处，不证明文件内容版本相同。旧缓存没有结构化physics时记 `physics=null`，不使用旧校验和推断物理配置，也不因此要求全量重算teacher。双方physics都已知时直接比较；缺来源与缺标签是不同状态，p/n和标签数值检查始终执行。微调初始化记录源checkpoint路径、大小、best/last角色、step、选模政策和可读定义；该来源随checkpoint与推理报告保存。旧checkpoint的校验和不被验证，恢复仍须满足模型结构、配置、已有物理/预处理、样本数量与状态的兼容规则。旧调度器若只有不透明身份、没有实际语义ID序列，不能证明同长度清单没有被重排；新版保存实际序列，可直接核对顺序。

小规模链路验证需独立复制运行YAML，显式减少steps/queries/累积并传 `--allow-partial-manifests`。该标记进入checkpoint和汇总，不能将其结果登记为完整100场景训练。不要改正式YAML来伪装完成预算。

固定test子集是正式选模配置，和任意缺帧smoke不同。子集checkpoint的scope、帧配置、覆盖和实例序列必须自洽；旧runtime身份也不能绕过子集的精确恢复比较。完整test评价报告分别记录 `evaluation_scope=complete_manifest` 与 `source_pretrain_selection_scope=fixed_test_subset`，微调来源继续保留源子集选择身份。

## 5. 目标域微调

`InstanceAttemptDataset`读取现有 `suction20.target.v1 / attempt_success` JSONL，并要求原始organized RGB-D。每行需有 `sample_id/group_id/split`、`success_definition`、`observation`、`attempt`；整份清单的非空成功定义必须唯一，沿用原采集清单的真实定义。下列是字段示例，数值、路径及成功定义占位符仅说明格式：

```json
{
  "schema_version": "suction20.target.v1",
  "label_kind": "attempt_success",
  "sample_id": "action_000001",
  "group_id": "capture_session_01",
  "split": "train",
  "success_definition": "replace_with_existing_manifest_definition",
  "actual_execution_verified": false,
  "observation": {
    "format": "rgbd",
    "rgb_path": "/data/target/rgb/000001.png",
    "depth_path": "/data/target/depth/000001.png",
    "instance_mask_path": "/data/target/masks/000001.png",
    "depth_scale_m": 0.001,
    "intrinsics": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
    "g_cam": [0, 1, 0]
  },
  "attempt": {
    "instance_id": 1,
    "point_cam_m": [0.0, 0.0, 0.6],
    "normal_out_cam": [0.0, 0.0, -1.0],
    "success": 1,
    "label_valid": true
  }
}
```

同观测、同目标实例的多次动作分组；保留原 `planned_pipeline_contact` 点、轴符号，仅单位化轴。在实例分组之前，规范化depth/RGB路径及显式capture UID须共同遵守同一split/group，同frame的不同instance也不能跨集合；训练器还核对分开的target/selection清单的已知共同来源及成功定义。中心支持使用共同常量 `distance <= 0.005 + BOUNDARY_ROUNDOFF_M`，`BOUNDARY_ROUNDOFF_M=1e-12 m` 只补偿float64边界舍入。无支持动作单独计数，不改写为失败。只有旧点云而没有原RGB-D/标定的记录须补齐资产，adapter不生成虚构图像。不做资产字节校验，也不宣称发现了不同路径、无共同ID的内容重复。

B示例；A换配置和run目录，并提供对应 `--backbone-root`：

```bash
python scripts/train_instance.py \
  --config configs/instance_b.yaml --stage finetune \
  --source-manifest "$CACHE/train/manifest_train.jsonl" \
  --target-manifest /data/target/attempts.jsonl \
  --selection-manifest /data/target/attempts.jsonl \
  --init-checkpoint runs/instance_b/source/best.pt \
  --run-dir runs/instance_b/finetune --device cuda
```

一个包含三种split的manifest可同时用于两个参数，loader分别取train和val。每步2个target观测与2个source回放观测，target各权重0.5、source各0.25；不再次除4。前200步仅头部，后800步解冻decoder/contact/heads，encoder全程冻结；A所有BN的统计与affine在微调中冻结。评价器进入前逐模块保存 `.training`，退出或抛错后逐模块恢复原标志，不用根模块 `train(True)` 递归覆盖已冻结的BN。head的Adam动量跨阶段保留。每100步用399条val动作的AUROC选best，BCE和更早步打破平局；87条目标test保持独立使用。

源任务头在预训练时没有成败监督，不能用它做成功概率排序。首版目标训练执行精确单动作BCE；20mm标签扩散与小姿态一致性正则均未启用。

## 6. 推理与native评价

从缓存清单无标签推理，teacher文件不被读取：

```bash
python scripts/infer_instance.py \
  --checkpoint runs/instance_b/source/best.pt \
  --manifest "$CACHE/test/manifest_test.jsonl" --split test \
  --output runs/instance_b/source/predictions.json --device cuda

python scripts/evaluate_instance.py \
  --checkpoint runs/instance_b/source/best.pt \
  --manifest "$CACHE/test/manifest_test.jsonl" --domain source --split test \
  --output runs/instance_b/source/native_test.json --device cuda

python scripts/evaluate_instance.py \
  --checkpoint runs/instance_b/finetune/best.pt \
  --manifest /data/target/attempts.jsonl --domain target --split test \
  --output runs/instance_b/finetune/target_test.json --device cuda
```

也支持单个 `--observation instance.npz`，或原图 `--rgb`、`--depth`、`--mask`、`--intrinsics camera.json`、`--depth-scale-m`、`--gravity gx gy gz`。多实例整数mask以 `--instance-id`选择目标。A推理可以用 `--backbone-root`更换骨干目录位置，原有固定Git blob SHA-1源码核验仍保留；该既有骨干机制不作为数据/标签校验和的替代方案。

默认source按q_inst、微调按task概率排序；显式 `--ranking mixed --alpha 0.5` 才使用混合分。排序分和概率分别保存。部署Top5执行方向敏感的10mm/15°双条件NMS；输出原始评分p/n和工具 `+Z=-n_out` 的接触SE(3)，不替换成邻近深度点，不暗中翻轴。空候选明确返回no_pose；读取/模型异常中止，不伪装空成功。

`infer_instance` 先读取 `query_valid`，只校验和单位化有效行，并只对这些行调用评分；无效槽位可含零/NaN占位，输出以原始query/source ID记录拒绝原因，不将占位数值写入JSON。有效行的非有限坐标或零法向仍是错误。支撑点少于模型门槛时，即使没有有效query，也统一报告 `insufficient_support`；支撑充分但全无效时报告 `no_eligible_queries`。正常或异常退出都恢复每个子模块原始training标志，保持微调BN冻结。

source评价保留少支持/无输出实例的失败分母；所需teacher标签未求值时停止完整评价，旧physics来源未知则单独报告；target输出AUROC、分类AP、BCE、Brier及逐动作支持和实际执行验证标志。这两个入口都不计算公开论文AP，`public_evaluation.status`始终deferred。

独立 `evaluate_instance.py` 始终按传入manifest评价并检查完整source split，不会因checkpoint用8帧选模而自动缩小报告集合。给它完整test manifest会评价全部90×256帧；只给选择子集会明确拒绝作为完整test报告。它不保存或重新选择best，完整报告继续披露test子集参与了选模。

## 7. 输出与验证范围

| 文件 | 内容 |
|---|---|
| `run_identity.json` | 完整配置、source/selection/target定义、运行合同版本、选择政策、初始化出处 |
| `training.jsonl` | 成功步、域损失、梯度范数、学习率、样本与跳过预算 |
| `evaluation_stepXXXXXX.json` | 选模指标、选择键/节点/范围、`timing` 与 `record_details`，不内嵌records数组 |
| `evaluation_stepXXXXXX.records.jsonl.gz` | 默认仅本次更新best或最终步保留的逐实例/动作明细 |
| `best.pt` | 按相应source test或target val选出的完整状态 |
| `last.pt` | 最近成功边界/预算终点的完整状态 |
| `summary.json` | 已完成与计划预算、best摘要、策略和来源 |

运行 `python -m pytest -q`。新的 `tests/test_instance_*.py`覆盖几何、缓存、teacher、真实B网络前后向、A显式测试骨干、损失、参数冻结、精确resume和CLI。B集成测试以合成的官方文件格式执行teacher→训练更新→256-query选best→保存/恢复→推理/评价。它验证接线与数值一致性，不是正式数据准确率实验。

当前未执行完整真实数据预训练/微调、A真实SpConv GPU验收、GPU时延或ONNX/TensorRT导出。训练数据与训练权重不提交GitHub。详细证据见[MODEL_VALIDATION.md](../MODEL_VALIDATION.md)。
