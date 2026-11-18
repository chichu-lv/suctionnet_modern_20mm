# A/B 代码运行说明

当前分支已经实现单实例数据、v3预处理、同姿态物理teacher、对应路线网络、预训练/微调、checkpoint恢复、推理与native评价。A在 `route-a-3d-rgb`，B在 `route-b-2d-projected-rings`；每分支只包含自己的模型、测试与运行配置，共同模块同步维护。

**当前政策：直接使用source全部100个训练场景；完整source test选best，标记 `test_selected`；保留last。公开评价只参数化吸盘半径的方案待定，不作为开训门槛。** 目标1676 train /399 val /87 test保持原划分。

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
  --output "$CACHE/train" --split train --camera realsense

python scripts/build_instance_dataset.py \
  --dataset-root "$DATASET" --label-root "$LABEL_RELEASE" \
  --output "$CACHE/test" --split test --camera realsense
```

默认train遍历scene0000–0099、test遍历scene0100–0189，各含全部256帧。每个可见实例生成固定最多256候选池；训练只计算其前128个标签，test计算完整池。前缀来自位反转Morton的16×16格顺序及不重复格内回填，不使用row-major上半区前缀。

产物为 `manifest_train.jsonl` / `manifest_test.jsonl`、同名 `.summary.json`、`observations/*.npz` 和 `labels/*.npz`。标签含S、W、SW、两种self及 `Q_inst=S*W*(1-C_self)`。原始p/n、query顺序、缓存字节、预处理规范、完整对象资产与物理代码均绑定hash；已有缓存身份不同时要求新输出目录，不会静默覆盖。未知标签保持NaN和无效mask，不改写为0。

当前builder按请求范围生成缓存，不宣称实现了全量异步/按需缓存服务。可先用 `--scenes 0 --frames 0 --train-queries 64` 做真实资产成本检查；相应清单被标记partial，正式训练默认拒绝。无标签推理缓存可用 `--without-labels`，不需要label-root。少于100支持点或无法向实例仍进入test清单并记no_pose；无GT实例帧单独登记在summary中。

## 3. 直接100场景预训练

B分支：

```bash
python scripts/train_instance.py \
  --config configs/instance_b.yaml --stage pretrain \
  --source-manifest "$CACHE/train/manifest_train.jsonl" \
  --selection-manifest "$CACHE/test/manifest_test.jsonl" \
  --run-dir runs/instance_b/source --device cuda
```

A分支：

```bash
python scripts/train_instance.py \
  --config configs/instance_a.yaml --stage pretrain \
  --source-manifest "$CACHE/train/manifest_train.jsonl" \
  --selection-manifest "$CACHE/test/manifest_test.jsonl" \
  --backbone-root external/graspness_modern \
  --run-dir runs/instance_a/source --device cuda
```

默认FP32、每次1实例前向、累积16个有监督实例、8750次成功更新，即140000次成功监督实例暴露。先均匀scene、再均匀frame、再均匀实例；无有效监督的draw单独计数并重抽，不充当有效预算。AdamW、438步warmup、3e-4峰值、cosine至1e-5。RGB增强共享到整幅实例及support RGB，几何和标签不变。

每875步在完整90×256帧的可见实例上做native INST20评价。`best.pt`优先最大化 `InstTop1_INST20_at_0_4`，其次selected-Q的frame→scene宏平均，再取更早步；该test确实参与选模。`last.pt`保存进度和最终预算状态。评价器不会将缺文件、未知teacher或漏场景伪装成低质量。

测试用于选模后的同集成绩带选择偏差，checkpoint和报告明确保留 `independent_test_estimate=false`、`paper_comparison_ready=false`。last的 `checkpoint_selection_role=budget_endpoint_or_progress` 与best区别保存，但同run已观察test的事实不被删除。

## 4. 暂停、恢复与身份检查

首次命令加 `--stop-after 875` 可在成功更新边界暂停，不改变原8750步预算身份。然后用原命令同样配置/清单/设备加：

```text
--resume runs/instance_b/source/last.pt
```

A使用自己的run目录并保留相同 `--backbone-root`。完整恢复模型、Adam状态、阶段、学习率、成功预算、数据采样器、RGB增强RNG，以及Python/NumPy/Torch/CUDA RNG。配置、代码、资产字节或物理/预处理身份变化会拒绝恢复；不能改总预算后继续声称精确resume。A/B、旧E5和新架构不能混用。

训练与test对共同物体的dense几何和Seal候选资产hash交叉核对。微调初始化额外记录源checkpoint文件SHA、best/last角色、step、源选模政策和源身份；该来源随微调checkpoint和推理报告保存。

小规模链路验证需独立复制运行YAML，显式减少steps/queries/累积并传 `--allow-partial-manifests`。该标记进入checkpoint和汇总，不能将其结果登记为完整100场景训练。不要改正式YAML来伪装完成预算。

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

同观测、同目标实例的多次动作分组；保留原 `planned_pipeline_contact` 点、轴符号，仅单位化轴。最近支持距离大于5mm等无支持动作单独计数，不改写为失败。只有旧点云而没有原RGB-D/标定的记录须补齐资产，adapter不生成虚构图像。既有group划分和数据字节身份会被检查。

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

一个包含三种split的manifest可同时用于两个参数，loader分别取train和val。每步2个target观测与2个source回放观测，target各权重0.5、source各0.25；不再次除4。前200步仅头部，后800步解冻decoder/contact/heads，encoder全程冻结；A所有BN的统计与affine在微调中冻结。head的Adam动量跨阶段保留。每100步用399条val动作的AUROC选best，BCE和更早步打破平局；87条目标test保持独立使用。

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

也支持单个 `--observation instance.npz`，或原图 `--rgb`、`--depth`、`--mask`、`--intrinsics camera.json`、`--depth-scale-m`、`--gravity gx gy gz`。多实例整数mask以 `--instance-id`选择目标。A推理可以用 `--backbone-root`更换骨干目录位置，源码仍须通过固定版本验证。

默认source按q_inst、微调按task概率排序；显式 `--ranking mixed --alpha 0.5` 才使用混合分。排序分和概率分别保存。部署Top5执行方向敏感的10mm/15°双条件NMS；输出原始评分p/n和工具 `+Z=-n_out` 的接触SE(3)，不替换成邻近深度点，不暗中翻轴。空候选明确返回no_pose；读取/模型异常中止，不伪装空成功。

source评价保留少支持/无输出实例的失败分母，unknown teacher停止完整评价；target输出AUROC、分类AP、BCE、Brier及逐动作支持和实际执行验证标志。这两个入口都不计算公开论文AP，`public_evaluation.status`始终deferred。

## 7. 输出与验证范围

| 文件 | 内容 |
|---|---|
| `run_identity.json` | 配置、source/selection/target及代码身份、选择政策、初始化来源 |
| `training.jsonl` | 成功步、域损失、梯度范数、学习率、样本与跳过预算 |
| `evaluation_stepXXXXXX.json` | 选模指标、逐实例/动作记录、选择键和身份 |
| `best.pt` | 按相应source test或target val选出的完整状态 |
| `last.pt` | 最近成功边界/预算终点的完整状态 |
| `summary.json` | 已完成与计划预算、best摘要、策略和来源 |

运行 `python -m pytest -q`。新的 `tests/test_instance_*.py`覆盖几何、缓存、teacher、真实B网络前后向、A显式测试骨干、损失、参数冻结、精确resume和CLI。B集成测试以合成的官方文件格式执行teacher→训练更新→256-query选best→保存/恢复→推理/评价。它验证接线与数值一致性，不是正式数据准确率实验。

当前未执行完整真实数据预训练/微调、A真实SpConv GPU验收、GPU时延或ONNX/TensorRT导出。训练数据与训练权重不提交GitHub。详细证据见[MODEL_VALIDATION.md](../MODEL_VALIDATION.md)。
