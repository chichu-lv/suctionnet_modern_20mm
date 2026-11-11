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

`configs/pretrain_e0.yaml` 至 `pretrain_e5.yaml` 实现 57,600 次更新的正式配置，按90/10场景拆分，每训练场景每轮抽64视角；本次交付没有运行这些完整实验。是否运行完整训练需要单独确定预算。

`last.pt` 保存 model、optimizer、AMP、步数、独立采样器、Python/NumPy/Torch/CUDA RNG、配置、冻结物理定义及代码身份；`best.pt` 按开发验证指标选取。`--init-checkpoint` 仅初始化兼容权重，`--resume` 恢复完整训练状态。恢复时可以迁移目录和设备，不能静默更改训练语义。

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
