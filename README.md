# SuctionNet Modern 20mm

面向 **半径 20 mm 单吸盘**的 SuctionNet 数据制作、公开数据预训练与少量自有场景微调项目。

当前已纳入的数据流程：SuctionNet 10 mm 标签对照（G0）→ 20 mm Seal 重标注（G1）→ 场景 Wrench/Collision → 每帧 RGB-D 点云与吸取候选映射 → GT-A/GT-B → 数据质量检查（G2）。

**当前状态（2026-11-12）：** 数据制作、只读适配、E0–E5 模型、预训练、两种微调、推理及 R20 评价已实现并完成代码复核。用户已用旧 `dc4ad6b00535` 方法完成训练数据制作，现有标签通过 adapter 直接复用。CD-009 v3 完整预训练正在分阶段运行；六组训练、完整开发验证及目标域效果实验尚未完成。实际部署代码身份、六份固定配置和策略见 [当前预训练](docs/CD009_PRETRAINING.md)，历史验证范围和数值见 [验证记录](docs/MODEL_VALIDATION.md)。

## Done（已完成）

- [x] 数据工程 G0–G2 的生成与检查代码；已有旧数据作为训练起点，保留冻结物理参数和 S/W/C/A/B 标签。
- [x] 模型、预训练、微调与实验的 [详细规格和设计演进图](docs/implementation/README.md)。
- [x] 只读旧数据 adapter：恢复原候选轴、重建观测法向、区分质量与轴有效掩码，无需重制标签。
- [x] E0–E5 模型与配置：稀疏主干、实例融合、五区接触特征、姿态与质量头及辅助监督。
- [x] 预训练、checkpoint 保存与恢复、无标签推理、NMS 和按实际姿态重算的 R20 评价。
- [x] `dense_proxy` / `attempt_success` 微调：200+800 步阶段、源域回放、固定查询验证及独立成功头。
- [x] CPU 测试复跑 **129 passed / 1 skipped**，梯度 checkpoint 一致性和两种微调跨阶段恢复复核；跳过项需要真实 release 路径。
- [x] 已有 H20 完整观测 smoke：**E1 2 次 + E5 18 次成功更新**，含重启与独立推理检查；32 个原候选的物理评价一致性检查通过。这里只记录链路验证，完整训练效果见下方 TODO。

- [x] v3 样本预算、硬件 profile、持久 supervisor、checkpoint/日志恢复与 CSV 分析导出；17 次零更新 profile 测量、六份 [正式配置及 SHA256](configs/runtime_profiles/cd009-v3/manifest.json) 已记录。

## TODO（待完成）

- [ ] **进行中：** 按 [实际 v3 预训练策略](docs/CD009_PRETRAINING.md) 完成 E0–E5 六组独立训练，每组 230,400 有效训练帧（名义 40 轮）；当前实测选择 E=4，派生为 57,600 次成功更新。全部六组与各 10 次完整验证验收后才完成。
- [ ] 按 [实验方案](docs/implementation/EXPERIMENTS.md) 比较验证指标与计算代价，选定用于迁移的 `gt_B` 模型；锁定方案后报告官方测试划分结果。
- [ ] 用已有少量自有场景数据运行选中模型的 zero-shot 与对应监督模式微调，并在独立目标场景比较效果。
- [ ] 进入业务执行阶段后，以相同动作参数完成正常工况效果比较。

可选 sidecar 缓存与 residual uncertainty 工具尚未实现，仅在实际需要时安排，不是上述训练的前置条件。现有测试和短训练记录没有替代全量数据生产验收或真实场景效果实验。

## 模型与训练实施文档

从 [实施文档总览](docs/implementation/README.md) 进入。优先阅读 [已完成旧数据的适配方案](docs/implementation/LEGACY_DATA_ADJUSTMENT.md)：已有 Seal/Wrench/Collision/GT-A/GT-B 原样复用，只在训练适配层恢复候选轴和补齐观测法向；可选派生缓存不要求重制整套标签。

| 内容 | 文档 |
|---|---|
| 原设计如何逐步修正 | [设计演进与图示](docs/implementation/DESIGN_EVOLUTION.md) |
| 输入、掩码、批数据 | [Data adapter](docs/implementation/DATA_ADAPTER.md) |
| 模型结构、维度、分组与损失 | [模型规格](docs/implementation/MODEL.md) |
| 正在运行的公开预训练 | [v3 固定配置、策略与边界](docs/CD009_PRETRAINING.md) · [原训练规格](docs/implementation/TRAINING.md) |
| 少量自有数据微调 | [微调接口与策略](docs/implementation/FINETUNING.md) |
| 消融、R20评价与目标验证 | [实验方案](docs/implementation/EXPERIMENTS.md) |

可执行命令见[运行指南](docs/MODEL_USAGE.md)。完整预训练和两种微调配置已提供；可选sidecar转换与residual uncertainty仍未实现。

## 目录

```text
configs/              # 标签物理参数、E0–E5预训练与有界smoke配置
suction20/             # 数据生成、只读适配、损失、训练状态与评价
scripts/               # 数据制作、训练、微调、评价和推理入口
tests/                 # 单元/合成场景测试
docs/                  # 数据字段、流程与项目计划
models/                # 稀疏主干封装、实例/接触分组与评分头
docs/implementation/   # 模型/预训练/微调/实验的详细规格与配图
```

## 数据制作入口

以下用于新增数据制作。用户已经制作完成的数据训练时直接走上述adapter，不再次执行此流水线。

```bash
pip install -r requirements.txt
pip install open3d pytest
git clone https://github.com/graspnet/suction_annotation.git /workspace/suction_annotation

export DATASET=/data/SuctionNet
export UPSTREAM=/workspace/suction_annotation
export OUT=/data/suction20_output

bash scripts/run_data_pipeline.sh
```

该命令需要本机已准备 SuctionNet 数据、`dense_point_clouds`、原始 Seal/Wrench 标注和场景 RGB-D 数据。输出写入 `$OUT`，不提交大型原始数据或生成标签至 GitHub。

具体步骤： [数据生成流程](docs/DATA_PIPELINE.md) · [字段定义](docs/DATA_SCHEMA.md) · [项目路线](docs/ROADMAP.md)。

## 重要约定

- G0 保持官方 10 mm Seal 方法和 ±1 mm 环带；G1 使用同样 72 扇区方法、20 mm 半径与 ±2 mm 环带。
- 20 mm 场景标签分别保存 `wrench_annot` 和 `wrench_eval`；GT-B 使用 `wrench_eval`。
- 当前producer用候选物体法向作为物理标签和主 `gt_normal`；PCA法向独立保存为 `gt_normal_pca`。旧缓存的同名 `gt_normal` 可能是PCA，训练时按旧数据适配方案恢复原候选轴。
- 场景点只通过同实例、空间距离、法向门控匹配最近候选。匹配失败使用 `gt_valid=False`，不是已知负样本。
- 不重新发布第三方原始标注脚本或数据；需要在本地拉取上游开源仓库。

原始数据使用和科研结果均需遵守 SuctionNet/GraspNet 的相应许可证。

