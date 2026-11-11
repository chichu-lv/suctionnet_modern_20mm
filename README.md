# SuctionNet Modern 20mm

面向 **半径 20 mm 单吸盘**的 SuctionNet 数据制作与后续抓取模型项目。

当前已纳入的数据流程：SuctionNet 10 mm 标签对照（G0）→ 20 mm Seal 重标注（G1）→ 场景 Wrench/Collision → 每帧 RGB-D 点云与吸取候选映射 → GT-A/GT-B → 数据质量检查（G2）。

**代码状态：** 已迁入数据制作实现。用户已用旧 `dc4ad6b00535` 方法完成数据集制作；本仓库的既有合成测试不等于已检查用户数据盘中的全量产物。只读适配、E0–E5模型、训练/微调、推理及R20评价代码已实现；真实数据有界验证见[验证记录](docs/MODEL_VALIDATION.md)。这不表示完成正式预训练或目标域效果验证。

## 模型与训练实施文档

从 [实施文档总览](docs/implementation/README.md) 进入。优先阅读 [已完成旧数据的适配方案](docs/implementation/LEGACY_DATA_ADJUSTMENT.md)：已有 Seal/Wrench/Collision/GT-A/GT-B 原样复用，只在训练适配层恢复候选轴和补齐观测法向；可选派生缓存不要求重制整套标签。

| 内容 | 文档 |
|---|---|
| 原设计如何逐步修正 | [设计演进与图示](docs/implementation/DESIGN_EVOLUTION.md) |
| 输入、掩码、批数据 | [Data adapter](docs/implementation/DATA_ADAPTER.md) |
| 模型结构、维度、分组与损失 | [模型规格](docs/implementation/MODEL.md) |
| 公开数据预训练 | [训练策略](docs/implementation/TRAINING.md) |
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

