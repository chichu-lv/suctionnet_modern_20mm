# SuctionNet Modern 20mm

面向 **半径 20 mm 单吸盘**的 SuctionNet 数据制作与后续抓取模型项目。

当前已纳入的数据流程：SuctionNet 10 mm 标签对照（G0）→ 20 mm Seal 重标注（G1）→ 场景 Wrench/Collision → 每帧 RGB-D 点云与吸取候选映射 → GT-A/GT-B → 数据质量检查（G2）。

**代码状态：** 已迁入数据制作实现；尚未在完整 SuctionNet 原始数据集上完成 G0–G2 的实测验收。模型训练和时序跟踪代码后续在本仓库维护。

## 目录

```text
configs/              # 吸盘尺寸、几何、标签计算参数
suction20/             # 数据生成 Python 包
scripts/               # 数据制作入口、稠密点云格式转换
tests/                 # 单元/合成场景测试
docs/                  # 数据字段、流程与项目计划
models/                # 后续模型实现与训练接口
```

## 快速使用

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
- 候选物体法向用于物理标签和主 `gt_normal`；PCA 法向独立保存为 `gt_normal_pca`。
- 场景点只通过同实例、空间距离、法向门控匹配最近候选。匹配失败使用 `gt_valid=False`，不是已知负样本。
- 不重新发布第三方原始标注脚本或数据；需要在本地拉取上游开源仓库。

原始数据使用和科研结果均需遵守 SuctionNet/GraspNet 的相应许可证。
