# 后续模型代码

这里预留 GSNet/MultiGraspNet 风格的吸盘模型实现。当前仅完成数据制作代码的迁移，不放入未经验证的网络实现。

后续新增模型时建议按 `models/backbones/`、`models/heads/`、`models/losses/` 模块组织，并在仓库根目录增加 `train.py` 或单独的 `training/` 入口。

使用本仓库逐帧 GT 时：`gt_B` 是主要质量监督；`gt_normal` 是与物理评分姿态一致的原始候选法向；`gt_normal_pca` 仅作独立几何对比，`normals_raw` 为相机观测估计。质量损失必须使用 `gt_valid` 掩码。
