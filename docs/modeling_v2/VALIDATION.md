# 方案 2 数据文档与参考工具验证

日期：2026-11-25。验证对象为本分支的实施文档、协议配置、清单/预检工具和几何参考；不代表生产数据已转换或新模型已训练。

## 实际执行结果

| 检查 | 结果与边界 |
|---|---|
| 几何参考自测 | **35项通过**；[逐项报告](reference/self_test_report.json) |
| 新预检工具pytest回归 | **4项通过**：3200帧计划、CP均匀选帧、缺GT不换帧、已有输出不覆盖 |
| 参考几何接入pytest | **1项通过**，调用上述35项参考检查，不重复累计为35个pytest case |
| 仓库禁止内容摘要门禁 | **60项通过**，覆盖新增可执行代码 |
| 实际运行plan CLI | 3200条唯一SN帧，2880 train / 320 dev，0条test帧 |
| CP合成文件fixture | 24scene，其中一个100帧取96、其余23个各3帧，总165，无重复补齐 |
| 缺选中GT的fixture | 退出2，帧选择不变，缺失数1、可用完整帧减少1 |
| 缺整个来源/输出占用 | 形成明确失败或拒绝覆盖；没有宣称训练可用 |
| 文档和文件静态检查 | 20个相对链接存在，8个Python/bash代码块语法通过，4个Python文件及协议JSON可解析 |

几何检查包含：离轴光轴深度与射线、两次plane-clamp、原始有效深度逐值不变、75 mm反例、80 mm加修正端点的覆盖、25/70/4 mm边界、22/28 mm半径、目标自碰与外物分项、解析plane、未知场景拒绝、输入白名单拒绝GT别名，以及双相机/物理瓶/分组跨split拒绝。

参考schema仅包含其显式输入字段，生产schema仍需新Adapter实现。点云碰撞参考只针对调用者提供并声明完整的几何，不证明真实场景完整。plan/audit仅进行路径/文件大小预检，不解码图像，不验证位姿、单位、CAD或Teacher。

## 复现命令

```bash
python docs/modeling_v2/reference/geometry_contract.py --self-test
python -m pytest -q tests/test_modeling_v2_data_plan.py tests/test_modeling_v2_geometry_contract.py tests/test_no_sha256_verification.py
python scripts/modeling_v2_data_plan.py plan --output /path/to/new/model2-plan
```

## 合并基线验证

历史代码先收拢再创建本分支。合并基线 `2e1349d85e7638db0bc35e8faf9a3d921ca1d8cd` 已完成 [GitHub 全量 CPU 测试](https://github.com/chichu-lv/suctionnet_modern_20mm/actions/runs/45604682267)：**1030 passed / 1 skipped，156.30秒**。CI 使用 Python 3.11.18、PyTorch 2.5.1+cpu，执行未删减的 `python -m pytest -q`。

本地分组回归为1027 passed /1 skipped，旧prefetch文件另两项单独通过；DataLoader spawn在本地阻塞，已如实保留记录，该项随后在CI全量测试中通过。没有为通过测试修改测试或增加skip。详细本地环境、命令、原生扩展及限制见[分支收拢记录](../BRANCH_CONSOLIDATION_20261125.md)。上述全量结果属于合并基线；本次新增工具的本地验证见上表，不混记为已在该次CI中执行。

本次没有读取训练服务器的真实数据，没有执行ClearPose/MGN批量转换、GPU训练或机器人吸取。原规格中的3–5倍提速、85%偏好准确率、90%空洞覆盖和4 mm补洞误差不属于本轮验证结果。
