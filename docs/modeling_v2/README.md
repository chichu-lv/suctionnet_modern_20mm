# 模型化方案 2：Suction20 V2-Belt70

本目录是新方案的开发入口。方案 2 采用单目标实例加局部背景、12 维几何输入、内置缺深修复，以及 R20 / 70 mm 的解耦质量与碰撞监督。它与历史 E5、Route A、Route B 分开版本管理。

## 从这里开始

1. 阅读 [数据处理实施文档](DATA_IMPLEMENTATION.md)。按阶段执行，并用每阶段的产物和验收条件判断是否完成。
2. 运行 [数据清单与资产预检工具](../../scripts/modeling_v2_data_plan.py)，配置见 [data_protocol_v1.json](../../configs/modeling_v2/data_protocol_v1.json)。所有命令从仓库根目录执行。
3. 运行 [几何合同参考检查](reference/geometry_contract.py)。它验证单位、射线、补洞边界、裁剪和 C70 规则，不代替真实数据验收。
4. 对照实施文档的“生产实现任务”逐项实现新 Adapter、Belt70 Teacher 和数据读取器。现有 A/B 构建器不会自动生成方案 2 数据。

## 本次交付边界

这次交付的是可执行的准备/预检工具、几何合同参考、明确到字段和算法的数据实施文档，以及完整保留的[原始规格附件](source/suction20_v2_belt70_reproducible_spec.html)。真实训练数据未随本次附件提供；没有据此宣称已完成服务器上的数据转换或模型训练。

历史路线的合并与保留情况见 [分支收拢记录](../BRANCH_CONSOLIDATION_20261125.md)。
