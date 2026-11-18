# 路线 B 实施卷宗：共享二维编码与投影环带

本目录是 **`route-b-2d-projected-rings`** 分支的开发入口，将 B 的数据准备、预处理、模型设计、后处理、训练和实验拆为六份可逐项落实的说明。共同起点为 [a9f4a84](https://github.com/chichu-lv/suctionnet_modern_20mm/commit/a9f4a8451c6bc42cfd8867dd96c051398644ad71)，其中已包含 v3 方案和 A/B 方法架构图。

**当前交付状态：工程设计、实施步骤和可运行 NumPy 数学参考。** B 生产模型、实例 teacher 接线、新训练器和部署引擎尚待实现；文档中的“验收”表示后续必须取得的证据，不表示已经训练或测速。新路线仍使用物理质量 `gt_INST20_pose_v2`，观测法向、候选和模型设计为 v3。[原始 B 规格](../ROUTE_B.md)与[设计 JSON](../reference/route_b.json)是本卷宗的参数依据。

![Route B v3 architecture](../../figures/instance-route-b-architecture.svg)

完整[架构图说明、双语图注和维度表](../METHOD_ARCHITECTURE_B.md)；下载 [SVG](../../figures/instance-route-b-architecture.svg)、[PDF](../../figures/instance-route-b-architecture.pdf)、[PNG](../../figures/instance-route-b-architecture.png)。

公开验证入口：[本方法逐一对照现有方法](EXPERIMENT_PLAN.md) · [统一公开协议](../PUBLIC_METHOD_COMPARISON.md) · [论文数值与来源](../reference/paper_baselines.json)。使用独立public20评分权重和三种匹配的前端/工作区推理身份；论文数值已填，本方法结果待测。

## 1. 六部分阅读顺序

| 顺序 | 文档 | 实施者应得到的结果 |
|---|---|---|
| 1 | [数据准备](DATA_PREPARATION.md) | 可核验的源/目标清单、共同 query 清单、同姿态 S/W/self/INST 标签及缓存身份 |
| 2 | [预处理](PREPROCESSING.md) | organized 几何与法向、候选回填、13 通道上下文图、5 通道原生纹理 ROI、坐标变换 |
| 3 | [模型设计](MODEL_DESIGN.md) | 共享编码器、Fq/G、投影读取、支持门控、96+18+12 接触描述、330 维融合和五头接口 |
| 4 | [后处理](POSTPROCESSING.md) | 稳定排序、米制 NMS、精确 query 位姿、输出字段与部署数值边界 |
| 5 | [训练方案](TRAINING_PLAN.md) | 唯一首轮配方、正确损失归约、参数冻结、预算、恢复与选模规则 |
| 6 | [实验方案](EXPERIMENT_PLAN.md) | 最小实验集合、统一分母/指标、边界与真实 teacher 检查、计时和结果记录 |

共同事实继续引用[数据与标签合同](../DATA_AND_LABELS.md)、[训练评价合同](../TRAIN_EVAL_REPRODUCE.md)和[v3 边界复核](../BOUNDARY_REVIEW.md)。本目录按实施环节重组这些规定，不建立第二套默认 teacher、采样器或损失参数。若实现发现合同缺口，应先记录问题并修订相应版本，不能在某个模块中隐式放宽门控或复用旧身份。

共同数据、teacher、采样及评价合同发生变更时，必须同步到 A/B 两个分支并记录相同版本，防止后续比较的输入和指标口径漂移。

## 2. 本分支固定设计

| 项目 | B v3 默认 |
|---|---|
| 全局图像 | 完整实例 letterbox，`13×128×128`，不宣称恒定毫米比例 |
| 原生纹理 | RGB3 + 相对深度1 + valid1；不 resize；stride1 stem 输出16通道 |
| 共享语义 | ResNet-18 encoder / GN8；L1/L2 各32通道查询采样得到 Fq64；L4 池化得到 G128 |
| 查询姿态 | 原始 p 与共同观测 n；历史动作直接保留 p_plan/n_plan |
| 接触读取 | 9半径×72角；r=0读取一次并广播；query chunk=64 |
| 接触编码 | 三个独立权重的 circular Conv1d / 每角 LayerNorm 分支，共96维 |
| 支持旁路 | 18维覆盖/最大循环缺口 + 12维门控前原始观测/高度 |
| 融合与输出 | 330→128→128；q、Seal、Wrench、ell、task 五个独立头 |
| 标签 | `Q_inst=S20*W20*(1-C_self)`；保留两个 self 分项和纯 SW 诊断 |
| 默认排序 | 源模型 q_inst；任务微调后 sigmoid(task_logit) |

B 的工程目标是共享二维编码和向量化查询。相机投影、深度反投影和姿态几何仍是必要计算；不恢复逐 query 的局部二维栅格/CNN，也不恢复全量三维球邻域 PCA。有限投影环不能证明隐藏部位或完整工具接近体积无碰撞。[结构与边界依据](../ROUTE_B.md)

## 3. 按依赖交付，而非同时开始所有模块

以下文件名是**待实现接口的计划**，当前不能作为已存在的训练命令运行。

| 阶段 | 计划实现 | 进入下一阶段前的证据 |
|---|---|---|
| B1 共用实例资产 | `instance_data.py`、`build_instance_data.py` | split/group/hash 无泄漏；原始 p/n、像素身份和标定一致；两图回读一致 |
| B2 同姿态标签 | `instance_labels.py`、`label_instance_queries.py` | 真实精确 query 的 S/W、两个 self 分项与冻结 evaluator 一致；缓存身份可拒绝旧数据 |
| B3 模型与查询 | `models/instance_b.py`，类 `InstanceSuction2DFastV3` | Fq 位置采样、支持门控、原始高度旁路、空 mask、330维及任意有向轴查询通过 |
| B4 训练接线 | `instance_training.py`、`train_instance.py` | 少量真实更新确实反向；成功步预算、阶段冻结、保存恢复保持同一数据/学习率轨迹 |
| B5 推理评价 | `evaluate_instance.py`、`infer_instance.py` | 排序及 NMS 可复核；输出 p/n 与评分输入相同；主指标含无输出分母 |
| B6 导出计时 | `export_instance_b.py` | 导出子图和组合输出通过真实数值对照；固定动态尺寸/精度合同后完成端到端计时 |

实施阶段和文件路径来源于[共同实施顺序](../TRAIN_EVAL_REPRODUCE.md)。共享 builder/teacher 应复用共同合同；A/B 网络不能因为都输出五个头而互换 checkpoint。

## 4. 现在即可运行的检查

在本分支仓库根目录、已安装 NumPy 的参考环境执行：

```bash
python docs/instance_routes/reference/geometry_contract.py selftest
python docs/instance_routes/reference/revision_contract.py selftest
python docs/instance_routes/reference/sampling_contract.py selftest
python docs/instance_routes/reference/contact_boundary_contract.py self-test
```

它们分别覆盖29、26、26、48项，共129项合成合同检查。旧栅格和旧法向检查保留作历史数学回归，不是 B v3 默认实现。参考环境为 Python 3.12.14 / NumPy 2.3.5，完整环境、demo 与结果边界见[复现说明](../TRAIN_EVAL_REPRODUCE.md)和[既有验证记录](../reference/VALIDATION.json)。

`reference/route_b.json` 的 `artifact_kind=design_spec`，不能直接交给现有 E5 训练脚本；后续运行配置必须显式校验本地资产和版本。真实数据、完整标签与 checkpoint 存放在外部数据/运行目录，仓库保留配置、实现、清单身份和必要汇总。
