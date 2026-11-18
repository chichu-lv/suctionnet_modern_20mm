# 路线 A 实施卷宗：单物体稀疏 3D + RGB

本目录是 `route-a-3d-rgb` 分支的开发入口，按数据到实验的顺序整理 `InstanceSuction3D`。共同设计基线为 [a9f4a84](https://github.com/chichu-lv/suctionnet_modern_20mm/commit/a9f4a8451c6bc42cfd8867dd96c051398644ad71)，其中已包含 v3 几何修订和方法架构图。分支中的新生产模型、实例 teacher、训练器和推理器仍待实现；当前可运行内容为继承的几何/schema 参考程序。

![路线 A 方法架构](../../figures/instance-route-a-architecture.svg)

图示说明、双语图注及 SVG/PDF 下载见[路线 A 方法架构图](../METHOD_ARCHITECTURE_A.md)。

公开验证入口：[本方法逐一对照现有方法](EXPERIMENT_PLAN.md) · [统一公开协议](../PUBLIC_METHOD_COMPARISON.md) · [论文数值与来源](../reference/paper_baselines.json)。使用独立public20评分权重和三种匹配的前端/工作区推理身份；论文数值已填，本方法结果待测。

## 1. 阅读顺序与交付物

| 顺序 | 文档 | 实施后应交付 |
|---|---|---|
| 1 | [数据准备](DATA_PREPARATION.md) | 固定实例/动作清单、资产与标定身份、同姿态 S/W/self/INST 标签 |
| 2 | [预处理](PREPROCESSING.md) | 原始支持、v3 法向、嵌套候选、主干点及关联诊断 |
| 3 | [模型设计](MODEL_DESIGN.md) | 12 通道稀疏编码、三区域与原始 lip 摘要、531 维姿态评分接口 |
| 4 | [后处理](POSTPROCESSING.md) | 分阶段排序、米制 NMS、稳定输出与相机系/工具系变换 |
| 5 | [训练方案](TRAINING_PLAN.md) | 8750 次预训练成功更新、200+800 次微调及可恢复记录 |
| 6 | [实验方案](EXPERIMENT_PLAN.md) | 几何/真实 teacher/短链路验收、正式比较、耗时与失败归因 |

六篇分别固定操作顺序、输入输出和完成条件。共同事实仍以[数据与标签合同](../DATA_AND_LABELS.md)、[训练评价合同](../TRAIN_EVAL_REPRODUCE.md)和 [A 设计 JSON](../reference/route_a.json)为准；本目录对 A 做实施展开。若未来变更共同物理或观测定义，应同步合同、版本及缓存身份后再进入两个路线分支，不能靠本地措辞修改默认值。

## 2. 本路线固定的设计

| 项目 | A 默认合同 |
|---|---|
| 设计身份 | `suction20.instance_routes.design.v3`；JSON 为 `design_spec` |
| 物理质量 | `gt_INST20_pose_v2`；`Q_inst=S20*W20*(1-C_self)` |
| 标签 schema | `suction20.instance_inst20.v2`，true 表示目标自碰 |
| 观测法向 | `organized_masked3x3_span3_v3`，只改变法向估计输入 |
| 候选 | `image_stratified_morton_refill_v3`；完整池最多 256，训练/回放默认前 128 |
| 主干 | 2 mm 体素代表点后 FPS 最多 2048，12 维输入，输出 F256 |
| 姿态评分 | Fq256 + G64 + 三区域192 + lip8 + pose10 + valid1 = 531 |
| 输出头 | q_inst、Seal、Wrench、Laplace log-scale、task logit |
| 历史动作 | 原记录 p/n 直接评分，几何锚点不移动姿态 |
| 输出 | 选中的原 p/n；工具 +Z 为 −n，roll 按固定约定补全 |

原始支持 `P_support`、稀疏主干点与共同 query 是三个集合。主干抽样控制网络成本；原始支持保留唇口细节；query 为 A/B 提供共同评测条件。`mu_vis` 是可见支持的几何均值，不是 teacher 的完整对象质量中心。[A 完整规格](../ROUTE_A.md)

## 3. 开发阶段与依赖

| 阶段 | 建议新增接口/模块（尚未接线） | 进入下一阶段的条件 |
|---|---|---|
| A1 数据与查询 | `suction20/instance_data.py`、`scripts/build_instance_data.py` | split、标定、原始像素身份及 query 回读一致 |
| A2 实例 teacher | `suction20/instance_labels.py`、`scripts/label_instance_queries.py` | 固定真实 p/n 的 S/W、自碰分项与冻结实现对齐，缓存拒绝错误身份 |
| A3 模型 | `models/instance_a.py` 与独立 12 通道 wrapper | 531/200/329/260 维头部接线及缺失支持行为通过 |
| A4 训练 | `suction20/instance_training.py`、`scripts/train_instance.py` | loss 归约、BN 冻结、优化器切换及下一批恢复一致 |
| A5 推理与评价 | `scripts/infer_instance.py`、`scripts/evaluate_instance.py` | 输出 p/n 等于实际评分条件，指标分母和无输出记录完整 |
| A6 正式实验 | 固定运行配置、预测、汇总与性能报告 | 完成[最少实验集合](EXPERIMENT_PLAN.md)，再决定是否追加诊断 |

生产接口沿用 `encode_instance(observation)` 与 `score_at_pose(encoded,p,n,query_batch_index)` 的共同约定。建议 CLI 与字段在共同合同中已列出；本文没有把未来文件名写成已经能运行的训练命令。原 E5 的 checkpoint、8 通道 wrapper 及旧配置不能直接恢复到 A，继续执行当前同身份恢复政策。[共同实施顺序](../TRAIN_EVAL_REPRODUCE.md)

## 4. 当前可以直接运行什么

使用已有 Python/NumPy 参考环境，在仓库根目录执行：

```bash
python docs/instance_routes/reference/geometry_contract.py selftest
python docs/instance_routes/reference/revision_contract.py selftest
python docs/instance_routes/reference/sampling_contract.py selftest
python docs/instance_routes/reference/contact_boundary_contract.py self-test
```

继承的四组检查共 29+26+26+48=129 项，记录见 [VALIDATION.json](../reference/VALIDATION.json)。它们覆盖数学与文件关系；真实资产来源、同姿态 teacher、神经模型及 GPU 时间需依照本卷宗继续验收。参考环境安装和演示命令见[共同复现说明](../TRAIN_EVAL_REPRODUCE.md)。

大体积 RGB-D、完整对象几何、标签缓存和 checkpoint 保存在外部数据/运行目录。分支提交源码、配置、必要汇总与可复核的运行身份；未来实验中的成功率和时延只填写实际完成的结果。
