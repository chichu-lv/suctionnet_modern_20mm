# A/B 与 SuctionNet 现有方法的成对比较

**公开评价仅参数化吸盘半径的方案现设为待定；当前先实现、训练和验证A/B。** 原论文数值和逐方法成对表保留，本方法公开AP均为“待测”。不重训参考方法，不制作参考方法之间的排行榜。旧E5结果见[既有方法比较](../SUCTIONNET_METHOD_COMPARISON.md)，不能复制到A/B结果行。

## 1. 当前执行决定

| 项目 | 当前规则 |
|---|---|
| 源域训练 | scene0000–0099全部100场景，全部256帧可进入采样；从头初始化 |
| 源域选模 | scene0100–0189全部90场景、256帧，按native INST20实例Top1选择 `best.pt` |
| checkpoint身份 | `selection_role=test_selected`、`independent_test_estimate=false`；保留 `last.pt` |
| 目标域微调 | 原1676 train /399 val /87 test不变；399 val选择微调best |
| 本轮评价器 | `scripts/evaluate_instance.py`：native INST20或历史动作成败 |
| 公开半径协议 | `status=deferred`、`radius_only_protocol=null`，不作为开训门槛 |
| 公开对比结果 | 论文值继续引用；本方法AP、差值和公开正式权重仍待确定与实测 |

源域test已经参与best选择，选模后的同集成绩带有选择偏差。保留last便于追溯固定预算终点，但同一次运行已观察test；不能仅把文件名改成last就称整个研究过程具有独立test估计。当前native单实例指标也不是论文的完整场景AP。

旧90/10开发→100场景refit→固定last的流程不再是新路线当前执行方案。原方案的仅半径修改白名单、NMS/关联/六指标和空帧边界保存在[待定技术备忘](PUBLIC_RADIUS_PROPOSAL_DEFERRED.md)，不在本轮实现。运行入口见[IMPLEMENTATION.md](IMPLEMENTATION.md)。

## 2. 保留的逐方法比较单元

| 比较单元 | 论文报告来源 | 原方案拟匹配的前端/工作区 | 当前状态 |
|---|---|---|---|
| A或B vs SuctionNet / S1B | v2 Table IV | UOIS、无GT工作区 | 论文参考，本方法待测 |
| A或B vs Diffusion-Suction | v2 Table I，SuctionNet部分 | UOIS、无GT工作区；对方未公开字段仍未知 | 论文参考，本方法待测 |
| A或B vs UOIS-Net + UISN | Table I | UOIS、UISN专属GT联合前景AABB工作区 | 论文参考，本方法待测 |
| A或B vs UOAIS-Net + UISN | Table I | UOAIS、UISN专属工作区 | 独立成对参考，不与另一UISN行比较 |
| A或B vs MultiGraspNet-vac. | v2 Table I，Vacuum部分 | 原计划UOIS、无GT工作区 | 物理协议未核实，reference-only |
| A或B vs MultiGraspNet joint | 同表joint行 | 同上；额外平行夹爪监督需说明 | 物理协议未核实，reference-only |

每分支 `EXPERIMENT_PLAN.md` 保留六组成对表。论文未报告指标用 `null` / “—”，本方法未运行用“待测”，两者都不能填0。论文半径证据按各自来源保留，不把R10论文值按系数换算为R20。已有90个指标数值与18个缺失项保持原转录，不做本轮数值重估。

## 3. 后续公开协议待定事项

用户要求的目标仍是：每次只比较本方法与一个既有方法，测试集、相机和指标计算尽量匹配，允许明确披露吸盘尺寸差异。采用哪套公开物理适配、如何报告test参与选模、用哪份权重以及分割前端资产身份尚未冻结。此处不把未公开的历史API/模型资产/训练细节假设为相同。

未来运行完整场景评价时需保留全部失败和空帧分母；预测分割不可替换成GT实例输入。UISN的GT联合前景工作区只属于其专属比较条件，不推广给S1B等其它比较。模型内部v3法向、18–22mm接触特征和target-only teacher也不自动定义公开评价器。现阶段不将native INST20或旧项目R20直接写成官方AP。

## 4. 结果与来源身份

[paper_baselines.json](reference/paper_baselines.json)保留逐方法原表页码、版本、相机、缺失字段与数字；`result_kind=paper_reported_not_reproduced`。当前运行元数据另写 `source_training_policy`，公开 `our_protocol` 为deferred。旧协议数值集中保存于 `deferred_radius_proposal`，不是有效运行配置。

后续本方法若形成公开结果，至少需保存checkpoint路径/大小、代码版本出处、实际训练/选模划分、完整测试manifest、前端/工作区资产路径与配置、预测和precision文件及语义ID、评价器版本、吸盘半径、六指标及失败覆盖。只有可比性边界明确且有真实测量后才能计算 `ours−本表参考` 的百分点差值；不计算第三方之间的排名或差值。

## 5. 论文值的来源与可比性状态

| 方法 | 原始表定位 | 上轮取证状态 | 仍需随结果说明的限制 |
|---|---|---|---|
| S1B | arXiv:2103.12311v2，Table IV，PDF第6页；斜杠右侧RealSense | 上轮直接核对原论文 | 采用最终表的28.31/38.56等值；不混其他版本或后续转引的小数差异 |
| Diffusion-Suction | arXiv:2502.07238v2，Table I，PDF第7页；斜杠右侧RealSense | 上轮直接核对原论文；作者声明沿用SuctionNet评价 | 输入为场景点云；完整SuctionNet训练recipe、checkpoint、mask过滤、历史API版本未披露充分；不拿Parcel配置替代 |
| UISN两种完整配置 | DOI:10.1109/LRA.2024.3385609，Table I，PDF第6页/刊页4939 | 沿用仓库已有原PDF核验转录；上轮直接核验作者代码，未重新取得全文 | 表注为Kinect/RealSense，与§IV-B一处反写冲突；沿用表注和S1B交叉核对；历史API版本未锁定 |
| MultiGraspNet-vac. / joint | arXiv:2602.06504v2，Table I Vacuum，PDF第5页，明确RealSense | 上轮直接核对原论文 | 只报告Top50三列；文中称seal coefficient，未确认等同S×W；完整实现仍未公开。joint额外使用平行夹爪监督 |

UISN 数值来源链：固定仓库版本的[既有转录](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/a9f4a8451c6bc42cfd8867dd96c051398644ad71/docs/experiments/COMPARISON-20261118/method_metrics.csv)及[逐表说明](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/a9f4a8451c6bc42cfd8867dd96c051398644ad71/docs/SUCTIONNET_METHOD_COMPARISON.md)；DOI/机构条目只证明论文身份，不能代替表格证据。若后续拿到原PDF，更新核验记录，不静默更改列值。

Diffusion 与 UISN 原论文中的半径没有独立实验清单时，应写“所引用官方协议默认R10”，不能称已取得作者运行日志。MultiGraspNet 半径与物理一致性未核实，JSON用 `null` 配合证据说明；Top1未报告也用 `null` / “—”，不补0、不推算。MultiGraspNet 两张表保留数值，但 `comparison_status=reference_only_protocol_unverified`，不输出正式增益结论。其余论文行也保留各自证据强度，不能用“所有条件完全相同”概括未公开字段。

不要求复现第三方模型来补未公开字段；能从原文/官方源码核实的先核实，仍未知的明确保留。已有 E5 是 GT 实例输入、90场景训练的历史结果，不是本方案严格匹配后的内部基线。UISN Only Depth 是原方法消融且相机取值需推断，当前不增加该比较单元。


## 8. 原始来源

- **[S1] SuctionNet**：[最终v2论文](https://arxiv.org/pdf/2103.12311v2)、[官方数据集](https://graspnet.net/suction)。Table IV 为本方法S1B的六指标来源。
- **[S2] Diffusion-Suction**：[v2论文](https://arxiv.org/html/2502.07238v2)、[作者固定仓库README](https://github.com/TAO-TAO-TAO-TAO-TAO/Diffusion_Suction/blob/c91f0c76b54aedeb22c29f0ecf100efe9198c827/README.md)。只取Table I SuctionNet值。
- **[S3] UISN**：[DOI](https://doi.org/10.1109/LRA.2024.3385609)、[作者机构记录](https://research.cuhk.edu.hk/en/publications/uncertainty-aware-suction-grasping-for-cluttered-scenes-2/)。原表转录证据见第5节。
- **[S4] MultiGraspNet**：[v2论文](https://arxiv.org/html/2602.06504v2)、[官方项目页](https://vandal-lab.github.io/multigraspnet-project/)。取Table I Vacuum，取证日期2026-11-18。
- **[S5] 官方API固定版本**：[物理/关联/NMS](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/utils/eval_utils.py)、[测试划分与聚合](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/suctionnet_eval.py)、[法向基及极点行为](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/utils/rotation.py)。
- **[S6] UISN固定版本**：[推理与预测分割](https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/inference.py)、[训练清单](https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/dataset/dataset.py)、[评价入口](https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/eval.py)、[GT联合前景工作区实现](https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/utils/data_utils.py)。
- **[S7] 官方预测文件**：[baseline导出代码](https://github.com/graspnet/suctionnet-baseline/blob/93d604dd752d2ef865c80c8ecba96218560b516f/neural_network/inference.py)。以实际`score,direction,point`数组顺序为准。
