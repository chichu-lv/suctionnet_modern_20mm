# B 实验方案：test选模、几何边界与公开方法参考

B模型及共同训练/推理链路已实现，实际网络CPU前向、反向和12项模型边界测试通过；真实完整source/target训练、GPU部署和公开AP尚未运行。本页把本轮可执行的实例/动作评价，与待定的论文比较协议分别标明；旧E5数值、数学参考和合成网络检查不能填入B实测行。

## 1. 当前执行策略与公开评价状态

| 项目 | 本轮执行 |
|---|---|
| 源训练 | scene 0–99全部100场景、全部256视角可抽样；从头训练，不再90/10开发后final refit |
| 预算 | 140000个成功受监督实例、8750次成功更新；128-query训练前缀 |
| best选择 | 每875成功步评价一次，共10次；source test scene 100–189每场景固定8帧，全部可见实例、每实例完整最多256-query；保留best与预算终点last |
| 成绩身份 | `selection_role=test_selected`、`selection_scope=fixed_test_subset`、`independent_test_estimate=false`；源test子集已参与模型选择 |
| 训练后报告 | 独立CLI对固定best评价完整90场景×256帧，不更新或重选best；分别记录本次评价范围与源预训练选择范围 |
| 目标域 | 1676 train、399 val选best、87 test冻结后评价；保留既有group和规划动作语义 |
| 本轮评价 | INST20实例Top1/selected-Q/coverage，以及目标固定动作AUROC/AP/BCE/Brier |
| 公开方法评价 | `deferred`；只参数化半径的方案待定，未实现、不作为开训前置门禁 |
| 论文表 | 保留各论文原始数值及来源；B公开结果待测，不能填入本轮实例指标 |

source test完整集为RealSense Seen 100–129、Similar 130–159、Novel 160–189，各7680帧、总23040帧。默认周期选模只取每场景固定 `0,32,64,96,128,160,192,224` 八帧，各类别240帧、总720帧；每次使用同一集合，不按模型分数换帧或删实例。帧数由`pretrain.selection_frames_per_scene`配置：8为当前默认，16包含这8帧，256使用全量；旧配置未设置该字段仍保持256帧语义。完整test manifest可由训练入口自动筛选，正式8帧selection缓存也可直接使用，只要全部90场景的所需帧齐全就不需要`--allow-partial-manifests`。

较小selection减少周期评价成本，也可能使指标接近的checkpoint排序波动；固定帧集合控制重复采样差异，不能保证该best在完整test上也最优。完整test只在训练后评价已固定best，不根据报告重新挑权重。当前GT实例物理指标与预测实例前端参与的全景公开AP不是同一实验；子集及完整报告均保留`test_selected`，不能包装成盲测或“除吸盘半径外完全一致”的公开比较。

用户要求的逐方法对照保留在第2节及后文MultiGraspNet两组表中，每表只有B与一个参考，不比较参考之间。未来恢复公开评价时，须逐pair冻结测试集/相机/指标、物理口径、前端和GT工作区权限；未披露的论文训练细节标unreported，未核实项保持unverified。半径-only方案及其R10对齐检查都仍待决定，不能提前宣称已匹配。

论文表的六项AP继续按原来源解释：`AP50`为阈值{0.2,0.4,0.6,0.8}与前缀k=1..50精度的平均，`AP0.4_50`/`AP0.8_50`固定阈值，`AP1`及其阈值列只取首候选；使用百分数。它们不是实例Top1、PR面积或仅P@50。公式与证据见[已有公开比较参考](../PUBLIC_METHOD_COMPARISON.md)及[论文基线JSON](../reference/paper_baselines.json)，本次没有重新运行任何论文方法或重新取得其数值。

后续前端候选方案仍需按方法审查：无GT工作区UOIS，以及为匹配UISN单独声明的UOIS/UOAIS加GT联合前景AABB。UISN权限不自动扩展到S1B或Diffusion；同名前端也不证明权重、工作区和候选一致。以下前端名仅保留为候选方案，不表示公开推理/导出已实现或协议已冻结。

## 2. 本路线与各公开方法的独立比较

每张表只比较本路线与一个参考方法。`待测` 表示本方法尚未执行，`—` 表示论文没有报告。以下论文值来自各自标注的表，不是本仓库复现值；完整出处和核验状态以[公共来源登记](../PUBLIC_METHOD_COMPARISON.md)为准。

### 2.1 B 与 SuctionNet / S1B

[SuctionNet-1Billion 原论文 v2](https://arxiv.org/pdf/2103.12311v2)，Table IV，PDF 第 6 页，取 Kinect/RealSense 斜杠后的 RealSense 数值。固定采用 2021-10-29 v2 主表，不混用旧版本或消融表的小数差异。原方法处理整帧 RGB-D、不需要 GT 实例；本pair候选前端为 `uois_no_gt_workspace`，尚待协议冻结，不使用 GT 前景工作区、GT 实例裁剪或补回漏检。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | SuctionNet / S1B / 论文 R10 | 28.31 | 38.56 | 3.41 | 43.98 | 60.81 | 7.79 |
| Similar | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | SuctionNet / S1B / 论文 R10 | 26.64 | 35.34 | 3.42 | 36.59 | 49.30 | 6.54 |
| Novel | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | SuctionNet / S1B / 论文 R10 | 8.23 | 10.29 | 0.35 | 14.32 | 19.95 | 0.38 |

### 2.2 B 与 Diffusion-Suction

[Diffusion-Suction 原论文 v2](https://arxiv.org/html/2502.07238v2)，Table I，PDF 第 7 页，取 RealSense 列；Table II 的 Parcel-Suction 结果不属于此比较。论文声明沿用 SuctionNet 在线 Seal × Wrench 协议；Table I 的完整训练/验证 recipe、RGB/实例过滤和 checkpoint 配置未逐项核实，不能从 Parcel 训练脚本推定。本pair候选前端暂记 `uois_no_gt_workspace`，尚未执行；Diffusion 原始工作区仍标“未核实”，不能据此宣称该项已匹配，也不继承 UISN 的 GT 前景工作区权限。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | Diffusion-Suction / 论文，官方默认R10 | 31.90 | 42.78 | 4.79 | 54.79 | 68.59 | 21.03 |
| Similar | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | Diffusion-Suction / 论文，官方默认R10 | 31.59 | 41.38 | 4.27 | 47.00 | 59.36 | 16.08 |
| Novel | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | Diffusion-Suction / 论文，官方默认R10 | 10.00 | 12.32 | 0.76 | 14.14 | 17.67 | 2.42 |

### 2.3 B 与 UISN (UOIS)

[UISN 原论文](https://doi.org/10.1109/LRA.2024.3385609)，Table I，PDF 第 6 页 / 刊页 4939，按表注取 RealSense 数值。该pair候选匹配方案为 `uois_uisn_gt_aabb20`，尚未执行：冻结 UOIS 预测分割，另按作者 `get_workspace_mask(..., outlier=.02)` 使用 GT 前景联合 AABB 工作区。只读取前景联合范围，不读取 GT 实例身份，也不用 GT 实例 mask 替换预测分割。该预测不能复用 `uois_no_gt_workspace` 的结果。原文转录与本轮全文访问状态见公共来源登记；本轮未能重新读取全文，使用仓库已登记的逐表转录。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | UISN (UOIS) / 论文，官方默认R10 | 37.98 | 49.83 | 9.99 | 63.88 | 76.64 | 31.72 |
| Similar | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | UISN (UOIS) / 论文，官方默认R10 | 37.83 | 49.35 | 8.89 | 48.68 | 60.01 | 16.72 |
| Novel | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | UISN (UOIS) / 论文，官方默认R10 | 9.47 | 12.63 | 1.17 | 11.83 | 15.47 | 2.76 |

### 2.4 B 与 UISN (UOAIS)

来源仍为 [UISN 原论文](https://doi.org/10.1109/LRA.2024.3385609) Table I 的 UOAIS 配置。这是独立pair，候选匹配方案为 `uoais_uisn_gt_aabb20`，尚未执行：同一冻结 source-only 评分权重，冻结 UOAIS 预测分割，加作者 `get_workspace_mask(..., outlier=.02)` 的 GT 前景联合 AABB 工作区；不读取 GT 实例身份。按该 profile 重新生成整帧预测，不复制任一 UOIS profile 的结果，不额外训练评分模型。前端、工作区和成本随结果保存。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | UISN (UOAIS) / 论文，官方默认R10 | 38.27 | 50.57 | 10.56 | 66.11 | 78.35 | 35.29 |
| Similar | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | UISN (UOAIS) / 论文，官方默认R10 | 36.54 | 47.77 | 8.74 | 49.20 | 61.45 | 17.20 |
| Novel | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | UISN (UOAIS) / 论文，官方默认R10 | 9.08 | 12.05 | 1.04 | 11.94 | 15.40 | 2.07 |

## 3. 执行顺序和必要的准入检查

| 阶段 | 目的/固定条件 | 输出证据 | 当前状态 |
|---|---|---|---|
| E0 合同与网络回归 | NumPy参考、B真实网络CPU前向/反向 | 几何反例、Fq梯度、330维、mask/chunk/相位/有限性 | 既有129项数学参考；B专属12项测试通过 |
| E1 真实数据/teacher | 先64–128个覆盖尺寸、dense规模和形状的真实实例；精确p/n | S/W/self对齐、身份、排除及成本报告 | 代码已接线，真实资产核验待运行 |
| E2 实际训练短链路 | 分片真实缓存到若干成功更新、保存恢复和推理 | 有效梯度、冻结、恢复、同姿态输出 | 使用partial运行身份，不作正式成绩 |
| E3 B-source100 | 全100场景train，128-query、140000实例/8750更新；每875步用90场景各固定8帧选best，共10次；结束后独立全量test报告 | test_selected / fixed_test_subset best、预算last、子集与完整报告各自分母及逐实例结果 | 待真实完整训练 |
| E4 B-target-ft | E3 best初始化，200+800，2 target+2 source回放 | 399 val选best；最终87 test | 待真实资产与训练 |
| E5 B公开方法对比 | 按各论文单独配对 | 明确测试/指标/半径/前端和选模身份，B六指标 | deferred，不阻塞E3/E4 |
| E6 B部署 | 冻结权重、动态ROI、精度和输出合同 | 导出差异、端到端p50/p95/显存 | ONNX/TensorRT未完成 |

E1的64–128是teacher成本和一致性剖析样本，不是训练收敛样本数。CPU合成检查与E2短链路只解除具体接线风险，不证明真实成功率或速度优势。任何合同变化须同步版本；不同候选轴、前端或evaluator变化不能归为网络收益。

## 4. 本地源域指标和明确分母

本节指标用于固定test子集选best，并以相同计算规则生成训练后完整test报告；两种范围均分Seen/Similar/Novel报告并标`test_selected`。主排序仍为`InstTop1_INST20@0.4`最大，再按selected-Q宏平均最大，仍平局选更早成功步。未参与梯度并不使其成为独立测试；继续调参或增加比较同样必须保留该选择身份。

周期选模记录`selection_scope=fixed_test_subset`。训练后`evaluate_instance.py`直接检查并评价完整test manifest，报告`evaluation_scope=complete_manifest`；同一报告的`source_pretrain_selection_scope=fixed_test_subset`保存checkpoint来源，不会把本次23040帧报告缩成720帧，也不重新计算best排名。各报告分别列出自己的帧数、实例数、空GT帧和无输出分母，不能把子集指标标成全量指标。

令每个可见GT实例i的Top1命中`h_i=1[有输出且所选姿态Q_inst>=0.4]`。当前B默认100点以下支持的实例、无合法观测法向、无输出实例的h_i=0，保留在原始实例分母。按frame内实例均值→scene内本次评价范围的有效frame均值→scene等权平均得到`InstTop1_INST20@0.4`；子集不混入未选帧，完整报告使用全部帧。空GT帧由显式声明计入覆盖，另计并排除于实例平均。另报实例微平均。[协议来源](../TRAIN_EVAL_REPRODUCE.md)

次级selected-Q使用相同宏平均层级：有输出取所选姿态Q_inst，no_pose取0；它用于同主指标时选模。另报**有输出条件下**所选Q的微平均及有效输出数，避免把两个均值混称。所有结果同时报告原始实例数、少支持/无normal/无输出计数及原因。

**teacher未知不能当Q=0。** 所选姿态或候选覆盖等指标所需query标签缺失/非法时，评价标为不完整并修复缓存，不能伪造负例或删除分母后宣告完成。no_pose的策略失败与有姿态但未知teacher的评价缺失是不同状态。

`Coverage_INST20@256`表示实例的固定最多256共同候选池中是否至少存在Q_inst≥0.4的query；另报“池内存在好候选时Top1选中率”，其分母仅为池内有好候选实例。实际K不足256时报告真实K，不复制候选。两个指标须用同一池统计，支持过滤和NMS前后分别报，不能混用分母。纯 SW 和 self 比例作本地解释项；本节 INST20 结果不能填入第 2 节官方族 R10/R20 六指标表。

## 5. 历史动作与最终测试

固定1676 train、399 val、87 test，继承group，不重新随机拆分。B-source100 没有受训练的 task 概率；可诊断其q对成败的排名关系，但不能用q填task BCE/Brier或以随机task头当微调前基线。B-target-ft 按任务概率报告AUROC、AP、BCE、Brier以及逐动作logit/概率、标签、姿态和身份。

始终列总动作数、支持有效数、正负数、各group数量和跳过原因。无几何关联不是动作失败，保留覆盖诊断。若另行开展内部模型对照，在相同有效交集进行并同时列各自覆盖；不悄悄改变每次验证分母。分组仅单类时AUROC记未定义。

399 val用于每100成功步选模及预注册的小对照；87 test在权重、排序规则、输入与标签版本冻结后一次性评价。目标87 test不用于选alpha、法向跨度、门控参数或学习率。当前记录是planned_pipeline_contact、actual_execution_verified=false，离线同动作排名改善不直接证明新候选的真实执行成功率改善。

## 6. B 必须单独解除的几何与数值风险

| 风险 | 样例/检查 | 必须保留的结果 |
|---|---|---|
| 法向噪声和边缘损失 | 固定真实平面、曲面、薄结构；有可靠参考才报角误差 | normal_valid率、原因、候选质量覆盖及标签分布；不能只报幸存像素误差 |
| 稀疏mask候选枯竭 | 细长/斜放实例、空mask、eligible不足 | 首轮/回填数、重复率、64/128前缀、米制间距和NMS后覆盖 |
| 投影径向/角向错槽 | 45°平面h=-6mm，18mm理想两侧实际rho约24.254558/12.254558mm；反向槽 | contact拒绝正确、raw负高度保留；拒绝不改teacher标签 |
| r=0与像素量化 | phase变化、18/22mm边界、重复source命中 | 中心mask相位无关、逐半径独立像素数/拒绝原因，不补虚假支持 |
| 缺测形状 | 连续半圈、交替缺测、跨71→0、全空 | coverage相同而gap不同；连续半圈(.5,.5)、交替(.5,1/72)、全空(0,1) |
| 归一化/空槽 | 非零LN affine、全空/部分空、不同query块 | LN仅跨每角16通道，层末原mask保持0，全空池化有限零值 |
| 中层query位置 | 单像素脉冲、同物体不同位置、fx≠fy | Fq来自各自L1/L2位置及正确梯度；stride4/8无half-pixel偏移 |
| 纹理实际分辨率 | 4mm带宽投影不足2原生像素的实例子集 | underresolved比例与该子集质量；放大图像不计新观测 |
| 坐标图和phase | 最小x/y分量切换面、备用+Z图、整数/连续角偏移 | 有限正交/有向轴不翻转；整数循环移位一致，连续phase的logit/Top1敏感性另报 |
| 输出与导出 | 严格NMS边界、近同分、不同ROI动态尺寸、FP32/FP16部署变体 | gate/mask、分数、Top1/NMS、原p/n、SE(3)和profile一致性 |

这些检查同时覆盖数学单元、真实图像读取组合与模型接线，不能拿NumPy通过替代GPU网络/teacher验证。阈值均来自既有合同；若需新验收容差，在运行前登记，不从测试结果反调通过标准。

## 7. 最小对照，避免展开参数笛卡尔积

当前执行B-source100与B-target-ft，公开pair留待协议确定。各路线独立执行，不将另一分支完成设为前置条件。追加对照只回答已经观察到的问题，每次仅改变一个因素并记录 variant、完整config与显式版本：

| 触发条件 | 允许的最小对照 | 保持不变 |
|---|---|---|
| teacher成本/内存阻碍默认训练 | 训练64对128 query | 同一256池及其前缀、140000实例预算、完整source-test最多256、同teacher及test_selected身份 |
| 历史动作对小偏移敏感且开发证据充分 | 默认关闭 vs 后续实现并预注册的2mm/2° pose-consistency | 一动作一次BCE、split、微调步数/回放；不复制源标签 |
| 需要检验混合排序是否有价值 | alpha=0 vs 预注册alpha=.5 | 同一冻结权重、候选、NMS；单独输出task概率 |
| 投影门控拒绝率/低分辨率子集异常 | 先分层报告、检查坐标/量化/支持；确需改阈值再建立独立配置 | 不在默认配置里静默改3mm、环半径或teacher |
| 工程部署要求降精度 | float64几何/FP32网络基线 vs明确部署精度 | 同权重、输入、ROI/padding、query及输出任务 |

不同时组合query预算、phase、法向、门控、alpha和正则形成大矩阵。新旧法向若作质量对照，必须各自重算精确p/n标签身份；这比较的是候选管线，不能说只消融网络。前三阶谐波属于A分支，不向B默认融合另增未经规范的特征。

## 8. 统一计时和导出报告

每实例batch1，50次预热后至少200个固定验证实例。报告model-only及instance端到端p50/p95和峰值显存；GPU阶段用CUDA events并同步，端到端使用明确墙钟同步边界。记录硬件、CPU线程、软件/driver、dtype、真实ROI/padded大小、对象数、raw有效像素数、实际K、chunk、缓存/拷贝及上游相机/分割是否计入。

分项至少包括：原图反投影、3×3平滑/±3差分及路径门控、候选回填、上下文encoder、native纹理stem、环位置生成/采样/门控/摘要、环网络/头、NMS/位姿。整帧多实例另报汇总，不将单实例model-only倒数称为机器人FPS。延迟比较要求相同硬件、候选预算和输出任务；2–5ms 是工程目标，缺失结果保持待测。公开整帧端到端计时还需计入 UOIS/UOAIS 前端和实例汇总；论文计时条件不匹配时不计算速度增益。

导出分别保留子图和组合差异；动态ROI保持真实padding合同。部署改几何精度时特别统计门控翻转及其Top1影响，不用平均分数误差掩盖离散决策变化。[后处理与导出边界](POSTPROCESSING.md)

## 9. 每次实验必须交付的记录

运行目录保存完整配置与环境、代码/checkpoint/manifest/teacher/precision身份、逐query/实例/动作预测、支持与排除统计、按scene/group聚合汇总、计时分项和失败样例索引。记录成功更新、成功监督实例、全部consumed draws和skip；另存固定selection帧ID、帧数、实例顺序、`selection_scope`和`selection_role`。训练后全量报告分别保存`evaluation_scope`与`source_pretrain_selection_scope`及实际固定best，不能把选择范围和评价范围合并为一个字段。可见输入和完整GT teacher的使用边界写入报告。

GitHub只同步必要汇总、配置和来源索引；完整资产/标签/权重留在数据与运行目录。公开对照的协议待定，先保留[来源参考](../PUBLIC_METHOD_COMPARISON.md)和[论文基线登记](../reference/paper_baselines.json)，不调用项目 `eval_r20` 冒充官方AP；第三方优先引用论文值，不要求重训。旧E5若做本地INST分析，须对它实际输出姿态以同一实例teacher重评，且仍不能写入B的任何实测行。

完成必要的E1/E2真实资产接线后执行E3/E4；E3固定test子集选择的best及其训练后全量报告均保留test_selected来源，E4另按目标399 val/87 test记录。E5公开协议待定，不构成E3/E4前置条件；E6实际导出与计时完成后才能陈述部署成本。真实新动作的业务成功率需要后续执行结果，不能从离线历史AUROC、合成法向噪声或局部卷积MAC直接推算。

## 10. 待协议核验的 MultiGraspNet 参考附录

[MultiGraspNet 原论文 v2](https://arxiv.org/html/2602.06504v2) Table I（PDF 第 5 页）Vacuum 部分报告 RealSense Top50 三列；没有报告 Top1。论文明确 100 train / 90 test 和前缀 k=1..50，但将 vacuum 成功阈值表述为 seal coefficient，尚不能从公开实现确认与官方 Seal × Wrench、完整碰撞和全部帧配置完全相同。[项目页](https://vandal-lab.github.io/multigraspnet-project/)的代码发布状态及核验日期见公共来源登记。

因此以下两对保留论文数值作为参考，协议证据补齐前不纳入正式增益结论，也不推算缺失 Top1。两张表彼此不计算差值。

### 10.1 B 与 MultiGraspNet-vac.

单任务 vacuum 变体；本路线后续候选方案为 `uois_no_gt_workspace` 与source-only评分权重，尚未执行，不使用 UISN 的 GT 前景工作区。该行的训练目标与 joint 行不同，不能混名引用。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | MultiGraspNet-vac. / 论文，尺寸待核验 | 28.26 | 36.39 | 3.60 | — | — | — |
| Similar | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | MultiGraspNet-vac. / 论文，尺寸待核验 | 31.37 | 41.63 | 3.88 | — | — | — |
| Novel | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | MultiGraspNet-vac. / 论文，尺寸待核验 | 8.07 | 10.03 | 0.48 | — | — | — |

### 10.2 B 与 MultiGraspNet joint

joint 行同时训练 parallel / vacuum，带额外平行夹爪监督；Table I 不启用 switching strategy。本路线后续候选方案为 `uois_no_gt_workspace` 与source-only评分权重，尚未执行，不使用 GT 前景工作区。即使后续 evaluator 核验通过，仍须保留额外监督说明。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | MultiGraspNet joint / 论文，尺寸待核验 | 28.07 | 35.81 | 3.62 | — | — | — |
| Similar | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | MultiGraspNet joint / 论文，尺寸待核验 | 31.22 | 41.65 | 3.53 | — | — | — |
| Novel | B / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | MultiGraspNet joint / 论文，尺寸待核验 | 7.38 | 8.92 | 0.46 | — | — | — |
