# A 实验方案：test选模、几何边界与公开方法参考

A模型及共同训练/推理链路已实现，13项CPU几何/模型测试通过；前后向使用显式注入测试backbone，wrapper通道测试mock外部加载，真实SpConv/GPU、完整source/target训练和公开AP尚未运行。本页把本轮可执行的实例/动作评价，与待定的论文比较协议分别标明；旧E5数值、数学参考和合成网络检查不能填入A实测行。

## 1. 当前执行策略与公开评价状态

| 项目 | 本轮执行 |
|---|---|
| 源训练 | scene 0–99全部100场景、全部256视角可抽样；从头训练，不再90/10开发后final refit |
| 预算 | 140000个成功受监督实例、8750次成功更新；128-query训练前缀 |
| best选择 | 每875成功步在source test scene 100–189完整256视角上评价，最多256-query；保留best与预算终点last |
| 成绩身份 | `test_selected` / `independent_test_estimate=false`；源test只评价、不反传，但已参与模型选择 |
| 目标域 | 1676 train、399 val选best、87 test冻结后评价；保留既有group和规划动作语义 |
| 本轮评价 | INST20实例Top1/selected-Q/coverage，以及目标固定动作AUROC/AP/BCE/Brier |
| 公开方法评价 | `deferred`；只参数化半径的方案待定，未实现、不作为开训前置门禁 |
| 论文表 | 保留各论文原始数值及来源；A公开结果待测，不能填入本轮实例指标 |

source test完整集为RealSense Seen 100–129、Similar 130–159、Novel 160–189，各7680帧、总23040帧。当前采用GT实例输入做实例物理选模；它与预测实例前端参与的全景公开AP并非同一实验。`test_selected`成绩只能按该身份报告，不能包装成盲测或“除吸盘半径外完全一致”的公开比较。

用户要求的逐方法对照保留在第2节及后文MultiGraspNet两组表中，每表只有A与一个参考，不比较参考之间。未来恢复公开评价时，须逐pair冻结测试集/相机/指标、物理口径、前端和GT工作区权限；未披露的论文训练细节标unreported，未核实项保持unverified。半径-only方案及其R10对齐检查都仍待决定，不能提前宣称已匹配。

论文表的六项AP继续按原来源解释：`AP50`为阈值{0.2,0.4,0.6,0.8}与前缀k=1..50精度的平均，`AP0.4_50`/`AP0.8_50`固定阈值，`AP1`及其阈值列只取首候选；使用百分数。它们不是实例Top1、PR面积或仅P@50。公式与证据见[已有公开比较参考](../PUBLIC_METHOD_COMPARISON.md)及[论文基线JSON](../reference/paper_baselines.json)，本次没有重新运行任何论文方法或重新取得其数值。

后续前端候选方案仍需按方法审查：无GT工作区UOIS，以及为匹配UISN单独声明的UOIS/UOAIS加GT联合前景AABB。UISN权限不自动扩展到S1B或Diffusion；同名前端也不证明权重、工作区和候选一致。以下前端名仅保留为候选方案，不表示公开推理/导出已实现或协议已冻结。

## 2. 本路线与各公开方法的独立比较

每张表只比较本路线与一个参考方法。`待测` 表示本方法尚未执行，`—` 表示论文没有报告。以下论文值来自各自标注的表，不是本仓库复现值；完整出处和核验状态以[公共来源登记](../PUBLIC_METHOD_COMPARISON.md)为准。

### 2.1 A 与 SuctionNet / S1B

[SuctionNet-1Billion 原论文 v2](https://arxiv.org/pdf/2103.12311v2)，Table IV，PDF 第 6 页，取 Kinect/RealSense 斜杠后的 RealSense 数值。固定采用 2021-10-29 v2 主表，不混用旧版本或消融表的小数差异。原方法处理整帧 RGB-D、不需要 GT 实例；本pair候选前端为 `uois_no_gt_workspace`，尚待协议冻结，不使用 GT 前景工作区、GT 实例裁剪或补回漏检。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | SuctionNet / S1B / 论文 R10 | 28.31 | 38.56 | 3.41 | 43.98 | 60.81 | 7.79 |
| Similar | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | SuctionNet / S1B / 论文 R10 | 26.64 | 35.34 | 3.42 | 36.59 | 49.30 | 6.54 |
| Novel | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | SuctionNet / S1B / 论文 R10 | 8.23 | 10.29 | 0.35 | 14.32 | 19.95 | 0.38 |

### 2.2 A 与 Diffusion-Suction

[Diffusion-Suction 原论文 v2](https://arxiv.org/html/2502.07238v2)，Table I，PDF 第 7 页，取 RealSense 列；Table II 的 Parcel-Suction 结果不属于此比较。论文声明沿用 SuctionNet 在线 Seal × Wrench 协议；Table I 的完整训练/验证 recipe、RGB/实例过滤和 checkpoint 配置未逐项核实，不能从 Parcel 训练脚本推定。本pair候选前端暂记 `uois_no_gt_workspace`，尚未执行；Diffusion 原始工作区仍标“未核实”，不能据此宣称该项已匹配，也不继承 UISN 的 GT 前景工作区权限。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | Diffusion-Suction / 论文，官方默认R10 | 31.90 | 42.78 | 4.79 | 54.79 | 68.59 | 21.03 |
| Similar | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | Diffusion-Suction / 论文，官方默认R10 | 31.59 | 41.38 | 4.27 | 47.00 | 59.36 | 16.08 |
| Novel | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | Diffusion-Suction / 论文，官方默认R10 | 10.00 | 12.32 | 0.76 | 14.14 | 17.67 | 2.42 |

### 2.3 A 与 UISN (UOIS)

[UISN 原论文](https://doi.org/10.1109/LRA.2024.3385609)，Table I，PDF 第 6 页 / 刊页 4939，按表注取 RealSense 数值。该pair候选匹配方案为 `uois_uisn_gt_aabb20`，尚未执行：冻结 UOIS 预测分割，另按作者 `get_workspace_mask(..., outlier=.02)` 使用 GT 前景联合 AABB 工作区。只读取前景联合范围，不读取 GT 实例身份，也不用 GT 实例 mask 替换预测分割。该预测不能复用 `uois_no_gt_workspace` 的结果。原文转录与本轮全文访问状态见公共来源登记；本轮未能重新读取全文，使用仓库已登记的逐表转录。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | UISN (UOIS) / 论文，官方默认R10 | 37.98 | 49.83 | 9.99 | 63.88 | 76.64 | 31.72 |
| Similar | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | UISN (UOIS) / 论文，官方默认R10 | 37.83 | 49.35 | 8.89 | 48.68 | 60.01 | 16.72 |
| Novel | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | UISN (UOIS) / 论文，官方默认R10 | 9.47 | 12.63 | 1.17 | 11.83 | 15.47 | 2.76 |

### 2.4 A 与 UISN (UOAIS)

来源仍为 [UISN 原论文](https://doi.org/10.1109/LRA.2024.3385609) Table I 的 UOAIS 配置。这是独立pair，候选匹配方案为 `uoais_uisn_gt_aabb20`，尚未执行：同一冻结 source-only 评分权重，冻结 UOAIS 预测分割，加作者 `get_workspace_mask(..., outlier=.02)` 的 GT 前景联合 AABB 工作区；不读取 GT 实例身份。按该 profile 重新生成整帧预测，不复制任一 UOIS profile 的结果，不额外训练评分模型。前端、工作区和成本随结果保存。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | UISN (UOAIS) / 论文，官方默认R10 | 38.27 | 50.57 | 10.56 | 66.11 | 78.35 | 35.29 |
| Similar | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | UISN (UOAIS) / 论文，官方默认R10 | 36.54 | 47.77 | 8.74 | 49.20 | 61.45 | 17.20 |
| Novel | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | UISN (UOAIS) / 论文，官方默认R10 | 9.08 | 12.05 | 1.04 | 11.94 | 15.40 | 2.07 |

## 3. 执行顺序与停止条件

| 阶段 | 最少工作 | 证据和当前状态 |
|---|---|---|
| E0 数学/CPU模型回归 | 数学/schema与A几何、接触、五头、BN检查 | 既有129项参考；A13项CPU测试通过，模型使用测试backbone |
| E1 真实数据和teacher | 固定约64–128个实例作成本/同姿态核验 | 代码已接线；真实标定、S/W、self和成本仍待测 |
| E2 真实SpConv短链路 | 实际稀疏编码、少量反向、保存恢复、历史异轴查询 | GPU/spconv、loss/梯度、BN/参数组和下一批恢复；尚未运行 |
| E3 A-source100 | 全100场景train，140000实例/8750更新，每875步完整source test选best | test_selected best、预算last、完整分母及逐实例结果 |
| E4 A-target-ft | E3 best初始化，200+800更新、2 target+2 source回放 | 399 val按AUROC/BCE/早步选best，最终87 test |
| E5 A公开方法比较 | 每方法单独pair，论文值保留 | 协议deferred，不阻塞E3/E4；A六指标待测 |
| E6 实际部署与计时 | 固定权重/硬件/输入边界，真实CPU几何+GPU模型 | p50/p95、显存、raw几何/稀疏编码成本；尚未运行 |

发现坐标、标签或身份不一致时先修复对应阶段，不能用继续训练掩盖。CPU注入backbone测试与少量成功更新只验证对应接线，不能作为SpConv实际运行、泛化或时延证据。[训练执行细节](TRAINING_PLAN.md)

E1需报告normal有效率、候选覆盖及主干关联缺失；有可靠参考时才报告法向角误差。self验证保留两个分项，与独立外部项合并重建旧总C。Seal核对使用完全相同p/n，改变轴导致标签下降不自动是teacher错误。E2记录singleton BN fallback次数，正常源BN路径保持更新；微调验证全部BN统计/affine冻结。

## 4. 本地实例与目标任务指标及分母

本节源指标用于完整source test选best，按Seen/Similar/Novel同时报告，并明确 `test_selected`。无反向传播不等于独立测试。目标域仍为1676 train/399 val/87 test，399 val选微调best，87 test待权重/规则冻结后评价，源预训练的选模来源继续保留。

| 任务 | 指标 | 固定解释 |
|---|---|---|
| 实例物理选择 | `InstTop1_INST20@0.4` | 所选姿态同teacher `Q_inst>=0.4`记命中，无输出记未命中 |
| 次级选择质量 | 所选Q_inst宏平均 | 与主指标同层级聚合；no_pose贡献0；用于预训练选模第二顺位 |
| 条件质量诊断 | 有输出条件下Q_inst微平均、SW、self比例 | 仅在实际输出姿态上取微平均，同时附输出数与无输出数 |
| 候选覆盖 | `Coverage_INST20@256` | 同一固定候选池中至少存在一个Q_inst>=.4 |
| 有好候选时选中率 | 条件Top1命中率 | 分母只包含同池存在好候选的实例 |
| 历史动作 | AUROC、AP、BCE、Brier | 保存总数/有效数/正负数；BCE/Brier仅使用task概率 |
| 成本 | model-only及instance端到端p50/p95 | batch1、固定输入边界、实际M/N/K、峰值显存 |

实例主指标先在每frame内对可见GT实例平均，再对同scene的有效frame平均，最后scene等权平均；另外报告实例微平均。少于100点、无法向或无输出实例仍在主分母；没有GT实例的frame单独计数，不进入平均。

次级所选Q_inst使用相同的三级宏平均，no_pose为0；有输出条件下Q微平均单列，不能替代次级选模指标。以上细化与共同协议同源，不更改Q_inst物理含义。teacher未知不是质量0：若所选姿态或Coverage所需query标签未知/非法，相应评价不完整，先修复标签；不得补负例或删分母后形成看似完整的结果。

候选覆盖与选择率必须用同一个固定池，分别保存支持过滤前后、NMS前后及实际K；不得用无输出率代替好候选覆盖。GT阈值.4不等于推理筛选阈值。历史几何不支持动作报缺失覆盖，不当作失败；若另行开展内部模型对照，使用固定有效动作交集并同时报告各自覆盖，不能以不同子集的 AUROC 直接下结论。单类group的AUROC未定义。

本节为原生实例/动作指标，不转换为论文全景AP。公开半径-only评价协议待定；旧项目R20、INST20或纯SW结果不能进入第2节公开比较表。source test-selected也不包装成盲测。[公开比较参考状态](../PUBLIC_METHOD_COMPARISON.md)

## 5. A 特有风险的诊断与最小对照

| 风险 | 先做的诊断 | 仅在需要时追加的对照 |
|---|---|---|
| 主干2048覆盖不足 | d1/d3分位数、10mm有效关联率、无主干组件比例，按尺寸/倾斜分层 | 明确命名4096或8192预算；保留相同FPS前缀与query，计入额外成本 |
| 水平吸取角向信息缺口 | 相同高度multiset不同排列、恒高缺角、少于7方向支撑、高阶盲区 | 对中心化RMS/谐波贡献做预注册单因素研究，声明配置/输入身份变化 |
| 法向噪声和边缘损失 | v3 normal有效率、可靠参考角误差、候选质量分布 | 旧±1仅为独立命名诊断；改变n须重算标签，不与默认缓存混用 |
| 历史规划落点偏差 | 支持距离分布、按group错误、原轴与观察轴差异 | 后续实现并预注册的2mm/2°一致性正则，当前关闭；只用399 val决定 |
| 训练query预算成本 | teacher秒/query、缓存命中、128前缀覆盖与验证质量 | 单独命名64-query训练配置；验证仍固定最多256 |

这些是由观测风险触发的候选研究，不是首轮必须全跑。任何改模型/法向/主干预算的对照均记录精确配置和额外计算；同一对照不同时改变多项再将收益归于单个模块。目标87条test不参与这些选择；源test用于best的成绩继续标test_selected。

## 6. 性能测量与工程瓶颈

每物体batch1，先50次预热，再测至少200个固定验证实例；分别报model-only和instance端到端p50/p95、峰值显存及实际query数量。GPU模型时间使用CUDA events并同步，端到端墙钟定义同步边界；不得把异步launch时间当执行时间。

A至少分项记录：原深度/XYZ处理、3×3平滑+±3法向/路径门控、候选格内回填、4连通组件、2mm代表点/FPS、特征关联、raw区域查询与lip谐波、稀疏网络、逐query头、CPU/GPU拷贝和NMS。记录原始点M、主干N、query K及实例尺寸；整帧多实例另报真实汇总。说明相机采集与上游分割是否包含在端到端边界内。

15–30ms 是既有工程目标，不是本方案已测值。延迟比较需相同硬件、候选预算、dtype 和计时边界；论文未提供可匹配计时条件时只记录来源，不能据此给出速度增益。若生产 FP32 几何替代 float64，先验证边界一致性并单独标 precision-contract。公开整帧端到端计时还需计入登记的 UOIS/UOAIS 前端与实例汇总。

## 7. 实验记录与当前复现

每个真实run保存配置/环境/代码/数据/标签/标定身份、成功及消费预算、参数组、checkpoint、逐实例/query/动作分数、排除和no_pose原因、各项有效分母与聚合方式、计时原始样本及汇总。结果表未执行项写“未运行”，不填推测数值；既有 E5 的 AUROC、AP 或其他改善不能归给 A；A无须等待B完成才能执行本轮源训练和目标适配，公开pair须等待评价协议确定。

当前在仓库根目录可运行A CPU测试（显式测试backbone）与数学参考：

```bash
python -m pytest -q tests/test_instance_route_a.py
python docs/instance_routes/reference/geometry_contract.py selftest
python docs/instance_routes/reference/revision_contract.py selftest
python docs/instance_routes/reference/sampling_contract.py selftest
python docs/instance_routes/reference/contact_boundary_contract.py self-test
```

继承验证详情见 [VALIDATION.json](../reference/VALIDATION.json)。生产数据builder、teacher、模型和新训练/原生评价入口已[实现](README.md)；真实SpConv、完整资产重算和正式训练尚未运行。模型实验完成后应把少量汇总、输入身份和可复核路径提交本分支；大型资产与checkpoint留在外部运行目录。

## 8. 待协议核验的 MultiGraspNet 参考附录

[MultiGraspNet 原论文 v2](https://arxiv.org/html/2602.06504v2) Table I（PDF 第 5 页）Vacuum 部分报告 RealSense Top50 三列；没有报告 Top1。论文明确 100 train / 90 test 和前缀 k=1..50，但将 vacuum 成功阈值表述为 seal coefficient，尚不能从公开实现确认与官方 Seal × Wrench、完整碰撞和全部帧配置完全相同。[项目页](https://vandal-lab.github.io/multigraspnet-project/)的代码发布状态及核验日期见公共来源登记。

因此以下两对保留论文数值作为参考，协议证据补齐前不纳入正式增益结论，也不推算缺失 Top1。两张表彼此不计算差值。

### 8.1 A 与 MultiGraspNet-vac.

单任务 vacuum 变体；本路线后续候选方案为 `uois_no_gt_workspace` 与source-only评分权重，尚未执行，不使用 UISN 的 GT 前景工作区。该行的训练目标与 joint 行不同，不能混名引用。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | MultiGraspNet-vac. / 论文，尺寸待核验 | 28.26 | 36.39 | 3.60 | — | — | — |
| Similar | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | MultiGraspNet-vac. / 论文，尺寸待核验 | 31.37 | 41.63 | 3.88 | — | — | — |
| Novel | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | MultiGraspNet-vac. / 论文，尺寸待核验 | 8.07 | 10.03 | 0.48 | — | — | — |

### 8.2 A 与 MultiGraspNet joint

joint 行同时训练 parallel / vacuum，带额外平行夹爪监督；Table I 不启用 switching strategy。本路线后续候选方案为 `uois_no_gt_workspace` 与source-only评分权重，尚未执行，不使用 GT 前景工作区。即使后续 evaluator 核验通过，仍须保留额外监督说明。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | MultiGraspNet joint / 论文，尺寸待核验 | 28.07 | 35.81 | 3.62 | — | — | — |
| Similar | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | MultiGraspNet joint / 论文，尺寸待核验 | 31.22 | 41.65 | 3.53 | — | — | — |
| Novel | A / 公开协议待定 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | MultiGraspNet joint / 论文，尺寸待核验 | 7.38 | 8.92 | 0.46 | — | — | — |
