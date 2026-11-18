# A 实验方案：公开方法比较、实例质量与失败归因

本页规划 A 的公开方法比较、单实例物理选择、历史动作成败预测和工程成本。公开比较执行[公共规范](../PUBLIC_METHOD_COMPARISON.md)；本地开发沿用[共同训练评价合同](../TRAIN_EVAL_REPRODUCE.md)、[数据合同](../DATA_AND_LABELS.md)和[v3 边界复核](../BOUNDARY_REVIEW.md)。当前没有 A 的新模型实测结果，论文数值仅出现在明确标记的参考行。

## 1. 公开方法比较的固定协议

正式对外比较使用 `A-public20`（公共运行类型 `ours-public20`），执行[公开方法比较规范](../PUBLIC_METHOD_COMPARISON.md)，数值及来源登记在[论文基线 JSON](../reference/paper_baselines.json)。下文的 INST20 实例指标、目标域历史动作指标和几何检查服务于本路线开发，不能填入公开 SuctionNet AP 表。

| 项目 | 本路线公开比较的固定条件 |
|---|---|
| 数据与模型 | 仅源域训练；先在 90 train / 10 dev 开发阶段锁定配置，再以登记的初始化重启模型和优化器，用官方全部 100 个训练场景做 final-refit；不得使用产线微调权重 |
| 预算与选模 | 在 final-refit 前冻结预算，默认 140000 个有效实例 / 8750 次成功更新；完成固定预算后使用 `last`，不按公开测试结果挑 checkpoint |
| 测试集与相机 | RealSense；Seen 100–129、Similar 130–159、Novel 160–189；每场景全部 256 帧，即各 7680 帧、合计 23040 帧；不以抽帧或可见实例子集代替 |
| 推理前端与工作区 | S1B / Diffusion / MultiGraspNet 使用 `uois_no_gt_workspace`；UISN (UOIS) 使用 `uois_uisn_gt_aabb20`；UISN (UOAIS) 使用 `uoais_uisn_gt_aabb20`；三个 profile 共用同一冻结 public 评分模型 |
| GT 使用边界 | 仅 UISN 两个 pair 为匹配作者流程使用 GT 前景联合 AABB 工作区，按 `get_workspace_mask(..., outlier=.02)`；只使用前景联合范围，不读取 GT 实例身份，不以 GT 实例掩码替换预测分割或补回漏检；S1B / Diffusion / MultiGraspNet 不获得该工作区权限 |
| 候选与排序 | 收集整帧各预测实例候选，用源域受监督的物理质量 `q_inst` 排序；不使用未经目标训练的 task 头，不调用产线混合排序；分割漏检/无姿态保留为空预测 |
| evaluator | 固定官方族 `sn-paper-radius-v1`：先通过 `radius=10 mm` 的官方等价门禁，再以相同实现评价本方法 `radius=20 mm`；参考方法采用论文的官方默认R10口径；MultiGraspNet尺寸仍待核验 |
| 唯一允许的评估变化 | 吸盘接触/碰撞半径及官方公式内直接由半径导出的量；NMS、关联、每物体上限、环宽、碰撞轴向区间、阈值、力矩算法和聚合保持公共规范锁定值 |
| 禁止混用 | 项目 `eval_r20`、`INST20` 或纯 `SW` 评价不能冒充该官方族六指标；旧 E5 结果不能填入 A 行 |

官方等价门禁使用相同预测文件，核对各帧候选保留/排序、关联、碰撞、物理分和聚合输出；空预测、候选不足 50、阈值边界等也须覆盖。合法空帧按预声明全零扩展独立验收，上游在此没有可直接数值对齐的输出；已知轴极点非有限物理质量按官方阈值未命中保留排名。未经登记的数值修正、异常跳帧或清理参考实现行为，都不能静默进入主协议。门禁通过前，本页公开比较表保持待测。

六指标均使用百分数：`AP50` 是阈值集合 {0.2, 0.4, 0.6, 0.8} 下、前缀 `k=1..50` 的精度平均；`AP0.4_50` / `AP0.8_50` 固定相应阈值，`AP1` 及其阈值列只取首个候选。判正用官方实现的 `>=`，不足 k 的分母仍为 k。它不是实例 Top-1、PR 曲线面积或仅 `Precision@50`。

每个 pair 都保留该论文自身的版本、相机、输入/前端、训练和 evaluator 证据状态。论文缺失的具体训练、分割或后处理配置写“未核实”；采用统一测试集并不能自动证明所有条件只差半径。只有公共规范的对应门禁通过，才填写本方法实测值和本方法相对该参考的差值。当前不计算任何增益，不建立参考方法排名或参考之间的差值。

同一冻结 public 评分模型生成三份 profile 独立预测：`uois_no_gt_workspace`、`uois_uisn_gt_aabb20`、`uoais_uisn_gt_aabb20`。前端模型及工作区完全相同的 pair 才可复用预测；只更换这些推理配置，不额外训练评分模型。UISN 的 GT 前景工作区不是 S1B 的共同步骤；Diffusion 原论文/实现的具体工作区尚未核实，不由 UISN 配置推定。

可以对每个 profile 的同一份冻结预测另用 R10 evaluator 评价，作为尺寸变化的附加解释；这不新增训练，不替代 R20 主行，也不等于专门训练的 R10 模型。

## 2. 本路线与各公开方法的独立比较

每张表只比较本路线与一个参考方法。`待测` 表示本方法尚未执行，`—` 表示论文没有报告。以下论文值来自各自标注的表，不是本仓库复现值；完整出处和核验状态以[公共来源登记](../PUBLIC_METHOD_COMPARISON.md)为准。

### 2.1 A 与 SuctionNet / S1B

[SuctionNet-1Billion 原论文 v2](https://arxiv.org/pdf/2103.12311v2)，Table IV，PDF 第 6 页，取 Kinect/RealSense 斜杠后的 RealSense 数值。固定采用 2021-10-29 v2 主表，不混用旧版本或消融表的小数差异。原方法处理整帧 RGB-D、不需要 GT 实例；本 pair 使用 `uois_no_gt_workspace`，不使用 GT 前景工作区、GT 实例裁剪或补回漏检。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | A-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | SuctionNet / S1B / 论文 R10 | 28.31 | 38.56 | 3.41 | 43.98 | 60.81 | 7.79 |
| Similar | A-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | SuctionNet / S1B / 论文 R10 | 26.64 | 35.34 | 3.42 | 36.59 | 49.30 | 6.54 |
| Novel | A-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | SuctionNet / S1B / 论文 R10 | 8.23 | 10.29 | 0.35 | 14.32 | 19.95 | 0.38 |

### 2.2 A 与 Diffusion-Suction

[Diffusion-Suction 原论文 v2](https://arxiv.org/html/2502.07238v2)，Table I，PDF 第 7 页，取 RealSense 列；Table II 的 Parcel-Suction 结果不属于此比较。论文声明沿用 SuctionNet 在线 Seal × Wrench 协议；Table I 的完整训练/验证 recipe、RGB/实例过滤和 checkpoint 配置未逐项核实，不能从 Parcel 训练脚本推定。本 pair 暂使用 `uois_no_gt_workspace`；Diffusion 原始工作区仍标“未核实”，不能据此宣称该项已匹配，也不继承 UISN 的 GT 前景工作区权限。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | A-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | Diffusion-Suction / 论文，官方默认R10 | 31.90 | 42.78 | 4.79 | 54.79 | 68.59 | 21.03 |
| Similar | A-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | Diffusion-Suction / 论文，官方默认R10 | 31.59 | 41.38 | 4.27 | 47.00 | 59.36 | 16.08 |
| Novel | A-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | Diffusion-Suction / 论文，官方默认R10 | 10.00 | 12.32 | 0.76 | 14.14 | 17.67 | 2.42 |

### 2.3 A 与 UISN (UOIS)

[UISN 原论文](https://doi.org/10.1109/LRA.2024.3385609)，Table I，PDF 第 6 页 / 刊页 4939，按表注取 RealSense 数值。该 pair 使用 `uois_uisn_gt_aabb20`：冻结 UOIS 预测分割，另按作者 `get_workspace_mask(..., outlier=.02)` 使用 GT 前景联合 AABB 工作区。只读取前景联合范围，不读取 GT 实例身份，也不用 GT 实例 mask 替换预测分割。该预测不能复用 `uois_no_gt_workspace` 的结果。原文转录与本轮全文访问状态见公共来源登记；本轮未能重新读取全文，使用仓库已登记的逐表转录。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | A-public20 / uois_uisn_gt_aabb20 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | UISN (UOIS) / 论文，官方默认R10 | 37.98 | 49.83 | 9.99 | 63.88 | 76.64 | 31.72 |
| Similar | A-public20 / uois_uisn_gt_aabb20 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | UISN (UOIS) / 论文，官方默认R10 | 37.83 | 49.35 | 8.89 | 48.68 | 60.01 | 16.72 |
| Novel | A-public20 / uois_uisn_gt_aabb20 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | UISN (UOIS) / 论文，官方默认R10 | 9.47 | 12.63 | 1.17 | 11.83 | 15.47 | 2.76 |

### 2.4 A 与 UISN (UOAIS)

来源仍为 [UISN 原论文](https://doi.org/10.1109/LRA.2024.3385609) Table I 的 UOAIS 配置。这是独立 pair，使用 `uoais_uisn_gt_aabb20`：同一冻结 source-only 评分权重，冻结 UOAIS 预测分割，加作者 `get_workspace_mask(..., outlier=.02)` 的 GT 前景联合 AABB 工作区；不读取 GT 实例身份。按该 profile 重新生成整帧预测，不复制任一 UOIS profile 的结果，不额外训练评分模型。前端、工作区和成本随结果保存。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | A-public20 / uoais_uisn_gt_aabb20 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | UISN (UOAIS) / 论文，官方默认R10 | 38.27 | 50.57 | 10.56 | 66.11 | 78.35 | 35.29 |
| Similar | A-public20 / uoais_uisn_gt_aabb20 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | UISN (UOAIS) / 论文，官方默认R10 | 36.54 | 47.77 | 8.74 | 49.20 | 61.45 | 17.20 |
| Novel | A-public20 / uoais_uisn_gt_aabb20 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | UISN (UOAIS) / 论文，官方默认R10 | 9.08 | 12.05 | 1.04 | 11.94 | 15.40 | 2.07 |

## 3. 执行顺序与停止条件

| 阶段 | 最少工作 | 产物与进入下一阶段的条件 |
|---|---|---|
| E0 数学/schema回归 | 继承四组参考检查，核对版本与有效性语义 | 129项参考记录及源码身份；不代表生产模型通过 |
| E1 真实数据和teacher | 固定约64–128个实例作成本/同姿态核验 | 标定回读、S/W对齐、self拆分对齐、缓存身份与成本报告 |
| E2 A短链路 | 真实实例编码/评分、少量反向、保存恢复、历史异轴查询 | 实际loss/梯度、BN/参数组、下一批恢复、p/n保持报告 |
| E3-N 本地开发预训练 | 90 train / 10 dev；140000 成功实例暴露、8750 更新、10 次固定 val | best/last、逐 query 预测与 INST20 曲线；仅服务本地选模 |
| E3-P 公开比较 final-refit | 配方冻结后全 100 训练场景，固定预算，source-only | A-public20 的 last、完整训练身份；不使用产线成败数据 |
| E4-N 目标域适配 | 从 E3-N best 初始化，200+800 更新，399 开发动作每100步评价 | 按 AUROC/BCE/早步冻结 best 及产线 ranking 模式；不用于论文 AP 主表 |
| E5-P 公开方法比较 | R10 等价门禁后，A-public20 覆盖 RealSense 全部 23040 帧 | 六指标、逐帧输出、完整分母、前端/evaluator 身份；分别填入对应 pair |
| E5-N 本地冻结测试与计时 | INST20 三源域子集；目标 87 动作一次；固定硬件 | 逐实例/动作输出、失败原因及 p50/p95；与公开 AP 独立 |

发现坐标、标签或身份不一致时先修复对应阶段，不能用继续训练掩盖。正式实现的损失下降与少量更新只能验证链路，不是泛化性能证据。[训练执行细节](TRAINING_PLAN.md)

E1需同时报告normal有效率、候选覆盖及主干关联缺失；有可靠参考时才报告法向角误差。self验证保留 `C_seal_self` 和 `C_target_envelope`，与独立外部项合并重建旧总C。Seal核对须用完全相同p/n；改变轴导致标签下降不自动是teacher错误。

## 4. 本地实例与目标任务指标及分母

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

本节全部为本地开发指标。第三方六指标统一执行第 1 节 `sn-paper-radius-v1`；旧项目 R20 / INST20 / 纯 SW 结果即使保留历史分析，也不能进入第 2 节公开比较表。[公开方法比较规范](../PUBLIC_METHOD_COMPARISON.md)

## 5. A 特有风险的诊断与最小对照

| 风险 | 先做的诊断 | 仅在需要时追加的对照 |
|---|---|---|
| 主干2048覆盖不足 | d1/d3分位数、10mm有效关联率、无主干组件比例，按尺寸/倾斜分层 | 明确命名4096或8192预算；保留相同FPS前缀与query，计入额外成本 |
| 水平吸取角向信息缺口 | 相同高度multiset不同排列、恒高缺角、少于7方向支撑、高阶盲区 | 对中心化RMS/谐波贡献做预注册单因素研究，声明配置/输入身份变化 |
| 法向噪声和边缘损失 | v3 normal有效率、可靠参考角误差、候选质量分布 | 旧±1仅为独立命名诊断；改变n须重算标签，不与默认缓存混用 |
| 历史规划落点偏差 | 支持距离分布、按group错误、原轴与观察轴差异 | 默认关闭的2mm/2°一致性正则；只用399开发动作决定 |
| 训练query预算成本 | teacher秒/query、缓存命中、128前缀覆盖与验证质量 | 单独命名64-query训练配置；验证仍固定最多256 |

这些是由观测风险触发的候选研究，不是首轮必须全跑。任何改模型/法向/主干预算的对照均记录精确配置和额外计算；同一对照不同时改变多项再将收益归于单个模块。87条test不参与这些选择。

## 6. 性能测量与工程瓶颈

每物体batch1，先50次预热，再测至少200个固定验证实例；分别报model-only和instance端到端p50/p95、峰值显存及实际query数量。GPU模型时间使用CUDA events并同步，端到端墙钟定义同步边界；不得把异步launch时间当执行时间。

A至少分项记录：原深度/XYZ处理、3×3平滑+±3法向/路径门控、候选格内回填、4连通组件、2mm代表点/FPS、特征关联、raw区域查询与lip谐波、稀疏网络、逐query头、CPU/GPU拷贝和NMS。记录原始点M、主干N、query K及实例尺寸；整帧多实例另报真实汇总。说明相机采集与上游分割是否包含在端到端边界内。

15–30ms 是既有工程目标，不是本方案已测值。延迟比较需相同硬件、候选预算、dtype 和计时边界；论文未提供可匹配计时条件时只记录来源，不能据此给出速度增益。若生产 FP32 几何替代 float64，先验证边界一致性并单独标 precision-contract。公开整帧端到端计时还需计入登记的 UOIS/UOAIS 前端与实例汇总。

## 7. 实验记录与当前复现

每个真实run保存配置/环境/代码/数据/标签/标定身份、成功及消费预算、参数组、checkpoint、逐实例/query/动作分数、排除和no_pose原因、各项有效分母与聚合方式、计时原始样本及汇总。结果表未执行项写“未运行”，不填推测数值；既有 E5 的 AUROC、AP 或其他改善不能归给 A；A 不需要等待 B 完成才执行自己的公开 pair。

当前在仓库根目录可运行：

```bash
python docs/instance_routes/reference/geometry_contract.py selftest
python docs/instance_routes/reference/revision_contract.py selftest
python docs/instance_routes/reference/sampling_contract.py selftest
python docs/instance_routes/reference/contact_boundary_contract.py self-test
```

继承验证详情见 [VALIDATION.json](../reference/VALIDATION.json)。生产数据builder、teacher、模型和新训练/评价入口尚待[分阶段实现](README.md)。模型实验完成后应把少量汇总、输入身份和可复核路径提交本分支；大型资产与checkpoint留在外部运行目录。

## 8. 待协议核验的 MultiGraspNet 参考附录

[MultiGraspNet 原论文 v2](https://arxiv.org/html/2602.06504v2) Table I（PDF 第 5 页）Vacuum 部分报告 RealSense Top50 三列；没有报告 Top1。论文明确 100 train / 90 test 和前缀 k=1..50，但将 vacuum 成功阈值表述为 seal coefficient，尚不能从公开实现确认与官方 Seal × Wrench、完整碰撞和全部帧配置完全相同。[项目页](https://vandal-lab.github.io/multigraspnet-project/)的代码发布状态及核验日期见公共来源登记。

因此以下两对保留论文数值作为参考，协议证据补齐前不纳入正式增益结论，也不推算缺失 Top1。两张表彼此不计算差值。

### 8.1 A 与 MultiGraspNet-vac.

单任务 vacuum 变体；本路线使用 `uois_no_gt_workspace` 和同一 source-only 评分权重，不使用 UISN 的 GT 前景工作区。该行的训练目标与 joint 行不同，不能混名引用。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | A-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | MultiGraspNet-vac. / 论文，尺寸待核验 | 28.26 | 36.39 | 3.60 | — | — | — |
| Similar | A-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | MultiGraspNet-vac. / 论文，尺寸待核验 | 31.37 | 41.63 | 3.88 | — | — | — |
| Novel | A-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | MultiGraspNet-vac. / 论文，尺寸待核验 | 8.07 | 10.03 | 0.48 | — | — | — |

### 8.2 A 与 MultiGraspNet joint

joint 行同时训练 parallel / vacuum，带额外平行夹爪监督；Table I 不启用 switching strategy。本路线使用 `uois_no_gt_workspace` 和同一 source-only 评分权重，不使用 GT 前景工作区。即使后续 evaluator 核验通过，仍须保留额外监督说明。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | A-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | MultiGraspNet joint / 论文，尺寸待核验 | 28.07 | 35.81 | 3.62 | — | — | — |
| Similar | A-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | MultiGraspNet joint / 论文，尺寸待核验 | 31.22 | 41.65 | 3.53 | — | — | — |
| Novel | A-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | MultiGraspNet joint / 论文，尺寸待核验 | 7.38 | 8.92 | 0.46 | — | — | — |
