# B 实验方案：公开方法比较、几何边界与部署成本

本页规划 B 的公开方法比较、单实例物理选择、历史动作评估和实际部署成本。公开比较执行[公共规范](../PUBLIC_METHOD_COMPARISON.md)；本地开发遵循[共同评价合同](../TRAIN_EVAL_REPRODUCE.md)、[数据准备](DATA_PREPARATION.md)和[训练方案](TRAINING_PLAN.md)。当前只有既有数学参考结果；论文数值仅写入参考行，旧 E5 和合成检查不能冒充 B 实测。

## 1. 公开方法比较的固定协议

正式对外比较使用 `B-public20`（公共运行类型 `ours-public20`），执行[公开方法比较规范](../PUBLIC_METHOD_COMPARISON.md)，数值及来源登记在[论文基线 JSON](../reference/paper_baselines.json)。下文的 INST20 实例指标、目标域历史动作指标和几何检查服务于本路线开发，不能填入公开 SuctionNet AP 表。

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
| 禁止混用 | 项目 `eval_r20`、`INST20` 或纯 `SW` 评价不能冒充该官方族六指标；旧 E5 结果不能填入 B 行 |

官方等价门禁使用相同预测文件，核对各帧候选保留/排序、关联、碰撞、物理分和聚合输出；空预测、候选不足 50、阈值边界等也须覆盖。合法空帧按预声明全零扩展独立验收，上游在此没有可直接数值对齐的输出；已知轴极点非有限物理质量按官方阈值未命中保留排名。未经登记的数值修正、异常跳帧或清理参考实现行为，都不能静默进入主协议。门禁通过前，本页公开比较表保持待测。

六指标均使用百分数：`AP50` 是阈值集合 {0.2, 0.4, 0.6, 0.8} 下、前缀 `k=1..50` 的精度平均；`AP0.4_50` / `AP0.8_50` 固定相应阈值，`AP1` 及其阈值列只取首个候选。判正用官方实现的 `>=`，不足 k 的分母仍为 k。它不是实例 Top-1、PR 曲线面积或仅 `Precision@50`。

每个 pair 都保留该论文自身的版本、相机、输入/前端、训练和 evaluator 证据状态。论文缺失的具体训练、分割或后处理配置写“未核实”；采用统一测试集并不能自动证明所有条件只差半径。只有公共规范的对应门禁通过，才填写本方法实测值和本方法相对该参考的差值。当前不计算任何增益，不建立参考方法排名或参考之间的差值。

同一冻结 public 评分模型生成三份 profile 独立预测：`uois_no_gt_workspace`、`uois_uisn_gt_aabb20`、`uoais_uisn_gt_aabb20`。前端模型及工作区完全相同的 pair 才可复用预测；只更换这些推理配置，不额外训练评分模型。UISN 的 GT 前景工作区不是 S1B 的共同步骤；Diffusion 原论文/实现的具体工作区尚未核实，不由 UISN 配置推定。

可以对每个 profile 的同一份冻结预测另用 R10 evaluator 评价，作为尺寸变化的附加解释；这不新增训练，不替代 R20 主行，也不等于专门训练的 R10 模型。

## 2. 本路线与各公开方法的独立比较

每张表只比较本路线与一个参考方法。`待测` 表示本方法尚未执行，`—` 表示论文没有报告。以下论文值来自各自标注的表，不是本仓库复现值；完整出处和核验状态以[公共来源登记](../PUBLIC_METHOD_COMPARISON.md)为准。

### 2.1 B 与 SuctionNet / S1B

[SuctionNet-1Billion 原论文 v2](https://arxiv.org/pdf/2103.12311v2)，Table IV，PDF 第 6 页，取 Kinect/RealSense 斜杠后的 RealSense 数值。固定采用 2021-10-29 v2 主表，不混用旧版本或消融表的小数差异。原方法处理整帧 RGB-D、不需要 GT 实例；本 pair 使用 `uois_no_gt_workspace`，不使用 GT 前景工作区、GT 实例裁剪或补回漏检。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | B-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | SuctionNet / S1B / 论文 R10 | 28.31 | 38.56 | 3.41 | 43.98 | 60.81 | 7.79 |
| Similar | B-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | SuctionNet / S1B / 论文 R10 | 26.64 | 35.34 | 3.42 | 36.59 | 49.30 | 6.54 |
| Novel | B-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | SuctionNet / S1B / 论文 R10 | 8.23 | 10.29 | 0.35 | 14.32 | 19.95 | 0.38 |

### 2.2 B 与 Diffusion-Suction

[Diffusion-Suction 原论文 v2](https://arxiv.org/html/2502.07238v2)，Table I，PDF 第 7 页，取 RealSense 列；Table II 的 Parcel-Suction 结果不属于此比较。论文声明沿用 SuctionNet 在线 Seal × Wrench 协议；Table I 的完整训练/验证 recipe、RGB/实例过滤和 checkpoint 配置未逐项核实，不能从 Parcel 训练脚本推定。本 pair 暂使用 `uois_no_gt_workspace`；Diffusion 原始工作区仍标“未核实”，不能据此宣称该项已匹配，也不继承 UISN 的 GT 前景工作区权限。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | B-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | Diffusion-Suction / 论文，官方默认R10 | 31.90 | 42.78 | 4.79 | 54.79 | 68.59 | 21.03 |
| Similar | B-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | Diffusion-Suction / 论文，官方默认R10 | 31.59 | 41.38 | 4.27 | 47.00 | 59.36 | 16.08 |
| Novel | B-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | Diffusion-Suction / 论文，官方默认R10 | 10.00 | 12.32 | 0.76 | 14.14 | 17.67 | 2.42 |

### 2.3 B 与 UISN (UOIS)

[UISN 原论文](https://doi.org/10.1109/LRA.2024.3385609)，Table I，PDF 第 6 页 / 刊页 4939，按表注取 RealSense 数值。该 pair 使用 `uois_uisn_gt_aabb20`：冻结 UOIS 预测分割，另按作者 `get_workspace_mask(..., outlier=.02)` 使用 GT 前景联合 AABB 工作区。只读取前景联合范围，不读取 GT 实例身份，也不用 GT 实例 mask 替换预测分割。该预测不能复用 `uois_no_gt_workspace` 的结果。原文转录与本轮全文访问状态见公共来源登记；本轮未能重新读取全文，使用仓库已登记的逐表转录。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | B-public20 / uois_uisn_gt_aabb20 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | UISN (UOIS) / 论文，官方默认R10 | 37.98 | 49.83 | 9.99 | 63.88 | 76.64 | 31.72 |
| Similar | B-public20 / uois_uisn_gt_aabb20 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | UISN (UOIS) / 论文，官方默认R10 | 37.83 | 49.35 | 8.89 | 48.68 | 60.01 | 16.72 |
| Novel | B-public20 / uois_uisn_gt_aabb20 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | UISN (UOIS) / 论文，官方默认R10 | 9.47 | 12.63 | 1.17 | 11.83 | 15.47 | 2.76 |

### 2.4 B 与 UISN (UOAIS)

来源仍为 [UISN 原论文](https://doi.org/10.1109/LRA.2024.3385609) Table I 的 UOAIS 配置。这是独立 pair，使用 `uoais_uisn_gt_aabb20`：同一冻结 source-only 评分权重，冻结 UOAIS 预测分割，加作者 `get_workspace_mask(..., outlier=.02)` 的 GT 前景联合 AABB 工作区；不读取 GT 实例身份。按该 profile 重新生成整帧预测，不复制任一 UOIS profile 的结果，不额外训练评分模型。前端、工作区和成本随结果保存。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | B-public20 / uoais_uisn_gt_aabb20 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | UISN (UOAIS) / 论文，官方默认R10 | 38.27 | 50.57 | 10.56 | 66.11 | 78.35 | 35.29 |
| Similar | B-public20 / uoais_uisn_gt_aabb20 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | UISN (UOAIS) / 论文，官方默认R10 | 36.54 | 47.77 | 8.74 | 49.20 | 61.45 | 17.20 |
| Novel | B-public20 / uoais_uisn_gt_aabb20 / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | UISN (UOAIS) / 论文，官方默认R10 | 9.08 | 12.05 | 1.04 | 11.94 | 15.40 | 2.07 |

## 3. 执行顺序和必要的准入检查

| 阶段 | 目的/固定条件 | 输出证据 | 当前性质 |
|---|---|---|---|
| E0 合同回归 | 四份参考程序，固定schema/float64 | 逐项通过、输入版本、几何反例 | 已有129项合成参考，可复跑 |
| E1 真实数据/teacher | 先64–128个覆盖尺寸、dense规模和几何形状的实例；精确p/n | S/W/self对齐、身份、排除、成本报告 | 需真实资产和接线 |
| E2 B短链路 | 一个真实实例到若干成功更新、保存恢复和推理 | 有效梯度、冻结、恢复、同姿态输出 | 模型实现后执行 |
| E3-N B-native-pre | 90 train / 10 dev；128 query、140000 监督实例 / 8750 成功更新；val 最多256 | 本地 best/last、逐 query 与实例结果 | 本地开发与选模 |
| E3-P B-public20 | 冻结配方后全100训练场景 final-refit，固定预算，source-only | last、完整训练身份；不使用产线数据 | 公开比较专用 |
| E4-N B-native-ft | 从 E3-N best 初始化，200+800，2 target+2 source 回放 | 399 val 选模、最终87 test；不填论文 AP 主表 | 目标域适配 |
| E5-P B 对公开方法 | R10 等价门禁后，B-public20 覆盖 RealSense 全部23040帧 | 六指标、逐帧输出与协议身份，分别填入对应 pair | 不依赖 A 的实现进度 |
| E5-N 本地冻结测试 | INST20 源三子集、冻结目标87动作 | 质量/覆盖/任务与分母报告 | 与论文 AP 独立 |
| E6 B部署 | 冻结权重、动态ROI、精度合同与输出边界 | PyTorch/导出差异、端到端p50/p95/显存 | 实际导出成功后 |

E1的64–128是teacher成本和一致性剖析样本，不是训练收敛样本数。E2只解除真实链路风险，不能替代E3/E4或宣告速度优势。本地共同合同变化需同步版本；公开比较协议独立锁定，不能把不同候选轴、前端或 evaluator 的变化归为网络收益。

## 4. 本地源域指标和明确分母

令每个可见GT实例i的Top1命中 `h_i=1[有输出且所选姿态Q_inst>=0.4]`。少于100支持点、无合法观测法向、无输出实例的h_i=0，保留在原始实例分母。按frame内实例均值→scene内有效frame均值→scene等权平均得到 `InstTop1_INST20@0.4`；空GT帧另计并排除。另报实例微平均。[协议来源](../TRAIN_EVAL_REPRODUCE.md)

次级selected-Q使用相同宏平均层级：有输出取所选姿态Q_inst，no_pose取0；它用于同主指标时选模。另报**有输出条件下**所选Q的微平均及有效输出数，避免把两个均值混称。所有结果同时报告原始实例数、少支持/无normal/无输出计数及原因。

**teacher未知不能当Q=0。** 所选姿态或候选覆盖等指标所需query标签缺失/非法时，评价标为不完整并修复缓存，不能伪造负例或删除分母后宣告完成。no_pose的策略失败与有姿态但未知teacher的评价缺失是不同状态。

`Coverage_INST20@256`表示实例的固定最多256共同候选池中是否至少存在Q_inst≥0.4的query；另报“池内存在好候选时Top1选中率”，其分母仅为池内有好候选实例。实际K不足256时报告真实K，不复制候选。两个指标须用同一池统计，支持过滤和NMS前后分别报，不能混用分母。纯 SW 和 self 比例作本地解释项；本节 INST20 结果不能填入第 2 节官方族 R10/R20 六指标表。

## 5. 历史动作与最终测试

固定1676 train、399 val、87 test，继承group，不重新随机拆分。B-native-pre 没有受训练的 task 概率；可诊断其q对成败的排名关系，但不能用q填task BCE/Brier或以随机task头当微调前基线。B-native-ft 按任务概率报告AUROC、AP、BCE、Brier以及逐动作logit/概率、标签、姿态和身份。

始终列总动作数、支持有效数、正负数、各group数量和跳过原因。无几何关联不是动作失败，保留覆盖诊断。若另行开展内部模型对照，在相同有效交集进行并同时列各自覆盖；不悄悄改变每次验证分母。分组仅单类时AUROC记未定义。

399 val用于每100成功步选模及预注册的小对照；87 test在权重、排序规则、输入与标签版本冻结后一次性评价。test不用于选alpha、法向跨度、门控参数或学习率。当前记录是planned_pipeline_contact、actual_execution_verified=false，离线同动作排名改善不直接证明新候选的真实执行成功率改善。

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

公开主结果是 source-only 的 B-public20 与第 2 节各参考的独立 pair；本地开发结果保留 B-native-pre 与 B-native-ft，各路线独立执行，不将另一分支的完成设为前置条件。追加对照只回答已经观察到的问题，每次仅改变一个因素并记录 variant/config/hash：

| 触发条件 | 允许的最小对照 | 保持不变 |
|---|---|---|
| teacher成本/内存阻碍默认训练 | 训练64对128 query | 同一256池及其前缀、140000实例预算、val256、同teacher |
| 历史动作对小偏移敏感且开发证据充分 | 默认关闭 vs 既定2mm/2° pose-consistency | 一动作一次BCE、split、微调步数/回放；不复制源标签 |
| 需要检验混合排序是否有价值 | alpha=0 vs 预注册alpha=.5 | 同一冻结权重、候选、NMS；单独输出task概率 |
| 投影门控拒绝率/低分辨率子集异常 | 先分层报告、检查坐标/量化/支持；确需改阈值再建立独立配置 | 不在默认配置里静默改3mm、环半径或teacher |
| 工程部署要求降精度 | float64几何/FP32网络基线 vs明确部署精度 | 同权重、输入、ROI/padding、query及输出任务 |

不同时组合query预算、phase、法向、门控、alpha和正则形成大矩阵。新旧法向若作质量对照，必须各自重算精确p/n标签身份；这比较的是候选管线，不能说只消融网络。前三阶谐波属于A分支，不向B默认融合另增未经规范的特征。

## 8. 统一计时和导出报告

每实例batch1，50次预热后至少200个固定验证实例。报告model-only及instance端到端p50/p95和峰值显存；GPU阶段用CUDA events并同步，端到端使用明确墙钟同步边界。记录硬件、CPU线程、软件/driver、dtype、真实ROI/padded大小、对象数、raw有效像素数、实际K、chunk、缓存/拷贝及上游相机/分割是否计入。

分项至少包括：原图反投影、3×3平滑/±3差分及路径门控、候选回填、上下文encoder、native纹理stem、环位置生成/采样/门控/摘要、环网络/头、NMS/位姿。整帧多实例另报汇总，不将单实例model-only倒数称为机器人FPS。延迟比较要求相同硬件、候选预算和输出任务；2–5ms 是工程目标，缺失结果保持待测。公开整帧端到端计时还需计入 UOIS/UOAIS 前端和实例汇总；论文计时条件不匹配时不计算速度增益。

导出分别保留子图和组合差异；动态ROI保持真实padding合同。部署改几何精度时特别统计门控翻转及其Top1影响，不用平均分数误差掩盖离散决策变化。[后处理与导出边界](POSTPROCESSING.md)

## 9. 每次实验必须交付的记录

运行目录保存完整配置与环境、代码/checkpoint/manifest/teacher/precision身份、逐query/实例/动作预测、支持与排除统计、按scene/group聚合汇总、计时分项和失败样例索引。记录成功更新、成功监督实例、全部consumed draws和skip。可见输入和完整GT teacher的使用边界写入报告。

GitHub只同步必要汇总、配置和来源索引；完整资产/标签/权重留在数据与运行目录。公开对照执行[公共比较规范](../PUBLIC_METHOD_COMPARISON.md)和[论文基线登记](../reference/paper_baselines.json)，不调用项目 `eval_r20` 替代官方族；第三方优先引用论文值，不要求重训。旧 E5 若做本地 INST 分析，须对它实际输出姿态以同一实例 teacher 重评，且仍不能写入 B 的任何实测行。

只有 E1/E2 通过后进入正式训练；E3-P 的 last 冻结及官方等价门禁通过后才能生成公开比较结果；E3-N/E4-N 权重冻结后生成本地测试结果；E6 导出通过后才能陈述部署成本。真实新动作的业务成功率需要后续执行结果，不能从离线历史AUROC、合成法向噪声或局部卷积MAC直接推算。

## 10. 待协议核验的 MultiGraspNet 参考附录

[MultiGraspNet 原论文 v2](https://arxiv.org/html/2602.06504v2) Table I（PDF 第 5 页）Vacuum 部分报告 RealSense Top50 三列；没有报告 Top1。论文明确 100 train / 90 test 和前缀 k=1..50，但将 vacuum 成功阈值表述为 seal coefficient，尚不能从公开实现确认与官方 Seal × Wrench、完整碰撞和全部帧配置完全相同。[项目页](https://vandal-lab.github.io/multigraspnet-project/)的代码发布状态及核验日期见公共来源登记。

因此以下两对保留论文数值作为参考，协议证据补齐前不纳入正式增益结论，也不推算缺失 Top1。两张表彼此不计算差值。

### 10.1 B 与 MultiGraspNet-vac.

单任务 vacuum 变体；本路线使用 `uois_no_gt_workspace` 和同一 source-only 评分权重，不使用 UISN 的 GT 前景工作区。该行的训练目标与 joint 行不同，不能混名引用。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | B-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | MultiGraspNet-vac. / 论文，尺寸待核验 | 28.26 | 36.39 | 3.60 | — | — | — |
| Similar | B-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | MultiGraspNet-vac. / 论文，尺寸待核验 | 31.37 | 41.63 | 3.88 | — | — | — |
| Novel | B-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | MultiGraspNet-vac. / 论文，尺寸待核验 | 8.07 | 10.03 | 0.48 | — | — | — |

### 10.2 B 与 MultiGraspNet joint

joint 行同时训练 parallel / vacuum，带额外平行夹爪监督；Table I 不启用 switching strategy。本路线使用 `uois_no_gt_workspace` 和同一 source-only 评分权重，不使用 GT 前景工作区。即使后续 evaluator 核验通过，仍须保留额外监督说明。

| 子集 | 方法 / 结果来源 | AP50 | AP0.4_50 | AP0.8_50 | AP1 | AP0.4_1 | AP0.8_1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Seen | B-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Seen | MultiGraspNet joint / 论文，尺寸待核验 | 28.07 | 35.81 | 3.62 | — | — | — |
| Similar | B-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Similar | MultiGraspNet joint / 论文，尺寸待核验 | 31.22 | 41.65 | 3.53 | — | — | — |
| Novel | B-public20 / uois_no_gt_workspace / 待实测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Novel | MultiGraspNet joint / 论文，尺寸待核验 | 7.38 | 8.92 | 0.46 | — | — | — |
