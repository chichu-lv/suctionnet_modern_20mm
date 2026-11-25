# 单物体吸取路线 A/B 工程规范（v3）

> **最新性能执行入口（2026-11-19）：** [L20三分钟实测后的优化与执行说明](L20_PIPELINE_OPTIMIZATION.md)，包括A/B优化分支、独立源NPY/静态几何缓存构建、磁盘预算、原生版本与命中检查，以及180秒完整更新配对。该优化不要求重新从头训练。

> **2026-11-19：H20 从头训练方案**已新增独立配置 `configs/instance_a_h20.yaml` / `configs/instance_b_h20.yaml`（现已同处收拢基线），采用真实微批 32、140000 个有效实例、4375 次更新、BF16 密集网络与持久 spawn 预取。启动、压测、精度边界与样本预算见 [H20_FROM_SCRATCH.md](H20_FROM_SCRATCH.md)。本页原 `instance_a.yaml` / `instance_b.yaml` 的单实例配方仍用于既有运行；新配置不直接续训旧检查点。

本规范定义单物体 RGB-D 吸取的两条实现路线：A复用稀疏3D骨干与物理接触分组；B采用共享2D图像编码、逐查询语义特征及投影环带读取。两者对明确的接触中心和有向轴预测Seal、Wrench、保留目标自碰的实例质量及历史规划动作成败。

**A/B的实例数据、teacher、网络、预训练/微调、checkpoint恢复和native推理/评价已实现。** 直接100场景训练，默认每个test场景固定8帧选best，标记 `test_selected`、`fixed_test_subset`；训练后单独评价完整test，不重新选模。公开仅半径评价待定。运行命令见[IMPLEMENTATION.md](IMPLEMENTATION.md)。当前完成CPU和合成资产链路验证，尚无新模型完整真实数据精度、GPU时延或TensorRT结果。

当前运行使用manifest v4，兼容旧v3并忽略旧校验和字段；数据/标签/checkpoint身份改为实际数组检查、可读配置和语义ID，不计算SHA-256，也不改用另一种摘要。source预训练仍为140000成功实例/8750更新，每875步在固定90×8=720帧上评价、共10次并保存last；默认仅保留当次best或最终步的gzip明细，并记录评价耗时。选择帧数可设8/16/256，旧配置缺省256；所选帧全部实例、每实例256候选池保持。builder支持同输出根下的独立手动分片、多进程与安全合并。微调初始化允许骨干路径/查询分块变化，普通resume仍严格；推理过滤无效查询槽位并保持子模块模式。A近邻向量化、5mm共同舍入边界、可配置支持门槛及上述运行边界见[运行说明](IMPLEMENTATION.md)。

生产代码审查基线为 [`87f29d2`](https://github.com/chichu-lv/suctionnet_modern_20mm/commit/87f29d29bea8fa5153d6df3f8253d9ced9e4b441)。初稿保留在 [`e495fd8`](https://github.com/chichu-lv/suctionnet_modern_20mm/commit/e495fd89678b5f696e1819437fef5ced5103aa57)；v2修订保留在 [`8e63e92`](https://github.com/chichu-lv/suctionnet_modern_20mm/commit/8e63e925bc54c442e8b5df6422ed51a69ce73fc1)；本页及子文档以v3为准。[逐项复核与修正](REVIEW_AND_CORRECTIONS.md)说明初稿缺陷和v2决策；[v3几何边界复核](BOUNDARY_REVIEW.md)说明本次四项修订、数值反例与限制。

## 收拢后的实施入口

- **A：稀疏 3D + RGB**：[六部分实施文档](route_a/README.md)。
- **B：共享二维编码 + 投影环带**：[六部分实施文档](route_b/README.md)。

2026-11-25 已将共同设计、两路线实现及最新优化收拢，详情见 [分支收拢记录](../BRANCH_CONSOLIDATION_20261125.md)。路线目录组织各自实施步骤，公共数据/teacher/查询/训练/评价以本目录共同文件为准。源域次级 selected-Q 继续使用相同三层宏平均，`no_pose=0`，未知 teacher 标签不作为零质量。

## 1. 阅读和执行顺序

| 文档 | 内容 |
|---|---|
| [A 方法架构图](METHOD_ARCHITECTURE_A.md) / [B 方法架构图](METHOD_ARCHITECTURE_B.md) | 三面板总览、接触几何、两阶段训练；SVG/PDF/PNG、双语图注与关键维度 |
| [v3几何边界复核](BOUNDARY_REVIEW.md) | 谐波摘要、投影视差/缺测门控、稳健法向与稀疏候选回填 |
| [结构复核记录](REVIEW_AND_CORRECTIONS.md) | 对用户各项质疑的结论、坐标基反例与成本计算 |
| [共同数据与标签](DATA_AND_LABELS.md) | 固定split、快速法向、共同候选、精确动作、自碰分项与标签缓存 |
| [路线A](ROUTE_A.md) | 12维稀疏输入、原始支持、覆盖诊断、无人工切向坐标的接触编码 |
| [路线B](ROUTE_B.md) | 删除U-Net/逐候选栅格CNN，Fq、共享原生纹理、批量投影环与循环池化 |
| [与公开方法成对比较](PUBLIC_METHOD_COMPARISON.md) | 保留原论文值与逐方法成对表；公开仅半径协议待定；[机器可读来源](reference/paper_baselines.json) |
| [代码运行说明](IMPLEMENTATION.md) | 实际builder/train/resume/finetune/infer/evaluate命令、输入字段和checkpoint输出 |
| [训练评价和复现](TRAIN_EVAL_REPRODUCE.md) | 128-query训练、损失、200+800微调、可选姿态一致性及真实可运行命令 |
| [v3法向/采样参考](reference/sampling_contract.py) | 26项检查：同实例平滑、宽差分、位反转顺序及格内回填 |
| [v3接触边界参考](reference/contact_boundary_contract.py) | 48项检查：谐波、门控、覆盖、原始高度与逐角归一化 |
| [v2参考程序](reference/revision_contract.py) | 26项保留检查；固定坐标图/理想环及物理v2标签校验仍复用；旧法向/采样仅作回归对照 |
| [基础几何参考](reference/geometry_contract.py) | 29项既有检查、投反影/姿态/manifest；切平面栅格仅为基础数学示例 |
| [A配置](reference/route_a.json) / [B配置](reference/route_b.json) | v3机器可读设计快照，不是现有训练器配置 |

## 2. 共同任务与物理v2目标

原总collision同时包含目标自身和外部环境。单物体输入不应失去目标自碰监督，故改为：

$$
C_{self}=C_{seal,self}\lor C_{target,envelope},\qquad
Q_{inst20}=S_{20}W_{20}(1-C_{self}).
$$

`true`表示碰撞。质量身份为`gt_INST20_pose_v2`；纯`q_sw=S*W`另存诊断。Seal和Wrench辅助目标不因自碰而被人为改成零。外部物体/桌面不进入新质量，原R10/R20完整评价保持原协议。没有self真值的行不能补false假装无碰撞。

默认法向使用 `organized_masked3x3_span3_v3`：同实例3×3原始XYZ带深度门控均值，至少5点；在平滑图上做±3像素中心差分，逐段核验原始路径的同实例支持与10mm跳变。只平滑法向输入，接触点和raw支持不动；不退回全量PCA或隐式±1。候选使用 `image_stratified_morton_refill_v3`：16×16分层、位反转Morton顺序，首轮每格一点，后续逐轮格内二维FPS回填至256或合法像素耗尽；训练/源回放取前128个，64为可选配置。历史动作仍按记录p/n直接评分。

物理函数与质量schema仍为v2；观测法向、候选序列和模型结构已变为v3，必须重建相应query/label缓存身份，不能复用旧pose缓存冒充同姿态监督。

## 3. 两路线的固定结构

| 维度 | A：单实例稀疏3D | B：共享2D编码与投影环 |
|---|---|---|
| 全局输入 | 完整可见支持；2mm代表点后FPS最多2048，12维 | 全实例13×128²语义图；另保留原生ROI纹理 |
| 主干 | 新12通道SpConv ResUNet14封装 | ResNet-18 encoder-only，GN8；没有U-Net |
| 查询语义 | 主干Fq256及G64 | 中间层位置采样Fq64及G128 |
| 精细接触证据 | raw支持的三区域共享点MLP，原始RGB旁路 | 一次stride1纹理stem，按姿态读取9半径×72角度 |
| 区域 | core 0–10mm，inner 10–18mm，lip 18–22mm | 9个投影半径；按实际半径、切向偏移及正径向半平面门控，保留门控前观测高度 |
| 局部方向处理 | rho/h等无任意切向基量，新增中心化高度RMS/前三阶谐波幅值与有效标记 | 固定±Z坐标图、每角跨通道LayerNorm、masked池化与显式覆盖/缺测摘要 |
| Query融合 | 531→256→256 | 330→128→128 |
| 学习输出 | q_inst、Seal、Wrench、Laplace log-scale、task logit | 同左 |
| 特征缺失 | 最多3个10mm内同组件邻居，至少1个即可；保留RGB/几何 | 记录原生深度/纹理缺测、投影有效性，不补出平面 |
| 默认在线几何 | organized法向、体素/FPS、局部3D分组 | organized法向、图像分层、向量化投影和读取；无空间树/逐query栅格 |
| 延迟 | 15–30ms只是工程目标 | 2–5ms只是工程目标 |

B每实例只计算一次全局和纹理编码。它仍使用相机投影、深度几何与任意姿态query，不能称为完全没有3D数学的无条件热力图。投影采样不等于完整3D接触积分，也不能认证隐藏自碰。B的价值由同硬件、同候选预算、同输入/输出边界的实测决定。

## 4. 为什么修订

初稿B恢复高分辨率特征后仅全局池化，缺少逐位置Fq；局部显式分区池化与64²→8²整图池化冲突；逐query栅格/CNN偏离快速图像路线。v2直接删除这些冗余路径，用共享中层Fq和原生纹理采样补足对应信息。

`argmin(abs(n))`有离散参考轴切换，跳角不固定90°；新A不向模型输入该基的x/y，新B减少坐标图切换并对环向共享权重。原144×128=18432个角度槽位权重被移除，不能将这项结构修正声称为已实测降低过拟合。

全量PCA与完整帧teacher的成本风险通过默认法向替换、128查询和按需缓存控制；Seal本身仍可能是离线主瓶颈。BN冻结原本已有规定，本次补执行验收，B全局/纹理保留GN，v3环分支改为仅跨通道LayerNorm。精确动作一次BCE仍是基线，轻微姿态一致性默认关闭且需开发集验证。

v3进一步处理：水平吸取时的显式局部角向信息不足；理想环投影与真实深度的错槽；短差分轴噪声；稀疏掩码的一格一点预算枯竭。已有位反转Morton前缀未发现row-major错误。前三阶谐波并不完整，3mm门控是工程初值，平滑法向也需报告有效率与边缘偏差。[完整边界记录](BOUNDARY_REVIEW.md)

## 5. 既有结果与新设计的边界

[方法比较](../SUCTIONNET_METHOD_COMPARISON.md)中，R20每实例无候选率约Seen0.0377%、Similar0.0134%、Novel0.0218%；无候选不等于缺少优质候选，现有数据不支持“25%优质实例因背景稀释而漏掉”的因果结论。[诊断原始汇总](../experiments/COMPARISON-20261118/diagnostics.json)

[目标数据汇总](../experiments/CD010-20261117/finetuning/data/summary.json)有1676/399/87条训练/验证/测试动作，关联的是`planned_pipeline_contact`、`actual_execution_verified=false`。旧E5验证AUROC改善不能填入A/B结果；新task输出首先解释为给定观测和规划命令的成败概率。

14万实例暴露是新工程初值，不由旧E5的2880×48个场景观测直接证明最优。单实例、RGB、法向、self目标、候选和结构均改变，公开验证逐一比较本方法与各参考方法，不将全部差值归于某一模块。无需新增大矩阵，先完成共同合同和必要风险检查。

## 6. 复用与实施顺序

现有`observation.py`、`eval_r20.py`、`full_data.py`、`target_dataset.py`和精确姿态评分思想可复用。旧骨干默认8通道保留，A显式指定12通道；新A/B使用独立模型、质量身份和训练入口，旧E5配置不变。保留[checkpoint政策](../CHECKPOINT_POLICY.md)，不恢复已退役的迁移兼容。

1. 固定实例/动作清单和同实例organized法向、二维候选，核对投反影和历史点/轴。
2. 拆分并核验target-only teacher；保留原Seal扫描、自碰和Wrench定义，先测少量真实实例的离线成本，再按需生成训练缓存。
3. 接通A的覆盖、raw RGB与任意姿态评分；接通B的Fq、共享纹理和投影环读取。
4. 已接通同姿态、mask、参数冻结、存取恢复与无输出路径；完整数据训练按100场景和test选best规则执行。
5. 报告native INST、历史动作分类、候选覆盖及端到端成本；[公开仅半径协议待定](PUBLIC_METHOD_COMPARISON.md)，不把native结果填成论文AP。

v3设计参考共129项数学/schema检查保留；新代码测试和B实际CNN合成资产链路见[验证说明](../MODEL_VALIDATION.md)。完整真实数据重算/训练和GPU性能仍待执行。
