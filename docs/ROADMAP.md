# 项目路线与验收

用户已说明按旧 `dc4ad6b` 完成数据制作。以下G0–G2列表保留为历史数据工程记录，不要求为了接入新模型重新运行；早期模型接入记录只覆盖指定release与少量真实帧，后续完整预训练、微调与测试另行记录，不把模型结果追溯为全量数据生产重验。训练优先复用已有分数，详见 [旧数据适配](implementation/LEGACY_DATA_ADJUSTMENT.md)。

## 数据工程 G0–G2（历史职责）

以下描述原数据流程的职责，不再作为接入模型的待办清单。用户已完成旧方法的数据制作；本仓库保留对应实现，已有模型验证只覆盖 [验证记录](MODEL_VALIDATION.md) 明示的 release 与样例。

- **G0 Seal**：在真实 10 mm 数据上校验原始方法；报告点数、候选 ID、Seal 得分与局部碰撞差异。
- **G0 Wrench**：重放官方 10 mm 标注，对齐公开场景标签并核对数值误差。
- **G1 Seal20**：生成全部 88 个物体的 20 mm Seal，确认 72 扇区、18–22 mm 环带及采样索引。
- **G2 场景标签**：生成 Wrench 双口径与三段式 Collision，并核对场景实例行号。
- **G2 逐帧 GT**：生成每相机 190×256 帧的 GT-A、GT-B 和有效掩码；检查法向、边界和负样本。

## 当前模型阶段：公开预训练 → 少量场景微调

更新：2026-11-18。E0–E5预训练、E5目标域开发微调及E5完整公开测试均已完成；当前结果见[方法效果对比](SUCTIONNET_METHOD_COMPARISON.md)和[实验资料](experiments/CD010-20261117/README.md)。早期P1–P3验证和配置按历史范围保留。

- [x] 形成 [可实施规格与图示](implementation/README.md)：数据适配、模型、预训练、微调和实验。
- [x] 实现只读旧缓存adapter，复用已有S/W/C/A/B，派生两类法向和训练掩码。
- [x] 实现E1基础模型、损失、训练入口与实际姿态R20评价。
- [x] 实现实例融合、物理分区及E0–E5配置。
- [x] 实现无标签推理、NMS及固定查询验证，完成CPU测试与代码复核；已有H20完整观测smoke为E1 2步、E5 18步。
- [x] 在32个原候选上核对实际姿态评价与已有标签的R20半径、冻结归一化、L2 Wrench和碰撞包络，并用8个旋转姿态检查重新计算。旧API的切向分量max不是本项目的数值协议。
- [x] 完成 v3 样本预算、17 次零更新硬件测量、六份固定 profile 及运行恢复/日志实现；配置与版本见 [当前预训练](CD009_PRETRAINING.md)。
- [x] 实现 [P3 原生接触几何](PERFORMANCE_PHASE3.md)：融合查询/筛选/FPS/打包与两候选线程，保留原点集和训练定义，接通 P2→P3 续训；本地 CPU 结果与训练机待完成项分别记录。
- [x] 已完成并记录 [P3实际接续与正常窗口](performance/P3_INTEGRATED_RESULTS.md)及[后续从头训练优化](PERFORMANCE_HISTORY.md)；各轮证据保持其实际模型与窗口范围，不再作为新训练前置。
- [x] 完成 E0–E5 六个独立、各230,400有效样本及10次开发验证的从头预训练；H20/E5采用有效batch48（4,800更新），L20/E0–E4采用有效batch12（19,200更新），见[正式配方](SELECTED_PRETRAINING_RECIPE.md)及[结果](experiments/CD010-20261117/pretraining/REPORT.md)。
- [x] 实现dense_proxy与attempt_success微调接口，完成合成初始化、跨200步阶段切换与恢复检查。
- [x] 按预定开发指标选定E5 best step2880；完成RealSense官方90测试场景、23,040帧的R10与R20评价，并发布[公开方法比较、UISN消融与候选覆盖](SUCTIONNET_METHOD_COMPARISON.md)。
- [x] 从E5 best初始化目标域微调并采用step1000；已完成399条历史动作开发验证及401帧配对推理，见[微调记录](experiments/CD010-20261117/finetuning/README.md)。
- [ ] 在已有87条未参与选模的有标签test历史动作上，比较冻结预训练与微调权重的同动作评分。
- [ ] 接入原定正常工况业务执行对比，以新动作的实际吸取和搬运结果判断部署收益。

当前预训练/微调不以额外逐物体质量质心标定、时序跟踪或极端工况专项为前置。动态抓取等功能等实际需求出现后另行规划。

可选sidecar缓存与residual uncertainty工具尚未交付，仅在实际需要时安排。最新完成项与待办摘要见 [README](../README.md)，正式运行命令见 [MODEL_USAGE](MODEL_USAGE.md)。

## 单物体路线 A/B 设计与实施

- [x] [A/B独立分支与六部分实施文档](instance_routes/README.md)，v3架构图、几何边界及物理v2目标自碰合同已整理。
- [x] 共同 `instance_geometry/data/teacher`、官方格式builder、精确p/n标签、对象资产/缓存身份；CPU与合成官方格式验证。
- [x] A的12通道SpConv包装、raw支持谐波/分组、531融合与精确姿态接口；CPU测试显式注入骨干，真实SpConv尚待GPU检查。
- [x] B原生Torch GN18/Fq/共享纹理、577槽投影、环带缺测/门控与330融合；真实CNN CPU更新和推理已验证。
- [x] 直接100场景source训练、90个test场景各固定8帧选择best并标记test_selected/fixed_test_subset、保留last；两阶段目标微调及严格暂停恢复、native推理/评价已接通。完整test由训练后独立CLI报告，不重新选模。[运行命令](instance_routes/IMPLEMENTATION.md)
- [x] [公开方法逐一对比](instance_routes/PUBLIC_METHOD_COMPARISON.md)的论文值与成对表保留；仅半径协议设置deferred，不作为训练门禁。
- [ ] 用完整真实资产执行缓存重算、既定8750步source预训练及1000步目标微调，记录teacher成本和候选覆盖；目标原划分不变。
- [ ] A真实CUDA/SpConv链路、同硬件GPU端到端成本及B的ONNX/TensorRT导出。15–30ms、2–5ms仍是未验证目标。
- [ ] 公开物理评价与权重规则待定后执行完整场景AP；当前native test选模成绩不称独立公开测试估计。

## 代码管理

- [x] 项目所有执行路径移除 SHA-256 计算和校验；当前[可读元数据与恢复规则](NO_CONTENT_CHECKSUMS.md)及根 AGENTS 约束持续生效。
- `main` 保持可阅读、可测试的主线。
- 新算法、标签公式或数据格式改动使用独立分支和 Pull Request。
- 代码不覆盖 SuctionNet 官方源文件；生成数据存储于外部 `$OUT`。
- Python 变更需运行 `pytest -q`；数据标签变更需附至少一个可视化示例和指标对比。
- 不将大型原始数据、隐私配置、训练权重默认提交进 Git。



