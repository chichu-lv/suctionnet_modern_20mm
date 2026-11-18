# SuctionNet Modern 20mm

面向 **半径 20 mm 单吸盘**的 SuctionNet 数据制作、公开数据预训练与少量自有场景微调项目。

当前已纳入的数据流程：SuctionNet 10 mm 标签对照（G0）→ 20 mm Seal 重标注（G1）→ 场景 Wrench/Collision → 每帧 RGB-D 点云与吸取候选映射 → GT-A/GT-B → 数据质量检查（G2）。

**当前实验状态（2026-11-18，北京时间）：** E0–E5 六组从头预训练均已完成，每组230,400有效样本及10次固定开发验证。按预定Top-1主指标采用 **E5 best step2880**；由它初始化的真实场景微调也已完成，当前采用 **step1000**，固定399条历史规划动作验证AUROC由0.6215提高到0.7121。两模型各401帧新候选的推理记录已发布；它们尚不代表新策略实际吸取成功率。完整数据见[实验资料](docs/experiments/CD010-20261117/README.md)，模型选择、模块收益、微调轨迹和动作变化见[独立分析报告](docs/experiments/CD010-20261117/analysis/RESULTS_ANALYSIS.md)。历史训练硬件配方与启动命令继续保留在[选定正式配方](docs/SELECTED_PRETRAINING_RECIPE.md)和[开训记录](docs/performance/CD010_SELECTED_TRAINING_START.md)。

**公开方法对比（2026-11-18）：** E5 best 已完成 RealSense 官方三个测试划分的全量评价（90场景、23,040帧）。官方 R10 的 Seen/Similar/Novel AP Top50 为 **16.33/20.22/5.69%**，AP Top1 为 **47.60/51.49/20.22%**。与 SuctionNet、Diffusion-Suction、MultiGraspNet、UISN 完整版及 Only Depth 的对照、R20候选覆盖分析见[方法效果对比](docs/SUCTIONNET_METHOD_COMPARISON.md)，紧凑数值见[结果表与来源](docs/experiments/COMPARISON-20261118/README.md)。这些是 GT 实例输入、R20 训练条件下的结果；论文值与本项目计算值分别标明。

## Done（已完成）

- [x] [完整测试与公开方法效果对比](docs/SUCTIONNET_METHOD_COMPARISON.md)：E5完整90场景官方R10及项目R20已计算，补齐UISN两种分割版本与Only Depth消融；发布六指标CSV、来源和候选覆盖分解图。已有precision重聚合与汇总一致，未复现对方模型。
- [x] [SuctionNet 方法对照与指标计算规范](docs/SUCTIONNET_COMPARISON_METRICS.md)：SuctionNet、UISN、Diffusion-Suction、MultiGraspNet 四项方法的 AP/Top-1、实机与效率指标；包含计算公式、官方输出接口，以及 R10/R20 和场景/实例 Top-1 的对应关系。

- [x] [E0–E5完整预训练与开发验证](docs/experiments/CD010-20261117/pretraining/REPORT.md)：每组230,400有效样本，合计100,800次成功更新与60个验证节点；E5 best step2880的Top-1为13.12%。E2已完成BF16 dense/FP32 sparse训练与原定开发验证。
- [x] [E5目标微调与401帧配对推理](docs/experiments/CD010-20261117/finetuning/README.md)：1000次成功更新，任务评分在固定历史动作验证上改善；[独立分析及结果图](docs/experiments/CD010-20261117/analysis/RESULTS_ANALYSIS.md)给出权重选择、收益与指标解释。

- [x] 数据工程 G0–G2 的生成与检查代码；已有旧数据作为训练起点，保留冻结物理参数和 S/W/C/A/B 标签。
- [x] 模型、预训练、微调与实验的 [详细规格和设计演进图](docs/implementation/README.md)。
- [x] [论文式方法架构图](docs/METHOD_ARCHITECTURE.md)：点云与稀疏 U-Net、实例上下文、五区接触编码、物理/任务双评分，以及预训练与目标适配；提供横版 SVG、PDF、高清 PNG、双语图注和绘图源代码。
- [x] 只读旧数据 adapter：恢复原候选轴、重建观测法向、区分质量与轴有效掩码，无需重制标签。
- [x] E0–E5 模型与配置：稀疏主干、实例融合、五区接触特征、姿态与质量头及辅助监督。
- [x] 预训练、checkpoint 保存与恢复、无标签推理、NMS 和按实际姿态重算的 R20 评价。
- [x] `dense_proxy` / `attempt_success` 微调：200+800 步阶段、源域回放、固定查询验证及独立成功头。
- [x] 模型实现阶段 CPU 测试复跑 **129 passed / 1 skipped**，梯度 checkpoint 一致性和两种微调跨阶段恢复复核；跳过项需要真实 release 路径。这里保留历史测试范围，各轮后续记录见 [验证说明](docs/MODEL_VALIDATION.md)。
- [x] 已有 H20 完整观测 smoke：**E1 2 次 + E5 18 次成功更新**，含重启与独立推理检查；32 个原候选的物理评价一致性检查通过。这里只记录历史链路验证，完整训练与测试效果见上方实验资料和方法对比。

- [x] v3 样本预算、硬件 profile、持久 supervisor、checkpoint/日志恢复与 CSV 分析导出；17 次零更新 profile 测量、六份 [正式配置及 SHA256](configs/runtime_profiles/cd009-v3/manifest.json) 已记录。
- [x] [P1 性能优化历史方案与实现](docs/PERFORMANCE_OPTIMIZATION.md)：精确 C FPS、core/lip 局部几何复用、CPU 几何上下文、按需读取与体素整理、有序预取和显式 checkpoint 迁移；该轮本地测试 **296 passed / 1 skipped**。四帧真实数据的共享骨干配对对照（含完整梯度回传）通过；H20/E5 在第 380 步保持学习状态迁移并正常续训。独立进程对照的数值差异及验收边界见方案；H20/E5、L20/E0、L20/E3 正常窗口中位数分别由 33.835→10.927、11.349→4.208、34.190→12.184 秒/更新。前后使用不同自然帧，E3 还改变了 GPU 放置；详见 [P1 真实检查与窗口证据](docs/performance/P1_REAL_RESULTS.json)。
- [x] P1 后 [11 组 batch/CPU 对照](docs/performance/P1_BATCH_CPU_RESULTS.json)：保留 B1A4、8 个逻辑核、2 个数据 worker；当时正常训练暴露的数据等待仅约 1–2 ms/update。公开记录对应 2026-11-12 的历史窗口；P3 后 E0/E1 的 loader 等待已重新显现。
- [x] [P2 全点 CPU 网格与坐标快路径](docs/PERFORMANCE_PHASE2.md) 通过真实四帧原容差对照并用于训练，最新 P3 来源 checkpoint 已确认其应用身份；voxel heads 的 AMP 梯度对照未通过，继续关闭。
- [x] 读取 [H20 P2 诊断](docs/performance/H20_P2_BATCH_DIAGNOSIS.md) 和 [L20 P2 诊断](docs/performance/L20_P2_BATCH_DIAGNOSIS.md)，完成 [batch 可行性分析](docs/H20_BATCHSIZE_ANALYSIS.md)：H20 B2 仅 +2.20%，L20 64k 下 B2 仍慢于 B1，当前继续 B1A4。L20/E5 的 16k→64k 吞吐 +28.40% 属于历史零更新短测；后续实际 E3 checkpoint 的 64k 配对未过原梯度门槛，当前选择 16k。
- [x] [H20/E2 历史启动记录](docs/performance/H20_E2_STARTUP.md) 已归档；E2 不启用接触分组。其既有 FP16 投影溢出仍是独立未解决问题，不能由启动记录推定当前恢复。
- [x] [P3 原生接触几何及 split 分块](docs/PERFORMANCE_PHASE3.md) 已整合、修复原生几何正负零字节差异并完成实际接续：H20/E5 使用 [64k 配置](configs/performance/cd009-p3-integrated.json)，L20/E0/E1/E3 使用 [16k 配置](configs/performance/cd009-p3-l20-16k.json)。各首 25 次正常更新的中位数分别为 **5.7462、1.8446、1.7423、3.2120 秒/更新**；E0/E1 无接触分组，改善主要出现在反向。实际运行代码为 `f4f7f0d`，详见 [配对、失败取舍与正常窗口](docs/performance/P3_INTEGRATED_RESULTS.md)。
- [x] [优化全程总览](docs/PERFORMANCE_HISTORY.md)：整理初始 profile、P1/P2/P3 的实现、实测、未采用路线和当前状态，并同步各运行指南。局部 CPU 原型、零更新短测和自然训练窗口分开呈现，不相乘为累计加速承诺。
- [x] supervisor 支持显式登记的 L20 GPU4 和独立 `control_repo`，管理代码可与固定 trainer checkout 分开。仍是单卡单组实验，不是多卡 DDP。
- [x] [E4 / L20 GPU4 启动](docs/performance/E4_GPU4_STARTUP.md)：使用固定 `f4f7f0d` trainer 与 16k + split + native，原预算内首 100 次成功更新、正常暂停 checkpoint 和最终备份通过，恢复首步 101 的采样连续性通过；首 100 步中位数为 4.4400 秒/更新。读取时观察到 107 步、当时持久化为 100 步；这是当时的启动与恢复记录，当前训练状态见上文。
- [x] [CD-010 从头训练性能实现](docs/FROM_SCRATCH_PERFORMANCE.md)及[真实 H20/L20 短测](docs/performance/CD010_FROM_SCRATCH_RESULTS.md)：split、共享体素头、分段实例广播反向、FP32原生几何、缓存与预取、显式精度、融合AdamW、每更新loss读回。H20/E5已测支持64k/geometry8/affinity16，B2A2比B1A4更快；128k只提高0.50%，生成器现保留64k。L20/E3支持geometry8/affinity16，E0支持6 workers，E2的BF16 dense/FP32 sparse已通过短程更新。详细数值、版本和范围以实测文档为准。
- [x] [CPU几何/GPU重叠实现](docs/GEOMETRY_GPU_OVERLAP.md)及[两机实测](docs/performance/CD010_GEOMETRY_OVERLAP_RESULTS.md)：H20/E5与L20/E3分别提速19.90%和33.82%，allocated峰值分别45.28和11.72 GiB。四项短测均正常结束，不再重复串行对照。
- [x] [显存/batch扫描](docs/performance/CD010_GEOMETRY_MEMORY_RESULTS.md)已执行全部12档，9档完成、3档OOM；人工显存上限已取消。H20/E5 microbatch4/depth2为本轮最快，L20/E3 B2只比B1快0.17%，保留microbatch1/depth2。两机的大batch单图方案都更慢，继续保留几何重叠。

- [x] [接触头可选实现](docs/CONTACT_HEAD_OPTIMIZATION.md)：v4 `contact_feature_gather_mode="zone"`按区域一次gather，原`chunk`路径保留。[最终执行入口](docs/FINAL_OPTIMIZATION_AND_TRAINING.md)使用`--suite final`，H20五档、L20四档均为有效batch8；`--run --start-training --training-root`在短测后自动选择并启动各机正式任务，新头未达3%收益则采用最快原路径。
- [x] [九档接触头实测与六组启动](docs/performance/CD010_FINAL_OPTIMIZATION_RESULTS.md)：全部完成，两机均选zone128，正式训练采用有效batch8；该结果只覆盖本轮有效batch8条件，没有测试B4A6/zone128。
- [x] [batch/累积组合补测入口](docs/BATCH_ACCUMULATION_SUPPLEMENT.md)：`--suite batch-combo`一次执行本机六档，固定zone128与depth2，保留整体及同有效batch的吞吐排名和样本预算换算。[十二档真实补测](docs/performance/CD010_BATCH_ACCUMULATION_RESULTS.md)已全部完成，暂用训练卡的E5/E3已按原配置恢复。
- [x] [选定正式配方及从头启动](docs/SELECTED_PRETRAINING_RECIPE.md)：`finalize_pretraining.py`支持读取已完成的batch-combo结果，选择各机最快完整候选，生成H20有效batch48、L20有效batch12的六组新配置；不再锁定有效batch8，预算从配置派生，旧final入口保留历史兼容。

## TODO（待完成）

- [ ] 在已有87条未参与选模的有标签test历史动作上，对冻结的E5预训练best与微调best完成一次同动作评分比较；开发验证和401帧新候选推理已完成。
- [ ] 接入原定正常工况业务执行对比，以新动作的实际吸取和搬运结果判断部署收益。

可选派生数据 sidecar 缓存与 residual uncertainty 工具尚未实现，仅在实际需要时安排，不是上述训练的前置条件。现有测试和短训练记录没有替代全量数据生产验收或真实场景效果实验。

## 模型与训练实施文档

从 [实施文档总览](docs/implementation/README.md) 进入。优先阅读 [已完成旧数据的适配方案](docs/implementation/LEGACY_DATA_ADJUSTMENT.md)：已有 Seal/Wrench/Collision/GT-A/GT-B 原样复用，只在训练适配层恢复候选轴和补齐观测法向；可选派生缓存不要求重制整套标签。

| 内容 | 文档 |
|---|---|
| 最新预训练、微调与推理结果 | [完整实验数据](docs/experiments/CD010-20261117/README.md) · [独立分析与选模](docs/experiments/CD010-20261117/analysis/RESULTS_ANALYSIS.md) |
| 公开方法效果与UISN对比 | [完整结果、输入条件与差距分析](docs/SUCTIONNET_METHOD_COMPARISON.md) · [机器可读表与来源](docs/experiments/COMPARISON-20261118/README.md) |
| 公开方法对照与指标算法 | [四项 SuctionNet 方法、官方 AP 计算与 R20 对应](docs/SUCTIONNET_COMPARISON_METRICS.md) |
| 论文式方法架构图 | [三子图总览、双语图注与代码对应](docs/METHOD_ARCHITECTURE.md) · [矢量 PDF](docs/figures/paper-architecture.pdf) · [可编辑 SVG](docs/figures/paper-architecture.svg) |
| 原设计如何逐步修正 | [设计演进与图示](docs/implementation/DESIGN_EVOLUTION.md) |
| 输入、掩码、批数据 | [Data adapter](docs/implementation/DATA_ADAPTER.md) |
| 模型结构、维度、分组与损失 | [模型规格](docs/implementation/MODEL.md) |
| 公开预训练原始规格与历史配置 | [v3 固定配置、策略与边界](docs/CD009_PRETRAINING.md) · [原训练规格](docs/implementation/TRAINING.md) |
| 缩短现有预训练耗时 | [优化全程与当前结果](docs/PERFORMANCE_HISTORY.md) · [P3 实施与接续](docs/PERFORMANCE_PHASE3.md) · [P2 历史反馈](docs/PERFORMANCE_PHASE2.md) · [H20 显存与 batch 可行性](docs/H20_BATCHSIZE_ANALYSIS.md) |
| 当前正式训练配方与性能结果 | [选定配置并从头启动](docs/SELECTED_PRETRAINING_RECIPE.md) · [十二档补测结果](docs/performance/CD010_BATCH_ACCUMULATION_RESULTS.md) · [接触头改动](docs/CONTACT_HEAD_OPTIMIZATION.md) |
| 少量自有数据微调 | [数据准备方案评估与正确字段](docs/TARGET_FINETUNING_DATA_REVIEW.md) · [微调接口与策略](docs/implementation/FINETUNING.md) |
| 消融、R20评价与目标验证 | [实验方案](docs/implementation/EXPERIMENTS.md) |

可执行命令见[运行指南](docs/MODEL_USAGE.md)。完整预训练和两种微调配置已提供；可选派生数据 sidecar 转换与 residual uncertainty 仍未实现。

## 目录

```text
configs/              # 标签物理参数、E0–E5预训练与有界smoke配置
suction20/             # 数据生成、只读适配、损失、训练状态与评价
scripts/               # 数据制作、训练、微调、评价和推理入口
tests/                 # 单元/合成场景测试
docs/                  # 数据字段、流程与项目计划
models/                # 稀疏主干封装、实例/接触分组与评分头
docs/implementation/   # 模型/预训练/微调/实验的详细规格与配图
```

## 数据制作入口

以下用于新增数据制作。用户已经制作完成的数据训练时直接走上述adapter，不再次执行此流水线。

```bash
pip install -r requirements.txt
pip install open3d pytest
git clone https://github.com/graspnet/suction_annotation.git /workspace/suction_annotation

export DATASET=/data/SuctionNet
export UPSTREAM=/workspace/suction_annotation
export OUT=/data/suction20_output

bash scripts/run_data_pipeline.sh
```

该命令需要本机已准备 SuctionNet 数据、`dense_point_clouds`、原始 Seal/Wrench 标注和场景 RGB-D 数据。输出写入 `$OUT`，不提交大型原始数据或生成标签至 GitHub。

具体步骤： [数据生成流程](docs/DATA_PIPELINE.md) · [字段定义](docs/DATA_SCHEMA.md) · [项目路线](docs/ROADMAP.md)。

## 重要约定

- G0 保持官方 10 mm Seal 方法和 ±1 mm 环带；G1 使用同样 72 扇区方法、20 mm 半径与 ±2 mm 环带。
- 20 mm 场景标签分别保存 `wrench_annot` 和 `wrench_eval`；GT-B 使用 `wrench_eval`。
- 当前producer用候选物体法向作为物理标签和主 `gt_normal`；PCA法向独立保存为 `gt_normal_pca`。旧缓存的同名 `gt_normal` 可能是PCA，训练时按旧数据适配方案恢复原候选轴。
- 场景点只通过同实例、空间距离、法向门控匹配最近候选。匹配失败使用 `gt_valid=False`，不是已知负样本。
- 不重新发布第三方原始标注脚本或数据；需要在本地拉取上游开源仓库。

原始数据使用和科研结果均需遵守 SuctionNet/GraspNet 的相应许可证。



