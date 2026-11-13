# SuctionNet Modern 20mm

面向 **半径 20 mm 单吸盘**的 SuctionNet 数据制作、公开数据预训练与少量自有场景微调项目。

当前已纳入的数据流程：SuctionNet 10 mm 标签对照（G0）→ 20 mm Seal 重标注（G1）→ 场景 Wrench/Collision → 每帧 RGB-D 点云与吸取候选映射 → GT-A/GT-B → 数据质量检查（G2）。

**当前状态（2026-11-13）：** 数据制作、只读适配、E0–E5模型、预训练、微调与R20评价已实现，旧数据继续复用。[12档batch扫描](docs/performance/CD010_GEOMETRY_MEMORY_RESULTS.md)已完成：H20支持B4/depth2，L20保留B1/depth2。用户已明确这是[最后一轮优化](docs/FINAL_OPTIMIZATION_AND_TRAINING.md)：两机统一正式有效batch8，共做9档定向短测（H20五档、L20四档），完成后无论新接触头是否提速，都立即采用所选配置从头启动正式训练，不再增加优化轮次或等待确认。仓库最后发布的完整训练状态仍为停止；当前聊天环境没有真实GPU执行通道，不宣称已经启动。

## Done（已完成）

- [x] 数据工程 G0–G2 的生成与检查代码；已有旧数据作为训练起点，保留冻结物理参数和 S/W/C/A/B 标签。
- [x] 模型、预训练、微调与实验的 [详细规格和设计演进图](docs/implementation/README.md)。
- [x] 只读旧数据 adapter：恢复原候选轴、重建观测法向、区分质量与轴有效掩码，无需重制标签。
- [x] E0–E5 模型与配置：稀疏主干、实例融合、五区接触特征、姿态与质量头及辅助监督。
- [x] 预训练、checkpoint 保存与恢复、无标签推理、NMS 和按实际姿态重算的 R20 评价。
- [x] `dense_proxy` / `attempt_success` 微调：200+800 步阶段、源域回放、固定查询验证及独立成功头。
- [x] 模型实现阶段 CPU 测试复跑 **129 passed / 1 skipped**，梯度 checkpoint 一致性和两种微调跨阶段恢复复核；跳过项需要真实 release 路径。这里保留历史测试范围，各轮后续记录见 [验证说明](docs/MODEL_VALIDATION.md)。
- [x] 已有 H20 完整观测 smoke：**E1 2 次 + E5 18 次成功更新**，含重启与独立推理检查；32 个原候选的物理评价一致性检查通过。这里只记录链路验证，完整训练效果见下方 TODO。

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

## TODO（待完成）

- [ ] 执行[最后一轮短测并直接启动训练](docs/FINAL_OPTIMIZATION_AND_TRAINING.md)。六组均有效batch8、230,400有效样本、28,800次成功更新；warmup1,440次更新、每2,880次更新一次原定验证，共10个节点。用户已授权测试结束后立即启动，无收益也采用原路径开始训练。
- [ ] 用最终选定设置记录长期吞吐与原定验证结果；已完成的资源/重叠短测和旧P3历史配对不重复，旧CD-009不再恢复。
- [ ] 完成E2的BF16 dense/FP32 sparse新训练与原定验证；短程有限更新已通过，不据此推定完整训练收敛。
- [ ] 仅用 **H20/E5、L20/E3各一个实验**完成最后9档比较：H20五档、L20四档。该轮结束即关闭优化阶段，不再扩batch/worker/allocator/trace矩阵，不追加恢复审计或训练前完整验证；CUDA几何与GPU增强不再作为本次开训前任务。
- [ ] 按 [实验方案](docs/implementation/EXPERIMENTS.md) 比较验证指标与计算代价，选定用于迁移的 `gt_B` 模型；锁定方案后报告官方测试划分结果。
- [ ] 用已有少量自有场景数据运行选中模型的 zero-shot 与对应监督模式微调，并在独立目标场景比较效果。
- [ ] 进入业务执行阶段后，以相同动作参数完成正常工况效果比较。

可选派生数据 sidecar 缓存与 residual uncertainty 工具尚未实现，仅在实际需要时安排，不是上述训练的前置条件。现有测试和短训练记录没有替代全量数据生产验收或真实场景效果实验。

## 模型与训练实施文档

从 [实施文档总览](docs/implementation/README.md) 进入。优先阅读 [已完成旧数据的适配方案](docs/implementation/LEGACY_DATA_ADJUSTMENT.md)：已有 Seal/Wrench/Collision/GT-A/GT-B 原样复用，只在训练适配层恢复候选轴和补齐观测法向；可选派生缓存不要求重制整套标签。

| 内容 | 文档 |
|---|---|
| 原设计如何逐步修正 | [设计演进与图示](docs/implementation/DESIGN_EVOLUTION.md) |
| 输入、掩码、批数据 | [Data adapter](docs/implementation/DATA_ADAPTER.md) |
| 模型结构、维度、分组与损失 | [模型规格](docs/implementation/MODEL.md) |
| 正在运行的公开预训练 | [v3 固定配置、策略与边界](docs/CD009_PRETRAINING.md) · [原训练规格](docs/implementation/TRAINING.md) |
| 缩短现有预训练耗时 | [优化全程与当前结果](docs/PERFORMANCE_HISTORY.md) · [P3 实施与接续](docs/PERFORMANCE_PHASE3.md) · [P2 历史反馈](docs/PERFORMANCE_PHASE2.md) · [H20 显存与 batch 可行性](docs/H20_BATCHSIZE_ANALYSIS.md) |
| 独立从头训练与最后一轮优化 | [最终测试并开训](docs/FINAL_OPTIMIZATION_AND_TRAINING.md) · [batch实测](docs/performance/CD010_GEOMETRY_MEMORY_RESULTS.md) · [接触头改动](docs/CONTACT_HEAD_OPTIMIZATION.md) |
| 少量自有数据微调 | [微调接口与策略](docs/implementation/FINETUNING.md) |
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


