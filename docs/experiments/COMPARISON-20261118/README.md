# SuctionNet 方法比较结果与数据说明

日期：2026-11-18。正文见 [方法比较报告](../../SUCTIONNET_METHOD_COMPARISON.md)，定义与官方公式见 [评价指标规范](../../SUCTIONNET_COMPARISON_METRICS.md)。

本目录重新聚合已经完成的 E5 测试结果，并摘录已核实的公开论文表格。本次发布没有启动训练、模型推理、物理评价或机器人实验，也没有复现其他论文的模型。论文值与本地计算值分别标注。这里的数值比较保留输入、训练目标、分割来源与评价实现的差异。

## 文件

| 文件 | 内容 | 数值单位 |
|---|---|---|
| [method_metrics.csv](method_metrics.csv) | 8 个完整配置或明确标注的消融配置 × 3 个测试 split，共 24 行；每行保留六项公开基准指标及条件、来源 | `units=percent`，例如 `16.328757735434` 表示 16.328757735434% |
| [diagnostics.json](diagnostics.json) | E5 的 R10/R20 汇总和排名/覆盖分解，R20 每实例指标，UISN 消融与证据定位，历史开发验证、微调分类、效率及参数摘要 | 各部分显式注明；项目分数通常为 `[0,1]` 小数，论文表格为百分数，差额为百分点 |

机器文件只保留汇总数值与必要出处。完整逐帧预测、训练记录、个人数据及原运行机器的绝对路径不包含在本目录中。`diagnostics.json` 的 `provenance.source_files` 保存源文件名、SHA-256 和用途；所列源文件是本次整理的输入标识，不表示这些原始文件全部随本目录发布。

## CSV 的列与缺失值

每行唯一键为 `method + variant + camera + split`，其中 `split` 为 `seen / similar / novel`。八个配置分别为 E5、S1B、Diffusion-Suction、MultiGraspNet-vac.、MultiGraspNet 联合版本、UOIS-Net+UISN、UOAIS-Net+UISN 和 UISN Only Depth 消融。

| 列 | 含义 |
|---|---|
| `method`, `variant` | 方法与完整配置标识；不同 UISN 分割器、MultiGraspNet 两种训练方式分别列出 |
| `camera`, `camera_evidence` | 相机及判定依据；Only Depth 的 `camera=realsense` 是与主表一致数值关联得到的推断，必须同时读取 `camera_evidence` |
| `AP_top50`, `AP04_top50`, `AP08_top50` | Top-50 的平均 AP、质量阈值 0.4 和 0.8 的 AP |
| `AP_top1`, `AP04_top1`, `AP08_top1` | 每帧全场景最高模型分数姿态的对应 AP |
| `result_kind` | 已有本地计算结果、未复现的论文主表值或论文消融值 |
| `input_features`, `input_mask_source`, `oracle_instance_input` | 输入模态、分割来源及是否使用 GT 实例；未核实的条件明确标注 |
| `training_target`, `train_scene_count` | 训练目标与已知训练场景数；不从测试结果推测训练设置 |
| `evaluator`, `evaluator_commit` | 评价器及已核实的实现版本；论文值未在本地重算，不能自动继承 E5 的固定提交 |
| `source_file`, `source_locator`, `source_url`, `notes` | 来源文件、论文表格/页码或 JSON 定位、公开网址及必要限制 |

论文值按原表的两位小数精度保存；E5 的百分数保留至小数点后最多 12 位，原始小数精度另保存在诊断文件。CSV 数值单元格不含 `%` 字符，方便程序直接读取。

**空单元格表示未报告或未知，不表示 0。** MultiGraspNet 的两个配置均没有主表 Top-1 三项，因此共留空 18 格。Only Depth 没有 AP₀.₄ / AP₀.₈ 的 Top-50/Top-1 分项，因此另留空 12 格。其他论文没有在本次摘录中核实的训练场景数和评价器提交同样留空。没有根据其他指标推算缺失值。

Top-50 AP 对 `k=1…50` 的 Precision@k 和质量阈值 `{0.2, 0.4, 0.6, 0.8}` 求均值；它不是 P@50，也不是 PR 曲线面积。Top-1 指全场景排名第一的姿态，不是每实例 Top-1。[官方评价代码](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/suctionnet_eval.py)

## E5 的条件与两套评价

E5 使用已有 `best step2880` 权重，checkpoint SHA-256 为 `5bc2c14c014fbf23cbef48062a0dac523ea243b6c18e540bb5af022ff7d0ea3a`。输入为观测深度、观测法向、重力及 **GT oracle-instance masks**；RGB 用于展示，不作为评分输入。训练目标为项目 R20，使用 90 个官方训练场景，另有 10 个开发验证场景。完整测试为 RealSense 的 90 个测试场景、23,040 帧；三个 split 各有 30 场景 ×256 视角。来源是 `report_data.json`、`official_summary.json` 和已汇总的 `numeric_notes.json`。

CSV 的 E5 行是 **R20-trained / official R10 evaluation / GT oracle-instance input**。官方 API 固定为 `0dac9f7e5b2f7ea801cddf9655cb82a259681976`。论文方法的输入与训练配方没有因此被统一，数值差额不能直接解释为某个模块的净贡献。

诊断文件另保留项目 R20 的已有结果。R10 与 R20 使用不同物理判定、NMS 和候选保留规则，二者差额不是单独改变吸盘半径的消融。`protocols.r20.*.metrics.AP_top1` 是四阈值平均的**场景级**指标；`r20_rows[].instance_Top1_Q20_at_0_4` 是固定阈值 0.4 的**每实例**指标。历史开发集的 10.60% AP-R20 与 13.12% 每实例 Top-1 仍属于 10 场景 ×32 帧的开发验证，不能替代完整测试行。

## 论文摘录与证据位置

| 方法 | 一手来源与取值位置 | 本目录保留的限制 |
|---|---|---|
| S1B | [SuctionNet 原论文](https://arxiv.org/pdf/2103.12311)，Table IV，RealSense 为斜杠后数值 | 按原论文保存；不替换为 UISN 中个别相差 0.01 个百分点的 Cao 等人基线行 |
| Diffusion-Suction | [论文 v2](https://arxiv.org/html/2502.07238v2)，Table I 的 SuctionNet 实验 | 不混用 Table II 的 Parcel-Suction 成绩；输入条件与 E5 未逐项对齐，当前摘录未核实精确输入模态 |
| MultiGraspNet | [论文 v2](https://arxiv.org/html/2602.06504v2)，Table I 的 Vacuum 部分 | 单吸取与联合版本分列；未报告 Top-1；论文的 seal coefficient 术语与官方 Seal×Wrench 之间的底层实现等价性尚未核实 |
| UISN | [论文 DOI](https://doi.org/10.1109/LRA.2024.3385609)，用户提供的原论文 PDF | Table I：PDF 第 6 页/印刷页 4939；Table III：同页；Table V：PDF 第 7 页/印刷页 4940；数值已按渲染后的表格核实 |

UISN PDF 的文件名为 `Uncertainty-Aware_Suction_Grasping_for_Cluttered_Scenes.pdf`，SHA-256 为 `17f3cf68c1aec51b076f27a46ca553bc94bc44778e06a7bc4232428f7acfd54c`。主表原列顺序为 AP、AP₀.₈、AP₀.₄，CSV 按字段含义统一为 AP、AP₀.₄、AP₀.₈。[作者评价脚本](https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/eval.py) 调用官方三个测试 split 的 API；该事实不代表本次重新运行了 UISN。

UISN 有两处必须连同数值保留的正文差异：

- Table I 的表注明确写 **Kinect / RealSense**，而 PDF 第 5 页/印刷页 4938 的 §IV-B 正文写反。这里采用表注，并以原 SuctionNet 基线结果交叉核对相机归属；斜杠后是 RealSense。
- Table V 没有注明相机。其 RGB-D 行的六个 AP 数值与 Table I 的 **RealSense UOIS-Net+UISN** 行完全一致，因此 Only Depth 与 RealSense 的关联是推断。该表 Novel Top-1 的 Only Depth 为 12.43%，RGB-D 为 11.83%，不能据正文概括为 RGB-D 在所有 split 的两项指标都提高。

Only Depth 以全 1 初始特征替代 RGB，保留论文架构。它与 E5 的输入模态更接近，但仍没有统一分割、训练和后处理条件。Table III 仅报告不确定性消融的 Top-50 AP；Table II 比较的是完整几何评分与网络评分，并非独立局部接触模块的消融。Tables II–V 中没有单独报告局部接触模块消融。证据与原表数字保存在 `diagnostics.json.uisn.ablations`。

## 诊断文件的结构与读法

| JSON 路径 | 内容 |
|---|---|
| `provenance.source_files`, `provenance.public_sources` | 输入文件 SHA-256 与公开论文/代码网址 |
| `e5_configuration` | E5 权重身份、输入条件与数据范围 |
| `protocols.official`, `protocols.r20` | 三个 split 的六项 AP、各阈值结果、完整性、排名/覆盖统计和分场景集中度摘要 |
| `r20_rows` | 三个 split 的 R20 场景指标及原有每实例 Top-1、碰撞率、缺候选率 |
| `uisn.main_table_evidence`, `uisn.paper_inconsistencies` | UISN 表注、列顺序、页码和正文差异 |
| `uisn.macro_percent`, `uisn.e5_minus_uisn_pp`, `uisn.ablations` | 单指标的三 split 宏平均、E5 减论文值的百分点差、Tables II–V 消融 |
| `development` | E0–E5 已有开发验证汇总；保留不同 checkpoint 步数 |
| `target_classification` | 历史动作的微调前后分类结果和 checkpoint 身份 |
| `inference_efficiency`, `model_sizes` | 已有推理时间、参数数目与文件大小摘要 |

覆盖分解从已有 precision 张量获得。固定阈值下，`k × precision@k` 是前 k 名的通过数；统计只涉及各自后处理**已经保留的 Top-50**。零增量既可能是碰撞/低质量，也可能是候选缺位，不能据此唯一归因。每个阈值的帧数满足：

`top1_good_frames + good_in_retained50_but_not_top1_frames + no_good_in_retained50_frames = 7680`。

`top1_capture_given_retained50_has_pass_each_threshold` 是“保留 Top-50 至少有一个通过时，Top-1 也通过”的条件比例；它与无条件 Top-1 AP、原始候选覆盖率不同。所有阈值数组顺序均为 `[0.2, 0.4, 0.6, 0.8]`。`macro_metrics` 对三个 split 的同一个指标作等权平均；不平均不同含义的指标。

`target_classification` 的 399 条历史动作记录参与了 checkpoint 选择，属于开发验证。AUROC 与 PR-AUC 衡量既有动作的评分关联，不能改写成新策略机器人成功率，也不能作为微调后的完整公开 R10/R20 成绩。`inference_efficiency` 使用已有 H20-3e、batch=1 记录，原深度与双目深度分别为 1,139 与 1,158 条，非同帧配对实验；计时不覆盖相机采集、前置分割或双目深度生成。`checkpoint_bytes` 包含训练/优化器状态，不能当作纯推理权重大小。以上限制均来自对应汇总文件的记录口径。

## 一致性核对

本目录核对了 24 个唯一方法配置/split 记录、六项指标的字段映射、百分数转换、30 个应留空指标，以及 R10/R20 各阈值覆盖帧数的分解。E5 六项汇总与已有保存结果之间只存在浮点舍入量级的差异，保留在 `summary_difference` 中。此次核对是文件与已有结果的完整性检查，不是新的模型实验。
