# A/B 与 SuctionNet 现有方法的公开对照协议

本页固定 **本方法 A 或 B 与每一个参考方法分别比较** 的执行方案。论文报告值优先，不要求重训或重新运行参考方法；不制作参考方法之间的排行榜、差值表，也不把 A/B 互比作为这一轮公开验证的必做实验。每个分支的 `EXPERIMENT_PLAN.md` 提供本路线的独立成对结果表。[机器可读论文数值及来源](reference/paper_baselines.json)是转录目录，不是已实现的训练配置。

**当前状态：方案和论文数值已整理，A/B 新模型尚未完成训练；本方法结果均为“待测”。** 既有 E5、原项目 R20、单物体 INST20 和这里的公开对照是不同结果身份。旧报告继续保留在[方法比较](../SUCTIONNET_METHOD_COMPARISON.md)，不能将 E5 数字复制到 A 或 B 行。

## 1. 比较对象、数量与结论范围

| 比较单元 | 本方法的实例前端 | 参考结果 | 本轮使用方式 |
|---|---|---|---|
| 本方法 vs SuctionNet / S1B | `uois_no_gt_workspace`，计入完整系统 | 原论文 v2 Table IV | 主对照；完整场景协议 |
| 本方法 vs Diffusion-Suction | `uois_no_gt_workspace` | 原论文 v2 Table I 的 SuctionNet 部分 | 论文协议对照；保留未公开实现字段 |
| 本方法 vs UOIS-Net + UISN | `uois_uisn_gt_aabb20` | UISN Table I 的完整 UOIS 配置 | 独立成对对照 |
| 本方法 vs UOAIS-Net + UISN | `uoais_uisn_gt_aabb20` | UISN Table I 的完整 UOAIS 配置 | 独立成对对照；不与 UOIS 参考行比较 |
| 本方法 vs MultiGraspNet-vac. | `uois_no_gt_workspace`，计入完整系统 | 原论文 v2 Table I Vacuum | 暂为论文参考，待评价物理语义核验 |
| 本方法 vs MultiGraspNet joint | 同上 | 同表独立 multitask 行 | 暂为论文参考；另注明平行夹爪额外监督 |

每条路线只需一份 source-only 的最终评分模型，配三份不同输入条件的完整推理：UOIS/无GT工作区、UOIS/UISN工作区、UOAIS/UISN工作区。只有前端与工作区身份均相同的表才复用同份预测；评分权重不因对手改变。没有为六张表安排六次模型训练，也不从两种 UISN 配置挑各列最高值拼成不存在的方法。

“保持一致”约束的是测试数据、可用观测与 oracle 边界、外部物理评价和指标计算。网络、内部点数、候选生成、法向估计、损失与优化器是各方法自身的设计，不强行改成相同算法。**主表允许本方法 R20 与论文 R10 的吸盘尺寸差异，结论属于跨尺寸完整系统对照；不是只改架构的因果消融，也不能将全部差值归因于网络。** 对手未披露的历史依赖、训练资产和前端版本必须写 `unreported`，不能把未知当一致。

## 2. 必须固定的数据与输入条件

| 项目 | 本方法公开对照的固定值 | 不允许混入 |
|---|---|---|
| 数据集 | SuctionNet-1Billion 官方场景及完整模型资产 | Parcel-Suction 或自定义物体子集 |
| 相机 | `realsense`；使用对应内参、深度单位、外参 | Kinect 数值、两相机混合平均 |
| Seen | scene0100–0129，30 场景 × 256 帧 = 7,680 帧 | 开发集、抽帧子集 |
| Similar | scene0130–0159，同上 | 仅容易实例或成功分割帧 |
| Novel | scene0160–0189，同上 | 仅指定可吸类别 |
| 全测试覆盖 | 每场景 view0000–0255，共 23,040 帧 | 用每实例平均替换全场景 AP |
| 模型开发 | 既定 90 train / 10 dev，仅使用 scene0000–0099 | 用官方 90 个 test 场景调参 |
| 公开最终训练 | 配置冻结后从头使用官方 100 train 场景、全部 256 帧的可用实例清单 | 继续把 90 场景 checkpoint 标作 100 场景模型 |
| 公开 checkpoint | `public20`、source-only、140,000 成功监督实例 / 8,750 成功更新，固定 `last` | 产线成败训练、目标域微调权重、测试选出的 best |
| 测试实例 | RGB-D → 冻结预测分割 → A 或 B → 全场景候选 | GT 实例掩码、GT 身份、GT 姿态或 teacher 质量筛选 |

100 场景 refit 后，原 10 个 dev 场景已经进入训练，不能继续凭它们挑 `best`。模型从头初始化，不从使用产线标签的权重接续；预算和冻结配置在 refit 前登记。native 90/10 与产线 1,676/399/87 的训练方案继续独立保留，不作为公开表中的评分模型。[共同训练合同](TRAIN_EVAL_REPRODUCE.md)

预测前端是完整系统的一部分：记录模型名、代码/权重 SHA、分割训练数据来源、预处理、mask 缓存 SHA 和失败数；前端使用的额外数据或预训练必须披露，不能因此宣称“所有训练信息也完全相同”。S1B / Diffusion 的全场景输入不要求它们改用我们的分割器；我们不能获得额外测试 GT 实例信息。UISN 对照则应复用作者对应 UOIS/UOAIS 设置；若原权重或缓存无法确认，保留 `frontend_match=unverified`，该行不能称严格匹配。

工作区必须按比较单元区分，不能统称“官方共同GT工作区”：

| 推理 profile | 工作区输入与顺序 | 适用比较 |
|---|---|---|
| `uois_no_gt_workspace` | 全图预测UOIS mask及有效RGB-D；不读取测试GT工作区/实例标签 | S1B、Diffusion；MG参考。Diffusion原工作区细节仍为unverified |
| `uois_uisn_gt_aabb20` | 复用UISN流程：先预测全图mask，再对GT前景 `seg>0` 的联合3D AABB外扩20mm，筛选cloud/color/net_seg，最后按预测实例分组 | 仅UISN-UOIS |
| `uoais_uisn_gt_aabb20` | 同一UISN工作区规则，换对应UOAIS预测mask | 仅UISN-UOAIS |

官方S1B推理未读取GT工作区；UISN的上述步骤会改变支持点和候选池，不能称它对query没有影响。后两种profile仅允许这项与UISN一致的GT联合前景工作区：先将实例标签坍成二值前景，不用GT ID拆实例、修正漏检、提供评分特征或按GT质量裁撤候选。除该明确的UISN工作区路径外，GT完整模型和场景姿态不进入模型侧；外部evaluator正常读取其所需GT模型、姿态与桌面几何。工作区使用的相机变换及扩张规则也按作者源码锁定。[S6]、[S7]

每profile保存 `workspace_mode/source/stage/outlier_m` 及预测SHA。漏分割、少点实例、无法向或零候选都产生真实的空输出并保留帧分母；缺预测文件、模型异常和资产读取失败属于未完成，不能伪装为空预测。官方数据与划分依据见 [S1]、[S5]。

## 3. 吸盘大小例外：单独建立官方物理评价族

协议名称固定为 **`sn-paper-radius-v1`**，本方法主表 `radius_m=0.020`，论文参考记录官方默认 `0.010` 或作者未明确披露的证据状态。本协议是待实现的项目适配层，不冒称官方已发布 R20 evaluator。

### 3.1 为什么不能直接调用已有 R20 / INST20

原项目 `suction20/eval_r20.py` 使用冻结 Seal 归一化、切向力矩 L2 范数以及自定义三段碰撞包络；新 `gt_INST20_pose_v2` 还去掉外部环境碰撞，主要统计实例 Top1。它们都不是“官方协议只改半径”。因此，**公开 AP 不读取缓存 Q_inst 或旧 Q20，不用它们判定命中**；外部 evaluator 根据输出的真实 p/n、官方完整几何重新评价。[旧指标差异说明](../SUCTIONNET_COMPARISON_METRICS.md)

模型内部的 18–22 mm lip、v3 法向和 target-only teacher 保持设计身份，属于模型训练方法。它们不覆盖公开外评的物理定义。公开 R20 的唇口检索仍为 **r ± 1 mm，即 19–21 mm**；不能顺手扩大为 18–22 mm。

### 3.2 最小改动白名单

基准源码冻结为 `graspnet/suctionnetAPI@0dac9f7e5b2f7ea801cddf9655cb82a259681976`。[S5] 中 `utils/eval_utils.py` 有三处独立半径来源，须统一显式传入 `radius_m`，不能只改模块常量后漏掉函数局部值。

| 源码位置 / 作用 | 唯一允许的几何参数变化 | 必须保持的定义 |
|---|---|---|
| 模块级 radius / wrench threshold | `0.01 → radius_m`；调用时算 `15.6 × π × radius_m` | 原重力坐标变换、完整稠密模型均值质心、切向力矩绝对值 max 范数 |
| `get_smoothness_score` 局部 radius | 圆周、ideal lengths、内部 `rho < r`、检索环共同使用 r | 原始 Seal，不做项目 min/max 归一化；72 扇区；检索半宽 1 mm；自碰 h > 5 mm |
| `collision_detection` 局部 radius | 柱体径向 `rho < r` 使用 r | 原场景/桌面碰撞；轴向起点 10 mm、终点 100 mm |

参数传播链为 `SuctionNetEval → eval_scene → eval_suction → collision_detection / get_suction_score → get_smoothness_score / get_wrench_score`。保留上游原文件快照，适配代码独立存放并保存 patch SHA。不要原地覆写环境里的官方包或运行中修改全局 radius；导入时已算出的 `wrench_thre` 也必须消除过期 R10 值。数值运算沿用固定API的dtype、公式和顺序；半径作为常量参数的重构不得顺带向量化或改精度。

不随吸盘尺寸缩放 NMS、环宽、轴向高度、阈值、模型体素或候选数量。阈值、严格/非严格边界、场景重力变换、稠密模型生成与采样顺序沿用固定 API；实际稠密几何文件/依赖/随机状态记 hash。原论文未披露历史 API SHA 或模型采样 hash 时写 `unreported`。**R10 一致性验收证明的是本适配层与本项目冻结 API 等价，不证明复原了作者未公开的历史环境。**

### 3.3 R10 准入与可选同尺寸补充

先在 `radius_m=0.010` 对未修改API可完成的非空案例核对逐候选 Seal/Wrench/碰撞、NMS及归属/保留索引、置信度排序和每帧 `[50,4]` precision；覆盖正常、碰撞、阈值边界和少于50候选。离散结果须相同；浮点容差在执行前固定为 `atol=1e-8, rtol=1e-7`（相同 dtype/环境），最终命中布尔值和 precision 必须相同。出现阈值翻转时修复实现，不能靠放宽容差放行。R20 再核对确实只有白名单中的半径传播发生变化。

原API不能正常返回全空帧precision，因此空帧另验收第4节预声明的全零扩展与完整分母，不能声称其与上游不存在的输出数值等价。

固定API还有必须忠实记录的数值边界：外向轴严格平行相机Z时，Seal/Wrench所用切平面基退化。原函数在72扇区平面合成样例上可产生Seal=0、Wrench/Q为NaN；`Q>=t`为false，所以该候选在全部阈值均未命中，仍占排序位置。这是官方算子的历史行为，不是可以删掉的空预测。R10准入须同时比较非有限值mask、有限值容差、命中布尔值及precision；诊断JSON用 `null` 加 `nonfinite_official_quality` 状态，不写裸NaN。无效输入/未知资产与这一已复现的上游数值边界分开处理。

不通过微扰法向、删候选/帧或静默移入生产模型的基fallback改善公开AP。若未来采用稳定数值修复，建立独立评价版本并披露偏离，不混入仅半径变化的主表。这里对上游函数的有限合成边界复核不等于R10适配层或全真实数据评价已经验收。

如需更直接的相同评价尺寸解释，可将**同一份**本方法预测再交未修改物理定义的官方 R10 evaluator（带同一空帧适配），列为附表“R20 训练 / R10 评价”；不需要第二次训练。该数值与 native R20 性能分开，不复制到主 R20 行。主表仍按用户允许的尺寸例外执行，不要求重算论文方法的 R20 成绩，也不将 R10 论文数值乘系数估成 R20。

## 4. 完整场景输出与六指标计算

### 4.1 导出合同

每个预测实例最多256个合法 query；将所有实例的候选直接汇合。使用 source-only 模型的 `q_inst`（下称q）作为跨实例排序分数，**不做每实例分数归一化**。跳过 native 部署用 Top5、10 mm/15° NMS、task 头与业务阈值；不先用 GT 排除碰撞或难例。模型内部候选生成是方法设计，外部官方 NMS/关联/截断必须一致。

每个profile使用独立`dump`根目录，再把该profile根目录传给API；固定官方相对路径为 `<dump>/<test_seen|test_similar|test_novel>/scene_XXXX/realsense/suction/YYYY.npz`，`arr_0` 存 `float32[N,7]`：

```text
[predicted_score, nx_out, ny_out, nz_out, px_m, py_m, pz_m]
```

n 是相机系向外单位法向，与官方 direction 一致；不是工具 `+Z=-n`。不写 GT object ID；合法空帧保存显式 `shape=(0,7)`。模型输出NaN/Inf、运行异常或未完成推理不能伪装为合法空帧；这是模型侧失败，与第3节记录的官方物理质量非有限边界分开处理。通用 `SuctionGroup` 的可视化八列类型、源码过时的17列注释都不能替代官方 baseline 实际 dump。[S7] 保存同时包含 frame/instance/query 身份的 sidecar，便于回读检查；sidecar 的预测实例 ID 不参与官方关联。

拟定生产接口为 `predict_scene(...) -> predictions[N,7]` 和 `evaluate_scene_official_family(..., radius_m) -> precision[50,4]`，目前不是已有 CLI 或已实现函数。接线时以本合同及两路线后处理文档验收。

### 4.2 evaluator 的顺序不可改变

1. 完整场景候选执行官方全局 NMS：距离 `0.02 m`、角度参数 `181°`；不按物体先 NMS，也不随 r 变40 mm。
2. 按官方5 mm体素化模型的最近点，将候选归到GT模型；不额外施加10 mm关联距离截断。
3. 每个GT物体最多保留10个，再用官方场景/桌面碰撞及 Seal/Wrench 评价。
4. 用**预测置信度**全场景排序，最多50个；碰撞质量置0但保留其排名位置，不能先删掉后让后续高质量候选补位。
5. 对全部帧计算前缀精度；不使用 teacher Q 排名，不把每实例Top1当作场景Top1。[S5]

若原 API 在全空候选上发生 `np.concatenate([])`，适配层只为已成功导出的空帧返回 `[50,4]` 全0；其余异常上报。空帧仍进入平均。缺文件不是空帧，所有23,040份预测和precision覆盖检查完成后才发布正式值。

### 4.3 明确公式和列序

设一帧经上述流程保留 M 个排序候选，质量为 Q_i；t 属于 T={0.2,0.4,0.6,0.8}：

\[
P_f(k,t)=\frac{1}{k}\sum_{i=1}^{\min(k,M)}\mathbf 1[Q_i\ge t],\quad k=1,\ldots,50.
\]

判正是 `>=`，不是 `>`；M<k 时分母仍为k。每个 split 有固定 F=7,680 帧：

\[
AP_{50}=\frac{100}{F\cdot50\cdot4}\sum_f\sum_{k=1}^{50}\sum_{t\in T}P_f(k,t),\qquad
AP_1=\frac{100}{F\cdot4}\sum_f\sum_{t\in T}P_f(1,t).
\]

`AP04_top50` / `AP08_top50` 固定 t=0.4 / 0.8 后平均50个前缀和全帧；`AP04_top1` / `AP08_top1` 固定阈值并仅取k=1。所有场景都是256帧，因此frame均值与scene等权后再均值相同。不能丢掉空帧后再称等价。

表格和JSON统一顺序：**AP50、AP04_50、AP08_50、AP1、AP04_1、AP08_1**，单位%。原论文常按 AP/AP08/AP04 排列，必须按列名重排。这里的 AP 不是分类 precision–recall 曲线面积，也不是单独 P@50。所有差值只计算 `ours − 本表参考方法`，单位百分点 pp；不计算参考方法彼此差值。只有聚合数字时，不编造配对置信区间或显著性检验。

## 5. 论文值的来源与可比性状态

| 方法 | 原始表定位 | 本次取证状态 | 仍需随结果说明的限制 |
|---|---|---|---|
| S1B | arXiv:2103.12311v2，Table IV，PDF第6页；斜杠右侧RealSense | 直接核对原论文 | 采用最终表的28.31/38.56等值；不混其他版本或后续转引的小数差异 |
| Diffusion-Suction | arXiv:2502.07238v2，Table I，PDF第7页；斜杠右侧RealSense | 直接核对原论文；作者声明沿用SuctionNet评价 | 输入为场景点云；完整SuctionNet训练recipe、checkpoint、mask过滤、历史API SHA未披露充分；不拿Parcel配置替代 |
| UISN两种完整配置 | DOI:10.1109/LRA.2024.3385609，Table I，PDF第6页/刊页4939 | 沿用仓库已有原PDF核验转录；本轮直接核验作者代码，未重新取得全文 | 表注为Kinect/RealSense，与§IV-B一处反写冲突；沿用表注和S1B交叉核对；历史API SHA未锁定 |
| MultiGraspNet-vac. / joint | arXiv:2602.06504v2，Table I Vacuum，PDF第5页，明确RealSense | 直接核对原论文 | 只报告Top50三列；文中称seal coefficient，未确认等同S×W；完整实现仍未公开。joint额外使用平行夹爪监督 |

UISN 数值来源链：固定仓库版本的[既有转录](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/a9f4a8451c6bc42cfd8867dd96c051398644ad71/docs/experiments/COMPARISON-20261118/method_metrics.csv)及[逐表说明](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/a9f4a8451c6bc42cfd8867dd96c051398644ad71/docs/SUCTIONNET_METHOD_COMPARISON.md)；DOI/机构条目只证明论文身份，不能代替表格证据。若后续拿到原PDF，更新核验记录，不静默更改列值。

Diffusion 与 UISN 原论文中的半径没有独立实验清单时，应写“所引用官方协议默认R10”，不能称已取得作者运行日志。MultiGraspNet 半径与物理一致性未核实，JSON用 `null` 配合证据说明；Top1未报告也用 `null` / “—”，不补0、不推算。MultiGraspNet 两张表保留数值，但 `comparison_status=reference_only_protocol_unverified`，不输出正式增益结论。其余论文行也保留各自证据强度，不能用“所有条件完全相同”概括未公开字段。

不要求复现第三方模型来补未公开字段；能从原文/官方源码核实的先核实，仍未知的明确保留。已有 E5 是 GT 实例输入、90场景训练的历史结果，不是本方案严格匹配后的内部基线。UISN Only Depth 是原方法消融且相机取值需推断，本轮不增加该比较单元。

## 6. 只运行本方法的执行顺序

| 步骤 | 必须做的工作 | 可复核产物 / 完成条件 |
|---|---|---|
| P0 论文引用冻结 | 核对相机、表版本、列序和缺失项 | `paper_baselines.json`、每对来源链接；不能把参考值记为本地重跑 |
| P1 数据和前端 | 建官方100train及90test清单，锁定UOIS/UOAIS资产和oracle边界 | manifest、mask/cache/hash、各帧处理状态；测试集不用于选择配置 |
| P2 本方法训练 | native开发冻结；public20从头100train、8750成功更新 | 独立checkpoint、完整配置/预算、last SHA；source-only |
| P3 导出/评价接线 | 原p/n与外向轴回读、全帧候选汇合；R10等价后R20半径检查 | exporter/API/patch版本、准入报告、空帧和缺文件区分 |
| P4 全测试推理 | 每推理profile、每路线完整23,040帧；重复用于多个成对表 | `.npz`、sidecar、完整覆盖；没有对手训练成本 |
| P5 全测试评价 | 冻结官方族R20；必要时同预测再做R10补充 | 每split `[30,256,50,4]` precision、逐候选物理诊断及六指标 |
| P6 成对报告 | 本方法与一个参考方法并列；保留相机/尺寸/前端/协议状态 | 仅本表差值、未公开项、失败分母；不做第三方总排名 |

不需要为了这组比较增加新的大规模消融矩阵。native 实例质量、产线动作分类、必要几何验收和部署计时继续按各自实验页执行，其指标不填入论文 AP。实机成功率只在相同机器人/物体/任务/试验定义下比较；本轮论文 AP 不推出实机成功率。异硬件、不同计时边界的论文耗时不直接算加速比；本方法计时应包括预测分割和整帧实例处理，并另列model-only。

## 7. 结果身份与发布要求

每份本方法结果至少保存：`route`、`run_id`、`result_kind=measured`、`training_profile=public20`、训练场景/数据hash、代码/checkpoint SHA、前端代码/权重/mask SHA及额外数据来源、工作区profile/来源/阶段/扩张参数、`camera`、完整test manifest SHA、`prediction_sha`、`radius_m`、`evaluator_family`、`ours_api_base_commit`、`adapter_patch_sha`及独立`empty_frame_adapter_sha`、R10等价验收ID、`precision_sha`、`score_field=q_inst`、normal/sampler/precision合同、dense资产hash/依赖/NMS版本、六指标与缺失帧/空帧/官方非有限质量候选数。参考方法单独保留 `result_kind=paper_reported_not_reproduced`、版本/表/页/相机、`paper_source_api_commit=null` 及未核实字段。

`null` 表示未报告/未测，须由 `status` 区分；不输出数值0占位。没有A/B真实预测和precision时，结果表保持“待测”，也没有可计算的pp差值。额外信息或评价语义尚未匹配的行可以列论文参考，不能标成已完成严格比较。

本次修改的验收限于文档合同、论文转录、机器可读结构、链接与分支一致性。生产 exporter、官方半径适配、100场景refit和全测试推理仍待实现/执行，不将文档审查或旧129项合成参考记录写成这些步骤已通过。

## 8. 原始来源

- **[S1] SuctionNet**：[最终v2论文](https://arxiv.org/pdf/2103.12311v2)、[官方数据集](https://graspnet.net/suction)。Table IV 为本方法S1B的六指标来源。
- **[S2] Diffusion-Suction**：[v2论文](https://arxiv.org/html/2502.07238v2)、[作者固定仓库README](https://github.com/TAO-TAO-TAO-TAO-TAO/Diffusion_Suction/blob/c91f0c76b54aedeb22c29f0ecf100efe9198c827/README.md)。只取Table I SuctionNet值。
- **[S3] UISN**：[DOI](https://doi.org/10.1109/LRA.2024.3385609)、[作者机构记录](https://research.cuhk.edu.hk/en/publications/uncertainty-aware-suction-grasping-for-cluttered-scenes-2/)。原表转录证据见第5节。
- **[S4] MultiGraspNet**：[v2论文](https://arxiv.org/html/2602.06504v2)、[官方项目页](https://vandal-lab.github.io/multigraspnet-project/)。取Table I Vacuum，取证日期2026-11-18。
- **[S5] 官方API固定版本**：[物理/关联/NMS](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/utils/eval_utils.py)、[测试划分与聚合](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/suctionnet_eval.py)、[法向基及极点行为](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/utils/rotation.py)。
- **[S6] UISN固定版本**：[推理与预测分割](https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/inference.py)、[训练清单](https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/dataset/dataset.py)、[评价入口](https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/eval.py)、[GT联合前景工作区实现](https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/utils/data_utils.py)。
- **[S7] 官方预测文件**：[baseline导出代码](https://github.com/graspnet/suctionnet-baseline/blob/93d604dd752d2ef865c80c8ecba96218560b516f/neural_network/inference.py)。以实际`score,direction,point`数组顺序为准。
