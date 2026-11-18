# 公开半径参数化评价：待定方案备忘

**状态：deferred，未批准为当前执行协议，未实现，不阻塞A/B训练。** 本页保留上一版的技术研究，便于后续决策；下文所有“固定”“必须”“准入”均只描述如果未来采用此提案时需要满足的内部条件，不构成本轮训练门禁。当前直接100场景训练、source test选best，运行规则见[代码说明](IMPLEMENTATION.md)。公开权重选择、半径适配及比较结论须另行确定。

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


## 8. 原始来源

- **[S1] SuctionNet**：[最终v2论文](https://arxiv.org/pdf/2103.12311v2)、[官方数据集](https://graspnet.net/suction)。Table IV 为本方法S1B的六指标来源。
- **[S2] Diffusion-Suction**：[v2论文](https://arxiv.org/html/2502.07238v2)、[作者固定仓库README](https://github.com/TAO-TAO-TAO-TAO-TAO/Diffusion_Suction/blob/c91f0c76b54aedeb22c29f0ecf100efe9198c827/README.md)。只取Table I SuctionNet值。
- **[S3] UISN**：[DOI](https://doi.org/10.1109/LRA.2024.3385609)、[作者机构记录](https://research.cuhk.edu.hk/en/publications/uncertainty-aware-suction-grasping-for-cluttered-scenes-2/)。原表转录证据见[来源登记](PUBLIC_METHOD_COMPARISON.md)。
- **[S4] MultiGraspNet**：[v2论文](https://arxiv.org/html/2602.06504v2)、[官方项目页](https://vandal-lab.github.io/multigraspnet-project/)。取Table I Vacuum，取证日期2026-11-18。
- **[S5] 官方API固定版本**：[物理/关联/NMS](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/utils/eval_utils.py)、[测试划分与聚合](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/suctionnet_eval.py)、[法向基及极点行为](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/utils/rotation.py)。
- **[S6] UISN固定版本**：[推理与预测分割](https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/inference.py)、[训练清单](https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/dataset/dataset.py)、[评价入口](https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/eval.py)、[GT联合前景工作区实现](https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/utils/data_utils.py)。
- **[S7] 官方预测文件**：[baseline导出代码](https://github.com/graspnet/suctionnet-baseline/blob/93d604dd752d2ef865c80c8ecba96218560b516f/neural_network/inference.py)。以实际`score,direction,point`数组顺序为准。
