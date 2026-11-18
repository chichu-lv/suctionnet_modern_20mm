# B 后处理：稳定选择、精确位姿与部署输出

本页接收[模型](MODEL_DESIGN.md)对明确 p/n 的逐 query 输出，形成推荐接触姿态与可审计结果。默认阈值和坐标约定来自[共同输出合同](../TRAIN_EVAL_REPRODUCE.md)；不添加未经开发集验证的 mask 腐蚀、任务概率门槛或质量门槛。

## 1. 输入与有效性分层

每个 query 至少提供 instance/query身份、原始 p/n、source ID或历史动作ID、q/Seal/Wrench/ell/task_logit、query_valid与支持诊断；encoded关联输入、标定、normal/sampler/model身份。

首先核验实例编码、p/n/坐标合法性和共同中心支持。canonical使用原像素深度及合法观测轴；历史动作使用记录轴与≤5mm同实例支持锚点。失效记录保留原因，不吸附或翻轴。query几何有效、环槽接触有效、teacher label有效、任务outcome有效是不同字段，不能复用一个bool。

模型输出NaN/Inf属于数值错误，应记录并排除该次输出；全无合法query时返回 `no_pose`。单个环或全部接触槽无支持时，模型内部遵循全零接触特征加显式摘要合同，不能在此擅自将其改成Seal=0或成功概率0。默认不要求某个覆盖率才能参与排序；若未来启用支持过滤，须独立登记配置并重测覆盖/失效率。

## 2. 阶段决定排名分数

| checkpoint身份 | 默认排名 | 必须同时保留 |
|---|---|---|
| 源域物理预训练 | q_inst | S/W、ell、query支持与身份 |
| 已完成目标任务微调 | sigmoid(task_logit)，alpha=0 | task_probability、q_inst和ranking_score分开保存 |

没有任务训练身份的模型不能使用随机task头排序。源q预测含目标自碰的INST质量，不能当作已校准产线成功率。ell描述源残差尺度，默认不加 `q-lambda*sigma` 或MC Dropout采样。

仅作为预注册的开发集对照，可用 `P_task*max(q_inst,1e-6)^0.5`。乘积是排序启发式，不是联合成功概率，不用于BCE/Brier。没有默认task≥0.5过滤；**Q_inst≥0.4是GT评价阈值，不是线上q门槛**。校准、阈值或alpha更改需要先冻结验证选择规则，87条test不能用于调参。

## 3. 稳定排序和米制NMS

按ranking_score降序，完全同分按query_id稳定排序。在同一实例内，只有当两个query的真实3D中心距离 **<10mm且有向轴夹角<15°** 时才视为重复。计算夹角前将点积限于[-1,1]，不得取点积绝对值把n和−n视为同一个方向。

按排序依次保留，最多Top5；Top1为默认推荐。池不足5时返回实际数量。候选不足、支持失效或数值错误导致无输出时返回明确原因，不对全invalid图argmax。记录过滤前后、NMS前后K和所选query_id，候选覆盖按各自同一池统计。

B可把已评分canonical query按source ID散射到显示图；未评分像素保持unknown。该可视化不是全图稠密监督预测，也不表示同像素任意其他动作轴的分数。NMS不使用固定11×11像素窗口，避免随深度改变物理距离。

## 4. 输出同一个被评分的姿态

最终中心就是query的原始p，外向轴就是本次评分的n。禁止在选中后读取5×5深度中位数移动位置；若需要稳健位置修正，应在生成query时完成，或修正后重建特征并重新评分。

定义工具接近方向 `z_tool=-n_out`。默认静态roll补全按与z最不平行的固定相机轴b，平局顺序x/y/z：

$$x=\operatorname{normalize}(b\times z),\quad y=z\times x,\quad R=[x,y,z].$$

验收单位正交、det(R)=+1及R第三列=−n，平移为p。这个补全只生成合法的相机系contact SE(3)，不表示网络学出了roll；参考轴切换可能不连续。特征坐标图与工具roll分别管理，连续执行由明确工具roll偏好或上一姿态连续运输处理。

$$T_{base,TCP}=T_{base,cam}\,T_{cam,contact}\,T_{contact,TCP}.$$

工具contact→TCP与相机→base来自真实标定；唇口中心不等于法兰原点。外部障碍、机器人可达性和接近轨迹由下游规划器处理，模型INST质量和投影环不能证明其通过。

## 5. 可审计的输出记录

统一使用共同 `instance_suction_prediction_v2` 语义，并保留以下字段；[共同文档](../TRAIN_EVAL_REPRODUCE.md)的示例数值仅为格式示例。

| 字段组 | 要求 |
|---|---|
| 身份 | instance_uid、query_id/source_point_id或attempt_id、route/model版本、checkpoint/data/calibration/normal/sampler hash |
| 状态 | ok/no_pose、有效性与各排除原因、实际候选数、过滤/NMS前后数量 |
| 姿态 | 原始 point_cam_m、normal_out_cam、tool_z_cam、4×4 contact姿态、roll_policy |
| 分数 | q_inst、seal、wrench、ell、task_probability、ranking_score、ranking_alpha、quality_contract |
| 观测证据 | 逐半径coverage/gap、raw高度摘要、独立source像素数、门控/插值回退/低分辨率诊断 |
| 执行边界 | pose_source、actual_execution_verified、标定变换身份、是否做分数校准、计时范围 |

历史动作预测保留 `planned_pipeline_contact` 与 `actual_execution_verified=false`。没有任务头训练身份时明确标记任务分数不可用于决策，不伪造有效概率。no_pose记录不含伪造的零坐标姿态；评价仍保留原始实例分母。诊断的coverage不是碰撞通过或密封概率。

## 6. 导出与线上实现顺序

`scripts/export_instance_b.py` 当前是待实现计划。先分别验证上下文encoder、native纹理stem、query读取/环网络/五头，再验证组合结果；允许circular padding用首尾slice+concat表达，但必须保持同一数学结果。原生ROI的动态profile支持合同规定的实际右下padding尺寸，最大范围由真实资产决定，不能静默改变GN统计。

首版几何正确性基线使用原始输入构造float64，网络FP32。若部署改FP32几何/FP16网络，另登记precision-contract，对照有效mask、门控、分数、排序/Top1、NMS、p/n和SE(3)；容差与近同分处理在测试前明确，不能只看平均logit误差。真值门控边界变化须报告，不能扩大毫米阈值掩盖精度差。

计时拆分预处理、两个共享encoder、环生成与采样、环网络/头、NMS/位姿与拷贝；每实例batch1，50次预热后至少200个固定验证实例，GPU用events并同步，端到端用明确墙钟边界。具体报告与A对照见[实验方案](EXPERIMENT_PLAN.md)。

## 7. 验收出口

必须验证稳定同分、严格NMS边界、有向轴反向区别、空query/no_pose、全空环有限值、评分p/n到输出原值一致、SE(3)手性、标定组合顺序和导出profile。默认不用新增概率门槛“清理”难例。后续真实部署以固定版本的输出日志证明效果和延迟，当前2–5ms仍只是工程目标。
