# A 后处理：排序、NMS与接触姿态输出

输入为模型对共同最多256个query或指定历史动作的逐query结果。输出应保持实际被评分的中心和轴，并携带独立的物理、任务及支持信息。依据为[共同排序与姿态合同](../TRAIN_EVAL_REPRODUCE.md)和 [A规格](../ROUTE_A.md)。

## 1. 评分有效性与排序模式

先核对checkpoint的架构、normal/sampler、质量及ranking身份，再检查query p/n、几何支持与模型输出有限性。错误身份或非有限评分必须记录原因；不对全invalid数组强行argmax。Fq缺失本身不是几何无效，合法raw几何仍可评分。

| 模型阶段 | 主排序分数 | 保存的附加分数 |
|---|---|---|
| 源域预训练，尚未完成任务适配 | q_inst | Seal、Wrench、ell、支持诊断 |
| 已完成目标任务微调 | `sigmoid(task_logit)` | q_inst、Seal、Wrench、ell、原task_logit |

默认alpha=0。GT `Q_inst>=0.4`用于实验指标，不是推理质量门控；当前合同没有默认任务概率0.5阈值、温度缩放或置信度惩罚。未训练task头的checkpoint不能使用随机task分数排名。

可选启发式为 `P_task*max(q_inst,1e-6)**alpha`、alpha=.5，只能在预先选定的开发验证对照中评估。它须独立命名与固定配置，不直接作为BCE/Brier的概率输入；这些指标仍读取task_probability。单次确定性前向为默认，ell用于残差诊断，不增加MC Dropout采样。

## 2. 米制NMS和返回数量

按ranking_score降序排序，同分以query_id稳定破同分。依次接受候选；仅当它与某个已接受候选**同时**满足中心距离严格<0.010m、有向轴夹角严格<15°时抑制：

```text
distance = norm(p_i - p_j)
angle = acos(clamp(dot(n_i, n_j), -1, 1))
duplicate = (distance < 0.010) AND (angle < 15 degrees)
```

不能用abs(dot)把相反轴当相同姿态，不能把<改成≤而不更新输出合同。每实例保留Top5，Top1为默认推荐；可用候选不足则返回实际数量。支持过滤前、NMS前后及Top5截断后的数量分别记录。

无可输出候选返回 `status=no_pose` 和原因。源域实验仍保留原GT实例分母；历史动作无几何关联不变成人工失败标签。候选NMS只合并相近建议，不认证杯体自碰、外部障碍或机器人可达性。

## 3. 保持实际评分的p/n

选择后的输出中心就是被评分query的原始p；不使用5×5深度中位数或最近主干点再次移动它。输出n也是同一有向轴。若以后采用深度修正，应在query生成时完成并绑定缓存身份，或在修正后重新构造特征并评分。

canonical查询由原始像素反投影给出p，v3法向给出n。历史动作的p/n来自记录，支持锚点不替换姿态。输出另保存source_point_id（存在时）、query_id、normal/sampler版本及精确pose hash，使实际评分条件可回读。

## 4. 构造相机系接触位姿

工具+Z定义为接近方向，因此 `z_tool=-n_out`。默认无状态roll约定：在相机x/y/z基向量中选择与z最不平行的b，同值按x/y/z顺序；令 `x=normalize(b×z)`、`y=z×x`，得到 `R=[x,y,z]` 和 `T_cam_contact=[R|p]`。

验收 `R.T@R≈I`、`det(R)≈1`、`R[:,2]=-n`、translation=p。点与轴本身只约束5个自由度，roll是序列化约定；该参考基可能切换，不保证连续运动，而且不进入模型接触学习特征。工具roll约束或前帧连续参考由后续控制器明确管理。

机器人坐标变换按：

```text
T_base_tcp = T_base_cam @ T_cam_contact @ T_contact_tcp
```

`T_contact_tcp`来自工具标定；唇口中心不直接等于法兰/TCP原点。部署接口保留相机/工具标定hash、坐标系和roll策略。模型结果送交下游时，外部场景碰撞、机器人约束与轨迹检查属于独立规划步骤。

## 5. 结果与错误账本

沿用共同建议schema `instance_suction_prediction_v2`；实际生产序列化器尚待实现。

| 字段组 | 至少包含 |
|---|---|
| 实例/查询身份 | instance_uid、query_id、source_point_id（可用时）、状态和原因 |
| 精确姿态 | point_cam_m、normal_out_cam、tool_z_cam、4×4相机系contact变换、roll_policy |
| 分数 | q_inst、seal、wrench、ell、task_probability（已适配时）、ranking_score、ranking_alpha |
| 合同身份 | quality_contract、checkpoint/代码/data/normal/sampler/标定hash、精度配置 |
| 支持诊断 | pose_support_valid、query_feature_valid、三区域raw数、lip缺口/harmonic_valid |
| 计时与数量 | 原始/有效query数、过滤/NMS数量、编码/评分/后处理时间 |

无效数值在JSON中使用明确null/状态，而不是输出非标准NaN或假概率0。A的lip点数和最大角间隙是观测摘要，不能用一个未定义的coverage数冒充密封或碰撞通过率。

## 6. 验收与当前可运行范围

现有 `geometry_contract.py selftest`覆盖基础SE(3)数学；完整A后处理仍需实现后核验：分阶段排序身份正确；同分顺序稳定；NMS严格边界及有向轴正确；Top5/无输出行为明确；返回p/n与实际评分字节身份一致；相机/TCP变换可回读；错误候选和历史动作分别记账。延迟统计计入候选过滤、NMS和序列化边界，见[实验方案](EXPERIMENT_PLAN.md)。
