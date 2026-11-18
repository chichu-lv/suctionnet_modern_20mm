# A 后处理：排序、NMS与接触姿态输出

输入为模型对共同最多256个query或指定历史动作的逐query结果。输出应保持实际被评分的中心和轴，并携带独立的物理、任务及支持信息。依据为[共同排序与姿态合同](../TRAIN_EVAL_REPRODUCE.md)和 [A规格](../ROUTE_A.md)。

第1–6节规定原生实例/产线输出；第7节另规定与论文对比的公共评测导出。两种模式共用被评分的原始p/n，但公共导出从业务NMS和Top5之前分流，完整协议见[公开方法对比](../PUBLIC_METHOD_COMPARISON.md)。

## 1. 评分有效性与排序模式

先核对checkpoint的架构、normal/sampler、质量及ranking身份，再检查query p/n、几何支持与模型输出有限性。错误身份或非有限评分必须记录原因；不对全invalid数组强行argmax。Fq缺失本身不是几何无效，合法raw几何仍可评分。

| 模型阶段 | 主排序分数 | 保存的附加分数 |
|---|---|---|
| 源域预训练，尚未完成任务适配 | q_inst | Seal、Wrench、ell、支持诊断 |
| 已完成目标任务微调 | `sigmoid(task_logit)` | q_inst、Seal、Wrench、ell、原task_logit |

默认alpha=0。GT `Q_inst>=0.4`用于实验指标，不是推理质量门控；当前合同没有默认任务概率0.5阈值、温度缩放或置信度惩罚。未训练task头的checkpoint不能使用随机task分数排名。

可选启发式为 `P_task*max(q_inst,1e-6)**alpha`、alpha=.5，只能在预先选定的开发验证对照中评估。它须独立命名与固定配置，不直接作为BCE/Brier的概率输入；这些指标仍读取task_probability。单次确定性前向为默认，ell用于残差诊断，不增加MC Dropout采样。

## 2. 原生实例模式的米制NMS和返回数量

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

## 7. 公共论文对比的独立导出接口（待实现）

### 7.1 全景候选出口与输入权限

公共导出使用 `sn-paper-radius-v1`，从每个预测实例的 `score_at_pose` 结果读取源域质量 `q_inst`。先在90/10源域开发划分上选参，再冻结配置、用官方100个训练场景完成source-only final refit；公共checkpoint不使用产线样本、产线任务微调或任务头排序。测试实例分组由固定预测分割前端产生，GT实例身份不输入模型；测试GT前景范围只允许用于下述UISN专属工作区配置。

每实例保留所有eligible query，最多256个；eligible只依赖观测输入、预先冻结的query合法性以及有限的p/n/模型输出。Fq缺失按原模型合同处理，不用GT质量、GT目标身份、GT位姿、GT碰撞或teacher支持筛选候选。将一帧内**全部预测实例**的候选合并为一个数组；不先作第2节NMS、Top5、概率/质量阈值过滤，不提前截断全景Top50，也不先删掉碰撞候选。预测分割的漏检、误检和拆分/合并属于完整方法的结果，不能用测试GT修补。

与SuctionNet-1Billion、Diffusion-Suction的全场景输入行比较时，A的预测分割前端、逐实例推理和候选合并共同构成被比较系统；不要求其他方法改用我们的分割器。同一source-only评分checkpoint采用三种独立推理profile：

| inference_profile | 工作区模式 | 使用范围 |
|---|---|---|
| `uois_no_gt_workspace` | UOIS预测分割，不读取测试GT工作区 | 默认S1B/Diffusion对照；未证实的对手权限不扩大 |
| `uois_uisn_gt_aabb20` | UOIS＋`uisn_gt_aabb20` | 仅对应UISN-UOIS配置 |
| `uoais_uisn_gt_aabb20` | UOAIS＋`uisn_gt_aabb20` | 仅对应UISN-UOAIS配置 |

`uisn_gt_aabb20`严格复用[UISN作者工作区步骤](https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/inference.py)：先读取全图预测分割，再由 `seg>0` 的GT前景并集在对齐坐标中求3D AABB并扩20mm，将该空间过滤用于点云/颜色/预测分割，之后仍按预测实例ID分组。该步骤**确实会改变query池**，是UISN专属输入协议例外，不是给S1B、Diffusion或其他方法对照的一般GT使用授权；不能传入GT实例ID、逐实例GT轮廓或GT质量。记录工作区来源、阶段、边界和缓存hash。

不同profile的预测文件和结果目录隔离，不将有GT工作区的结果复用于无GT工作区的表。同一profile结果可复用；不因对手改变评分权重。UISN对应前端权重/预处理若不能确认，标记 `frontend_match=unverified`。输入模态或协议尚未确认的论文行，包括待核实的MultiGraspNet行，只能保留为论文参考，不能标为已匹配对照。

### 7.2 相机系NPZ合同与法向符号

公开预测文件为 `float32[M,7]`，列顺序固定：

```text
[q_inst, n_out_x, n_out_y, n_out_z, p_cam_x_m, p_cam_y_m, p_cam_z_m]
M = sum(eligible_query_count over all predicted instances in this frame)
profile_root = dump_root/inference_profile
path = profile_root/{test_seen|test_similar|test_novel}/scene_XXXX/{camera}/suction/YYYY.npz
key = arr_0
```

使用 `np.savez(path, predictions)` 所生成的 `arr_0` 键；将 `profile_root` 传给官方 `dump_folder`。`YYYY`为0000–0255；相机系位置单位为米，方向为同次评分的单位外法向。**API方向存储 `n_out`，工具的 `R[:,2]=-n_out` 不得直接写进方向列。** 不移动p、不翻转n、不附加GT object_id。逐query身份和错误账本另存sidecar；文件必须保存精度转换后的实际p/n，并记录转换身份。

这一7列顺序依据[官方基线实际导出](https://github.com/graspnet/suctionnet-baseline/blob/93d604dd752d2ef865c80c8ecba96218560b516f/neural_network/inference.py)及[API切片实现](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/suction.py)。通用Suction对象的8列定义包含额外object_id，旧注释中出现的17列及README中相反的p/n文字顺序不能代替实际公共预测接口。

### 7.3 只改变吸盘半径的外部评价

本路线原生R20输出用 `sn-paper-radius-v1` 的R20参数评价，与论文R10报告值逐方法对应；必须显式标注半径不同。公共评价不调用项目现有 `eval_r20.py` 或INST20 teacher作为最终AP计算器，因为它们的物理分数/碰撞口径还存在其他差异。

在[固定官方API基线](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/utils/eval_utils.py)上仅参数化三处半径：模块wrench阈值的半径、`get_smoothness_score`局部半径、`collision_detection`局部半径。统一显式传入 `radius_m`，同步重算 `15.6*pi*radius_m`，避免导入期阈值仍停留R10。保持官方唇口带**±1mm**，因此公共R20评价取19–21mm；本模型18–22mm的接触输入/训练环仍保留其自身合同，不替换公共评价环。

其他参数固定：72扇区、自碰轴向>5mm、场景碰撞柱径向<r且轴向10–100mm、原始Seal、dense模型均值质心、原重力变换和max切向力矩、5mm模型关联采样。官方全景流程保持20mm/181° NMS、最近模型归属、每对象最多10、碰撞质量置0但保留排名位置、按预测分数全景排序、Top50前缀及 `{0.2,0.4,0.6,0.8}` 阈值聚合；不把P@50当AP，也不把每实例Top1平均当场景Top1。具体公式与论文行条件见[公开方法对比](../PUBLIC_METHOD_COMPARISON.md)。

### 7.4 空帧、身份与导出验收

每个测试帧均须有结果。分割无实例或观测几何无合法query时保存 `(0,7)` 数组及原因；独立空帧adapter在进入官方NMS前产生该帧 `50×4` 全零precision矩阵，并保留全部帧分母。该规则是对上游空输入故障的显式扩展，不声称原API本身输出相同零矩阵。模型/运行异常造成无输出、缺失npz文件、未完成推理或未记录异常均表示评测不完整，不能静默替代为合法空帧，也不能删除帧。每相机三个测试子集各30×256帧；RealSense与Kinect分别统计。

导出manifest至少固定 `protocol=sn-paper-radius-v1`、`radius_m=0.020`、`score_field=q_inst`、source-only checkpoint与100场景训练清单、`inference_profile`、工作区模式/GT权限/阶段/边界/hash、预测分割前端/权重、输入模态与相机、帧清单、normal/sampler/precision身份、API基线commit、radius patch/empty adapter各自hash、dense几何和依赖/NMS版本、非有限诊断计数及dump hash。论文未公开的历史API commit、依赖和dense点云hash写 `unreported`，不能声称与本地冻结版本完全相同。

实现后必须通过以下门禁，当前文档不表示这些测试已运行：

1. NPZ往返读取准确保留7列，API的 `directions()` 与 `translations()` 分别恢复n和p；归一化/精度转换在评分合同中登记。
2. 采用具有单侧凸起/碰撞物的非对称几何fixture验证n与−n的有向差异，确认API采用物体外法向、工具+Z取其负向；不能只用正反等价的无障碍平面验收。
3. 将参数化评价器设为R10，对正常非空及边界案例与冻结官方API核对逐候选S/W/碰撞标记、保留顺序及每帧 `50×4` precision，比较容差预先登记；非有限掩码、最终命中和precision另要求一致。合法空帧独立验收预声明零矩阵扩展，不纳入“上游原样数值等价”的宣称。门禁未通过不得填R20论文对比结果。数值边界补丁需单独版本化，不能混入“仅改半径”的修改中。
4. 核对业务NMS/Top5未被调用、每实例≤256且全景无额外截断、缺候选仍用前缀k作分母、碰撞仍占排名位置；验证空帧零矩阵、模型/运行异常和缺文件报错分别生效，三profile的工作区权限和输出目录不混用。

可选“R20训练/R10评价”附表对同一profile的同一份预测使用物理定义不变的官方R10评价加同一空帧adapter；不重新训练，不把带GT工作区的dump换到无GT工作区行。官方轴极点等历史数值行为按共同协议记录并在R10/R20一致处理，不删候选或帧。

现阶段只形成接口与验收说明；A公共导出器、参数化评价器和完整公开AP结果均待实现/运行。
