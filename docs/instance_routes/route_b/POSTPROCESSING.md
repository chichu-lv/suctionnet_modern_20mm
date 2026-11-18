# B 后处理：稳定选择、精确位姿与部署输出

本页接收[模型](MODEL_DESIGN.md)对明确 p/n 的逐 query 输出，形成推荐接触姿态与可审计结果。默认阈值和坐标约定来自[共同输出合同](../TRAIN_EVAL_REPRODUCE.md)；不添加未经开发集验证的 mask 腐蚀、任务概率门槛或质量门槛。

第1–7节规定原生实例/产线输出；第8节规定与论文对比的公共评测导出。公共导出使用完整预测实例集合，在业务NMS与Top5之前分流，依据[公开方法对比](../PUBLIC_METHOD_COMPARISON.md)的独立评价合同。

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

## 3. 原生实例模式的稳定排序和米制NMS

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

计时拆分预处理、两个共享encoder、环生成与采样、环网络/头、NMS/位姿与拷贝；每实例batch1，50次预热后至少200个固定验证实例，GPU用events并同步，端到端用明确墙钟边界。具体报告与本路线对公开方法的比较见[实验方案](EXPERIMENT_PLAN.md)；公开完整管线计时另包含预测分割及全景候选合并。

## 7. 验收出口

必须验证稳定同分、严格NMS边界、有向轴反向区别、空query/no_pose、全空环有限值、评分p/n到输出原值一致、SE(3)手性、标定组合顺序和导出profile。默认不用新增概率门槛“清理”难例。后续真实部署以固定版本的输出日志证明效果和延迟，当前2–5ms仍只是工程目标。

## 8. 公共论文对比的独立导出接口（待实现）

### 8.1 全景候选出口与输入权限

公共导出使用 `sn-paper-radius-v1`，读取每个预测实例 `score_at_pose` 的源域 `q_inst`。90/10源域开发划分只用于选参；冻结配置后用官方100个训练场景完成source-only final refit。公共checkpoint不使用产线数据或产线任务微调，不能用task_logit或任务概率替换公开排序分数。测试实例分组来自固定预测分割前端，GT实例身份不输入模型；测试GT前景范围仅限下述UISN专属工作区配置。

每实例导出所有eligible query，最多256个，再合并本帧**全部预测实例**。eligible仅依据观测输入、已冻结的query合法性以及有限p/n/输出；单环或全部槽位缺测继续遵守第1节的模型合同，不临时增加覆盖率/GT质量门槛。禁止用GT对象身份、位姿、碰撞或teacher支持筛选候选；不调用第3节10mm/15°业务NMS，不作Top5、任务概率/质量阈值过滤或提前全景Top50截断，不先删碰撞候选。query显示图和固定像素窗口NMS均不是公开导出入口。

预测分割的漏检、误检和拆分/合并计入完整方法结果。对SuctionNet-1Billion、Diffusion-Suction的全场景输入行，B的预测分割前端、逐实例推理和候选合并一起参加系统比较，不要求其他方法改用相同分割器。同一source-only评分checkpoint固定三种推理profile：

| inference_profile | 工作区模式 | 使用范围 |
|---|---|---|
| `uois_no_gt_workspace` | UOIS预测分割，不读取测试GT工作区 | 默认S1B/Diffusion对照；不推定对手有GT权限 |
| `uois_uisn_gt_aabb20` | UOIS＋`uisn_gt_aabb20` | 仅对应UISN-UOIS配置 |
| `uoais_uisn_gt_aabb20` | UOAIS＋`uisn_gt_aabb20` | 仅对应UISN-UOAIS配置 |

`uisn_gt_aabb20`对应[UISN作者工作区步骤](https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/inference.py)：读取全图预测分割后，将 `seg>0` 的GT前景并集变换到对齐坐标，求3D AABB并扩20mm，过滤点云/颜色/预测分割，再按预测实例ID分组。该工作区**会改变query池**；其GT使用只为匹配UISN行，不授权S1B/Diffusion等行使用相同GT步骤。GT实例ID、逐实例GT轮廓和GT质量均不流入B的编码或环特征。记录工作区模式、输入来源、应用阶段、边界和缓存hash。

三profile隔离预测/评价目录，仅同profile复用输出；评分checkpoint不因对手改变。UISN对应前端权重或预处理无法确认时登记 `frontend_match=unverified`。待确认输入/评价协议的论文行，包括待核实的MultiGraspNet行，只作为论文参考，不标记为已匹配对照。

### 8.2 精确姿态与7列文件合同

公共预测数组为 `float32[M,7]`：

```text
[q_inst, n_out_x, n_out_y, n_out_z, p_cam_x_m, p_cam_y_m, p_cam_z_m]
M = sum(eligible_query_count over all predicted instances in this frame)
profile_root = dump_root/inference_profile
path = profile_root/{test_seen|test_similar|test_novel}/scene_XXXX/{camera}/suction/YYYY.npz
key = arr_0
```

使用 `np.savez(path, predictions)`，官方 `dump_folder` 传入 `profile_root`，帧号为0000–0255。保持同次评分的原始相机系p（米）与单位外法向n；**API方向列存 `n_out`，不是工具 `R[:,2]=-n_out`**。禁止在导出时再反投影名义环、移动query中心、读取邻域深度修正p或翻轴。公开数组不附GT object_id；精度转换、query ID、环槽有效性和错误原因另存sidecar。

顺序依据[官方基线实际导出](https://github.com/graspnet/suctionnet-baseline/blob/93d604dd752d2ef865c80c8ecba96218560b516f/neural_network/inference.py)与[API切片实现](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/suction.py)。通用Suction对象的8列结构另含object_id；旧17列注释和README中相反的p/n文字顺序不能替代公共预测接口。

### 8.3 R20外部评价与固定官方条件

采用 `sn-paper-radius-v1` 的R20评价本路线预测，与论文R10报告值按方法对应，明确半径差异。INST20训练teacher及项目旧 `eval_r20.py` 不能代替公共AP评价器：它们还改变了物理质量或碰撞口径。

对[固定官方API基线](https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/utils/eval_utils.py)只参数化三处半径：模块wrench阈值的半径、`get_smoothness_score`局部半径、`collision_detection`局部半径。显式传入同一 `radius_m`，wrench阈值同步计算 `15.6*pi*radius_m`。官方唇口带宽固定**±1mm**，故公共R20评价使用19–21mm；B输入中的名义采样环和18–22mm分区继续属于本方法模型设计，不据此改公共评价宽度。

其他条件保留：72扇区、自碰轴向>5mm、场景碰撞径向<r/轴向10–100mm、原始Seal、dense模型均值质心、原重力变换、max切向力矩、5mm模型关联采样。全景官方流程为20mm/181° NMS、最近模型归属、每对象最多10、碰撞质量置0但保留排名、按预测分数全景排序、Top50前缀和 `{0.2,0.4,0.6,0.8}` 阈值聚合。AP不是P@50，也不是每实例Top1均值，具体公式与匹配条件见[公开方法对比](../PUBLIC_METHOD_COMPARISON.md)。

### 8.4 空帧、结果身份与验收

一帧无预测实例或观测几何无合法query时保存 `(0,7)` 和原因。独立空帧adapter在官方NMS之前返回该帧 `50×4` 全零precision矩阵，仍计入完整帧分母；这是对上游空输入故障的明确扩展，不声称上游原API已有相同输出。模型/运行异常导致无输出、缺文件、未完成推理或未记录异常均须报告评测未完成，不能静默变成合法空帧或删除该帧。每相机Seen/Similar/Novel各30×256帧，RealSense/Kinect分别统计。

manifest固定 `protocol=sn-paper-radius-v1`、`radius_m=0.020`、`score_field=q_inst`、source-only checkpoint/100场景训练清单、`inference_profile`、工作区模式/GT权限/阶段/边界/hash、前端/权重/输入模态、相机与帧清单、normal/sampler/precision身份、API基线commit、radius patch/empty adapter各自hash、dense几何和依赖/NMS版本、非有限诊断数及dump hash。论文历史API commit、依赖或dense点云hash未披露时标 `unreported`；本地锁定版本不构成论文曾使用相同版本的证据。

待实现验收要求：

1. NPZ往返确认direction恢复n、translation恢复p，单位和精度身份可追溯；用单侧凸起/碰撞物的非对称fixture验证n与−n区别及工具+Z=-n，不只用正反等价的平面验收符号。
2. 参数化评价器在R10下对正常非空及边界案例与冻结官方API验证逐候选S/W/碰撞、保留顺序和每帧 `50×4` precision等价，预先登记容差；非有限掩码、最终命中及precision须一致。合法空帧独立检验全零扩展规则，不声称上游在空输入时已有相同数值。门禁通过后才运行R20；数值边界补丁单独版本化，不混入“仅改半径”差分。
3. 验证业务NMS/Top5未调用，全部预测实例参与且每实例≤256，没有额外全景截断；缺候选仍用前缀k作分母，碰撞仍占位；空帧零矩阵、模型/运行异常与缺文件报错分开测试。
4. 验证公开导出始终采用source-only q_inst，除已登记UISN工作区外测试GT不参与前端/候选筛选，三profile权限和输出隔离；未核实的论文行不自动通过匹配状态。

可选“R20训练/R10评价”附表使用同一profile同一份预测，交物理定义不变的官方R10评价加同一空帧adapter；不新增训练、不跨工作区配置复用结果。官方轴极点等历史数值行为按共同协议记录并在R10/R20一致处理，不能删除候选或帧。

本节仅定义公共导出与验证接口；B公共导出器、半径参数化评价器和公开AP数值尚待实现/运行。
