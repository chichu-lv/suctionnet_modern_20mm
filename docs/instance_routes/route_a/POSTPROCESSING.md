# A 后处理：排序、NMS与接触姿态输出

输入为模型对共同最多256个query或指定历史动作的逐query结果。输出应保持实际被评分的中心和轴，并携带独立的物理、任务及支持信息。依据为[共同排序与姿态合同](../TRAIN_EVAL_REPRODUCE.md)和 [A规格](../ROUTE_A.md)。

第1–6节由共同 `suction20/instance_inference.py` 和原生infer/evaluate CLI实现；第7节仅记录公开协议待定边界。半径-only公共评价为deferred，不能把当前实例输出直接作为论文AP。第8节给出实际命令，需要真实训练产生的checkpoint。

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

canonical查询由原始像素反投影给出p，v3法向给出n。历史动作的p/n来自记录，支持锚点不替换姿态。输出另保存source_point_id（存在时）、query_id、normal/sampler版本及语义pose_id及原始p/n，使实际评分条件可回读。

## 4. 构造相机系接触位姿

工具+Z定义为接近方向，因此 `z_tool=-n_out`。默认无状态roll约定：在相机x/y/z基向量中选择与z最不平行的b，同值按x/y/z顺序；令 `x=normalize(b×z)`、`y=z×x`，得到 `R=[x,y,z]` 和 `T_cam_contact=[R|p]`。

验收 `R.T@R≈I`、`det(R)≈1`、`R[:,2]=-n`、translation=p。点与轴本身只约束5个自由度，roll是序列化约定；该参考基可能切换，不保证连续运动，而且不进入模型接触学习特征。工具roll约束或前帧连续参考由后续控制器明确管理。

机器人坐标变换按：

```text
T_base_tcp = T_base_cam @ T_cam_contact @ T_contact_tcp
```

`T_contact_tcp`来自工具标定；唇口中心不直接等于法兰/TCP原点。部署接口保留实际相机/工具标定参数及来源、坐标系和roll策略。模型结果送交下游时，外部场景碰撞、机器人约束与轨迹检查属于独立规划步骤。

## 5. 结果与错误账本

逐实例序列化使用 `instance_suction_prediction_v2`；CLI外层报告为 `instance_suction_inference_v1`，已实现原子写入、有效性/选模身份与JSON非有限值检查。

| 字段组 | 至少包含 |
|---|---|
| 实例/查询身份 | instance_uid、query_id、source_point_id（可用时）、状态和原因 |
| 精确姿态 | point_cam_m、normal_out_cam、tool_z_cam、4×4相机系contact变换、roll_policy |
| 分数 | q_inst、seal、wrench、ell、task_probability（已适配时）、ranking_score、ranking_alpha |
| 合同身份 | quality_contract、checkpoint路径/大小、显式运行/normal/sampler版本、数据/标定定义和精度配置 |
| 支持诊断 | pose_support_valid、query_feature_valid、三区域raw数、lip缺口/harmonic_valid |
| 计时与数量 | 原始/有效query数、过滤/NMS数量、编码/评分/后处理时间 |

无效数值在JSON中使用明确null/状态，而不是输出非标准NaN或假概率0。A的lip点数和最大角间隙是观测摘要，不能用一个未定义的coverage数冒充密封或碰撞通过率。

## 6. 验收与当前可运行范围

现有 `geometry_contract.py selftest`覆盖基础SE(3)数学；共同生产后处理已实现，相关原生推理/评价测试核验：分阶段排序身份正确；同分顺序稳定；NMS严格边界及有向轴正确；Top5/无输出行为明确；返回p/n与实际评分数组逐元素一致；相机/TCP变换可回读；错误候选和历史动作分别记账。延迟统计计入候选过滤、NMS和序列化边界，见[实验方案](EXPERIMENT_PLAN.md)。

## 7. 公开比较导出与半径评价：待定

本轮没有实现公开7列NPZ导出、官方评价器半径参数化、R10等价门禁或空帧adapter。`public_evaluation.status=deferred`，公开比较准备不阻塞源/目标训练。论文数值仍列于[实验方案](EXPERIMENT_PLAN.md)，A行保持待测；不再将 `sn-paper-radius-v1` 当作本轮已经冻结的执行合同。

当前 `best.pt` 是source test-selected；同一test上的实例Top1/selected-Q不能转换为全景AP或独立测试估计。后续若恢复论文对比，应先决定半径/物理评价合同、完整测试集、指标计算、每方法输入权限、分割前端与工作区；必要时重新规划独立评估。不能只靠相同scene范围宣称“除了吸盘半径其余完全一致”。

可保留的接口设计约束是：全景候选出口应在业务NMS/Top5之前分流；保持同次评分原p/n；外向法向与工具+Z=-n区别必须明确；无预测实例/无有效候选的合法空帧与运行错误、缺文件应分别处理；漏检不能缩小完整帧分母。未来与UISN匹配的GT联合前景工作区权限不自动授权其他方法pair，未核实项继续标为未核实。这些是未来公开接口的审查清单，当前原生推理CLI不执行这套公共协议。

## 8. 原生推理与评价命令

对一份已缓存的实例观测推理不需要teacher标签：

```bash
python scripts/infer_instance.py --checkpoint /data/runs/a_source100/best.pt --observation /data/instance_v3/test/observations/realsense_s0100_f0000_i000.npz --output /data/runs/a_source100/instance_prediction.json --backbone-root /data/backbones/graspness_modern --device cuda
```

也可传 `--manifest /data/instance_v3/test/manifest_test.jsonl --split test`，替换 `--observation`，推理器不会读取label文件。默认 `--ranking auto` 按checkpoint阶段选择源q或微调任务概率，`--top-k 5`；无合法query输出明确no_pose。`--max-instances` 只用于manifest显式分片，输出记录partial；不能据此宣称全test完成。

原始RGB-D入口保持真实像素与标定，下面的尺度和重力须改成该观测的实际值：

```bash
python scripts/infer_instance.py --checkpoint /data/runs/a_source100/best.pt --rgb /data/frame/rgb.png --depth /data/frame/depth.png --mask /data/frame/instance.png --instance-id 1 --intrinsics /data/frame/K.json --depth-scale-m 0.001 --gravity 0 1 0 --output /data/frame/prediction.json --backbone-root /data/backbones/graspness_modern --device cuda
```

源实例物理评价与目标固定动作评价分别运行：

```bash
python scripts/evaluate_instance.py --checkpoint /data/runs/a_source100/best.pt --manifest /data/instance_v3/test/manifest_test.jsonl --domain source --split test --output /data/runs/a_source100/source_test_selected.json --backbone-root /data/backbones/graspness_modern --device cuda
python scripts/evaluate_instance.py --checkpoint /data/runs/a_target/best.pt --manifest /data/attempts/test.jsonl --domain target --split test --output /data/runs/a_target/target_test.json --backbone-root /data/backbones/graspness_modern --device cuda
```

源评价要求完整最多256候选的精确姿态标签；train缓存默认前128标签不能直接充当完整候选评价缓存。source partial检查须显式 `--allow-partial-manifests` 并记录其非正式身份。目标评价要求finetune checkpoint，不能用随机task头冒充概率模型。报告保存checkpoint选模来源、数据身份、完整分母/排除原因和 `public_evaluation.status=deferred`，这些命令不计算论文表中的公开AP。

A加载仍需固定外部SpConv代码与兼容运行环境；`--backbone-root` 可显式指定外部源码位置，blob身份不变。CPU注入backbone仅限单元测试，不由这些CLI自动启用。
