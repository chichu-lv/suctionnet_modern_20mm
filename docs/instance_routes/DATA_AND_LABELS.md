# 单实例数据与同姿态物理标签

本页定义 A/B 共用的输入资产、实例数据、物理 teacher 和目标动作监督。模型只接收观测；完整对象几何与冻结物理参数用于生成和评价标签。新增质量目标名为 `gt_SW20_pose_v1`，不得复用 `gt_B` 名称后改变其含义。

## 1 原始资产与划分

源域需要已下载的 SuctionNet RGB-D、相机内参、实例掩码、场景与相机位姿、完整对象 `dense_point_clouds`、既有 R20 Seal 候选及冻结物理配置。GitHub 中的汇总数据不包含这些完整训练资产。RGB、深度和分割必须在同一已校正相机网格，标定文件及深度单位随 manifest 固定。

沿用当前 `suction20/datasets.py` 的划分，先划分 scene，再展开 frame 和实例：

| split | 场景 | 视角 |
|---|---|---|
| train | 0–99 且 `scene_id % 10 != 9`，90场景 | 0–255，训练 sampler 固定 |
| val | 9、19、…、99，10场景 | 0、8、…、248，每场景32个固定视角 |
| test_seen | 100–129 | 每场景256视角 |
| test_similar | 130–159 | 每场景256视角 |
| test_novel | 160–189 | 每场景256视角 |

同帧不同实例不跨 split；这不是 CAD 物体身份互斥的划分。验证按完整可见实例展开，预先保存固定 query 和顺序。

上表的test_seen/test_similar/test_novel是官方评价子集名称。参考manifest校验器的`split`只接受`train/val/test`；因此源域测试实例写`split="test"`，另存`test_subset="seen"|"similar"|"novel"`。不能将test_seen直接填进split后调用参考校验命令。train/val省略test_subset；评价适配器依据该字段恢复对应官方子集。

目标域保留原有清单，不能重新随机拆动作凑比例：

| split | 动作数 | 成功 | 失败 |
|---|---:|---:|---:|
| train | 1676 | 1051 | 625 |
| val | 399 | 209 | 190 |
| test | 87 | 65 | 22 |

数据共接受2162条、排除629条；后者的逐条原因不在公开汇总中。验证对应399个观测、3个group，未公开的训练/test group数量不推测。来源见[汇总](../experiments/CD010-20261117/finetuning/data/summary.json)和[微调记录](../experiments/CD010-20261117/finetuning/README.md)。

每个原始 capture 的不同深度版本、裁剪、相邻帧和同一摆放/session 继承既有 group；同一动作不能以 RGB-D 和双目深度两个样本跨 split。新增内容 SHA-256 检查，补充现有按规范路径识别同观测的检查。多个实例共享同一观测 hash 合法，但必须同 split。

## 2 坐标和传感器合同

| 字段 | 约定 |
|---|---|
| `point_cam_m` | 校正相机系米制坐标；相机 x 向右、y 向下、z 向前 |
| `normal_out_cam` | 单位外向法向，形状[3]；历史动作保持其有向符号 |
| `gravity_cam` | 单位重力方向，形状[3]；原值、来源与是否近似同时记录 |
| `K` | 当前 RGB/深度共同网格上的3×3内参；缩放/校正后使用更新后的K |
| `depth_m` | 沿相机z轴的深度，不是到相机的欧氏距离；有限且>0 |
| `rgb` | 与深度同像素/时间，RGB顺序，原始浮点值[0,1]；各路线编码器的后续归一化随配置固定，不使用BGR顺序 |
| `T_a_b` | 把b系坐标变换到a系的4×4矩阵 |
| `instance_uid` | 场景、相机、帧、分割版本和实例标识构成的稳定身份 |
| `source_point_id` | 原始像素 `v*W+u` 或明确的点云行号；排序、tie-break与回读依据 |

对于校正且无畸变的像素，`p=Z K^{-1}[u,v,1]^T`。若原数据有畸变，先完成与深度一致的校正，再进入本合同；不能把未校正RGB与另一相机的深度直接拼成颜色点云。[相机模型](https://docs.opencv.org/4.7.0/d9/d0c/group__calib3d.html)

现有目标域重力来自 `negative_shared_belt_plane_normal_approximation`，不是独立惯性测量。两路线沿用该近似并记录，不将相机系恒定 `g_z` 当作完整重力。若做刚性坐标增强，点、查询中心、法向和重力一起旋转；平移不作用于方向。

## 3 实例支持与法向

从有效实例深度取得完整 `P_support[M,3]`，RGB与点ID一一对应。全体有限有效观测的均值 `mu_vis` 用作平移原点；不要在随机2048点上重新估计中心，否则输入和还原随采样变化。`mu_vis` 不是物理质心，米制尺度不按物体尺寸归一化掉。

训练原始支持点少于100的实例记 `insufficient_support`，不进入常规源域采样；验证/测试仍保留其身份，另报无输出并计入主指标的失败分母。不可通过跳过小物体抬高指标。点数是观测充分性的工程下限，不保证唇口可见或可吸取。

默认观测法向估计：

1. 在原分辨率、同实例的有效3D支持点中取15 mm欧氏球邻域，不能将背景或无效深度补入邻域。球邻域本身不保证隔离同一实例的所有孔洞和深度断层；若增加连通性筛选，必须固定其规则并更新normal-spec和缓存。
2. 去除重复位置后至少20点；PCA协方差的最小特征向量作为法向。
3. 协方差按邻域点数N归一化，特征值按 `0<=lambda0<=lambda1<=lambda2` 升序排列，法向取lambda0对应特征向量。有效条件为特征值有限且中间特征值 `lambda1>1e-10 m²`，排除线状/点状退化；保存曲率 `lambda0/(lambda0+lambda1+lambda2)`，首版不按该曲率设硬筛选，更不能按真值质量筛掉负例。
4. 仅对自动生成的候选，以 `dot(n,-p)>=0` 选择朝相机的外向符号。历史动作轴只归一化，绝不执行此翻转。
5. 未满足条件则 `normal_valid=false`，值置零供存储；不生成标准法向候选，但该观测点仍可作为RGB/几何支持。参数进入normal-spec hash。

15 mm与20点都是首版固定参数，不是对每种传感器精度的普适保证。RGB增强不改变几何；几何坐标增强不改变 S/W 对应的物理相对关系。首轮只使用光度增强，避免从纯重投影近似中制造新的真实动作标签。

## 4 候选与查询的两种来源

**规范候选**用于预训练与普通推理：从 `normal_valid` 的同实例支持点中，以固定tie-break的FPS选最多256个不重复接触点，轴取相应观测法向；不足256时padding加mask，重复点不增加候选数。固定规则为先按source_point_id升序，相同XYZ只保留最小ID，首点取最小ID，其后按到已选集合的最小平方距离最大取点，距离并列仍取最小ID；离散选点采用CPU float64距离。A/B读取同一份缓存的点、轴与 `query_id`。A的2048主干点采用同一规则但从完整支持池选取。种子选择是预算，不是表面全覆盖证明。

**动作查询**来自历史规划：`p_plan`、`n_plan` 与目标实例直接传入 `score_at_pose`。即使不在256候选中，也单独建立查询。可以在同实例5 mm范围寻找特征锚点，但锚点不能替换动作中心或法向。超过关联范围、轴不合法、标定缺失等记录独立 `pose_support_valid` 和原因；不得偷偷吸附到最近像素或改成观测法向。

微调的一次动作是一次BCE。20 mm圆盘、环带或高斯核可以汇聚该动作的观测特征，不表示周围其他接触中心/轴具有同一成败标签。机械动作结果还包括转运，正负例不自动拆成Seal、Wrench或碰撞标签。

## 5 同姿态 SW teacher

定义
$$
Q_{\mathrm{SW20}}(p,n)=S_{20}(p,n)\,W_{20}(p,n).
$$

S20保留冻结的72角度分箱和18–22 mm唇口定义；W20保留当前R20冻结物理实现。当前 Wrench 使用完整原候选的几何均值作为中心、默认1 kg基准质量，它是协议代理，不是逐物体实测COM。可见裁剪均值仅作网络输入，不替代 teacher 的中心。[实际评价实现](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/eval_r20.py)

去掉旧C不等于只去掉外部障碍：现有C还包含目标自身碰撞。这里明确使用无碰撞项的SW；自碰、场景碰撞或原Q20可保留为诊断字段，不进入SW标签和本版模型loss。S、W、SW缺失与数值零严格区分。

缓存生成按固定新query执行，不能直接挪用旧候选标签：

```python
# 现有真实接口；需要本地已准备的SuctionNet、R20标签和冻结配置。
from suction20.eval_r20 import load_scene_eval_context, evaluate_poses

context = load_scene_eval_context(
    dataset_root, label_root, scene_id,
    camera="realsense", frozen_cfg=frozen_cfg,
)
result = evaluate_poses(
    context, frame_id, query_point_cam_m,
    query_normal_out_cam, gt_instance_row,
)
seal = result["S"]
wrench = result["W"]
q_sw = seal * wrench
```

数组的具体 dtype、shape 和 GT 实例行号必须依此版本实现转换；`gt_instance_row` 不是任意分割instance ID。GT实例行号映射在数据构建器中明示。该函数可能同时计算旧碰撞诊断，只提取S/W不改变原始文件。后续若实现只计算SW的快速teacher，应先与此参考在同一姿态下对齐。

以下都要求新缓存：接触点改变、法向改变、冻结物理参数改变、candidate sampler或normal estimator改变。`gt_B / C` 在C=0时无定义，不能恢复SW；即使有单独旧S/W，也只可在原点原轴保持不变时复用。禁止先把标签稠密插值再当成任意姿态真值。

teacher只使用训练或评价时允许的完整对象资产；网络observation中不得出现GT密封值、Wrench真值、完整不可见表面、真值COM或成功标签。源域GT实例掩码与目标域预测分割的分布差异应随结果说明。

## 6 数据文件与缓存身份

数据留在本地数据根目录，不提交原始影像、权重或整套生成标签。建议manifest逐行一个实例观测，动作表一行一次历史尝试；字段是新增v1合同，不是现有TargetDataset直接接受的字段。

```json
{
  "schema_version": "instance_observation_v1",
  "sample_id": "example_capture_instance_003",
  "instance_uid": "example_capture_mask_v1_003",
  "group_id": "example_session_01",
  "split": "train",
  "observation_sha256": "原始观测资产规范打包后的64个小写十六进制字符SHA256",
  "capture_uid": "example_capture",
  "observation_npz": "observations/example_capture.npz",
  "mask_id": 3,
  "source_kind": "target_stereo",
  "calibration_sha256": "同一标定文件的SHA256",
  "normal_spec_sha256": "法向配置的SHA256",
  "query_cache": "queries/example_capture_instance_003.npz",
  "label_cache": null
}
```

示例中的hash文字是字段解释，真正输入必须替换为64位十六进制值。参考程序demo生成的是合法的合成manifest；现有真实清单需要显式adapter，不能简单重命名字段。

原始观测NPZ至少含RGB、米制depth、实例mask、K、gravity及valid。source query缓存含 `point_cam_m[K,3]`、`normal_out_cam[K,3]`、`query_valid[K]`、`source_point_id[K]`、`query_id[K]`。参考label缓存使用下表的精确字段，见[校验程序](reference/geometry_contract.py)。生产查询字段通过[路线A示例](ROUTE_A.md)显式映射，不能只改字段名后省略姿态核对。

| 参考label字段 | shape与类型 | label_valid=True行的约束 |
|---|---|---|
| `points` | [K,3]数值数组，推荐float64 | 米制相机系查询点，全部finite |
| `normal_out` | [K,3]数值数组，推荐float64 | 外向轴，全部finite，`abs(norm-1)<=1e-4` |
| `seal` | [K]数值数组 | finite且在[0,1] |
| `wrench` | [K]数值数组 | finite且在[0,1] |
| `q_sw` | [K]数值数组 | finite且在[0,1]；与seal*wrench比较，atol=1e-6、rtol=1e-5 |
| `label_valid` | [K]，必须bool | 已存在且可监督的共同姿态标签掩码 |

所有数组都禁止Inf；mask=False的行允许NaN，有限值也仍然不参与监督。全false或空缓存可通过结构检查，但返回`has_supervision=false`，不能据此宣称可用于监督训练。参考器核对字段、形状、范围和乘积关系，不会重算真实物理teacher或验证缓存身份sidecar。

本版teacher对一个有效姿态同时生成S与W，因此三项质量共享label_valid。生产缓存可另存seal_valid、wrench_valid等分量可用性；参考器只验证上述共同mask，不能证明额外分量合同成立。若采用部分分量监督，应显式定义q有效性为所需分量mask的交集，更新训练schema和对应检查；未知/padding行不能补零当负例。

每条动作另外保存：

| 字段 | 内容 |
|---|---|
| `attempt_id / sample_id / group_id` | 动作与实例、观测及split关联 |
| `point_cam_m / normal_out_cam` | 原规划接触点及有向轴，单位与符号固定 |
| `outcome / label_valid` | 0或1；未记录不能补成0 |
| `pose_source` | `planned_pipeline_contact` |
| `actual_execution_verified` | 本次既有数据为false |
| `pose_support_valid / exclusion_reason` | 几何关联与标签有效性分别记录 |
| `pose_transform_sha256` | 原动作系转换到相机系的标定和适配代码身份 |

源标签sidecar包含：模型不可见的对象资产身份、冻结配置文件完整SHA256、代码commit、normal spec、候选点/轴字节SHA256、数据split清单SHA256、`quality_contract=gt_SW20_pose_v1`、`radius_m=0.020`、`units=m`。每次读取先比对身份，不能通过修改名称强制使用旧缓存。

## 7 B 的标签投影合同

B预训练默认只对共用K个明确姿态的query loss回传，不需要把稀疏3D标签涂成全图密集监督。全局图可以显示候选热力分数，但未标注像素是unknown。

若未来改为全像素canonical姿态监督，必须逐个有效像素定义 `(p,n_obs)` 并由teacher生成其标签。投影时执行同实例、正深度、可见性和z-buffer检查；多个GT点落同像素不能平均成一个不属于任何真实姿态的标签。插值绘图、近邻补洞和真实标签掩码分别存储。

B历史动作也不使用 `p_plan` 投影像素处的无条件热力值代替姿态评分。动作轴与该像素观测法向可以不同，局部patch必须由记录的轴构建。

## 8 数据验收

首次实现只需完成直接影响结果的检查：

- 原始投反影、坐标系变换、单位和实例行号；所有有效方向单位化。
- 数据按scene/group划分后展开，跨split同内容hash或同group被拒绝。
- 对固定真实query，缓存S/W与参考teacher在指定容差一致；改动轴后旧缓存身份不再匹配。
- A/B使用相同query和label缓存；padding、无效mask、空实例不产生伪监督。
- 历史动作没有被吸附/翻轴，真实失败不自动作为周围候选负例。
- 统计总数、有效数、各类跳过数及成功/失败分布；无输出实例仍保留评价分母。

合成工具能检查数学和schema，不能替代RGB标定、真实teacher值、数据来源与逐记录去重的验收。

