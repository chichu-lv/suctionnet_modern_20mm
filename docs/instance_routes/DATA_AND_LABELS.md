# 单实例数据、法向与同姿态物理标签（v2）

本页定义 A/B 共用的数据、查询姿态、离线 teacher 和目标动作监督。**v2 默认综合质量保留目标自身碰撞，去掉其他物体与桌面的碰撞项**：`quality_contract=gt_INST20_pose_v2`，参考标签 schema 为 `suction20.instance_inst20.v2`。纯 `q_sw=S20*W20` 继续保存，供物理诊断和与上一版比较；不能把旧 `gt_SW20_pose_v1` 文件改名后当作 v2。

本页是新路线的实现合同。现有源码提供 organized normals、Seal/Wrench 和原混合碰撞评价器，尚未提供完整 v2 数据构建或训练入口。新增实例 teacher、候选 sampler、缓存身份和训练适配必须按本页实现，不能把下面的伪代码写成已有命令。模型只接收当前观测，完整对象几何只用于离线标签与评价。

## 1. 原始资产与固定划分

源域需要已下载的 SuctionNet RGB-D、相机内参、实例掩码、场景与相机位姿、完整对象 `dense_point_clouds`、既有 R20 Seal 候选及冻结物理配置。GitHub 汇总不含这些完整训练资产。RGB、深度、分割必须在同一已校正相机网格，标定文件及深度单位随 manifest 固定。

沿用当前 `suction20/datasets.py` 的划分，先划分 scene，再展开 frame 和实例：

| split | 场景 | 视角 |
|---|---|---|
| train | 0–99 且 `scene_id % 10 != 9`，90 场景 | 0–255，训练 sampler 固定 |
| val | 9、19、…、99，10 场景 | 0、8、…、248，每场景 32 个固定视角 |
| test_seen | 100–129 | 每场景 256 视角 |
| test_similar | 130–159 | 每场景 256 视角 |
| test_novel | 160–189 | 每场景 256 视角 |

同帧不同实例不跨 split；这不是 CAD 物体身份互斥的划分。验证按完整可见实例展开，预先保存固定 query 与顺序。

参考 manifest 的 `split` 只接受 `train/val/test`。上表测试子集写成 `split="test"`，另存 `test_subset="seen"|"similar"|"novel"`；不能把 `test_seen` 直接填进 split。train/val 省略 test_subset，评价适配器依据该字段恢复官方子集。

目标域保留原有清单，不重新随机拆动作凑比例：

| split | 动作数 | 成功 | 失败 |
|---|---:|---:|---:|
| train | 1676 | 1051 | 625 |
| val | 399 | 209 | 190 |
| test | 87 | 65 | 22 |

共接受 2162 条、排除 629 条，排除明细不在 GitHub 汇总内。验证对应 399 个观测、3 个 group；未公开的训练/test group 数量不推测。来源见[数据汇总](../experiments/CD010-20261117/finetuning/data/summary.json)和[微调记录](../experiments/CD010-20261117/finetuning/README.md)。

每个 capture 的不同深度版本、裁剪、相邻帧和同一摆放/session 继承既有 group；同一动作不能以 RGB-D 和双目深度两份样本跨 split。内容 SHA-256 补充现有按规范路径识别同观测的检查。多个实例共享同一观测 hash 合法，但必须同 group、同 split。hash 校验只验证输入清单声明，不能代替真实资产字节校验和采集过程核对。

## 2. 坐标与传感器合同

| 字段 | 约定 |
|---|---|
| `point_cam_m` | 校正相机系米制坐标；相机 x 向右、y 向下、z 向前 |
| `normal_out_cam` | 单位外向法向，形状 [3]；历史动作保留有向符号 |
| `gravity_cam` | 单位重力方向，形状 [3]；原值、来源和是否近似同时保存 |
| `K` | RGB/深度共同网格上的 3×3 内参；缩放/校正后使用更新后的 K |
| `depth_m` | 沿相机 z 轴的深度，有限且大于零，不是欧氏相机距离 |
| `rgb` | 与深度同像素/时间，RGB 顺序，原始浮点值 [0,1]；编码器归一化随配置固定 |
| `T_a_b` | 将 b 系坐标变换到 a 系的 4×4 矩阵 |
| `instance_uid` | 场景、相机、帧、分割版本及实例标识组成的稳定身份 |
| `source_point_id` | 原图像素 `v*W+u`；排序、tie-break 和回读依据 |

对校正且无畸变像素，`p=Z K^{-1}[u,v,1]^T`。原数据存在畸变时，先完成与深度一致的校正，不能把未校正 RGB 与另一相机的深度拼接。[相机模型](https://docs.opencv.org/4.7.0/d9/d0c/group__calib3d.html)

目标域当前重力来源为 `negative_shared_belt_plane_normal_approximation`，不是独立惯性测量。两路线保留并记录该近似，不把恒定 `g_z` 当作完整重力。若使用刚性坐标增强，点、查询中心、法向、重力一致旋转；平移不作用于方向。首版默认只做共同的光度增强。

## 3. 原始支持与默认快速法向

### 3.1 保留原始观测，分别控制计算预算

完整有效实例 RGB-D 是共同观测资产。A 可物化 `P_support[M,3]`；B 可保留 organized XYZ 图并按像素读取，**不要求建立无序点云、KD-tree 或逐 query 栅格**。所有 RGB、XYZ、mask、法向和原像素身份保持对应。

全体有限有效实例观测的均值 `mu_vis` 用作平移原点，抽样后不重算。它不是物理质心，米制尺度不能按对象大小消除。训练原始有效支持少于 100 点的实例记 `insufficient_support`，不进入常规源域采样；验证/测试仍保留身份，分别报告无输出，并计入主指标失败分母。

A 的 2048 主干点、法向估计支持、接触几何支持是不同预算。用于降低索引成本的 2 mm 体素不能替代保存细节的原始唇口 RGB/深度；体素化也不保证任何大小对象都有同样点数。

### 3.2 v2 默认：同实例 organized finite differences

默认 normal spec 为 `organized_instance_fd_v2`，按已有 [observation.py](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/observation.py) 的 `organized_normals(point_map, keep)` 数值语义执行。**每个目标传入自己的 `keep = target_instance_mask & valid_depth`**，使中心、上下左右五点必须都属于同一实例。不能用全景 depth-valid mask 后再仅裁中心。

对非边界像素，固定一像素中心差分：

```text
dx = P[v, u+1] - P[v, u-1]
dy = P[v+1, u] - P[v-1, u]
raw = cross(dx, dy)
```

有效条件同时满足：五点 keep 为真、`norm(raw)>1e-8`、`abs(dx.z)<0.010 m`、`abs(dy.z)<0.010 m`。归一化后仅对自动观测法向按 `dot(n,-p)>=0` 定号；图像边界及不满足条件的位置 `normal_valid=false`，存储值为零。这些位置仍提供有效 RGB/深度上下文，但不生成默认法向候选。

默认没有 Gaussian smoothing、全量 15 mm 球查询或 PCA；一次向量化差分处理所有像素。多实例实现可每帧只计算一次XYZ/dx/dy/raw，再用中心与四邻居的实例ID相同且大于0施加mask，数值须与逐实例keep调用一致，避免每物体重复扫描整帧。像素 stencil 不是恒定 15 mm 物理支持，深度噪声和曲率误差仍须在真实资产上核验。若增加平滑、换 stencil、修改跳变阈值或用其他法向估计器，必须更新 normal-spec 和姿态标签缓存，不得把变化隐蔽在预处理内部。

### 3.3 可选 PCA 精度配置，不进入 B 默认路径

若 A 后续确需 PCA，采用独立配置：先在同实例 2 mm 体素代表点上建立一次空间索引，仅对已配置主干预算（默认最多2048）及最多256个查询点拟合 15 mm 邻域，禁止在全量原始支持上逐点拟合。保持原始支持供局部细节读取；normal-invalid、法向回填和候选改变都要明示。

PCA 参数、体素代表规则、最少独立邻点数及退化判断进入独立 normal-spec，缓存不得与默认配置互换。A/B 的主对比使用同一默认候选位置和轴；如果比较 PCA 精度配置，必须另列结果与预处理耗时，不能一边改变输入轴一边声称只比较网络。

## 4. 共同候选与精确动作查询

### 4.1 图像分层候选：不用 3D FPS

共同 sampler 固定为 `image_stratified_16x16_bit_reversed_morton_v2`，两路线从同一 RGB-D/实例观测生成相同 canonical 查询：

1. 在原图目标实例 mask 上计算完整包围框，包含 mask 内无深度位置。像素中心为整数坐标，框边界为 `(u_min-0.5, v_min-0.5)` 到 `(u_max+0.5, v_max+0.5)`。
2. 将该框均分为 16×16 个半开网格，最外边界闭合以处理浮点舍入。候选资格为同实例、有效深度、`normal_valid=true`。
3. 每格至多取一个合格原始像素：到该格中心的原图二维像素平方距离最小，距离并列取最小 `source_point_id`。不将深度空洞补成候选，不将重复像素计为新候选。
4. 格坐标 `cx,cy` 都为 0…15。Morton 编号将 cx 的 4 个 bit 放在偶数位、cy 放在奇数位：`m=sum(((cx>>b)&1)<<(2*b) | ((cy>>b)&1)<<(2*b+1) for b in range(4))`。将 m 的 8 bit 反转后按升序排列格，空格省略。该固定顺序使较短前缀分散在框内；它不保证物理表面积均匀或覆盖所有可吸区域。
5. 从选中像素读取精确 `p=depth*K^{-1}[u,v,1]` 与对应 `n_obs`，保存原像素、格号、查询顺序及 p/n identity。空间统计仍使用真实米制坐标。

**训练与源域回放默认取前 128 个；64 个是单独命名的轻量配置。验证、测试和推理取最多 256 个。** 少于预算时不复制，padding 只加 mask。训练 query 数与开发/评价候选预算分别保存，不能把 128-query 训练耗时和 256-query 推理耗时混为一谈。64/128 是否足够收敛属于待验证结果。

A 的 2 mm 代表点池与主干 FPS 是另一独立步骤，不改变共同 query；B 默认不运行 3D FPS。采样布局、排序、预处理或法向变化都会改变候选身份，需要新缓存。

### 4.2 历史动作不受候选预算限制

历史规划的 `p_plan,n_plan` 直接传入 `score_at_pose`，不必出现在 256 候选中，也不挤占候选名额。canonical查询直接用原像素身份；历史查询用同实例原始XYZ距记录p最近且≤5mm的点作为观测支持锚点，距离并列取最小source ID，锚点不能替换动作中心或法向。B对通常Q=1的历史查询可做一次向量化距离归约，不必建空间树；A另用锚点确定观测组件。无此锚点记pose_geometry_invalid。坐标、标定、轴合法性和观测支持检查分别保存 `pose_support_valid` 与原因；不偷偷吸附最近像素或替换成观测法向。

一次动作默认产生一次 BCE。20 mm 是接触几何尺度，不是定位噪声分布；一个历史结果不能扩散成整片邻域的独立成败标签。记录 `pose_source=planned_pipeline_contact`、`actual_execution_verified=false`。小幅姿态扰动只能作为显式的可选噪声/一致性正则，实际幅度不能从这个布尔字段推得；具体归约与默认关闭设置见[训练规范](TRAIN_EVAL_REPRODUCE.md)。

## 5. 保留目标自碰的实例 teacher

### 5.1 质量与碰撞分量

所有碰撞字段约定 **true 表示发生碰撞**：

$$
C_{\mathrm{self}}=C_{\mathrm{seal,self}}\lor C_{\mathrm{target,envelope}},\qquad
Q_{\mathrm{inst20}}=S_{20}W_{20}(1-C_{\mathrm{self}}).
$$

纯 `Q_SW20=S20*W20` 仍存储。五个模型头保持 q、Seal、Wrench、Laplace log-scale、task，q 的目标改为 Q_inst20；首版不增加独立碰撞头。未知 self 标签不是 false，未知也不等于碰撞负例。

S20 保留冻结的 72 个开区间角度分箱与 18–22 mm 唇口定义。W20 保留当前 R20 的完整原 Seal 候选位置均值、默认 1 kg 基准质量以及评价阈值 `wrench_k*radius*pi`，这是协议代理，不是实测 COM。[实际 teacher](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/eval_r20.py)

完整 GT 对象、不可见表面、真值物理分和 teacher 中心不得输入网络。源域 GT mask 与目标域预测分割的差异随评价报告说明。

### 5.2 旧自碰定义必须准确保留

`seal_at_pose(dense_points,p_obj,n_obj,radius_m=.020)` 已在同一完整对象扫描中返回 `(raw_seal, self_collision)`。其自身判定为：

```text
C_seal_self = any(rho < 0.020 m AND h > 0.005 m)
```

该判定没有 h 上限，且这里是 20 mm 半径，不是仅 `rho<18 mm`。Seal 环带本身也没有轴向高度截断，不能用网络局部窗口的 ±40 mm 截断重定义 teacher。

`C_target_envelope` 复用 [full_data.py](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/full_data.py) 中 `collision_mask_for_instance` 对 **目标自身**的判定，保持原 3 mm 障碍体素规则、配置和比较边界。设 `h=(p_obstacle-p_query)·n_out`，rho 是横向距离，默认配置为：

| 项目 | 冻结默认值 |
|---|---|
| `self_height_threshold_m` | 0.004 m |
| cup `(zmin,zmax,radius)` | (-0.002, 0.025, 0.022) m |
| mount | (0.025, 0.070, 0.028) m |
| approach | (0.070, 0.150, 0.028) m |
| obstacle voxel | 0.003 m |

原函数先检查任一自身点 `h>self_height_threshold && h<=segments[-1,1] && rho<=segments[0,2]`；再对每段检查 `h>max(zmin,self_height_threshold) && h>=zmin && h<=zmax && rho<=segment_radius`。拆分函数须同时保留这两部分，不能只抄三段数组。

其他实例与桌面产生 `C_external`，不进入单物体新质量。原合并 collision 不能逆推出各分量。兼容验证需独立计算分量，再检查 `C_old_total == C_self OR C_external`，而不是拿旧总 flag 当成 self。

以上是冻结吸盘包络和表面模型的代理，不保证真实杯体弹性、隐藏几何或整机路径可行性。若修改夹具尺寸、杯腔形状、接近高度或原 Seal self 规则，另建 teacher 版本，不在本合同名下悄悄换物理意义。学习特征使用的局部坐标基也不得改变 teacher 冻结的对象系 72 分箱规则。

### 5.3 实例专用计算路径

现有 `evaluate_poses` 会同时构建/查询外部场景碰撞，结果只返回总 collision；它能提供 S/W 的参考，却不能直接返回正确的 self 分项。新增实例 teacher 的实现目标为：

```text
load/reuse target dense geometry + source candidate mean + transforms + frozen cfg
for exact query p,n:
    raw_seal, C_seal_self = seal_at_pose(full_dense_object, p_object, n_object)
S = frozen normalization(raw_seal)
W = wrench_scores_dual(exact_world_poses, frozen_center, baseline_mass)["wrench_eval"]
C_target_envelope = target-only frozen envelope query
C_self = C_seal_self OR C_target_envelope
q_sw = S*W
q_inst = q_sw*(NOT C_self)
```

这里是待实现逻辑，不是可运行新 API。可用原 `collision_mask_for_instance` 对仅包含目标体素、所有 row 都为该目标的树建立参考；传入全 false 的初始 self 数组得到包络分量，再与 `seal_at_pose` 的 self 合并。实际重构需验证与冻结原函数的目标部分一致。

场景/对象变换可复用；每个对象的完整 dense 数组、原候选均值及目标包络索引尽量复用，不加载桌面或其他物体树。Wrench 批量向量化。Seal 仍是完整 dense 扫描，其 `chunk_points` 只限制临时内存，不减少参与点数；去掉外部碰撞并不证明整个 teacher 已经快。任何批量或索引优化必须保留数值合同，不可用有限球裁剪替代无高度上限的环柱。

## 6. 缓存预算与身份

训练预算按真正需要的不同 `(instance_uid,p,n)` 生成，默认每实例 128 查询；重复的 instance draw 直接读缓存。先生成固定采样计划及其 hash，缓存可按需生成、校验后持久保存，不要求把 90 场景所有帧、所有实例的 256 查询一次性算完才能开始训练。验证/测试独立固定 256-query 清单，不受训练减量影响。

首次真实资产接入时，先在约 64–128 个覆盖不同实例尺寸与 dense 点数的样本上测量：加载时间、Seal 秒/query、self-envelope 时间、秒/instance、缓存大小和命中率。然后按实际不同实例及查询数估计总成本。该数量是成本剖析样本，不是收敛证明或新的性能结果；不能直接许诺几小时重算或训练。

缓存身份绑定：观测与实例身份、完整对象资产 hash、冻结物理配置 hash、teacher 代码 commit、normal-spec、sampler-spec、精确查询点/轴字节 hash、split 清单 hash、质量合同和单位。以下变化需要新身份：点或法向改变、normal/sampler 变化、物理参数/自碰定义变化。RGB 光度增强不改标签；刚性坐标变换须一致作用 p/n/g 并保存逆变换。

旧 S/W 仅能在其原精确 p/n 保持不变时复用；旧总 collision 不能拆出 self，`gt_B` 也不能通过除法恢复被门控抹去的质量。不可先把标签稠密插值再称为任意姿态真值。

## 7. 数据文件和 v2 标签字段

原始数据、权重及整套标签留在数据根目录，不提交 GitHub。manifest 一行一个实例观测，动作表一行一次历史尝试。以下为新增合同字段，现有 TargetDataset 不能直接读取这个新格式。

```json
{
  "schema_version": "instance_observation_v2",
  "sample_id": "example_capture_instance_003",
  "instance_uid": "example_capture_mask_v1_003",
  "group_id": "example_session_01",
  "split": "train",
  "observation_sha256": "原始观测资产规范打包后的64位小写十六进制SHA256",
  "capture_uid": "example_capture",
  "observation_npz": "observations/example_capture.npz",
  "mask_id": 3,
  "source_kind": "target_stereo",
  "calibration_sha256": "标定文件SHA256",
  "normal_spec_sha256": "organized_instance_fd_v2及参数的SHA256",
  "sampler_spec_sha256": "图像分层候选配置的SHA256",
  "query_cache": "queries/example_capture_instance_003.npz",
  "label_cache": null
}
```

示例 hash 文字是解释，真实文件必须使用合法的 64 位十六进制值。原始观测至少含 RGB、米制 depth、实例 mask、K、gravity 与各有效性标记。query 缓存保存 `point_cam_m[K,3]`、`normal_out_cam[K,3]`、`query_valid[K]`、原像素、`source_point_id[K]`、格号及 `query_id[K]`。

v2 参考标签 NPZ 的共同监督字段如下，文件内另保存 `schema_version="suction20.instance_inst20.v2"` 与 `quality_contract="gt_INST20_pose_v2"`：

| 字段 | shape / 类型 | `label_valid=true` 的约束 |
|---|---|---|
| `points` | [K,3] 数值，建议 float64 | 相机系米制查询点，全部 finite |
| `normal_out` | [K,3] 数值，建议 float64 | 外向单位轴，全部 finite，`abs(norm-1)<=1e-4` |
| `seal`, `wrench` | 各 [K] 数值 | finite 且位于 [0,1] |
| `q_sw` | [K] 数值 | 与 `seal*wrench` 一致 |
| `collision_self_seal` | [K] bool | 完整 dense 的原 Seal 自碰 |
| `collision_self_envelope` | [K] bool | 冻结 target-only 包络自碰 |
| `collision_self` | [K] bool | 与两个 self 分量的逻辑 OR 一致 |
| `q_inst` | [K] 数值 | 与 `q_sw*(~collision_self)` 一致 |
| `label_valid` | [K]，必须 bool | p/n、S/W 和两个 self 分项都已求值且合法 |

q 的浮点乘积核对采用 `atol=1e-6, rtol=1e-5`；合法质量在 [0,1]。所有数值数组禁止 Inf，mask=false 行允许 NaN；布尔分项在无标签行只是占位，**不得把其 false 占位解释为已确认无自碰**。q_inst 的有效性需要 S、W、自碰分项共同已知，首版全部监督使用共同 label_valid。部分分量已知时可保留诊断，但不能在本合同里补零形成完整负例。

空文件或全 false 可通过结构检查，但 `has_supervision=false`，不等于可用于训练。参考校验工具只能检查数值/schema 关系，不能重算真实 teacher、校验隐藏资产或证明 sidecar 身份真实。旧 SW 校验入口仅用于旧文件兼容核对；必须用 `revision_contract.py check-labels` 检查新质量，参见[复现说明](TRAIN_EVAL_REPRODUCE.md)及[v2参考程序](reference/revision_contract.py)。

动作表另存：

| 字段 | 内容 |
|---|---|
| `attempt_id / sample_id / group_id` | 动作、实例、观测及 split 关联 |
| `point_cam_m / normal_out_cam` | 原规划接触点和有向轴 |
| `outcome / label_valid` | 0/1 与标注有效性；未记录不补零 |
| `pose_source` | `planned_pipeline_contact` |
| `actual_execution_verified` | 现有数据为 false |
| `pose_support_valid / exclusion_reason` | 观测支持及排除原因，与成败标签分开 |
| `pose_transform_sha256` | 原动作系到相机系的标定和适配代码身份 |

## 8. B 的监督与投影

B 默认只在共同的明确 query 姿态上计算源域 loss，不要求把稀疏标签涂成稠密全图真值。编码器特征图或显示用热力图不等于每像素均已具备监督；未标注位置保持 unknown。

若未来采用全像素 canonical 监督，必须对每像素定义 `(p,n_obs)` 并生成其真实标签。投影需检查同实例、正深度、可见性和 z-buffer；多个不同姿态投到同像素不能平均成一个虚构姿态标签。

历史动作也不能只读取其投影像素上的无条件 canonical 热力分数。B 的 pose-conditioned 评分应接收记录的 p/n，区别观测 normal 和动作 axis；实现方式见 [ROUTE_B.md](ROUTE_B.md)。

## 9. 实现验收

优先完成直接影响合同的核验：

- 原始投反影、单位、实例行号、轴向符号与工具变换一致。
- organized normal 严格拒绝跨实例 stencil、无效深度和指定跳变；历史动作没有翻轴或吸附。
- 图像网格候选排序、tie-break、128 前缀与 256 完整序列确定；A/B 使用相同 query 与 label identity。
- 固定真实 query 的 S/W 与冻结原 evaluator 一致，self 分项与原碰撞函数目标部分一致；独立外部项与 self 合并后复原旧总 C。
- 修改点、轴、normal/sampler/teacher 配置后旧缓存身份被拒绝。
- 标签未知、padding、重复点和空实例不产生伪监督，未观测区域不能补成已确认无碰撞。
- scene/group/hash 不跨 split；统计总数、有效数、各类排除和正负例分布，无输出实例保留在评价分母。
- 分项报告法向、候选构建、teacher、编码、查询评分及端到端耗时；无实测时只标预算/目标。

合成工具能验证数学与文件合同，不能替代真实标定、teacher 数值对齐、原始数据来源与逐记录核验。
