# 单实例数据、法向与同姿态物理标签（v3 几何修订）

本页定义 A/B 共用的数据、查询姿态、离线 teacher 和目标动作监督。**当前综合质量保留目标自身碰撞，去掉其他物体与桌面的碰撞项**：`quality_contract=gt_INST20_pose_v2`，参考标签 schema 为 `suction20.instance_inst20.v2`。纯 `q_sw=S20*W20` 继续保存，供物理诊断和与上一版比较；不能把旧 `gt_SW20_pose_v1` 文件改名后当作 v2。

本次 v3 更新法向与候选构建，物理 teacher 的 S/W/self 定义未变，因此质量身份仍为 `gt_INST20_pose_v2`、标签 schema 仍为 `suction20.instance_inst20.v2`。normal-spec 与 sampler-spec 改为 v3，必须重建或逐精确姿态核验缓存，不能把旧姿态标签直接改名复用。

本页对应已实现的 `suction20/instance_geometry.py`、`instance_data.py`、`instance_teacher.py` 和 `scripts/build_instance_dataset.py`。真实命令见[代码运行说明](IMPLEMENTATION.md)。模型只接收当前观测，完整对象几何只用于离线标签与评价；下文公式和伪代码描述实现合同，不是额外 CLI。


数值基线：v3参考及首版几何正确性验收从原始深度值/内参以float64构造XYZ、canonical query和理想环，在转为网络FP32特征前完成接触边界判定。历史p/n从原记录以float64读取并保持原姿态，不从深度重建或覆盖。已存为FP32的XYZ升精度不能恢复原值；FP32部署几何须另记precision-contract并核验。参考1e-12m仅处理float64边界舍入，不是像素/深度误差容限。[边界与精度说明](BOUNDARY_REVIEW.md)

当前直接使用官方100个训练场景；原90/10开发划分停止用于新路线。源域完整test参与best选择，checkpoint标记 `test_selected`。以下为native INST20合同；[公开论文评价](PUBLIC_METHOD_COMPARISON.md)的仅半径参数化方案待定，不阻塞数据构建或训练。

## 1. 原始资产与固定划分

源域需要已下载的 SuctionNet RGB-D、相机内参、实例掩码、场景与相机位姿、完整对象 `dense_point_clouds`、既有 R20 Seal 候选及冻结物理配置。GitHub 汇总不含这些完整训练资产。RGB、深度、分割必须在同一已校正相机网格，标定文件及深度单位随 manifest 固定。

新入口使用 `suction20.instance_data` 的独立划分；旧E5 `datasets.py` 保留原实验定义。先划分scene，再展开frame和实例：

| split | 场景 | 视角 |
|---|---|---|
| train | 0–99，全部100场景 | 0–255，训练 sampler 固定 |
| test_seen | 100–129 | 每场景 256 视角 |
| test_similar | 130–159 | 每场景 256 视角 |
| test_novel | 160–189 | 每场景 256 视角 |

同帧不同实例不跨 split；这不是CAD物体身份互斥划分。源域无单独val，test用于选模；test按完整可见实例展开、保存固定query与顺序，选模后的成绩不能称独立test估计。

生产source manifest只使用 `split="train"|"test"`，另存 `test_subset="seen"|"similar"|"novel"`；不将 `test_seen` 写入split。目标域保留train/val/test。

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

### 3.2 v3 默认：带掩码平滑与宽基线 organized 法向

默认 normal-spec 改为 `organized_masked3x3_span3_v3`。旧 [observation.py](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/observation.py) 的 ±1 差分作为原实现和噪声对照，不再作为新路线默认。新算法仍只需要固定图像偏移、掩码求和与叉乘，不引入三维球查询或 PCA。

**仅供法向计算的平滑，不修改接触点或原始支持。** 设原始相机系 XYZ 图为 P、原始轴向深度为 Z、`keep=target_instance_mask & valid_depth`。对每个原始中心像素 c：

1. 中心必须 keep 为真。取3×3窗口中的原始像素 q，仅当 q 与 c 同实例、keep 为真且 `abs(Z[q]-Z[c])<0.010 m` 时计入均值；比较严格小于，不是小于等于。
2. 窗口中至少5个合格像素（包括中心）才令 `smooth_valid[c]=true`。以合格原始 XYZ 的算术均值生成 `P_smooth[c]`，不独立混合某一位置的XY和另一位置的Z。图外、非有限或无效数据不参与求和；禁止用 `NaN*0` 实现mask。
3. 无效平滑位置置零并保留mask，不补洞、不生成新的有效深度像素。

然后使用固定 ±3 像素中心差分：

```text
dx = P_smooth[v, u+3] - P_smooth[v, u-3]
dy = P_smooth[v+3, u] - P_smooth[v-3, u]
raw = cross(dx, dy)
```

一个新法向有效，当且仅当以下条件均成立：

- 中心以及水平/竖直±3端点的 `smooth_valid` 为真；中心原始 keep 也为真。
- 横向 `u-3...u+3` 和纵向 `v-3...v+3` 的全部原始像素均在图内、keep 为真且属于同一实例。不能越过掩码空洞只连接两端。
- 沿两条原始7像素路径，每一对相邻像素均满足 `abs(Z_next-Z_prev)<0.010 m`。不再要求两端总深度差小于10mm，否则扩大基线会额外排除正常倾斜平面。
- raw 有限且 `norm(raw)>1e-8 m²`。

归一化后，仅对自动观测法向按原始中心 p 执行 `dot(n,-p)>=0` 定号。不满足条件时 `normal_valid=false`、法向值为零，**没有隐式退回±1差分的路径**。这些像素仍可提供原始 RGB/深度支持，只是不提供 canonical 候选轴。

原始 `depth_m`、`P_support`、`mu_vis`、query 的 `p_cam_m`、RGB及 teacher 的接触中心全部保持不变；P_smooth只用于估计 n。历史动作继续使用记录的有向轴，不被平滑法向替换。

固定±3像素并不保证恒定毫米支持。在 Z=0.6m、fx=600px 的正对平面上，两端跨度约6mm；更远、更近和倾斜表面均不同。3×3窗口的深度门控还可能在斜面或边缘造成非对称权重；“邻点深度差小于10mm”也不能隔离所有小于10mm的真实台阶。宽stencil在细长物体、孔洞附近会降低法向有效率。必须报告法向误差、有效率和候选覆盖，不能只报告平滑后的均值角误差。

normal-spec包含窗口3×3、最少5点、span=3、严格10mm深度门槛、逐相邻路径门控、叉积阈值、计算精度和无回退策略。任何变化都需要新normal身份与查询缓存。多实例优化可一次处理原图并比较相邻实例ID，但必须与上述逐实例定义一致；没有真实测速前不承诺“毫秒级”或固定3–5倍降噪。

### 3.3 噪声与 teacher 条件必须分清

用户给出的 `atan(0.5mm/2mm)≈14.04°` 是一种端点深度差情形，不是已测传感器总体噪声。若两端独立且各自噪声标准差为 sigma_z，则差值标准差为 `sqrt(2)*sigma_z`，局部斜率误差约为该差值除以差分物理基线；匹配波纹的空间相关性、量化和真实曲率会改变这一估计。扩大基线与平均只在明确噪声假设下具有可推导收益，真实改善必须测量。

按错误候选轴重评后，teacher给低分往往是**该轴条件下正确的物理评价**；问题是法向噪声制造了差的候选姿态，并降低训练标签的重复性，不能把所有低分直接称为伪负例。平面上14°偏差在固定20mm表面半径的轴向投影约4.84mm；原self按query横向rho<20mm限域，对无限平面对应的最高h约为 `20mm*tan(14°)=4.99mm`，仍低于5mm。约 `atan(5/20)=14.04°` 才跨此理想边界；上述界限假定query中心恰在真实平面上，中心深度误差会额外平移h；有限对象、离散采样和工具包络另有影响。因此不能声称14°必触发5mm自碰或Seal一定变成0。

新法向用于候选构建后，仍在同一个精确 `(p,n)` 上生成S/W/self。对比噪声方案应分别测量法向稳定性、候选质量覆盖、标签分布和同姿态teacher一致性；不能通过给差轴强行赋高分来“净化标签”。

### 3.4 可选 PCA 精度配置，不进入 B 默认路径

若 A 后续确需 PCA，采用独立配置：先在同实例 2 mm 体素代表点上建立一次空间索引，仅对已配置主干预算（默认最多2048）及最多256个查询点拟合 15 mm 邻域，禁止在全量原始支持上逐点拟合。保持原始支持供局部细节读取；normal-invalid、法向回填和候选改变都要明示。

PCA 参数、体素代表规则、最少独立邻点数及退化判断进入独立 normal-spec，缓存不得与默认配置互换。A/B 的主对比使用同一默认候选位置和轴；如果比较 PCA 精度配置，必须另列结果与预处理耗时，不能一边改变输入轴一边声称只比较网络。

## 4. 共同候选与精确动作查询

### 4.1 图像分层候选与无重复回填

共同 sampler 改为 `image_stratified_morton_refill_v3`。v2已经使用反转Morton顺序，行优先截取不是现有参考代码的错误；本次新增的是稀疏占空比下的回填。两路线始终读取同一份完整候选序列。

1. 在原图目标实例mask上计算完整包围框，包含mask内无深度位置。整数像素中心的框外边界为 `(u_min-0.5,v_min-0.5)` 到 `(u_max+0.5,v_max+0.5)`，均分为16×16半开网格，最外边界闭合。
2. eligible像素满足同实例、原始有效深度及v3 `normal_valid=true`。保存唯一原图 `source_point_id=v*W+u`，一个原像素仅属于一个格，不复制、不补洞。
3. 固定格顺序与v2相同：cx、cy各为0…15，`m=sum(((cx>>b)&1)<<(2*b) | ((cy>>b)&1)<<(2*b+1) for b in range(4))`，将m的8个bit反转后升序排列。空格跳过。第一轮每个非空格最多取一个点：到本格中心的原图二维像素平方距离最小，并列取最小source ID。
4. 第一轮不足256时，以**相同格顺序逐轮回填**。每轮每个尚有剩余eligible像素的格最多取一个。第2个及后续格内点用二维像素最远点规则：对每个剩余像素，计算其到本格已选像素集合的最小平方距离，选该值最大的点；并列取最小source ID。只需维护每个格的剩余点及最小距离，不建立三维空间树。已耗尽的格跳过。
5. 达到256个候选或所有eligible像素耗尽即停止。先确定这份最多256的序列，训练128和轻量64都只取它的前缀；不能依据请求K改变网格数、格内规则、排序或随机种子。第一轮若已填满256则不执行回填。
6. 每个候选从其原始像素读取精确p与对应v3观测法向n；不使用P_smooth作为接触位置。保存source ID、格号、轮次、查询顺序及p/n身份，距离和NMS仍用真实米制坐标。

对应逻辑为：

```text
buckets = eligible pixels grouped by the fixed 16x16 cells
cell_order = occupied cells sorted by bit-reversed Morton code
selected = []
for round_index = 0, 1, ...:
    added = 0
    for cell in cell_order:
        if cell exhausted: continue
        if cell has no selected pixel: choose nearest to cell center
        else: choose farthest from this cell's already selected pixels
        break ties by smallest original source ID
        append pixel once; update that cell's remaining distances
        added += 1
        if len(selected) == 256: stop
    if added == 0: stop
train128 = selected[:128]
train64 = selected[:64]
```

只要eligible像素足够，回填会达到256；否则输出所有可用唯一像素。显式记录 `eligible_pixel_count`、`nonempty_cell_count`、`first_pass_count`、`refill_count`、`refill_rounds`、最终query数及不足预算原因。数组padding只补mask，不计为候选。

例如128×128的斜条掩码 `abs(u-v)<=4` 有1132个eligible像素，v2一格一点仅产生46个候选；v3应产生256个互不重复的候选，且64/128序列与完整池前缀一致。这个例子仅验证sampler，eligible资格在真实数据中仍由深度和法向共同决定。

反转Morton与格内最远点会使前缀更分散，但**不保证对任意稀疏形状都均匀覆盖物理表面积**；多轮回填也可能增加相近候选，须同时报告有效像素分布、米制最近邻间距、NMS前后覆盖与实际query数。掩码面积占空比不严格等于含有效像素的格占比，不能用“40%面积”直接推断恰好多少格有候选。

训练与源域回放默认前128，64是独立命名配置；验证、测试和推理最多256。64/128是否足够收敛仍需测量。A的2mm代表点池和主干FPS独立于共同query；B默认不运行3D FPS。

新normal、回填和格内取点会改变精确p/n清单，因此缓存身份绑定v3 sampler；不能在相同数据名下静默用旧46点缓存当作新256点缓存。重复instance draw仍复用相同query缓存。

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

该逻辑已在 `InstanceTeacher.evaluate` 实现：仅包含目标体素、所有row都为该目标的树复用冻结碰撞函数，传入全false初始self数组得到包络分量，再与 `seal_at_pose` 的self合并。合成官方格式资产的测试核对S/W及target-only结果；完整真实资产的重算与成本测量仍待执行。

场景/对象变换可复用；每个对象的完整 dense 数组、原候选均值及目标包络索引尽量复用，不加载桌面或其他物体树。Wrench 批量向量化。Seal 仍是完整 dense 扫描，其 `chunk_points` 只限制临时内存，不减少参与点数；去掉外部碰撞并不证明整个 teacher 已经快。任何批量或索引优化必须保留数值合同，不可用有限球裁剪替代无高度上限的环柱。

## 6. 缓存预算与身份

当前builder按所选scene/frame范围构建固定 `(instance_uid,p,n)` 缓存，默认每训练实例128个标签；重复instance draw复用相同缓存，训练调度器的状态和顺序身份保存在checkpoint。按需异步生成缓存仍属于后续优化，不要求为100训练场景每实例计算256个标签；训练默认128，test固定256。当前builder按所请求的scene/frame范围完整生成缓存；正式trainer要求完整100场景清单。可分范围生成做成本试验，但只有显式 `--allow-partial-manifests` 才能用于链路验证，不冒充正式训练。

首次真实资产接入时，先在约 64–128 个覆盖不同实例尺寸与 dense 点数的样本上测量：加载时间、Seal 秒/query、self-envelope 时间、秒/instance、缓存大小和命中率。然后按实际不同实例及查询数估计总成本。该数量是成本剖析样本，不是收敛证明或新的性能结果；不能直接许诺几小时重算或训练。

缓存身份绑定：观测与实例身份、完整对象资产 hash、冻结物理配置 hash、teacher 代码 commit、normal-spec、sampler-spec、精确查询点/轴字节 hash、split 清单 hash、质量合同和单位。以下变化需要新身份：点或法向改变、normal/sampler 变化、物理参数/自碰定义变化。RGB 光度增强不改标签；刚性坐标变换须一致作用 p/n/g 并保存逆变换。

旧 S/W 仅能在其原精确 p/n 保持不变时复用；旧总 collision 不能拆出 self，`gt_B` 也不能通过除法恢复被门控抹去的质量。不可先把标签稠密插值再称为任意姿态真值。

## 7. 数据文件和 v2 标签字段

原始数据、权重和整套标签留在数据根目录，不提交GitHub。source manifest由 `scripts/build_instance_dataset.py` 自动生成，一行一个实例观测；目标动作JSONL沿用独立 `suction20.target.v1` schema，示例见[运行说明](IMPLEMENTATION.md)。不要手写解释性hash占位符作为生产清单。

| source manifest字段 | 实际内容 |
|---|---|
| `schema_version` / `manifest_version` | `instance_observation_v2` / `suction20.instance_manifest.v3` |
| `sample_id/instance_uid/group_id/capture_uid` | 稳定实例、场景分组及帧身份 |
| `scene_id/frame_id/instance_row/obj_id/camera` | 官方场景0–189、帧0–255、实例行/对象ID和相机 |
| `split/test_subset/source_kind` | train或test；seen/similar/novel或null；`suctionnet_rgbd` |
| `normal_spec/sampler_spec/precision_spec/quality_contract` | 代码导出的四项精确合同字符串 |
| `observation_npz/observation_sha256/observation_identity` | 实例NPZ、实际文件字节hash、来源身份 |
| `query_cache/query_identity` | query与观测在同一个NPZ，独立精确p/n/顺序身份 |
| `label_cache/label_sha256/label_identity/labelled_queries` | 标签NPZ、文件hash、物理/代码/对象/视角身份、已标注前缀长度；无标签模式为null/0 |
| `source_hashes/calibration_sha256` | 原始RGB/depth/mask与标定来源的hash |
| `support_count/query_count/trainable/status` | 原始支持点数、实际K与样本状态 |

原始观测包含RGB、米制XYZ图、实例mask、intrinsics/gravity及有效性标记。query字段实际为 `query_points[K,3]`、`query_normals[K,3]`、`query_valid[K]`、`query_uv[K,2]`、`query_source_ids[K]`、`query_ids[K]`、`query_components`、`query_cell_ids`和`query_rounds`。它们由 `prepare_instance` 生成并直接供两模型消费。

生产NPZ把数组作为独立键、字符串和字典放在 `metadata_json`，`load_observation/load_labels`解包后返回统一字典。参考demo旧NPZ把schema字符串放顶层，两种序列化不能混用。生产数据可直接校验：

```python
from suction20.instance_data import InstanceDataset
source = InstanceDataset('/data/instance-v3/train/manifest_train.jsonl',
                         split='train', training_queries=128)
item = source[0]  # 校验文件hash、精确query、label数值关系和身份
```

v2标签解包后的共同监督字段如下；生产NPZ的metadata保存 `schema_version="suction20.instance_inst20.v2"` 与 `quality_contract="gt_INST20_pose_v2"`：

| 字段 | shape / 类型 | `label_valid=true` 的约束 |
|---|---|---|
| `points` | [K,3] 数值，默认 float64 | 相机系米制查询点，全部 finite |
| `normal_out` | [K,3] 数值，默认 float64 | 外向单位轴，全部 finite，`abs(norm-1)<=1e-4` |
| `seal`, `wrench` | 各 [K] 数值 | finite 且位于 [0,1] |
| `q_sw` | [K] 数值 | 与 `seal*wrench` 一致 |
| `collision_self_seal` | [K] bool | 完整 dense 的原 Seal 自碰 |
| `collision_self_envelope` | [K] bool | 冻结 target-only 包络自碰 |
| `collision_self` | [K] bool | 与两个 self 分量的逻辑 OR 一致 |
| `q_inst` | [K] 数值 | 与 `q_sw*(~collision_self)` 一致 |
| `label_valid` | [K]，必须 bool | p/n、S/W 和两个 self 分项都已求值且合法 |

q 的浮点乘积核对采用 `atol=1e-6, rtol=1e-5`；合法质量在 [0,1]。所有数值数组禁止 Inf，mask=false 行允许 NaN；布尔分项在无标签行只是占位，**不得把其 false 占位解释为已确认无自碰**。q_inst 的有效性需要 S、W、自碰分项共同已知，首版全部监督使用共同 label_valid。部分分量已知时可保留诊断，但不能在本合同里补零形成完整负例。

空文件或全 false 可通过结构检查，但 `has_supervision=false`，不等于可用于训练。参考校验工具只能检查数值/schema 关系，不能重算真实 teacher、校验隐藏资产或证明 sidecar 身份真实。生产文件通过 `load_labels` / `InstanceDataset` 调用 `validate_labels`校验。`revision_contract.py check-labels`只用于其自身的参考demo格式，不能直接读取生产metadata_json版NPZ。参见[复现说明](TRAIN_EVAL_REPRODUCE.md)。

目标动作输入与运行输出字段分别为：

| 字段 | 内容 |
|---|---|
| 输入 `sample_id / group_id / split / success_definition` | 动作、分组和唯一非空成功定义 |
| `point_cam_m / normal_out_cam` | 原规划接触点和有向轴 |
| 输入 `attempt.success / attempt.label_valid` | 0/1与标注有效性；输出行使用outcome |
| `pose_source` | `planned_pipeline_contact` |
| `actual_execution_verified` | 现有数据为 false |
| 输出 `pose_valid / pose_support_valid / supported` | 数值有效、观测支持及最终可评价状态，与成败标签分开 |
| checkpoint数据身份 | manifest和原始RGB/depth/mask字节hash；相机系p/n由原清单提供，禁止静默替换 |

## 8. B 的监督与投影

B 默认只在共同的明确 query 姿态上计算源域 loss，不要求把稀疏标签涂成稠密全图真值。编码器特征图或显示用热力图不等于每像素均已具备监督；未标注位置保持 unknown。

若未来采用全像素 canonical 监督，必须对每像素定义 `(p,n_obs)` 并生成其真实标签。投影需检查同实例、正深度、可见性和 z-buffer；多个不同姿态投到同像素不能平均成一个虚构姿态标签。

历史动作也不能只读取其投影像素上的无条件 canonical 热力分数。B 的 pose-conditioned 评分应接收记录的 p/n，区别观测 normal 和动作 axis；实现方式见 [ROUTE_B.md](ROUTE_B.md)。

## 9. 实现验收

优先完成直接影响合同的核验：

- 原始投反影、单位、实例行号、轴向符号与工具变换一致。
- v3法向只平滑normal输入，原始p/support不变；3×3至少5支持及±3路径严格拒绝跨实例、空洞和指定相邻跳变，缺少支持不回退±1；历史动作没有翻轴或吸附。
- 图像网格、回填轮次、格内二维最远点与tie-break确定；稀疏格有足够eligible时填满256且不重复，64/128仍是完整序列前缀；A/B使用相同query与label identity。
- 固定真实 query 的 S/W 与冻结原 evaluator 一致，self 分项与原碰撞函数目标部分一致；独立外部项与 self 合并后复原旧总 C。
- 修改点、轴、normal/sampler/teacher 配置后旧缓存身份被拒绝。
- 标签未知、padding、重复点和空实例不产生伪监督，未观测区域不能补成已确认无碰撞。
- scene/group/hash 不跨 split；统计总数、有效数、各类排除和正负例分布，无输出实例保留在评价分母。
- 分项报告法向、候选构建、teacher、编码、查询评分及端到端耗时；无实测时只标预算/目标。

合成工具能验证数学与文件合同，不能替代真实标定、teacher 数值对齐、原始数据来源与逐记录核验。
