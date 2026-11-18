# 路线 A：单物体稀疏点云与 RGB（v2 修订）

## 1. 目标、范围与实现状态

路线 A 保留稀疏 3D 骨干，在目标物体的可见 RGB-D 上预测指定接触姿态的 Seal、Wrench、含目标自碰约束的实例质量，以及历史规划动作成败。它不输入其它物体或场景背景。共同数据、法向、查询和 teacher 定义见 [DATA_AND_LABELS.md](DATA_AND_LABELS.md)，训练与评价见 [TRAIN_EVAL_REPRODUCE.md](TRAIN_EVAL_REPRODUCE.md)。

既有生产代码基线仍为 [`87f29d29`](https://github.com/chichu-lv/suctionnet_modern_20mm/commit/87f29d29bea8fa5153d6df3f8253d9ced9e4b441)。本文及 [route_a.json](reference/route_a.json) 是**待实现的 v2 设计合同**，不是已经训练、部署或计时的新模型。新增模型建议命名 `InstanceSuction3D`；旧 E5 的输入、损失和配置校验不能通过改几个字段直接变成这一模型。

相对 v1，本次纠正了五件事：保留目标自碰；不再对全量原始点逐点做球邻域 PCA；学习输入不依赖任意切平面基的角度槽位；明确稀疏特征覆盖与原始 RGB 旁路；历史动作在主干特征缺失但原始几何仍有效时继续评分。

## 2. 质量目标：排除外部碰撞，保留目标自碰

冻结 teacher 的目标自碰结果记为 `C_self∈{0,1}`，其中 1 表示目标自身几何与工具发生碰撞：

$$
Q_{\mathrm{inst20}}=S_{20}W_{20}(1-C_{\mathrm{self}}),\qquad
Q_{\mathrm{SW20}}=S_{20}W_{20}.
$$

模型 q 头预测前者，质量身份为 `gt_INST20_pose_v2`；后者仍保存为诊断标签。Seal 和 Wrench 继续独立监督，不能把碰撞导致的 q=0 反推为 Seal 或 Wrench 必定为零。

`C_self` 包含冻结协议的目标密封腔自碰与目标工具包络碰撞；它不包含其它物体、桌面或场景外部包络项。不得把整个旧 `collision_flag` 清零，也不得通过 `gt_B/C` 反推标签。目标自碰的具体几何、符号和容差遵循共同 teacher 合同，而不是另外写一个“中心高度大于零”的近似门槛。[既有碰撞实现](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/full_data.py)

单目可见支撑不能证明工具与完整目标不碰撞；完整对象几何只供离线 teacher 使用。新 q 是学习到的实例质量代理，不是已验证的机器人可达性或完整接近轨迹证明。

## 3. 原始支撑、主干点与输入特征

### 3.1 原始支撑和中心

一个源样本对应 `scene_id + frame_id + instance_id`。原始有效深度像素、相机系 XYZ、配准 RGB、像素 ID 和实例身份逐行对应，形成 `P_support[M,3]`。仍保留原分辨率 organized point map 供法向计算，不为了建立点云而丢掉图像拓扑。

默认训练至少100个有效原始点；不足时记录 `insufficient_support`。验证/测试保留这些实例身份，无输出计入主指标失败分母。中心是抽样前全部有效原始点的均值 `mu_vis_cam_m`，只用于平移归一化，不是物理质量中心或 teacher 的完整对象中心。物理半径始终使用米，不按物体尺寸缩放到单位球。

原始支撑点不因主干抽样而删除。非有限 RGB 不使有效深度点消失：RGB 在归一化后置零，记录 `rgb_valid=False`；无效观测法向同样置零并记录 `normal_valid=False`。不能将这些零当作补出来的观测或额外样本。

### 3.2 主干预算与覆盖

先以2 mm体素去重，再对体素代表点做确定性 FPS，默认最多2048点。体素坐标为 `floor((p-mu_vis)/0.002)`；每体素取距离体素中心最近的原始点，距离并列取最小 `source_point_id`。代表点保留该点的真实 XYZ/RGB/法向，不将独立通道从不同点拼接。

FPS 使用唯一代表点，首点为最小 source ID，之后最大化到已选集合的最小欧氏距离，并列取最小 source ID；几何选择用 CPU float64。2048不足时可以显式启用4096或8192预算，并保持同一 FPS 前缀。预算进入配置/checkpoint和计时报告，**不在运行时悄悄增加预算或放大关联半径**。不足预算时用 mask 表示padding，重复点不作为覆盖证据。

记录原始支撑到同一4连通组件主干点的最近距离及第三近距离的 p50/p95/max、最大最近邻覆盖半径、10 mm内有效关联比例；不足三个邻居或整组件无主干点时分别计数，不把未定义距离补零。物体400 mm宽不能单独推出失效率；真正需要测的是这些距离与覆盖。若固定2048预算不足，允许局部 RGB/几何旁路继续工作，并如实记录主干特征缺失。

### 3.3 12维主干输入

| 特征 | 通道 | 数值处理 |
|---|---:|---|
| 相对位置 `p-mu_vis` | 3 | 米制坐标除以固定0.1 m |
| 配准 RGB | 3 | [0,1]后按共同mean/std归一化，无效位置随后置零 |
| 观测外向法向 | 3 | 单位向量；无效置零 |
| 相机系单位重力 `g_cam` | 3 | 每点广播完整方向 |

`rgb_valid` 和 `normal_valid` 是旁路掩码；局部编码显式使用，不增加上述主干输入通道。RGB增强对同一实例的所有原始支撑和主干点一致，参数见共同训练规范。

主干稀疏坐标使用 `int32[V,4]`、顺序 `batch,z,y,x`；特征为 `float32[V,12]`，保留原始点/体素/主干对应关系。输入体素已唯一时，不再用含padding的均值改变真实颜色或位置。[现有骨干 wrapper](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/models/backbones/spconv_resunet14.py)

## 4. 默认法向与共同候选

### 4.1 在线默认法向

v2 默认复用已有 `organized_normals(point_map, keep=single_instance_valid)`：相邻±1像素 XYZ差分叉乘，叉积模长大于 `1e-8`，对向深度跳变均小于10 mm，相关像素均满足实例和有效深度mask；边界、缺测或退化位置标为无效。最后使 `dot(n,-p_cam)>=0`。准确计算顺序和边界见共同合同及[原预处理实现](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/observation.py)。

这是一遍 organized 图像运算，不是数万个原始点的逐点15 mm球查询/PCA。它也不保证得到“物理15 mm平滑法向”；不应继续沿用这一说法。需要 PCA 时，采用独立命名的可选方案：2 mm体素支撑上仅对已选查询/所需主干点估计局部法向，单独记录参数、耗时和缓存身份。不可把不同法向方案混用同一姿态标签。

历史动作始终保留记录的 `normal_out_cam` 方向，只做有限性、非零和单位化检查，不重新朝相机翻转，不替换为观测法向。[历史动作读入](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/target_dataset.py)

### 4.2 共同候选与标签预算

A/B共用 `image_stratified_16x16_bit_reversed_morton_v2` 查询池：完整实例像素bbox划分16×16格，每格选择距离格中心最近、法向有效的实例像素，并列取最小source ID；空格略过。格顺序按8 bit Morton编号的bit reversal固定，Morton的x位在偶数位置、y位在奇数位置。这样前64/128项分布在整个bbox，而不是集中在一角。像素边界和完整排序定义以共同合同为准。

训练默认取非空查询序列的前128项，64项为显式可选预算；验证/推理最多256项。每个候选保留实际原始点和观测法向。减少训练标签预算不改变验证池，不把模型是否收敛写成既成结论。历史动作直接走指定姿态接口，不需要落入或注入候选池。

## 5. 离线 teacher 与缓存

teacher 始终对实际参与监督的同一 `(p_cam_m,n_out_cam)` 计算 S、W和目标自碰。所需资产为完整对象密集几何、原R20候选、冻结物理配置、场景和相机位姿；完整对象几何不得输入网络。

**v2生成器必须把目标和外部场景碰撞拆开**：保留目标密封腔与目标包络检查，删除仅对其它实例/环境做的检查，再生成 `q_inst=S*W*(1-C_self)`。Seal和Wrench复用 `seal_at_pose`、`wrench_scores_dual` 的冻结定义。生成器与缓存schema为待实现模块，执行合同见 [DATA_AND_LABELS.md](DATA_AND_LABELS.md)。

旧 `evaluate_poses` 是核对同姿态 S/W及原完整协议的参考路径，仍会计算外部碰撞，不能充当“已优化的 v2生成器”。原 v1中仅丢弃C、生成SW的示例不再作为本路线生产标签命令。本页不提供尚不存在的可执行生成命令。[既有任意姿态评价器](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/eval_r20.py)

先用固定少量真实场景/帧比较新旧 S/W一致性、目标碰撞组成和每查询耗时，再按128查询训练清单生成缓存。缓存按观测/查询身份去重复用并允许断点续建；不预先重算所有训练帧的256查询。改变法向、点位置、工具几何或容差必须改变缓存身份。旧参考工具的SW校验只覆盖其明确声明的schema，不能冒充已验证的自碰teacher。

## 6. 网络与维度

### 6.1 主干和全物体上下文

新增12通道 ResUNet14 wrapper，保留512维骨干输出和512→256投影，每个真实主干点得到 `F[N,256]`。已有wrapper输入硬编码8通道，须新增正确入口，不编辑受blob校验约束的外部源码。

真实主干点的masked mean/max拼成512维，经 MLP `512→128→64` 得到对象上下文 `G[64]`。它表达可见形状上下文，不调用旧 `instance_fusion`，也不等同完整质量分布。

### 6.2 特征关联：允许少于三个邻居，保留原始旁路

原始有效实例像素按4连通计算组件ID。每个支撑点从**同组件**主干点中选择最多三个10 mm内的最近邻，距离相同按source ID排序；至少一个邻居即可有效插值。非零距离按 `1/d` 归一化，零距离直接取同点特征。不能因第三邻居超界而丢掉前两个有效邻居。

无主干邻居时 `F_local=0, feature_valid=False`，原始位置、RGB和观测法向仍保留。相同实例/4连通约束减少跨不连通区域关联，但不是完整薄壁层识别或碰撞保证；不能以扩大距离掩盖关联失效。

canonical query的组件来自其原始像素。历史query用目标实例内距记录点最近、距离≤5 mm的原始几何点确定组件，并列取最小source ID；这只是关联锚点，**不移动记录的p、不改变记录的轴**。找不到此锚点时记 `pose_geometry_invalid` 并报告；找到几何锚点但 `Fq` 无10 mm内主干邻居时，以零 `Fq` 和 `query_feature_valid=False` 继续评分。

### 6.3 三径向区域与无任意切平面方向的描述

对同实例、同组件支撑点x，直接计算：

$$
d=x-p,\quad h=d^Tn,\quad d_t=d-hn,\quad\rho=\|d_t\|.
$$

不把 `argmin(abs(n))` 生成的切平面x/y坐标送入网络。网络支持限于 `|h|≤40 mm`；这是观测窗口，不改变teacher的物理定义。冻结target-only包络可达150mm，而原Seal self柱体无高度上限；局部窗口不覆盖全部自碰，Fq/G提供额外可见上下文，q_inst仍是学习代理而非几何认证。

| 区域 | 径向边界 | 最多采样数 | 区域输出 |
|---|---|---:|---:|
| core | 0≤rho<10 mm | 32 | 64 |
| inner | 10≤rho<18 mm | 32 | 64 |
| lip | 18≤rho≤22 mm | 64 | 64 |

区域内先按source ID唯一化，再对真实XYZ做与主干相同破同分规则的确定性FPS；不足时padding并mask。区域原始计数在截断前记录。主干特征的 `256→64` 投影结果与原始RGB三维、下列八维标量拼接：

```text
rho/r, h/0.040, n_obs·n, (d_t·g_cam)/r, n_obs·g_cam,
feature_valid, normal_valid, rgb_valid
```

r为该区域外半径；无效法向的点积置零，RGB在归一化后按其mask置零，缺失主干特征在投影后再次置零以避免bias伪造有效特征。**输入维度为64+3+8=75**，经 `75→64→64`。区域masked mean/max，加 `log1p(原始点数)`和非空标记共130维，再经 `130→64`。空区域特征置零，mask和计数不由padding增加。

原始RGB旁路使无主干特征的点仍能贡献颜色和局部几何；它不能独自保证学到某种纹理，真实纹理辨别仍需验证。

唇口额外只提供三个标量：`log1p(原始唇口点数)`、最大角间隙/`2π`、非空mask。最大间隙在真实非零切向方向上求排序后的循环角差最大值；可使用任意正交切平面基来计算角度，但角度序列、起始槽位及该基均不进入网络。空集间隙置0且mask=0；只有一条角射线时为1。该最大间隙在数学上不依赖切平面换基，浮点误差按容差核验。它是可见覆盖摘要，不是Seal真值或连续圆周都已被观测的证明。

teacher的72角度分箱仍沿冻结规则运行，不能据此宣称teacher或整个相机系网络具有严格SO(2)旋转不变性。本修订去除的是人工切平面坐标选择带来的无意义跳变。

### 6.4 Query融合与五个头

`Fq[256] + G[64] + 三块区域特征[192] + lip摘要[3] + 姿态描述[10] + query_feature_valid[1]` 共 **526维**。姿态描述为 `(p-mu_vis)/0.1, n, g_cam, n·g_cam`。经 `526→256→256` 得到h。

| 头 | 输入/层宽 | 语义 |
|---|---|---|
| q_inst | 256→128→1，sigmoid | 同姿态 `S20*W20*(1-C_self)` 代理 |
| seal | 三块区域与lip摘要，共195→128→1，sigmoid | S20 |
| wrench | h、G、delta_p/0.1、n、g，共329→128→1，sigmoid | 冻结协议W20 |
| ell | 256→128→1 | Laplace残差尺度的对数 |
| task | h、n、n·g，共260→128→1 | 历史规划动作成败logit |

新增MLP隐藏层统一ReLU、带bias、无dropout；输出激活如表。ell初始化为 `log(0.1)`、前向限制到[-6,1]。没有单独的碰撞分类头；自碰通过q_inst标签进入监督。S/W两个辅助头用于辨别质量来源。现有 `heads.py` 不包含上述完整新结构。[既有模型](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/models/suction20net.py)

## 7. 损失与历史动作

源域单query损失为：

$$
L=e^{-\ell}|\hat q_{\rm inst}-Q_{\rm inst20}|+\ell
+0.5\,\mathrm{SmoothL1}_{.05}(\hat s,S_{20})
+0.25\,\mathrm{SmoothL1}_{.05}(\hat w,W_{20}).
$$

ell是Laplace log-scale，不是log-variance。v2首版全部物理项使用共同label_valid，要求S、W及两个目标自碰分项均已求值；部分分量仅保留诊断，不补零形成监督。源实例/目标观测归约和预算依共同训练规范，不以无效标签乘零处理NaN。

待实现接口保持：

```python
encoded = model.encode_instance(observation)
scores = model.score_at_pose(encoded, p_cam_m, n_out_cam, query_batch_index)
```

历史动作默认以原 `planned_pipeline_contact` 的一个task logit计算一次BCEWithLogits，源回放系数0.5。`actual_execution_verified=false`意味着输出应解释为给定规划命令及观测的成败概率，不是已知真实执行落点的成功概率。只有完全缺少合法几何关联的动作才跳过并报告原因；不能仅因 `Fq` 缺失就删除历史样本。

轻度位姿微扰是可选正则，按共同训练规范启用，默认关闭；不能复制S/W标签到新姿态，也不把一个动作拆成多个独立成败样本。前200步和后800步的参数组必须覆盖本模型的实际新增层，不照搬旧E5组名。

## 8. 推理、排序与6-DoF输出

对共同最多256查询评分，源域按q_inst，目标域默认按task probability排序。返回q_inst、Seal、Wrench、ell及主干/几何支持诊断。可额外输出 `seal_hat*wrench_hat` 作为两个预测头的乘积，但不能称其为独立训练的q_SW头或等同teacher真值。

NMS采用真实3D距离10 mm、有向轴夹角15°，按query ID稳定破同分；无合法几何候选返回 `status=no_pose` 及原因。输出接触中心就是被评分的p，不在选择后用深度中位数偷偷移动它。

工具+Z为接近方向，故 `z_tool=-n_out`。参考工具继续用最不平行的相机基向量b构造 `x=normalize(b×z_tool), y=z_tool×x`，同值按x/y/z固定。这只是**无状态SE(3)序列化的roll约定**，不进入上述学习特征，不保证相邻输出roll连续，也不提供机器人连续运动策略。机器人若需要连续roll，应由控制器采用工具约束或上一帧切向参考运输并处理退化；不得把该控制策略默认为已经实现。

`T_base_tcp=T_base_cam*T_cam_contact*T_contact_tcp`，标定变换和接触中心/TCP定义必须与历史动作一致。点与有向法向仅约束5自由度，roll是约定或工具约束，不是网络额外预测出的物理量。

## 9. 实施与验收

新增实例builder、12通道wrapper、三区域编码、原始RGB旁路、姿态评分、实例质量teacher与loss、参数分组和训练入口。保持独立架构/schema/checkpoint，不复活已经退役的旧迁移兼容路径。

最低检查包括：同姿态S/W/目标自碰对齐；候选前128/256固定且可复现；跨任意切平面换基时上述局部标量与最大角间隙不变；1/2/3个有效主干邻居均正确插值；无Fq但有原始几何/RGB的动作仍有有效评分；完全缺少5 mm内几何锚点时有明确原因；所有输出中心/轴与被评分姿态一致。

报告batch1的model-only和端到端p50/p95，分别列出organized法向、体素/FPS、支撑查询、网络、数据搬运和NMS耗时，并记录原始点数、主干预算与查询数。15–30 ms仍是未经实测的工程目标，不能作为本修订已达到的速度。
