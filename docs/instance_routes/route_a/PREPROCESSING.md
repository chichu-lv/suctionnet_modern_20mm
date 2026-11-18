# A 预处理：原始支持、主干采样和共同查询

本流程消费[数据准备](DATA_PREPARATION.md)的已校验实例观测，产生 `encode_instance` 和 `score_at_pose` 所需的真实几何、主干输入及query。物理单位、normal/sampler身份固定为[共同 v3 合同](../DATA_AND_LABELS.md)，参数快照见 [route_a.json](../reference/route_a.json)。

共同organized几何、法向和候选已实现在 `suction20/instance_geometry.py::prepare_instance`；A体素/FPS、组件邻居索引、分区与lip摘要实现在 `models/instance_route_a_geometry.py`。当前确定性几何使用CPU float64与SciPy cKDTree；这是正确性实现，不能由CPU测试或CUDA骨干存在推定完整预处理已达到15–30ms。

## 1. 三个集合与数值入口

| 集合 | 构建方式 | 用途 |
|---|---|---|
| 原始支持 `P_support[M,3]` | 目标实例所有有限有效深度像素，保留原source ID | 精细RGB/深度、局部几何与lip摘要 |
| 主干点 | 2mm体素代表点后确定性FPS，默认最多2048 | 稀疏编码预算 |
| query | 共同16×16图像分层与回填，最多256 | A/B同条件训练、验证与推理 |

query不要求属于主干点，原始支持不会因主干抽样而删除。历史动作是额外的直接姿态查询，不注入候选池。`source_point_id=v*W+u`稳定贯穿反投影、体素、抽样与输出。

用原始米制深度和内参以float64构造 `P=Z*K^-1*[u,v,1]`，保留organized XYZ图。`mu_vis`在抽样前由全体有效支持求均值；只作平移归一化。原始有效支持少于100点时源训练记排除原因，验证/测试保留实例与无输出账本。网络特征在边界判定完成后转FP32；参考 `1e-12m`只处理float64舍入，不是传感器容差。

## 2. 默认法向逐步操作

`normal_spec=organized_masked3x3_span3_v3`。设 `keep=target_mask & valid_depth`，输入XYZ和Z均为原始值：

1. 对原始有效中心取3×3邻域，只纳入同实例、keep=true且 `abs(Z_neighbor-Z_center)<0.010m` 的点；包括中心至少5点才有效。先选择合法点再求XYZ等权均值，不用NaN乘零。
2. 以平滑XYZ的水平、竖直±3像素端点作中心差分，`raw=cross(dx,dy)`。
3. 中心和四端点均需smooth_valid；横纵两条完整7像素原始路径均在图内、有效且同实例；路径每对相邻原始Z差严格<10mm。
4. raw有限且模长严格>1e-8m²才单位化；canonical法向按原始p执行 `dot(n,-p)>=0` 定号。
5. 不满足时法向置零、normal_valid=false，不回退±1、不补洞。该像素已有RGB/深度支持继续保留。

P_smooth只供法向估计。原始p、P_support、query中心、mu_vis和历史动作轴均保持原值。固定±3像素并非恒定毫米半径；对边缘、孔洞和细物体分别记录无效比例。历史n仅检查有限、非零和单位化，保留其有向符号。[法向边界解释](../BOUNDARY_REVIEW.md)

## 3. 共同 query 序列

先在完整实例mask（包含深度缺失位置）上确定bbox像素外边界 `(umin-.5,vmin-.5)` 至 `(umax+.5,vmax+.5)`；分成16×16格，内部半开、最外边界闭合。eligible要求同实例、有效原深度及normal_valid。

cx/cy的4bit交错为8bit Morton码（x占偶数bit、y占奇数bit），按该码bit reversal后的值升序访问格子。首轮每非空格取离格中心最近的eligible原像素；同分取最小source ID。后续每轮按同一格顺序，在未耗尽格取离本格已选集合最远的未选像素，以最小平方像素距离为准；同分仍取最小source ID。

达到256或eligible耗尽即停止。先形成唯一完整序列，再得到 `train128=full[:128]`、可选 `train64=full[:64]`。不能按请求K改变排序、格数或随机化；padding不计为query。每项保存原始p、v3观测n、source/query ID、格号和轮次。

必须保存 eligible数、非空格数、首轮数、回填数、回填轮数、最终K和停止原因。稀疏斜条有足够eligible时应回填满256；二维分层不保证三维物理表面积均匀，覆盖仍由[实验方案](EXPERIMENT_PLAN.md)评价。

## 4. 主干点和12维输入

体素键为 `floor((p-mu_vis)/0.002)`。每体素选距离体素中心最近的原始点，同分取最小source ID；保留该行真实XYZ/RGB/normal，不平均混合不同点的属性。

对唯一代表点做float64确定性3D FPS：首点为最小source ID，之后最大化到已选集合的最小欧氏距离，同分按source ID。默认最多2048点；不足时mask padding，不重复计数。4096/8192属于显式备选预算，沿同一FPS前缀，须另存配置与性能报告；线上不能自动扩点或放大关联半径。

| 特征 | 维数 | 数值处理 |
|---|---:|---|
| 相对位置 | 3 | `(p-mu_vis)/0.1m` |
| RGB | 3 | [0,1]后以mean=(.485,.456,.406)、std=(.229,.224,.225)归一化 |
| 观测法向 | 3 | 单位外向n，无效置零 |
| 相机系重力 | 3 | 广播完整单位方向，保留来源 |

非有限RGB不删除深度支持，归一化后置零并存rgb_valid=false；normal_valid、rgb_valid为旁路mask，不增加12通道。训练实例级光度增强固定brightness/contrast/saturation=.2、hue=.05、灰度概率.2，raw支持与主干点使用同一次变换。首版不做米制尺度归一化或几何缩放。

构造spconv输入时稀疏坐标为int32 `[V,4]`、顺序batch/z/y/x，特征float32 `[V,12]`。保存source↔voxel↔backbone映射；由位置/体素坐标产生的任何内存布局偏移必须可逆，不能改变物理p。

## 5. 连通组件与特征关联

在原始有效实例mask上计算4连通组件。每个raw支撑点只从同组件主干点中找最多3个10mm内最近邻；1个、2个、3个均可有效。非零距离用1/d归一化；零距离直接使用同点特征；同分依source ID。

找不到主干邻居时局部特征置零、feature_valid=false，原始RGB、几何和normal仍保留。投影MLP之后按mask再次清零，避免bias伪造有效特征。不要因第三个邻居超界而放弃前两个。

canonical query组件来自原像素。历史动作以原记录p为中心，在目标实例raw支持内找最近且距离≤5mm的点作组件锚点；同分取最小source ID。锚点不修改p/n。无几何锚点记 `pose_geometry_invalid`；有锚点但Fq无10mm主干邻居时，以Fq=0、query_feature_valid=false继续评分。

诊断保存同组件最近距离d1与第三近距离d3的p50/p95/max、最大最近覆盖半径、10mm内有效关联率、无主干组件比例及不足3邻居比例。不存在的d3不补零。组件约束不等同完整薄壁识别，缺失比例高时先定位数据/采样原因，再评估显式主干预算。

## 6. 输出与验收

输出观测容器至少保留raw XYZ/RGB/source ID/component ID/validity、mu_vis、K/g及身份、主干坐标与12D特征、各映射、normal图和query表；张量padding不能改变计数、均值或候选身份。局部分区与lip摘要由[模型设计](MODEL_DESIGN.md)在每个精确p/n上读取raw支持。

可立即运行的参考为：

```bash
python docs/instance_routes/reference/sampling_contract.py selftest
python docs/instance_routes/reference/contact_boundary_contract.py self-test
```

完整生产预处理验收还需：原支持未被平滑/抽样覆盖；法向跨孔洞/实例/跳变拒绝；候选前缀和回填稳定且不重复；体素代表与FPS破同分固定；1/2/3邻居插值正确；有raw支持但缺Fq的动作仍可评分；无合法支持有原因；dtype改变单独验证物理边界。分项计时包含法向、候选回填、体素/FPS、组件及关联，不能只报告网络前向。

已实现的CPU单元测试 `tests/test_instance_route_a.py` 覆盖确定性体素/FPS、同组件邻居与并列、缺Fq仍用raw旁路、原始lip摘要顺序和空/非法支持；共同数据测试另覆盖organized法向/候选接线。模型测试用注入backbone，不能替代真实SpConv和真实相机资产验收。生产读入保留 `instance_mask`（含缺深度位置）与 `mask`（有效深度）及独立RGB/normal有效性，避免混淆bbox、原始支持和候选。
