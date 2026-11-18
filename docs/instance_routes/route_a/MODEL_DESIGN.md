# A 模型设计：InstanceSuction3D

模型输入为单个目标实例的可见RGB-D，输出指定接触中心p和外向轴n的物理及任务分数。完整定义来源为 [ROUTE_A.md](../ROUTE_A.md)、[机器可读设计](../reference/route_a.json)和[方法架构图](../METHOD_ARCHITECTURE_A.md)。实现位于 `models/instance_route_a.py`，确定性几何位于 `models/instance_route_a_geometry.py`。13项CPU测试通过，其中模型前后向使用显式注入的测试backbone；真实SpConv/GPU运行、完整训练和时延尚未验证。

## 1. 模块接口与边界

已实现单实例接口：

```python
from models.instance_route_a import InstanceSuction3D

model = InstanceSuction3D(backbone_root="/data/backbones/graspness_modern")
encoded = model.encode_instance(observation)
scores = model.score_at_pose(encoded, p_cam_m, n_out_cam, query_batch_index=None)
```

`encode_instance`每个实例计算一次稀疏特征与对象上下文，并保留raw支持及必要索引。`score_at_pose`接收任意合法的精确p/n，返回逐query的q_inst、seal、wrench、ell、task_logit、几何/主干支持状态与query身份。query可分块，但分块不应改变支持定义和评分结果。

默认 `query_chunk_size=32`；单实例 `query_batch_index` 省略或全0，其他值报错。返回 `points` 为原始中心，`normal_out` 为保留方向符号的单位轴；`valid/pose_geometry_valid` 与 `query_feature_valid` 独立。`log_scale` 同义于ell，`log_variance=log(2)+2*ell`，invalid行只是数值占位。真实构造必须有已校验 `backbone_root`；测试注入backbone是显式调用选择，CLI不会自动换成CPU替代网络。

canonical使用共同候选的p/n；历史规划动作直接调用相同接口。几何锚点只关联组件及支持，不移动记录中心或替换有向轴。网络不读取完整CAD、teacher中心或物理真值，不调用旧instance_fusion、q0或学习轴头。

## 2. 共享稀疏编码

消费[预处理](PREPROCESSING.md)输出的最多2048主干点、2mm体素坐标和12维特征。`models/backbones/spconv_resunet14.py` 已增加可选 `in_channels`，默认仍为8以保留旧E5行为；A显式传12。保留骨干512维输出及512→256投影，恢复每个真实主干点 `F[N,256]`；旧8通道checkpoint不能用改键或补零方式冒充新架构。

真实主干点做masked mean/max，拼成512维，再经 `512→128→64` 得到 `G[64]`。padding不参与池化。F通过同组件最多3邻居、10mm门槛关联到query和raw支持；缺失保留mask及原始RGB/几何，详见预处理。

新A接触、池化、融合和头部MLP隐藏层使用带bias线性层和ReLU，不用dropout；保留的backbone 512→256投影沿用原wrapper中的LayerNorm/ReLU。外部 `resnet.py` / `backbone_resunet14.py` 保留既有固定Git blob SHA-1源码核验，避免加载偏离固定骨干的源码。它不是本轮新增的数据校验方式；被移除的数据/标签/运行SHA-256不以另一种摘要替代。

默认 `singleton_bn_policy=running_stats`：仅在训练模式某一BatchNorm1d输入每通道值数≤1时，改用该层已有running mean/variance，避免单活动体素的BN异常；不更新这次的running统计或batch计数。其余输入沿用普通BN前向及统计更新，不把源预训练全部BN冻结。策略写入架构身份，`error`可作为显式替代配置。微调期按训练器规定冻结全部BN running stats及affine，和这个退化输入保护是两件事；单例保护不证明真实SpConv已验证。

## 3. 精确姿态下的三区域

对与query同实例、同组件的raw点x计算：

```text
d = x - p
h = dot(d, n)
d_t = d - h*n
rho = norm(d_t)
```

接触窗口为 `abs(h)<=0.040m`，只用于网络观测，不能截断teacher。三个区域固定：

| 区域 | 径向范围 | 区域FPS上限 | 编码输出 |
|---|---|---:|---:|
| core | 0≤rho<10mm | 32 | 64 |
| inner | 10≤rho<18mm | 32 | 64 |
| lip | 18≤rho≤22mm | 64 | 64 |

每区先按source ID唯一化、记录截断前点数，再对真实XYZ做确定性FPS；首点、距离并列均遵循最小source ID规则。不足时mask padding。三个区域合计192维。

将raw点关联特征经 `256→64` 投影，再拼RGB3及以下8项，得到75维逐点输入：

```text
rho/r, h/.040, n_obs·n, (d_t·g)/r, n_obs·g,
feature_valid, normal_valid, rgb_valid
```

r是该区外半径。三区域共享逐点MLP `75→64→64`，其后各区做masked mean/max，拼 `log1p(raw_count)`及nonempty得到130维，再经 `130→64`。空区输出零。无效法向点积、无效RGB和投影后的缺失特征显式置零；padding和bias不能制造观测。

网络不接收人为切平面坐标x/y。上述标量对内部切平面基不敏感，但相机系骨干与重力条件并不因此获得全模型严格旋转不变性。网络窗口仅覆盖±40mm；原Seal self无高度上限，target包络可达150mm，所以q_inst仍是学习代理。

## 4. 区域FPS前的8维原始lip摘要

使用全部同实例/同组件、source ID去重后的raw lip支撑，径向18–22mm且|h|≤40mm，不能从64点截断后的lip重算。对M个点先取 `h_tilde=h-mean(h)`，计算：

```text
RMS = sqrt(mean(h_tilde**2))
Ak = abs(mean(h_tilde * exp(1j*k*theta))), k = 1, 2, 3
```

theta只由任意右手正交切平面基供内部计算，角序列和基不进入MLP。等点权重描述当前观测采样，缺测和密度仍会改变摘要。中心化避免常数高度在缺角支持上产生伪谐波；绝对高度由逐点h保留。

按[边界合同](../BOUNDARY_REVIEW.md)，循环排序角度的间隙严格大于1e-6rad时计为独立角向支撑，非空至少记1。独立支撑≥7才有 `harmonic_valid=true`；否则A1/A2/A3置零但保留非空RMS。最大间隙使用原始未合并角度，包括跨2π接缝；单射线归一最大间隙为1，空集全部摘要为0。

固定8维顺序为：

```text
log1p(M), max_circular_gap/(2*pi), lip_nonempty,
RMS/.040, A1/.040, A2/.040, A3/.040, harmonic_valid
```

前三阶幅值可区分部分同高度多重集但角向排列不同的表面，保留低阶结构且对整体换基旋转不变；它丢掉相位和高阶信息，不能完整恢复环形表面或保证密封。摘要自身无可学习参数，接入融合和Seal首层增加1920个权重，不能称整个改动零增参。

## 5. 531维融合及独立头部

| 融合分量 | 维数 |
|---|---:|
| Fq | 256 |
| G | 64 |
| core/inner/lip | 192 |
| 原始lip摘要 | 8 |
| `(p-mu_vis)/.1, n, g, n·g` | 10 |
| query_feature_valid | 1 |
| 合计 | **531** |

融合为 `531→256→256`，输出h。valid只计一次。各头的输入不同：

| 头 | 输入与层宽 | 输出 |
|---|---|---|
| q_inst | h，256→128→1 | sigmoid，实例物理质量代理 |
| Seal | 三区域192+lip8，**200→128→1** | sigmoid，S20 |
| Wrench | h256+G64+Δp3+n3+g3，**329→128→1** | sigmoid，W20；Δp按0.1m归一化 |
| ell | h，256→128→1 | log b；初值log(.1)，前向限制[-6,1] |
| task | h256+n3+n·g1，**260→128→1** | 原始logit，BCEWithLogits监督 |

q_inst不是预测Seal和Wrench的乘积；没有独立碰撞分类头或q_sw头。目标自碰通过Q_inst监督q。ell是Laplace残差尺度的对数，不是log方差或任务成功率。精确loss与参数冻结见[训练方案](TRAINING_PLAN.md)。

## 6. 空支持、历史动作及实现验收

| 情况 | 固定行为 |
|---|---|
| 区域空/主干padding | 输出零并保留mask/计数；不构造重复观测 |
| raw点缺主干关联 | 局部F置零、feature_valid=false；RGB/几何仍参与 |
| 历史query有合法几何锚点但Fq缺失 | Fq置零、query_feature_valid=false，继续评分 |
| 历史query无≤5mm同实例raw锚点 | pose_geometry_invalid，记录原因，不伪造任务失败 |
| 源标签未知 | 不通过特征mask推断label_valid或质量零 |
| 历史轴与观测法向不同 | 采用记录轴重建区域并评分 |

模型级验收包括：所有维数与JSON一致；五头读取正确输入；同一encoded下改变轴会重建接触条件；raw-lip摘要在FPS前；换基幅值/最大间隙在数值容差内不变；恒高缺测支持的Ak为零；低于7方向支撑valid=false；padding/缺特征不会污染池化；空区域和极端点数输出有限；同查询分块前后结果一致。

`parameter_groups()` 导出互斥且覆盖全部参数的encoder/decoder/contact/heads：encoder/decoder继承稀疏wrapper分组，G投影属于decoder；局部特征投影、逐点MLP、区域投影和融合属于contact；五个输出头属于heads。架构、输入、摘要、singleton BN和质量身份进入checkpoint，跨A/B或旧8通道E5不能resume。

`python -m pytest -q tests/test_instance_route_a.py` 当前通过13项CPU测试，覆盖体素/FPS并列、1/2/3邻居和组件边界、分区/谐波、531/200/329/260维、真实A接触/头部反向、任意轴/chunk、缺Fq旁路、raw-lip在FPS前、空/非有限输入及singleton BN。网络测试显式注入测试backbone；wrapper通道测试mock外部加载，验证默认8与A传12，不代表真正执行SpConv。真实稀疏骨干/GPU、同姿态真实teacher和完整训练仍须验收。[阶段计划](README.md)
