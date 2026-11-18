# A 数据准备：实例、历史动作与同姿态物理标签

本文定义进入 A 预处理和训练前应形成的数据资产。依据为[共同数据合同](../DATA_AND_LABELS.md)、[A v3 设计](../ROUTE_A.md)和[设计 JSON](../reference/route_a.json)。数据 builder 与实例 teacher 尚待接线；现有参考命令只能检查文件结构与数学关系。

## 1. 输入资产与固定划分

源域输入包括 SuctionNet 配准 RGB-D、GT 实例掩码、内参、相机/场景/对象位姿、完整对象 `dense_point_clouds`、原 R20 Seal 候选及冻结物理配置。完整对象几何和原候选均值仅供 teacher/评价，模型只接收目标实例的当前可见观测。

先按 scene 划分，再展开相机视角与实例：

| 集合 | scene | frame/view |
|---|---|---|
| train | 0–99 且 scene%10!=9，共90场景 | 0–255，按固定 sampler 取样 |
| val | 9、19、…、99，共10场景 | 0、8、…、248，每场景32个 |
| test / seen | 100–129 | 每场景256个 |
| test / similar | 130–159 | 每场景256个 |
| test / novel | 160–189 | 每场景256个 |

manifest 的 `split` 只写 `train/val/test`；测试另存 `test_subset=seen|similar|novel`。同帧实例不跨 split，这也不等于 CAD 身份互斥划分。源训练有效原始支持不足100点的实例记录 `insufficient_support`；验证和测试保留其身份与无输出结果，计入主指标分母。

目标域继承现有动作清单：

| split | 全部 | 成功 | 失败 |
|---|---:|---:|---:|
| train | 1676 | 1051 | 625 |
| val | 399 | 209 | 190 |
| test | 87 | 65 | 22 |

这些是既有汇总计数，不代表新预处理后可用条数；新链路必须另报原数、合法几何支持数和排除原因。现有数据使用 `pose_source=planned_pipeline_contact`、`actual_execution_verified=false`。关联的不同深度版本、裁剪、相邻帧及同一摆放/session 继承原 group；不重新随机拆动作。来源及未公开明细范围见[共同合同](../DATA_AND_LABELS.md)。

## 2. 实例与动作文件

一行实例 manifest 对应一次观测中的一个目标实例；同观测多个实例可以共用资产，但必须同 group/split。

| 文件/字段 | 准备内容与校验 |
|---|---|
| 实例 manifest | `schema_version=instance_observation_v2`、sample/instance/capture/group ID、split、source_kind |
| 观测资产 | RGB、米制轴向 depth、实例 mask、K、完整 gravity_cam 及有效性；RGB/深度在同一校正网格 |
| 身份 | `observation_sha256`、`calibration_sha256`、normal/sampler spec SHA-256；校验真实字节与清单一致 |
| query 索引 | `query_cache`；source ID、原像素、格号、轮次、query_id、精确 p/n 与 query_valid |
| label 索引 | 源域 `label_cache`；目标历史动作没有物理真值时为空，不能补零 |
| 历史动作 | attempt/sample/group ID、原相机系 p/n、outcome、label_valid、pose_source、actual_execution_verified |
| 支持和变换 | `pose_support_valid`、排除原因、`pose_transform_sha256`；与 outcome 有效性分开 |

相机坐标采用 x向右、y向下、z向前；深度为相机 z 轴距离。使用更新后的 K 反投影，长度单位为米。历史姿态用原标定变换到相机系，以 float64 读取并保留有向轴；最近点只可验证支持，不能吸附记录中心。当前重力是负传送带平面法向近似，来源标记随样本保留。

建议数据 release 内按 observations、manifests、queries、labels、attempts 和 reports 分类保存，具体根路径由运行配置给出。所有路径、哈希和原始资产均需实际核验；本目录不虚构本地数据根或完成清单。

## 3. 按实际需要构建缓存

1. 核对源资产及目标原始记录，固定 split/group/标定和规范化资产 hash；输出接受与排除清单。
2. 按[预处理方案](PREPROCESSING.md)生成 v3 法向和完整最多256 query序列。源训练/回放取前128；验证、测试使用固定完整池；历史动作另存精确 p/n。
3. 固定训练实例抽样计划及其 hash。对实际会访问的不同实例/query生成标签；重复 draw复用相同缓存，不要求提前对90场景全部帧、全部实例重算256个点。
4. 在约64–128个覆盖不同实例尺寸及 dense点数的真实样本上剖析 teacher成本；记录加载时间、Seal秒/query、self-envelope时间、秒/实例、缓存大小及命中率。
5. 同姿态与 self拆分验收通过后，按需扩展缓存，校验完成后再发布为可消费的release；中断恢复时拒绝身份不符的旧项。

训练128、可选64必须是同一完整池前缀。新normal/sampler改变精确姿态后，质量合同名相同也不能复用旧pose标签。

## 4. 保留目标自碰的 teacher

每个源标签对应网络实际评分的精确 `(p_cam_m,n_out_cam)`：

```text
raw_seal, C_seal_self = seal_at_pose(full_dense_object, p_object, n_object)
S = frozen_normalization(raw_seal)
W = frozen wrench_scores_dual(...)["wrench_eval"]
C_target_envelope = frozen target-only envelope query
C_self = C_seal_self OR C_target_envelope
q_sw = S * W
q_inst = q_sw * (NOT C_self)
```

这是待实现的运算顺序；p_object/n_object由精确相机系p/n按已校验变换得到，raw_seal须经冻结归一化后才成为S20标签。旧 `evaluate_poses`仍计算混合碰撞，可核对 S/W，但不能直接返回新 self分项。冻结约定包括：

- Seal保留72个开区间角度分箱和18–22mm唇口；Seal环柱没有网络式轴向截断。
- `C_seal_self=any(rho<20mm AND h>5mm)`，没有 h上限。
- target-only包络沿用3mm障碍体素、self_height_threshold=4mm；cup为(-2,25,22)mm、mount为(25,70,28)mm、approach为(70,150,28)mm，元组顺序为(zmin,zmax,r)。
- 包络先执行原自身点总条件 `h>4mm && h<=150mm && rho<=22mm`，再执行每段原边界条件；不能只复制三段数组。
- W使用冻结原 R20 Seal候选位置均值、1kg基准质量及原评价阈值，不将可见点均值当成真实质心。

目标对象几何、变换及包络索引尽量复用；W可批量处理，Seal仍需保持完整dense参与。移除外部查询不自动意味着 teacher已足够快。网络的±40mm接触窗口不改变任何上述物理定义。[精确拆分规则](../DATA_AND_LABELS.md)

## 5. 标签 schema 与缓存身份

标签 NPZ 固定 `schema_version=suction20.instance_inst20.v2`、`quality_contract=gt_INST20_pose_v2`。

| 字段 | shape / 约束 |
|---|---|
| `points`, `normal_out` | [K,3]，默认float64；有效行有限，轴单位范数误差≤1e-4 |
| `seal`, `wrench`, `q_sw`, `q_inst` | [K]；有效行位于[0,1] |
| `collision_self_seal`, `collision_self_envelope`, `collision_self` | [K] bool；self为两项OR；true表示碰撞 |
| `label_valid` | [K] bool；要求p/n、S/W及两个self分项都已求值 |

乘积一致性使用 `atol=1e-6, rtol=1e-5`。Inf禁止；mask=false行可存NaN，布尔false仅是占位。部分标签未知时保留诊断，不补成负例；全false结构文件须标 `has_supervision=false`，不能计入训练有效实例。

缓存身份绑定：观测/实例、完整对象资产、物理配置、teacher代码commit、normal-spec、sampler-spec、精确p/n字节、split清单、质量合同与单位。几何正确性基线从原深度/内参构造float64，之后才转网络FP32；已经量化的FP32 XYZ升精度不能恢复原数值。光度增强不改变标签，任何p/n变更必须重新匹配缓存身份。

## 6. 可运行检查与完成条件

在仓库根目录对真实准备好的文件执行（尖括号为需要替换的路径）：

```text
python docs/instance_routes/reference/geometry_contract.py check-manifest <manifest.jsonl>
python docs/instance_routes/reference/revision_contract.py check-labels <labels.npz>
```

这两个入口已存在。manifest检查核对声明关系，标签检查核对schema与数值关系；真实资产hash、隐藏表面、teacher一致性需新增链路核验。builder/teacher拟新增文件见[总览](README.md)，尚无可直接启动的新数据生成命令。

完成本阶段需有：固定清单及排除账本；scene/group/hash不跨split；标定与原姿态回读报告；真实同p/n的S/W对齐；self两分项与原目标部分对齐；独立外部项与self合并可重建旧总C；错误法向/姿态/teacher身份的缓存被拒绝；正式训练所需缓存预算与实测成本报告。
