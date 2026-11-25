# 模型化方案 2 数据处理实施文档

**方案：Suction20 V2-Belt70 / 协议：modeling_v2_data_protocol_v1 / 日期：2026-11-25**

本文把用户提供的[原始规格](source/suction20_v2_belt70_reproducible_spec.html)细化为数据工程实施依据。数据负责人可按本文准备资产、生成清单、完成预检；开发人员可按给定的数据合同和算法实现后续转换，并逐项验收。文中所有路径命令都从仓库根目录执行。

**状态必须区分：本分支已提供清单/文件预检工具与几何参考实现；生产级多源 Adapter、补洞构建器、新 Belt70 Teacher、方案 2 训练读取器仍需按第 17 节实现。** 不把设计中的命令写成已存在的入口，也不把合成检查通过称为真实数据转换完成。原始 SuctionNet、ClearPose、现场偏好数据和训练服务器不在本次工作区内。

## 0. 执行顺序与完成定义

| 阶段 | 执行内容 | 必须留下的产物 | 放行条件 |
|---|---|---|---|
| D0 | 固定代码、协议与数据根 | Git revision、协议 JSON、源版本和路径记录 | 路径和单位来源明确，不覆盖旧发布 |
| D1 | 原始资产预检 | `audit.csv`、`audit.json`、选中帧清单 | 所选 P0 图像与元数据无缺失；仅说明文件层面通过 |
| D2 | 固定分组切分与抽帧 | 场景 split、相机帧清单、排除账本 | 同物理场景、双相机、重复采集不跨 split |
| D3 | 标准化坐标、实例、平面与 GT | frame / objects / calibration 记录 | 单位、位姿、实例行号、重投影和支撑面通过 |
| D4 | 原始观测与监督分流 | `observations/` 和 `supervision/` | 网络输入没有 GT 深度、CAD、物体位姿或质量标签 |
| D5 | 预填、背景裁剪、候选池 | fill、12 通道、候选 bank 与有效掩码 | Zero-Touch、射线和平面约束、裁剪覆盖通过 |
| D6 | R20 / Belt70 重算 | S/W、C70 分项、标签姿态及来源 | 无 C150 冒充 C70、无 unknown 冒充 free |
| D7 | 小规模 pilot | 数值报告、可视化、耗时/容量估计 | 第 16 节数据验收通过后才扩量 |
| D8 | 分片构建与发布 | 完整 manifest、summary、拒绝原因、READY | 读取器回读、split 检查和覆盖账本通过 |
| D9 | 现场偏好数据准备 | A 原动作、B 像素、分组、冻结表面版本 | A 不漂移，B 空洞可追踪，偏好不冒充物理成功 |

顺序是 `D0→D1→D2→D3→D4→D5→D6→D7→D8`。D9 可以先整理二维标注，必须在 Stage 1 几何模型冻结后生成最终 B 三维邻域。

## 1. 本轮固定的范围与规格修订

### 1.1 数据源分工

P0 使用 **SuctionNet + ClearPose**。SuctionNet主要提供普通物体几何、遮挡、R20 基础资产和合成缺深；ClearPose提供真实透明缺深与渲染表面监督。55% / 45% 是训练时按**实例 draw**的抽样比例，不是硬盘帧数比例，也不保证两库每个样本被等次数读取。

P1 再接入 MetaGraspNetV2 的仿真子集和压扁瓶合成。现场约 400+ 帧人工偏好属于 Stage 2；该数量来自原规格，尚未核验实际标注文件。先前讨论的 E5 + TransCG 微调是另一条实验线，不自动纳入本 P0 清单。

### 1.2 与原 HTML 的明确差异

| 项目 | 本文采用的实施决定 | 原因与依据 |
|---|---|---|
| 局部 AABB 外扩 | **80 mm**，并先包含允许的补洞修正端点 | 任意法向圆柱的最远半径为 `sqrt(70²+28²)=75.3923 mm`；75 mm 不足以严格覆盖；只改80 mm也不能忽略后续±40 mm修正 |
| SuctionNet 帧预算 | 每场景每相机16视角，双相机合计32 camera-frames | 100×32×2=6400，不是3200；本轮锁100×16×2=3200 |
| ClearPose 帧预算 | 官方24个训练scene，每scene最多96帧，总预算2304 | 每scene只取10–20帧无法得到约2200；保持官方测试分组不进入训练 |
| 开发集 | 新方案专用90/10 SuctionNet scene切分，ClearPose19/5 scene切分 | 历史 A/B 是100个train场景+test subset选模；新schema独立实现，不改写历史实验口径 |
| ClearPose 真值 | 人工CAD与位姿渲染得到的表面监督，显式记录可信区域 | 官方说明模型手工建立，`depth_true`为渲染物体深度覆盖raw；不称其为独立实测真值 [S2] |
| MGN 资产规模 | 按实际版本枚举；论文为82物体，不设置80–120瓶罐硬配额 | 原规格“989物体”与原论文不符 [S4] |
| MGN 版本与单位 | 锁定修正错位后的 `MetaGraspNetV2_V2`；scene深度和translation由cm转m | 官方2025-06-04更新与读取代码 [S3][S5] |
| 碰撞 | 重算cup/mount/plane分项；不 OR 旧无上界Seal自碰布尔值 | 旧C150和旧self结果不能可靠倒推出C70 |
| 质量标签姿态 | P0明确为 `canonical_gt`，另保存label pose；实测预测姿态另行重评 | 随网络变化的p_ref/n_hat不等于离线标注姿态，不能冒称exact-pose标签 |

这些是本次针对可实施性作出的 v1 决定，不是新数据上的消融结论。修改任何影响标签/几何的值都要增加协议版本并建立新发布目录。

## 2. 现在就能执行的准备步骤

### 2.1 进入新开发分支并固定环境

```bash
git fetch origin
git switch modeling/plan2-belt70
git pull --ff-only
git rev-parse HEAD

python --version
python scripts/modeling_v2_data_plan.py --help
python -m pip install numpy
python docs/modeling_v2/reference/geometry_contract.py --self-test
```

清单工具仅需 Python 标准库；几何参考需 NumPy，上述安装应在计划使用的 Python 环境中执行。需要运行仓库既有物理函数时，再安装 `python -m pip install -e '.[dev]'`。GPU与SpConv不是文件预检的前提。实际生产环境依赖以成功pilot记录为准，记录Python/NumPy/SciPy/渲染器版本和Git revision。

### 2.2 生成不依赖原始数据的计划

```bash
python scripts/modeling_v2_data_plan.py plan \
  --output ./artifacts/model2-plan-v1
```

输出目录必须是新目录。工具应拒绝覆盖已有结果。计划包含3200条SuctionNet相机帧、24个ClearPose场景预算和保留测试场景；它不是训练manifest，也没有证明文件存在。ClearPose的具体帧ID在读取实际文件目录后确定。

| 实际文件 | 生成时机 |
|---|---|
| `plan.json`、`config_snapshot.json`、`scenes_suctionnet.csv`、`scenes_clearpose.csv` | plan和audit都生成 |
| `frames_suctionnet.jsonl` | plan生成；audit仅在传入SuctionNet根时生成 |
| `frames_clearpose.jsonl`、`selection_clearpose.csv` | audit且传入ClearPose根 |
| `audit.json`、`audit.csv` | audit生成 |

帧记录声明`schema=plan_v1,training_ready=false`。audit状态只有`file_preflight_passed`或`incomplete_assets`，plan没有audit状态。后文的`source_asset_audit.csv`是补充语义审核的目标产物，不是该工具已完成的内容。

### 2.3 指向真实数据并做文件预检

把以下路径换为实际已解压的目录；不根据历史聊天推断服务器挂载已存在。

```bash
SN_ROOT=/data/raw/SuctionNet
CP_ROOT=/data/raw/ClearPose
MODEL2_RELEASE=/data/derived/suction20-model2/p0-v1

python scripts/modeling_v2_data_plan.py audit \
  --suctionnet-root "$SN_ROOT" \
  --clearpose-root "$CP_ROOT" \
  --output "$MODEL2_RELEASE/preflight-01"
```

可只传一个来源，另一来源后续单独预检。退出码0表示本次所选资产路径存在且大小满足文件预检；退出码2表示缺失/空文件或输入不完整。读报告中的明细并补齐原始资产后换一个结果目录重跑。**文件存在不代表K、位姿、GT和CAD正确；D3验收仍必须执行。** 工具不下载数据，不展开大型归档，也不读取保留test帧。

### 2.4 现有代码的可执行读入检查

例如确认SuctionNet的源数据读取链路，可使用旧builder生成少量**无标签的旧格式观测**，与新Adapter输出做有界数值比较：

```bash
python scripts/build_instance_dataset.py \
  --dataset-root "$SN_ROOT" \
  --output "$MODEL2_RELEASE/legacy-readback-pilot" \
  --split train --camera realsense \
  --scenes 0 --frames 0,128 \
  --without-labels --workers 1
```

读取stdout JSON的`manifest`字段获得真实路径；显式scene/frame选择会创建范围子目录。该命令不会生成ClearPose数据、方案2的dev、补洞点或Belt70标签。它只是现成读入通路的检查，不能把输出直接交给尚未实现的新训练器。

**到此可立即运行的内容已经列全。D3–D9的生产转换需要第17节模块实现后接通；本文不提供不存在的 `--belt70`、`--clearpose` 或 `--fill-holes` 参数。**

## 3. 资产组织、下载范围与来源记录

### 3.1 目录约定

原始目录只读，衍生数据独立写入。推荐逻辑布局如下；不同磁盘挂载可通过路径配置替换。

```text
raw/SuctionNet/
raw/ClearPose/
raw/MetaGraspNetV2_V2/                 # P1
raw/target_preferences/               # 现场原始记录与二维点击
derived/suction20-model2/p0-v1/
  preflight-01/                       # 本分支工具实际输出
  inventory/                          # 下面各项由生产实现生成
  splits/
  objects/
  observations/
  supervision/
  instances/
  labels/
  manifests/
  reports/
  release.json
  READY
```

`release.json`记录：`dataset_release`、`code_revision`、`protocol_version`、来源绝对根路径、相对资产路径、文件大小、schema版本、精度、物理小型配置、构建时间、接受/拒绝计数。遵守仓库[AGENTS.md](../../AGENTS.md)：不新增SHA-256或替代全文件内容摘要；这些元数据也不被解释为内容完整性证明。

### 3.2 SuctionNet需要哪些文件

从官方[SuctionNet页面][S1]取得对应资产；RGB-D使用与GraspNet相同的场景文件。P0只需选定训练场景/视角，但**物体完整dense几何不可由局部观测替代**。

| 文件或目录 | 用途与关键注意 |
|---|---|
| `scenes/scene_XXXX/{realsense,kinect}/{rgb,depth,label,meta}/FFFF.*` | 单相机帧，RGB/depth/label网格一致；meta读取factor_depth与K |
| `camK.npy`、`camera_poses.npy`、`cam0_wrt_table.npy` | 预检均要求存在；K与逐帧meta交叉检查；`T_table_cam=cam0_wrt_table @ camera_poses[frame]` |
| `object_id_list.txt`、meta中的对象pose | 对象身份与0-based实例行号；重复类别必须有独立实例映射 |
| `models/000..087/` | 网格、可视化与必要的GT投影 |
| `dense_point_clouds/000..087.npz` | 已有R20 Seal求值所需完整物体采样，保留原资产 |
| 旧R20发布的`seal_r20/poisson/`和`config_frozen.json` | 核验后复用候选点、法向、原始Seal及中心定义；不能直接复用总碰撞 |

官方网站的原碰撞目录名为`suction_collision_label`，部分上游API历史实现使用不同路径。新流程不依赖该旧二值标签生成C70。label像素0是背景，1..88是类别编码；`object_id`与`instance_row`不可混用 [S1]。

### 3.3 ClearPose需要哪些文件

依据[官方仓库][S2]准备`set*/scene*/metadata.mat`、每帧`*-color.png`、`*-depth.png`、`*-depth_true.png`、`*-label.png`与对应模型。`*-normal_true.png`可作诊断，生产GT法向建议从经过检查的几何独立计算。

`metadata.mat`是整scene容器，按帧编号字符串取记录。检查实际结构的字段名：`cls_indexes`、`camera_intrinsics`、`factor_depth`、`poses`、`rotation_translation_matrix`等。使用字段名和shape检查，不照抄位置解包。公开训练分支出现的逐帧`*-meta.mat`或`*-tableplane_depth.png`是预处理结果，不属于必须从官方包中找到的原始文件 [S2][S6]。

单帧探索示例只读，不假定所有MAT打包版本相同：

```python
from pathlib import Path
from scipy.io import loadmat

scene = Path('/data/raw/ClearPose/set1/scene1')
container = loadmat(scene / 'metadata.mat')
frame_keys = sorted(k for k in container if k.isdigit())
record = container[frame_keys[0]][0, 0]
print(frame_keys[0], record.dtype.names)
```

若字段名/shape与预期不同，记录该资产版本并写显式适配映射；禁止猜测转换后直接批量生成标签。

### 3.4 数据负责人先填写的表

`source_asset_audit.csv`至少含：`source,release,scene,frame,camera,object_id,instance_row,split,path,size_bytes,depth_unit,mesh_unit,pose_translation_unit,K_source,pose_convention,plane_source,gravity_source,gt_provenance,status,reject_reason`。预检工具输出的是其中的路径/大小层；D3后补齐语义层。

对象表使用命名空间，例如`suctionnet:000`、`clearpose:bottle_1`，不要让两个来源的object 1指向同一个缓存目录。对象表同时记录模型物理尺寸、dense采样版本、候选顺序、candidate_id定义、是否参与目标训练及是否作为障碍物。

## 4. 固定切分与抽帧：先分组，后展开实例

### 4.1 SuctionNet v1

| split | scene ID | 每相机视角 | 相机 | camera-frames |
|---|---|---|---|---:|
| train | 0–89 | `0,16,32,...,240`，16个 | realsense+kinect | 2880 |
| dev | 90–99 | 同上 | 两相机 | 320 |
| reserved_test | 100–189 | P0不展开或选模 | 保留官方分组 | 不计入本轮构建预算 |

这是**新方案的新版本切分**。历史A/B的`train=0..99`、`selection来自test`不被重命名为该dev。新方案如果从已看过90–99的E5/A/B权重启动，要记录`pretraining_exposure=true`，不能再把这10个scene宣称为完全未见数据；从头训练按本协议则无此历史暴露问题。

同scene的两相机、所有视角、所有目标、所有增强都继承`group_id=suctionnet/scene_XXXX`。scene互斥并不等于CAD身份互斥；官方seen/similar/novel测试仍分开报告。

预检工具当前字段名是`split_group=suctionnet:scene_XXXX`（ClearPose同理为`clearpose:set1/scene1`）。生产Adapter显式转换为上述group_id，保留原split_group来源；不能把plan行直接当训练记录。

### 4.2 ClearPose v1

只从官方训练部分划出内部dev。官方训练为set1/scene1–4与set4–7/scene1–5，共24个scene [S2]。

| split | scene名单规则 | 数量 | 每scene上限 | 总预算 |
|---|---|---:|---:|---:|
| dev | set1/scene4、set4/scene5、set5/scene5、set6/scene5、set7/scene5 | 5 | 96 | 480 |
| train | 上述24个scene扣除5个dev | 19 | 96 | 1824 |
| reserved_test | 官方其余测试分组 | 单列 | 不进入P0选模 | 不计入2304 |

按该scene实际`*-depth.png`文件的frame ID排序，令`k=min(96,N)`，选索引`floor((i+0.5)*N/k)`，`i=0..k-1`。不要假设帧编号连续，不复制短序列凑数。审计报告记录实际数与预算差额；缺对应GT/label/RGB的选中帧必须报错，不静默换另一帧凑齐。

如果资产说明显示多个scene来自同一摆放/setup，把这些scene连成一个group后整体分配；分配前发现的关联以防泄漏为优先，此时修改并冻结新split版本，不在训练中临时改名单。

### 4.3 总量口径与去重

两库合计上限为**5504 camera-frames**，其中train4704、dev800。一个camera-frame有多个实例，最终instance-example数由实际mask和支持审核决定，不能预先写成5504个训练样本。

依次核对`source+scene+camera+frame`、`capture_id`、实际注册观测路径、`group_id`，防止副本跨集合。用已知身份、文件路径与版本关系去重，不引入全文件内容摘要。每个实例、候选、B⁺邻域及增强版本都继承父观测split。

## 5. 坐标、尺度、注册与支撑面

### 5.1 全管线约定

相机坐标为x向右、y向下、z向前；深度是相机Z轴距离，统一使用米。像素`u`为列、`v`为行，原图左上角像素中心记为`(0,0)`，数组访问为`D[v,u]`。`T_a_b`定义为把b系点变换到a系：`p_a=R_a_b p_b+t_a_b`；所有位姿保存4×4矩阵。

反投影采用：

```text
ray_z1 = inv(K) @ [u, v, 1]       # 第三分量为1，不归一化
p_cam = depth_z_m * ray_z1
p_ref = p_fill + delta_z_m * ray_z1
```

若用单位射线乘`z_gt-z_fill`，离轴像素会出错。参考检查包含该反例。resize/crop需要同时更新K和像素坐标；现场RGB点击必须先还原到注册RGB-D原图，不能直接使用界面缩放后的像素。

### 5.2 各来源的单位与位姿确认

| 来源 | 深度 | mesh | pose translation | 必做检查 |
|---|---|---|---|---|
| SuctionNet | 读meta的`factor_depth`，常见1000；`D_m=D/factor` | 官方模型/已有dense按米核验 | 按官方MAT的米制约定核验 | cam0/table组合、逐实例重投影 |
| ClearPose | 读对应frame的`factor_depth` | 不凭扩展名猜单位；先验证bbox尺寸与投影 | 与所选模型和相机记录配套核验 | metadata scene容器、原模型轴、pose方向 |
| MGN-Sim P1 | 官方代码`depth/100` | 官方mesh加载后乘100用于厘米采样；项目米制通常不再缩放mesh | 场景平移`/100`，旋转不缩放 | 锁定新版，确认相机轴转换与instance映射 [S5][S7] |
| 现场 | 读取设备单位与注册配置 | 不要求每只瓶子有CAD | 历史A按记录的工具/相机变换链换算 | 时间戳、K、RGB–Depth网格、工具TCP偏移 |

矩阵检查：形状、有限值、末行、`RᵀR≈I`、`det(R)≈1`；建议旋转误差绝对阈值1e−4。超限则拒绝或按显式修复政策重标版本，不静默正交化。使用实际矩阵求逆，避免把非精确正交存储矩阵直接当成转置逆。

每个来源至少审核20个覆盖不同scene/对象/深度的camera-frames：可视化CAD轮廓对齐、前景实例色、深度差图；相机投影往返误差应在float64下≤1e−6 pixel。实际CAD与图像边缘差不是数值误差，先报告分布；以3 pixel为pilot调查阈值，不把不准CAD强制拟合成“通过”。明显错轴/尺度100倍/实例串行直接阻断。

### 5.3 支撑平面与重力

平面统一为`h(p)=n_b·p+d_b`，`||n_b||=1`，正方向指向物体所在半空间。补洞后要求`h>=0.002 m`。有效原始点可贴近平面；不要为满足补洞阈值把整幅深度抬高2 mm。

SuctionNet从table系确定平面与重力。若`T_world_cam`把camera转到world，平面齐次系数满足`pi_cam=T_world_cam.T @ pi_world`，随后规范化；重力方向按旋转逆变换。

ClearPose不要从“存在相机外参”推断“存在桌面真值”。在全部物体mask之外并外扩排除5 pixel，使用原始有效背景点拟合支撑面。v1建议参数为RANSAC距离3 mm、最多2000假设、至少500个支持点、inlier ratio≥0.70、inlier残差P95≤4 mm；这是本项目的起点，pilot失败要调查原因并新版本调整。跨帧使用相机轨迹汇聚同一scene的平面证据，防止每帧平面晃动。

仅在场景可以合理视为水平支撑时设置`g_cam=-n_b`，并记录`gravity_source=assumed_horizontal_support`，不标记为实测重力。非平面、透明覆盖、背景不足的样本记录`plane_unreliable`；可靠的depth-only监督可隔离归档，但W和涉及平面的free标签设为未知。当前含g_cam且强制plane-clamp的P0主路径不消费这些隔离样本；以后要用需独立depth-only输入路径，不能补默认平面/重力继续前向。

## 6. 标准数据合同：输入与监督分开保存

### 6.1 Frame和对象表

生产schema使用独立名称：`suction20.model2.frame.v1`、`suction20.model2.instance.v1`、`suction20.model2.labels.v1`。预检的plan schema与这三个训练schema不同。

| 字段 | 类型/shape | 定义 |
|---|---|---|
| `source,source_release,frame_id,capture_id,group_id,split` | 非空字符串 | source/capture/group可追溯；split为train/dev/test |
| `camera_id,image_size_hw,K` | string、[2]、float64[3,3] | 注册后观测的相机定义 |
| `depth_raw_m,sensor_valid` | float64[H,W]、bool[H,W] | 原始米制深度及原始有效性；原文件不被改写 |
| `instance_map` | int32[H,W] | 0背景、正值为该frame的实例编码，另表映射object namespace |
| `plane_cam,plane_valid,g_cam,gravity_valid` | [4]、bool、[3]、bool | 不存在时显式未知，不能用默认水平面冒充 |
| `object_rows` | 显式列表 | instance row→object ID→模型/位姿；是Teacher sidecar，网络不可读 |
| `depth_gt_m,depth_gt_valid,gt_provenance` | 监督侧数组与来源 | sensor-observed/synthetic-paired/rendered-CAD等分开统计 |
| `normal_gt,normal_gt_valid` | float64[H,W,3]、bool | 外向单位法向；由可信GT表面计算 |

ClearPose的`depth_gt_valid`不能简单等于`depth_true>0`：要求目标实例映射正确、渲染表面位于该像素的最前可见目标、排除边界不确定带（v1先腐蚀2 pixel）、无已识别遮挡/错位。GT可用性只决定监督有效性；不能用GT去改变网络的原始深度可靠性mask。

### 6.2 观测工作副本与12维输入

合成挖洞只改本次draw的`depth_input_m`副本。保留`sensor_valid`原始身份；`is_raw_valid`表示**本次增强后仍保留的观测深度**。合成删除位置因此可为`sensor_valid=1,is_raw_valid=0`，真实缺深为二者均0。

固定通道顺序：

| 索引 | 内容 | 规则 |
|---|---|---|
| 0:3 | `(p_cam-mu_vis)/0.10` | mu_vis取本次保留的目标原始有效点均值；不含背景或GT |
| 3:6 | `n_obs` | 原始点用观测法向，补洞点用预填面法向；无效为0，另存normal_obs_valid |
| 6:9 | `g_cam` | 单位重力；主质量集需有效 |
| 9 | `is_target` | 目标含合法补洞为1；其它物体/支撑面观测为0 |
| 10 | `is_raw_valid` | 本次有效深度为1；填充值为0 |
| 11 | `hole_dist_norm` | raw点和背景为0；补洞按第7节计算 |

输出`X_in[M,12]`为float32，但投影、平面、Teacher和标签身份比较使用float64。`normal_obs_valid`、`fill_valid`等属于sidecar，不暗中增加第13通道。进入稀疏骨干的体素坐标按2 mm物理网格构造；不把xyz归一化后的数值误当米来量化。

RGB保留用于注册、QA和人工点击，本方案几何骨干不吃RGB。`depth_gt`、`normal_gt`、完整CAD、对象6DoF、质量分数和teacher碰撞几何只能从监督入口读取。

### 6.3 逐候选标签

| 字段 | 约束 |
|---|---|
| `query_id,query_uv,pixel_id` | 一一对齐、唯一；pixel_id=`v*width+u` |
| `candidate_bank_row` | 当前候选在本实例固定bank中的行号；与mesh ID无关 |
| `label_point_cam_m,label_normal_out_cam` | 明确该标签实际评价的姿态；不可写成p_fill |
| `label_pose_source` | P0为`canonical_gt`；以后exact snapshot另版本 |
| `seal_raw,seal,wrench,q_sw` | 已知量finite且[0,1]，`q_sw=seal*wrench` |
| `c_self_cup,c_self_mount,c_ext_cup,c_ext_mount,c_plane,c70,p_free` | 布尔分项；已知时C70为OR，p_free=1−C70 |
| `seal_valid,wrench_valid,quality_valid,collision70_valid` | 独立bool；quality_valid=seal_valid&wrench_valid |
| `normal_gt_valid,hole_gt_valid` | 不与quality_valid强制绑成同一个mask |
| `hole_delta_z_m,hole_delta_in_range` | z_gt−z_fill；超±40mm保留原差值并标超范围，不把截断值当真值 |
| `teacher_spec,physics,geometry_version` | 精确小型配置、来源版本、全场景几何列表 |

未知浮点可存NaN，未知bool可存false占位，但必须由对应valid解释。损失计算先索引有效行再求值；不能用`NaN*0`掩盖未知。padding候选不计入监督、排序或覆盖分母。

## 7. Zero-Touch预填与缺深修正

### 7.1 原始有效性的定义

`sensor_valid = finite(D_raw) & (D_raw>0) & sensor_range_ok`。SuctionNet沿用已审查的0.2–1.2 m范围；ClearPose先按设备/数据元信息明确有效范围，pilot绘制分布后冻结，不能无条件照搬SuctionNet的截断。源文件中的非零但越界深度仍保留在原始资产中，并记录range-invalid原因。

网络使用的`is_raw_valid`在没有合成挖洞时等于`sensor_valid`；增强时还要去掉本次synthetic_hole。目标原始支持点不足100返回`insufficient_support`。评价manifest保留该实例并记录无输出，不从指标分母删除。

**非零穿透回传不会由本版本自动修复。** 透明瓶像素若返回桌面深度但仍满足传感器有效条件，Zero-Touch会保留它。可以利用GT在报告里统计这种错深，但不得用GT修改训练输入mask。后续若扩展修正范围，应另立`surface_reliability`契约和新的训练/推理一致规则。

### 7.2 确定性的局部拟合顺序

以下为**待实现的生产v1算法参数**，写入实际构建配置快照；几何参考工具只验证预填后的约束，不实现曲面拟合。

1. 在目标mask内求8连通分量。一个待填像素只使用同一目标、同一连通分量的原始有效点；不借邻物或桌面当瓶子表面。
2. 先取像素距离最近的最多16个支持点，以其深度中位数为`z_ref`，用`z_ref*(du/fx,dv/fy)`定义局部二维米制距离。局部支持搜索半径30 mm，最多128点，按距离再按pixel_id稳定排序。
3. 取至少12个支持点、覆盖至少3个二维象限时，拟合二次深度曲面。自变量除以30 mm改善条件数，基为`[1,x,y,x²,xy,y²]`，权重为`1/(d²+0.002²)`。用加权最小二乘；要求rank=6、加权设计矩阵条件数≤1e6、深度残差RMS≤3 mm。
4. 拟合预测须处于局部支持深度范围外扩10 mm以内。若拟合不合格，回退到IDW：至少6个支持、至少3象限，使用相同距离和权重。IDW不得迭代使用新填点充当原始支持。
5. 最近原始支持的上述二维物理距离超过30 mm、支持不足、跨越明确的实例/折痕屏障、或者两种方法均失败，设置`fill_valid=false`，保留原因。大洞不能用任意常数或传送带深度冒充瓶子表面。
6. 对预测深度沿本像素射线施加第7.3节平面约束；记录`fill_method=quadratic|idw|invalid`、支持数、拟合残差、最近支持距离、是否被平面截断。

折痕屏障以原始支持邻接深度跳变10 mm为起点，沿同实例邻接路径检查。透明大空洞内部没有可观测折痕时不要宣称知道其表面拓扑；由GT误差、hole_dist和单独大洞分层反映不确定性。

这些数值不是所有透明瓶上的最优结果。pilot若因局部支持不足大量拒绝，应输出失败分布，调整协议版本后重新构建，不能一边训练一边静默放宽阈值。

### 7.3 平面截断与残差

设`a=n_b·ray_z1`，则`h(z)=a*z+d_b`。补洞点约束为`a*z+d_b>=0.002`。顶视常见`a<0`，因此是**深度上界**`z<=(0.002-d_b)/a`；不要按文字“下界”直接写`max(z,z_belt)`。近平行、无合法正深度交点或方向异常时显式标invalid。

顺序为：

```text
z_fill = plane_clamp(local_fit(raw_support))
z_ref  = plane_clamp(z_fill + delta_z)       # 仅合法缺深点，模型残差有±40mm界
z_out[is_raw_valid] = z_input[is_raw_valid]  # 原始有效点逐值不变
```

训练目标保留真实`z_gt-z_fill`；绝对差>40 mm记`hole_delta_in_range=false`。这类像素计入超范围率与整体补洞误差，不把截断后的40 mm当作它的真实标注。

### 7.4 hole_dist与法向

最近有效目标像素选择采用像素欧氏距离，平局按pixel_id。米制距离明确定义为：

`dist_m=z_fill*sqrt(((u-u_valid)/fx)^2+((v-v_valid)/fy)^2)`。

`hole_dist_norm=clip(dist_m/0.030,0,1)`；原始有效点和非目标背景为0。它是基于投影的表面距离代理，不宣称是曲面测地距离。

原始点使用既有`organized_masked3x3_span3_v3`规则：同实例3×3至少5点，差分跨度±3，邻接深度跳变<10 mm，不能跨raw缺深路径或临时改成±1回退。新填点则在同实例`is_raw_valid | fill_valid`曲面上估计法向，允许路径经过合法填点，并标记`normal_source=filled`；若仍沿用raw-only有效掩码，洞中心将永远无法取得法向。所有合法法向朝相机可见侧：`dot(n,-p)>0`。缺少法向的点仍可进骨干，但不能生成需要完整接触姿态的有效候选；不能用GT法向筛选网络候选资格。

## 8. 目标加背景裁剪与保守覆盖

### 8.1 先得到目标范围，再保留背景

保留完整原始frame直到预填结束。有效原始目标点的位置固定；每个可修正点先对`z_fill−0.040`和`z_fill+0.040`分别沿原相机射线执行平面截断，以两端合法正深度反投影得到允许位置端点。对所有这些端点及原始目标点取AABB，再向六方向外扩80 mm。此规则与参考`conservative_crop_bounds`一致。

调用crop参考接口前只选择有效raw或两端均合法的fill目标点；无合法端点的fill不进入可评分池，仍保留在完整mask及失败/覆盖账本中。不能把无效填值塞入几何接口，也不能因此删掉评价分母。

从完整frame保留该AABB内**全部可观测背景**：承载面、其它实例、未列为训练目标的物体。不要从“已经过目标筛选的物体列表”构造碰撞背景。没有观测的邻物背面也不能被标为已知空闲。

crop仅用于网络观测，不用于缩小Teacher的全场景几何。传送带以真实观测点和显式plane元数据保留；如果需要向网络注入解析平面采样，必须另立`plane_synthetic`来源，不能伪标为raw传感器有效点，本v1默认不注入。

### 8.2 为什么需要80 mm和修正端点

轴向z≤70 mm、半径ρ≤28 mm圆柱的任意方向投影不超过75.3923 mm。对一个确定单位轴n，更精确的第i轴界可写为`70*abs(n_i)+28*sqrt(1-n_i²)`（mm）。因此80 mm留有4.6 mm以上余量。

但若目标中心从p_fill移动40 mm、裁剪仍围绕旧中心，80 mm本身不能补偿这个变化。必须使用上面的**允许修正端点并集**，或者另版本实现第二次动态裁剪。对于离轴像素，XYZ位移不是简单的各轴40 mm，必须乘`ray_z1`。

稀疏体素合并时不允许把不同实例的标志平均成0.5。几何坐标grid保持统一，特征聚合策略要记录目标/背景混合体素的比例并提供目标优先的确定规则；接触判断仍使用去体素前带实例身份的点表。实现时评估混合体素对碰撞边界的影响，不能用网络2 mm voxel替代Teacher几何。

## 9. 候选bank、双配额与合成挖洞

### 9.1 固定bank解决在线增强与离线标签对应

本v1新增**每实例最多1024个UV的固定候选bank**。这不是推理输出数，推理仍最多256、训练最多128。bank只由2D目标mask构造，不依赖GT深度、GT法向或质量分数；这样不同合成洞draw可以从同一bank取子集，离线质量标签不需要每个epoch重算。

沿用16×16图像格、8bit bit-reversed Morton格顺序。每轮每格取一个尚未选择像素，首点取距格内mask像素中心最近者，后续按与已选点的最小二维距离最大者，平局取较小pixel_id；轮询至1024或用尽。细长目标按既有refill思想补齐，禁止重复像素凑数。固定bank足够覆盖洞中心与否是pilot要报告的结果，不以1024自动担保。

预填后将bank按观测资格分池：`raw_pool=bank & is_raw_valid & normal_obs_valid`；`hole_pool=bank & fill_valid & filled_normal_valid`。再按`raw,hole,raw,hole,...`交织，各自保持确定的空间分层顺序；某池耗尽就从另一池补齐。最终完整序列最多256，训练取前128，保持前缀一致。两池都充足时各占50%，只有一种时全取该种。不足256保留真实数量与padding mask，不复制点。

标签按`candidate_bank_row`连接，不能以`candidate_id`直接索引。对于复用旧mesh候选缓存，`candidate_row`是候选数组行号、`candidate_id`只是原始网格采样身份。两套索引分别命名，避免混用。

### 9.2 合成缺深的确定配方

仅train施加；dev使用固定raw版本，并另发布固定缺深诊断版本。SuctionNet与P1合成数据每draw有50%概率挖洞，ClearPose默认保留真实洞分布，不在首轮再叠加强洞。

v1椭圆参数：1–2个；**全短轴**12–32 mm均匀取值；长短轴比1–2；角度0–180°；中心优先位于距目标2D边界较远的区域，但不得用GT质量筛“好点”。根据局部raw深度和fx/fy转像素椭圆，裁到目标mask内。单次删除的原始有效像素不超过目标原有效点的40%，且保留至少100点；最多重采样10次，失败则本draw不挖洞并记录原因。

先保存clean观测监督，再创建工作副本并置空。合成洞GT来自挖洞前该像素的有效传感器深度；不能把真实raw缺深像素的fill当GT。统一记录`hole_origin=none|real|synthetic`。

随机状态用显式`seed=42`、固定manifest sample ordinal、epoch、draw index构成SeedSequence，保留清单顺序；不要使用跨进程不稳定的Python `hash()`。不同worker数不应改变同一draw的增强身份。

### 9.3 几何增强的正确位置

深度噪声、挖洞先作用于输入工作副本，然后重新预填与计算观测法向。刚体旋转随后同时作用于输入点、n_obs、g_cam、plane、GT label pose和Teacher场景：绕重力0–360°、小倾角±8°为默认预算。这里的小倾角是整套坐标/场景刚体变换，不能只转物体而保持重力/障碍物不动还复用旧W/C标签。

`delta_z`始终表示原注册相机的光轴残差。计算`p_ref_aug=R @ (p_fill+delta_z*ray_z1)+t`；plane-clamp在原相机系执行，然后整体变换。等价地可保存`ray_aug=R@ray_z1`，它通常不再满足z=1，不得重新归一化后继续乘同一个delta_z。参考投影/clamp/crop接口在原相机系调用，不能把增强后的XYZ当作原像素反投影结果。

本v1不使用随机物理缩放、不改变R20尺寸。任何实际改变相对场景或接触姿态的增强，都需要重新求标签或明确关闭相应监督；不得只保留旧S/W/C数值。

## 10. R20 / Belt70 Teacher的构建规范

### 10.1 先分清三个历史口径

1. 旧E5全场景producer有R20与含approach的碰撞。
2. 旧A/B `InstanceTeacher`为`frozen_r20_target_self_v1`，只评目标自碰，明确不加载桌面或其它物体。
3. 新Teacher必须使用完整场景，输出`q_sw`与`p_free`两个监督量，并严格按70 mm包络定义。

不能仅修改旧JSON中的末段就宣称完成新Teacher。旧Seal自碰结果的柱体高度无上界，旧总collision也没有足够分项供逆推。新Teacher单独命名为`belt70_canonical_gt_v1`，最终实际pose评估命名为`belt70_exact_pose_eval_v1`。

### 10.2 S20与W20

SuctionNet复用已核验的完整dense和原R20候选分布；候选均值用于W基准中心，不以可见点均值或crop中心替换。新来源在米制CAD上准备完整表面dense与固定候选集：先以0.5 mm表面采样间距作为pilot起点，记录采样器/种子/数量；核对半间距复算的Seal差异后才冻结。没有足够密度的mesh顶点不能直接冒充dense。

R20 Seal沿用现有`seal_at_pose`的72个开区间角度桶、18–22 mm环带、变形与拟合分数；**只取其raw seal返回值**。第二个无上界self布尔值单列旧诊断，不并入新C70。该Seal仍具有历史环柱取点定义，不能把网络core/lip的±8 mm窗口当作已改变了Teacher算法；若以后改Seal高度范围必须单独重命名和对照。

归一化采用一份显式冻结上下界。为延续既有R20口径，首版只用新SuctionNet train的scene0–89所涉及对象的R20 raw scores，取1/99百分位。ClearPose/新对象复用此尺度，并报告超界截断比例；不得各来源各自归一化后直接混在一起。`fit_normalization`现有Python函数接受`train_scene_ids`，但旧CLI默认0–99；需要显式调用或实现新封装：

```python
from pathlib import Path
from suction20.full_cli import fit_normalization
from suction20.full_data import config_from_file

# 运行前完成对象资产与R20候选审核，路径替换为实际值。
cfg = config_from_file('configs/suction20.json')
fit_normalization(
    Path('/data/raw/SuctionNet'),
    Path('/data/derived/r20-existing/seal_r20/poisson'),
    cfg,
    Path('/data/derived/suction20-model2/p0-v1/normalization'),
    train_scene_ids=range(90),
)
```

这段现有函数只产生Seal归一化及其旧格式配置记录；其中旧envelope不得作为Belt70配置使用。生产新Teacher从冻结结果显式读取`seal_norm_min/max`，其余物理参数按新协议构造。输出目录首次使用，勿用旧实验目录覆盖。

W20继承项目的标准化基准：mass=1 kg、gravity=9.8 m/s²、k=15.6、r=0.020 m。给定中心c、接触p、外法向n与重力力矢量F：

```text
tau = (c-p) × F
tau_t = tau - dot(tau,n)*n
W20 = 1 - min(1, ||tau_t|| / (r*k*pi))
q_SW = S20 * W20
```

该W是统一基准评分，不是每只现场瓶子的真实质量/真空泵成功概率；也不声称与任何官方在线评估器完全数值等价。官方对比继续使用独立官方评估协议，工程Belt70指标另报。

### 10.3 C70的精确定义

对障碍点x与接触姿态(p,n)，令`h=(x-p)·n`，`rho=sqrt(max(0,||x-p||²-h²))`，n为物体外法向。

| 分项 | 判定 |
|---|---|
| `c_self_cup` | 目标自身，rho≤22 mm，−2≤h≤25 mm，且h>4 mm |
| `c_self_mount` | 目标自身，rho≤28 mm，25<h≤70 mm |
| `c_ext_cup` | 任意其它物体，rho≤22 mm，−2≤h≤25 mm |
| `c_ext_mount` | 任意其它物体，rho≤28 mm，25<h≤70 mm |
| `c_plane` | 任一实体cup/mount圆柱与支撑平面相交或进入其下方 |

`C70=OR(上述五项)`，`p_free=1-C70`。严格上界70 mm之外的障碍不参与C70；approach碰撞只允许另存诊断。沿边界采用float64与1e−12 m舍入容差；这不是传感器误差容限或安全间隙。

plane不应仅靠稀疏桌面点判断。对每段(z_min,z_max,r)计算：

`h_min=n_b·p+d_b + min(z_min*(n_b·n),z_max*(n_b·n)) - r*sqrt(max(0,1-(n_b·n)²))`。

`h_min<=0`为相交。参考实现提供此解析检查。点云障碍teacher使用完整几何；3 mm voxel可用于加速索引，但最终near-boundary判断应回到对应原几何块，否则薄障碍可能被采样漏掉。若采用近似碰撞版本，必须另记算法版本、分辨率与和dense基准的分歧率，不与精确参考混写。

只要完整场景几何或plane不可信，就不能确认`free=1`。碰撞见证存在时可以保存“已知碰撞”的诊断，但P0首版仅在全场景审核通过时设置`collision70_valid=true`，避免单一bool掩码混淆部分证据。

### 10.4 canonical_gt与预测姿态的对应

对bank内UV，优先使用可信GT表面p_gt和外法向n_gt求S/W/C。SuctionNet无独立完整depth GT时，可将该像素有效原始深度作为`sensor_observed`表面监督，但必须先检查与对应目标CAD表面距离≤5 mm、目标归属与接触面有效，再由已验证对象几何计算法向/物理分；非零背景返回不自动成为表面GT。真实缺深没有可靠渲染交点时对应标签invalid。ClearPose使用通过审核的渲染表面。所有质量标签保存实际label pose。

网络仍只读取raw/fill与mask。P0训练中允许把canonical标签用于预测姿态的邻近监督，但显式记录：

- 质量loss初始门控：预测接触距label点≤5 mm、法向夹角≤30°，门控detach。
- 碰撞loss用更严格的2 mm /5°初始门控，仍是canonical监督近似，不是碰撞不变证明。
- 法向/补洞损失按各自GT有效性计算，不受质量门控的零覆盖阻断。
- 每个batch/source/raw-hole分层报告门控覆盖。没有合格query时该loss为0，不能补负样本；若覆盖不足，先进行几何warmup再启用质量loss。

70 mm末端即使小角度误差也会移位，30°更可移动约35 mm，因此绝不能把门控当安全证明。**最终dev/test结果必须对实际预测(p_ref,n_hat)在完整场景中重新求S/W/C70**；不能查同UV的canonical分数当最终物理结果。

还必须检查`contact_on_target`：预测点到对应目标完整GT表面的距离≤5 mm、实例归属正确且接触面有效。旧Seal只依赖环带相对高度变化，沿法向悬空可能仍有高S，C70也可能free；因此“不接触目标”的Top1必须判失败并留在分母，不能凭S×W×free较高放行。评价报告保存接触距离与失败原因。

若后续需要严格同姿态训练，另设“冻结几何模型快照→生成预测pose→exact-pose标签→训练质量头”的独立实验版本。不要在一个manifest里混称两种标签为同一合同。

## 11. 困难样本、数据分层与监督降级

### 11.1 首轮只标记困难程度，不自动删难例

原规格要求过滤极端深埋实例，但目前没有实测的过滤分布。为避免在首次转换中删除大量碰撞负例，P0 v1固定`difficulty_filter=tag_only`：语义/几何不合法的样本拒绝，深埋与高遮挡合法样本保留并打标签。pilot结束后才决定是否发布train-only的瘦身v2；dev/test始终保留所选帧的全部实例身份与失败状态。

统一困难度统计如下：

- `visible_ratio`：完整场景z-buffer里该目标可见像素数，除以单独渲染该目标的amodal投影像素数。只用于标签/统计，不作为网络输入。
- `top_cover_ratio`：在支撑面二维投影上以2 mm格子、沿反重力方向比较目标顶面与其它物体；分母为目标投影占用格，分子为有其它物体高于目标且间隙处于(0,70] mm的格。使用面积比例，不按点云点数计比。
- `near_obstacle`：目标候选周围半径40 mm、沿外法向高度(0,70] mm存在外物的比例；40 mm明确为半径，仅作困难标记，不替代C70。
- `raw_valid_ratio`、`real_hole_area_ratio`、`largest_hole_diameter_m`、`fill_failure_ratio`、`nonzero_wrong_depth_rate`分别记录。

缺amodal/完整场景时上述标签为unknown，不把unknown写成无遮挡。若未来启用深埋删除，可从`visible_ratio<0.20 AND top_cover_ratio>0.50`的候选规则开始审查，先看其对碰撞正负分布的影响，再冻结，不凭这个建议直接改当前v1清单。

### 11.2 部分监督如何使用

| 可用资产 | 允许的监督 | 不允许的推断 |
|---|---|---|
| 可靠raw与synthetic clean pair | hole depth、观测法向相关监督 | 不把原本raw hole的fill作为GT |
| 可靠渲染深度/法向，CAD位姿暂不够精确做质量 | hole/normal | q_SW与free仍unknown |
| CAD、接触pose可信，但重力未知 | Seal | 不用默认重力伪造W |
| 目标CAD可信，其它障碍缺几何 | Seal及已知W | 不因目标自碰为false就填free=1 |
| 完整场景、支撑面、对象映射均可信 | S/W/C70 | 不能因此宣称能预测真实机器人执行成功 |

主P0质量训练manifest和depth-only隔离manifest分开，summary分别计数；同一实例多任务监督共用split身份。当前P0路径不消费缺plane/gravity的隔离样本，后续独立输入路径实现后再显式混采，不通过补零伪装成完整标签。

## 12. 现场400+偏好对的数据准备

### 12.1 最小记录格式

另建`suction20.model2.preference.v1`，不复用旧`attempt_success`格式。每条必须记录：

| 类别 | 字段 |
|---|---|
| 身份 | preference_id、capture_id、instance_id、physical_bottle_id（若可追踪）、collection_batch_id、group_id、split |
| 图像与标定 | RGB/depth路径和时间戳、注册K、图像尺寸、界面到原图的变换、plane版本 |
| 历史A | 原`p_plan,n_plan`、坐标系、工具变换链、pose_source、actual_execution_verified、原成功/失败记录及其有效性 |
| 人工B | 原图`u_B,v_B`、preferred=true、标注者/标注轮次、review_status；不预填虚构z |
| 偏好性质 | `label_kind=relative_preference`、`preference_valid`、人工原因（如中央平整区、远离瓶口等） |
| 表面版本 | geometry_checkpoint_revision、fill_spec、候选/邻域规则；Stage1未冻结时为pending |

本项目历史A为planned pipeline contact，若没有实际执行反馈验证仍保留`actual_execution_verified=false`。人工B优于A不等于B在机械臂上必然成功，也不等于A的S/W物理分必然为0。

### 12.2 分组与80/20

把同一physical_bottle、collection_batch、相邻连续采集、同一capture关联成连通分量；一个连通分量只能在一个split。信息不全时至少按batch/capture，报告无法验证的实物身份关联，不能声称完成了严格物体互斥。

固定seed42，按group数量约80/20切分，再报告帧数/实例数实际比例。先构建关联关系再切分，不能先按帧随机切好后仅检查两个集合的group字符串。若连通分量太大导致无法留出有代表性的dev，应补采独立批次，而不是拆开同一物体骗出20%。

### 12.3 A不改，B经固定表面生成邻域

A按原变换链转到统一相机坐标，保留原点和有向轴；最近点仅用于检查支持，不能把A吸附到另一点然后仍称历史动作。

Stage1几何模型冻结后，B像素走与部署相同的raw→prefill→delta_z修正；B落在真实缺深时不能沿用旧`raw 5 mm support`门限直接丢弃。B⁺采用7 mm首版半径（原规格5–8 mm范围内），同时满足同实例、同局部连通面、与B法向夹角≤20°；至少4个唯一有效点，否则记录`preference_region_insufficient`。邻域分数采用有效点算术平均，记录点数和半径。

如果A/B距离太近、邻域重叠导致相同点同时成为正负例，或B不在目标mask/注册网格内，置`preference_valid=false`并送回标注审核。这里不自动移动人工B。全部排除保留账本，包括`A_pose_unsupported`、`B_fill_failed`、`registration_missing`、`overlapping_regions`。

冻结checkpoint版本变化后必须重建B三维与B⁺；二维点击无需重标。人工点击不用于创造深度GT。偏好draw不进行新的大空洞删除或会改变优劣的大幅几何增强。

## 13. P1数据接入条件

### 13.1 MetaGraspNetV2仿真子集

锁定官方`MetaGraspNetV2_V2`修正版本，记录实际下载源、版本声明和模型清单。论文物体集为82，不是989；先按实际存在资产列出瓶/罐/盒/容器、最大尺寸≥50 mm的候选，再人工审核可用对象数量，不预定“至少80–120瓶罐” [S3][S4]。

原始读取涉及`<view>.npz`、`<view>_rgb.png`、`<view>_camera_params.json`、`<view>_scene.hdf5`。重点核对`depth`、instance map和`objects/poses_relative_to_camera`的实例顺序。官方读取代码将深度与场景平移从厘米转米；不能把mesh再同倍率缩一次 [S5]。

原吸盘分不能混训：公开采样器的接触半径、碰撞外形及评分模型与本项目不同，不能简化为“原标签一定R10”。本项目只保留几何和场景，用同一S20/W20/C70定义重算。精选列表仅限制**训练目标及Seal预计算**；同帧其它物体仍保留为碰撞障碍。

按独立布局scene分组，选择每scene2–4视角作为P1起点，总2500–3000 camera-frames是预算，不是已确认可构建规模。先抽20个scene审核尺度和新版对齐；未通过不得扩量。P0单独能完成闭环，MGN不阻塞P0发布。

### 13.2 压扁瓶合成

只在P0完成后实施。为每个基础网格保留来源、可用许可、base_mesh_id；变形版保留父ID、参数与随机状态。原规格的轴向压缩0.35–0.75、局部凹陷和弯折仅是采样范围，不能把单轴缩放直接等同真实PET力学。

网格检查包括自交、面朝向、合理厚度/尺度、可用表面与支撑接触。瓶口可以物理开放，不为了计算体积而自动封口并改变密封几何；watertight不是所有真实瓶子的必要属性。若需要质量/质心却没有闭合体积，用本项目明确的统一基准中心政策，不伪造真实材料参数。

布局可按1–4个瓶子+1–3个异物落到水平支撑面；记录稳定姿态、penetration、深度/实例/GT及完整场景对象列表。先生成50个场景审核，不直接开2000帧全量任务。按base_mesh_id关联切分可评价形态外推；若按scene切分则明确是同基础物体新布局，不把它报告成新物体泛化。

## 14. 缓存、分片、容量与断点恢复

### 14.1 哪些能缓存

| 项目 | 离线缓存 | 不能混入同一静态缓存的量 |
|---|---|---|
| 原始解码、K、mask、物体/scene身份 | 可以；camera-frame共享 | 各次随机增强结果 |
| 固定bank UV及canonical标签 | 可以；包含bank版本和label pose | 当前预测p_ref/n_hat、模型特征 |
| 无增强raw的预填和几何 | 可以；单列normal/fill/crop版本 | 换洞mask后仍复用旧fill |
| 全场景Teacher几何与索引 | 可以；按scene与对象版本 | 网络裁剪后的不完整障碍物 |
| 现场B三维邻域 | Stage1冻结后可以 | 未固定的几何checkpoint |

在线挖洞会改变is_raw_valid、预填、法向、hole_dist、mu_vis、局部crop与候选raw/hole配额；这些量必须重新计算或按该明确增强身份命中缓存。canonical标签只要UV、GT姿态和物理场景不变，可以从固定bank取对应行。

### 14.2 存储与读取

观察到历史A/B对共享挂载NPZ回退读取敏感，本方案优先在**实际所需字段**上建设有界NPY/mmap镜像，不默认把全部大数组无界展开。先用pilot实测每camera-frame、每实例、每bank标签的逻辑字节/占盘字节，估算全量并保留至少20%磁盘余量。

几何大数组按camera-frame共享，instance记录保存mask ID、crop索引和bank索引，避免每个实例重复保存整图。监督目录和观察目录在loader代码中使用白名单分开打开，防止GT泄漏。

数据发布写到新版本staging目录；单scene/worker独占输出分片。文件写临时名后原子rename，全部语义检查通过后才发布manifest、summary，最后写READY。中断分片不被训练读取。恢复时比较schema、版本、小型物理参数、相机/帧/实例身份、数组shape和pose对齐；不计算内容摘要。

不要直接把新manifest交给旧`build_instance_source_cache.py`或`build_instance_geometry_cache.py`，它们理解旧A/B schema。可复用其原子发布、预算和读取设计，但新数据要独立实现格式识别并拒绝错误schema。

### 14.3 运行性能验收怎么做

在同一训练负载条件下分别测 cold read、warm OS cache、NPY mirror，记录：文件打开/解压、array materialization、fill/normal/crop、Teacher、IPC、H2D和训练等待。比较时固定worker数、候选预算与后台作业；不能把单跑与双任务并发的吞吐差当成缓存净收益。

首轮先测64–128个真实实例，估计Seal每query耗时、1024-bank总成本和磁盘预算。若1024-bank代价过大，不能静默缩到128后仍沿用bank版本；应选定更小bank、报告候选覆盖/质量覆盖差，再建立新协议版本。缓存覆盖不足要在正式训练前解决，别只报告“缓存开关已打开”。

## 15. pilot的具体执行包

### 15.1 固定pilot样本

SuctionNet：scene0、1（train）及scene90（dev），每scene取视角0、128、两相机，共12 camera-frames。ClearPose：set1/scene1、set4/scene1（train），set1/scene4、set4/scene5（dev），每scene从已选96列表中取首/中/末三个，共12 camera-frames。没有足够帧时按实际数报告，不跨split借图。

在24 camera-frames中保留全部目标身份。为人工QA至少选择：正常不透明表面、中心缺深、小瓶口/凹陷、贴带、邻物高出50–70 mm、多个同类实例、遮挡严重、低支持和GT失效各一例；若pilot不包含某类型，则补同split的样本并记录，不能伪造覆盖。

### 15.2 每例应查看的图

一张排版图至少包含：RGB+实例边界、raw valid/real hole/synthetic hole、raw/fill/GT深度差、plane相对高度、raw与hole候选、目标加背景crop、预测前label pose、四C分项的碰撞见证。GT视觉层带“仅监督”标识。另给可旋转3D检查文件或明确的多视角截图，检查法向有没有朝瓶内部。

QA记录每个异常的sample_id、截图路径、原因与处理，不只挑最好的几个例子展示。数据问题通过修改适配器或显式reject解决，不能手工改最终label数值让报告变好看。

### 15.3 pilot交付清单

`pilot_report.json`包含每源/每split的frame、instance、bank、有效quality/free/hole计数；所有拒绝原因；重投影/平面残差；raw不变检查；补洞超范围率；原始非零错深比例；全场景C70见证；CPU耗时和磁盘估计。还需`pilot_review.csv`人工审核表与`release_config.json`完整实际参数。

只有pilot通过，才按P0完整帧清单扩量。不要先跑全库R20再发现实例映射或深度单位错误。

## 16. 数据验收与模型效果验收分开

### 16.1 数据进入训练前的硬门槛

| 检查 | 放行条件 |
|---|---|
| 文件与结构 | 所选清单全部可读；required字段/shape/dtype一致；无未处理异常 |
| 分组 | train/dev/test的scene/setup/capture/bottle关联无交叉；双相机同归属 |
| 单位与坐标 | 米制、K与pose方向已通过真实可视化；投影数值往返通过 |
| 输入隔离 | loader输入白名单；GT/CAD/位姿/质量不进入X_in |
| Zero-Touch | 同一增强draw中is_raw_valid=true的深度/坐标在预填和残差后逐值不变 |
| 缺深 | fill有效性显式；非法ray/无支持不输出假表面；所有合法hole点h≥2 mm |
| 裁剪 | 包含允许修正端点+80 mm；任意合法姿态的C70包络不被裁掉 |
| 候选 | UV唯一、raw/hole配额与借额可重现；128是256前缀；bank到label索引一致 |
| 标签 | q_SW与S×W一致；p_free与C70一致；五C分项OR正确；未知不参与loss |
| 物理定义 | 所有来源radius=20 mm、C70上界70 mm、同归一化/重力基准；无旧C150掺入 |
| 正负分布 | 各来源/各split报告S/W/C与有效率；没有因筛难例导致碰撞负例消失 |
| 发布 | 逐分片summary、拒绝账本、读取器回读通过；仅完整发布写READY |

`NaN*0`、丢弃无输出实例、用GT筛raw输入、在Teacher里删未精选邻物等属于阻断问题。数据负责人应修复对应阶段并更新受影响产物，不跳过断言继续训练。

### 16.2 模型训练后的效果门槛

原规格的偏好准确率≥85%、Top1命中好区净增≥15pp、源域Top1掉点≤1.5pp、空洞表面中位误差<4 mm等都保留为**研究/工程目标**，本次未测得这些结果，也不用于声称数据转换已通过。

“空洞候选覆盖≥90%”在本方案明确定义为：对原始target mask内有可信GT的真实/合成hole像素，是否存在同实例hole候选，其投影米制距离≤10 mm。分母包括fill失败的hole像素；不能只在fill成功处算覆盖。先报告支持≥100的可推理实例，再另报包括insufficient_support实例的端到端覆盖。median、P90和各洞尺寸区间分别给出。

同时报告bank覆盖、fill/normal资格通过后的覆盖、最终128/256采样覆盖，定位失败发生在哪一层。指标中的投影米制距离使用该hole像素可信GT深度作为离线度量尺度；模型输入的hole_dist仍只使用fill深度，两者用途分开。

Top1物理通过率基于实际输出姿态重评，必须同时通过contact_on_target，分母包含所选评价实例的无输出；质量阈值要在首次训练前显式冻结，不靠测试结果反复调整。偏好成对准确率按独立group汇总，同时给group数、帧数和置信区间，避免400帧的重复采集造成虚高。

普通稀疏卷积与离散采样不自动保证90°旋转后分数差<1e−4。数据侧验证刚体坐标/物理标签的一致性；网络旋转稳定性另报误差分布和排序一致率，不把原HTML的该数值当架构已经保证的性质。

## 17. 按依赖顺序实现的生产任务

下面是开发者执行清单。方括号未勾选意味着本次尚未实现；文件名是建议落地位置，不是当前CLI。完成一项需要满足右列产物和检查，不能只创建同名空脚本。

| 顺序 | 建议模块 | 具体实现内容 | 完成/验收 |
|---|---|---|---|
| 已完成 | `scripts/modeling_v2_data_plan.py` | 场景/抽帧计划、路径存在与大小预检 | 标准库可运行；不冒充训练manifest |
| 已完成 | `docs/modeling_v2/reference/geometry_contract.py` | ray、ZeroTouch、plane、crop、C70边界与声明分组参考 | 自测报告；明确不是完整producer |
| [ ] M1 | `suction20/model2/schema.py`、`adapters/suctionnet.py` | 新frame/instance/label合同，SN单位/实例/pose映射；独立dev | 真实pilot与旧readback XYZ/身份对照，无GT输入 |
| [ ] M2 | `adapters/clearpose.py`、`calibration.py` | MAT容器、模型/pose、渲染GT来源、plane与gravity审核 | 24scene清单可消费，unit/GT/plane诊断齐全 |
| [ ] M3 | `preprocess.py`、`sampling.py` | 局部fit/IDW、支持与valid、修正端点crop、12通道、1024bank+双配额 | 第7–9节全部断言与真实图检查 |
| [ ] M4 | `teacher.py` | 多源dense/中心/归一化；canonical S/W；完整场景五C分项 | 70mm边界、邻物/plane、旧self不混入、GT pose身份 |
| [ ] M5 | `build.py`、`dataset.py` | D3–D8串联，分片、锁、预算、原子发布、input/label白名单 | pilot端到端；失败账本与READY；训练mini-batch可回读 |
| [ ] M6 | `preferences.py` | group连通切分、原A、二维B、冻结表面B⁺ | 透明洞B不会被旧raw support静默skip |
| [ ] M7 | `adapters/metagraspnet.py`、合成管线 | P1按独立源版本接入 | 20scene/50scene先验收，再扩预算 |

先完成M1–M5的P0数据路径，再开始主模型开发/训练；M6可先整理二维记录。所有新增可执行代码都运行仓库要求的：

```bash
python -m pytest -q tests/test_no_sha256_verification.py
```

新增生产入口后再把其真实`--help`和一条成功pilot命令补入本文第2节，届时才可把“生产转换待实现”状态改为“可运行”。本次不以未来计划占位脚本冒充完成。

参考工具不是生产schema完整validator。调用时显式映射：`depth_raw_z_m←depth_input_m`、`raw_valid←本draw is_raw_valid`、`target_mask←instance_map==target_instance_id`、`depth_kind='optical_z_m'`，K/plane/g保持原相机系。合成挖洞后不能把raw_valid错误映射回未挖洞的sensor_valid。

## 18. 操作失败时如何处理

| 状态/现象 | 处理动作 | 是否允许继续质量构建 |
|---|---|---|
| preflight退出2 | 查看audit.csv的缺失路径，补齐所选原始资产后换新输出目录重跑 | 否 |
| CP metadata字段或pose shape不符 | 保留版本与字段名，更新显式adapter mapping并做重投影 | 否 |
| mask同类多实例混淆 | 提供instance-row标注或受控几何渲染映射；禁止类别当实例 | 否 |
| 支撑面未知/非平面 | 保留depth-only，单列plane_unreliable | 主质量集否 |
| raw支持<100 | 记录insufficient_support；评价保留身份与无输出 | 不进入该实例的模型前向 |
| fill失败/Δz超40mm | 保留失败像素与原因，回看洞尺寸/局部拟合；不造GT | 可保留该实例其它有效任务 |
| 旧C150或旧INST20标签 | 只作历史对照，用新Teacher重算 | 不可混用 |
| bank与label行数/UV不符 | 立即拒绝加载，重建相应分片 | 否 |
| 难例大比例被过滤 | 回到tag_only，检查筛选与正负分布 | 未查明前不发布 |
| NPY展开超过预算 | 减少重复整图、共享frame数组、分片；比较真实覆盖率 | 不无界写满磁盘 |
| 现场B落洞但无三维 | 核对注册、fill与冻结几何版本；保留原B送回审核 | 不把B改成最近raw点 |

## 19. 本次核验记录与来源

本次已检查用户HTML、现有仓库的数据/teacher/缓存实现和下列官方来源。只执行了本地参考、工具fixture与分支合并回归；实际数据资产、GPU训练、400+点击记录和机器人结果未在本轮获取或运行。具体本地验证见[VALIDATION.md](VALIDATION.md)，历史路线合并见[分支收拢记录](../BRANCH_CONSOLIDATION_20261125.md)。

### 项目内依据

- [旧数据流程](../DATA_PIPELINE.md)与[旧数据schema](../DATA_SCHEMA.md)：R20候选、row/id、物理配置与标签未知规则。
- [A/B共同数据合同](../instance_routes/DATA_AND_LABELS.md)与[A数据准备](../instance_routes/route_a/DATA_PREPARATION.md)：历史split、同姿态teacher与原始支持语义。
- [instance_teacher.py](../../suction20/instance_teacher.py)：旧target-only teacher的边界。
- [eval_r20.py](../../suction20/eval_r20.py)、[full_data.py](../../suction20/full_data.py)：Seal/Wrench及旧包络实现。
- [L20读取与缓存记录](../instance_routes/L20_PIPELINE_OPTIMIZATION.md)：负载条件、源读取与缓存覆盖问题。

### 官方一手来源

- [S1：SuctionNet官方数据格式与下载][S1]。
- [S2：ClearPose官方README、GT来源、数据结构与split][S2]。
- [S3：MetaGraspNet官方仓库及V2_V2更新][S3]。
- [S4：MetaGraspNetV2论文，§III-A对象集合与合成方法][S4]。
- [S5：MGN官方点云读取代码][S5]。
- [S6：ClearPose官方预处理代码][S6]。
- [S7：MGN官方抓取采样器与尺度转换][S7]。

[S1]: https://graspnet.net/suction
[S2]: https://github.com/opipari/ClearPose
[S3]: https://github.com/maximiliangilles/MetaGraspNet
[S4]: https://publikationen.bibliothek.kit.edu/1000168221/152209863
[S5]: https://github.com/maximiliangilles/MetaGraspNet/blob/master/Scripts/colored_pcl.py
[S6]: https://github.com/opipari/ClearPose/blob/main/data/preprocess.py
[S7]: https://github.com/maximiliangilles/MetaGraspNet/blob/master/grasps_sampling/scripts/sample_grasps.py
