# B 数据准备：共同实例清单与同姿态标签

本页将[共同数据合同](../DATA_AND_LABELS.md)落实为 B 分支的数据交付顺序。其产物供[预处理](PREPROCESSING.md)、[训练](TRAINING_PLAN.md)和[评价](EXPERIMENT_PLAN.md)使用。当前实例 builder 与 teacher 拆分是待实现接口；已存在的参考检查只核验数学、字段和清单关系。

## 1. native 输入资产和冻结划分

源域准备 SuctionNet 的配准 RGB-D、实例掩码、K、场景/相机/对象变换、完整对象 `dense_point_clouds`、既有 R20 Seal 候选和冻结物理配置。完整对象只用于离线 teacher 和评价；网络读取可见 RGB-D。目标域准备既有双目/深度、RGB、目标 mask、标定、原规划接触姿态和人工结果。

本节保留 native 实例物理研究及产线适配的既有划分；第7节另定义公共论文比较的 `public20` 数据 profile，两者不共用训练清单身份。先按 scene 或既有 group 固定 split，再展开 frame/实例/动作；不得先裁图后随机分样本。

| 源域集合 | 固定范围 |
|---|---|
| train | scene 0–99 中 `scene_id % 10 != 9` 的90场景；frame 0–255 |
| val | scene 9、19、…、99；frame 0、8、…、248，共32视角/场景 |
| test seen / similar / novel | scene 100–129 / 130–159 / 160–189；各256视角/场景 |

manifest 的 `split` 只填 `train/val/test`；测试子类写入 `test_subset=seen|similar|novel`。同帧实例同 split；该划分不是 CAD 身份互斥。

| 目标集合 | 动作总数 | 成功 / 失败 |
|---|---:|---:|
| train | 1676 | 1051 / 625 |
| val | 399 | 209 / 190 |
| test | 87 | 65 / 22 |

保持原始清单，不为配比重新拆分。同 capture 的深度版本、裁剪、相邻帧和同摆放/session 继承既有 group；共享同一观测的多个实例必须同 group、同 split。用原始资产 SHA-256 补充路径检查，不以声明的 hash 代替真实字节核验。目标数量和已知 group 情况以[共同数据页及其原始汇总来源](../DATA_AND_LABELS.md)为准；不推测缺少的 group 数量。

## 2. 逐阶段数据产物

以下为数据根目录内的逻辑产物，实际路径写入 manifest；这些不是新的 CLI 已实现承诺。

| 产物 | 必须包含 | 交给谁 |
|---|---|---|
| 原始观测包 | RGB、米制轴向 depth、实例 mask、K、gravity、有效性、标定身份 | 共用预处理 |
| 实例 manifest | 一行一个实例观测；见下表 | builder、训练/评价 sampler |
| normal/query 缓存 | v3 normal/spec、完整最多256候选、原像素 ID、精确 p/n、格号/轮次/顺序、有效标记 | A/B 共用 |
| source label 缓存 | 每个精确 query 的 S/W/self/SW/INST；物理v2 schema 与身份 | 源预训练、源回放、实例评价 |
| target attempt 表 | 一行一次原始规划动作及结果、支持与排除原因 | 目标 BCE 与同动作评价 |
| 准备报告 | split/group/hash 统计、支持/法向/候选有效率、排除原因、teacher 成本和校验结果 | 训练准入与实验追溯 |

实例字段至少为 `schema_version=instance_observation_v2`、`sample_id`、`instance_uid`、`capture_uid`、`group_id`、`split`、`observation_sha256`、`observation_npz`、`mask_id`、`source_kind`、`calibration_sha256`、`normal_spec_sha256`、`sampler_spec_sha256`、`query_cache`、`label_cache`。目标实例没有物理标签时 `label_cache=null`，不伪造空物理分。所有真实 hash 必须为规范字节生成的合法 SHA-256。

## 3. 实例、观测与 query 对齐

保持相机 x 向右、y 向下、z 向前；深度为 z 轴米制距离。统一 RGB/深度/mask 的校正相机网格和时间，`T_a_b` 表示 b→a。原像素身份固定为 `source_point_id=v*W+u`。相机系 gravity 保存完整3分量及来源；现有目标域为传送带平面法向的近似来源，不能写成已测惯性方向。[坐标依据](../DATA_AND_LABELS.md)

源训练原始有效支持少于100点的实例标记 `insufficient_support`，不进入常规采样；验证/测试保留实例身份并纳入失败分母。`mu_vis` 为全部有效可见支持的均值，在候选或图像缩放前计算；它不是物理 COM。

由[预处理步骤](PREPROCESSING.md)生成共同最多256个 query。训练/源回放只取完整序列前128个，64是独立可选配置，验证/测试最多256。query 缓存保留原始反投影的 p 和 v3 法向 n，不用平滑 XYZ 替换接触中心。B 不需要生成稠密监督热力图，也不把同像素的不同有向姿态标签平均。

## 4. 只针对精确姿态计算物理标签

固定质量和文件身份：

$$
C_{\rm self}=C_{\rm seal,self}\lor C_{\rm target,envelope},\quad
q_{\rm sw}=S_{20}W_{20},\quad
Q_{\rm inst20}=q_{\rm sw}(1-C_{\rm self}).
$$

`quality_contract=gt_INST20_pose_v2`，标签 `schema_version=suction20.instance_inst20.v2`。碰撞 true 表示碰撞发生；未知 self 不能补 false。先把相机 query 的 p/n 按冻结变换转到相应对象/世界坐标，再评测**同一姿态**。

1. 完整目标 dense 调用冻结 `seal_at_pose`，保留 raw Seal 与 Seal self；沿用冻结归一化和72开区间分箱。
2. `wrench_scores_dual` 使用同一精确姿态、完整原 Seal 候选均值与默认1 kg基准质量，取 `wrench_eval`；可见均值不代替其中心。
3. 单独复用 target-only 包络自碰；不建立外部物体/桌面索引来生成新 INST 标签。
4. 分别保存两个 self、OR 后 self、SW 与 INST；Seal/Wrench 辅助标签不会因 self 使质量归零而被改成零。

Seal self 冻结为 `any(rho < 0.020 m AND h > 0.005 m)`，无 h 上限。target-envelope 保留3mm障碍体素、4mm自身高度阈值、cup/mount/approach 参数及原函数前置检查；不能仅复制三段数组或套用模型的局部窗口。精确比较条件见[共同 teacher 合同](../DATA_AND_LABELS.md)。旧 `evaluate_poses` 可校验 S/W，但其总 collision 不能分解回 self；禁止用除法从旧 gt_B 恢复新标签。

| 标签数组 | 形状 / 语义 |
|---|---|
| `points`, `normal_out` | [K,3]，精确米制 p、外向单位轴；首版 float64 |
| `seal`, `wrench`, `q_sw`, `q_inst` | [K]，有效行有限且在[0,1] |
| `collision_self_seal`, `collision_self_envelope`, `collision_self` | [K] bool；有效行满足 OR |
| `label_valid` | [K] bool；仅在 S/W/两个 self 均合法已求值时为真 |

核验乘积 `atol=1e-6, rtol=1e-5`，轴范数误差≤1e-4。全数组禁止 Inf，mask=false 行可含 NaN；空/全无效文件可通过结构检查，但必须标记无监督，不能进入正常源损失。真实 q=0 是有效负例。

## 5. 历史动作保留原 p/n

attempt 表保存 `attempt_id/sample_id/group_id`、`point_cam_m/normal_out_cam`、`outcome/label_valid`、`pose_source=planned_pipeline_contact`、`actual_execution_verified=false`、`pose_support_valid/exclusion_reason` 和 `pose_transform_sha256`。

历史 p/n 从原记录以 float64 读取，保留有向符号。使用同实例原始 XYZ 中≤5mm的最近点验证中心支持，并列取最小 source ID；锚点不替换原动作中心或轴。无锚点标记 `pose_geometry_invalid`，不当成吸取失败。每条动作仅提供一次直接 BCE；不注入候选池，也不向20mm圆盘扩散标签。

## 6. 缓存、成本与准入

缓存身份绑定观测/实例、完整对象资产、冻结物理配置、teacher commit、v3 normal/sampler、精确 p/n 字节、split 清单、质量合同与单位。光度增强不改物理标签；改变法向、候选、p/n、物理规则或精度合同须更新身份。旧 S/W 仅在原精确姿态和对应身份不变时可复用。

按固定采样计划对真正不同的实例/query 按需建缓存，重复 instance draw 复用，不要求先重算90场景所有帧的256查询。先在64–128个覆盖尺寸和 dense 点数的真实实例测加载、Seal 秒/query、包络时间、秒/instance、缓存大小/命中率，再估算总成本。

目前可运行：`geometry_contract.py check-manifest` 与 `revision_contract.py check-labels`；用法见[共同复现命令](../TRAIN_EVAL_REPRODUCE.md)。前者不读取原始资产字节，后者不重新执行真实 teacher。进入训练前仍须完成真实 p/n 回读、S/W 与冻结 evaluator 对齐、两个 self 分项对齐、`C_old_total=C_self OR C_external` 的独立兼容检查、split 泄漏核验及全部排除统计。

## 7. 公共论文比较的数据 profile：public20

`public20` 用于本方法分别与既有方法的论文报告值比较。共同测试集、六项AP、半径例外及外部评价合同统一引用[公共方法比较协议](../PUBLIC_METHOD_COMPARISON.md)，不以本页 native 实例指标替代。该 profile 保留本路线模型和 `gt_INST20_pose_v2` teacher 定义；变化的是正式训练场景范围、选模阶段和测试实例来源。

| 阶段 | 数据范围与使用方式 | 独立身份 |
|---|---|---|
| 源域开发 | 沿用第1节90个train场景与10个dev场景；仅用于确定源域配置和冻结推理规则 | development manifest、抽样计划、teacher release、运行目录 |
| 正式源域重训 | RealSense scene 0–99全部100场景，每场景frame 0–255均在可抽样集合；从头训练，不继承开发或产线权重 | final-train manifest、100场景抽样计划、teacher release、checkpoint |
| 正式公开测试 | RealSense Seen 100–129、Similar 130–159、Novel 160–189；每场景256帧，总计23040帧 | 固定frame manifest、前端mask release、预测文件及完整评价结果 |

全部256视角进入训练可抽样集合不代表逐帧跑完一个epoch：正式预算仍是140000个成功受监督实例暴露，按[训练方案](TRAINING_PLAN.md)执行。100场景训练包含原dev场景，因此原10场景不再是其独立验证集。正式测试不得抽帧、筛选成功分割帧或把错误/漏检帧删除后缩小分母。

### 7.1 预测实例前端与输入权限

源域训练可使用GT实例裁剪及离线teacher所需完整目标几何。公开测试按配对对象固定以下推理profile；三者共用同一份source-only B评分权重，不重训评分网络，但必须分别推理、保存预测文件及SHA-256。

| 推理profile | 本方法配对对象 | 实例前端与工作区 |
|---|---|---|
| `uois_no_gt_workspace`（默认） | S1B、Diffusion-Suction、MultiGraspNet | 冻结UOIS；完整图像与有效depth输入，无GT工作区；其他过滤使用预先冻结的本方法规则 |
| `uois_uisn_gt_aabb20` | UOIS-Net + UISN | 冻结UOIS；复用UISN的GT联合前景3D AABB工作区，outlier=0.02m |
| `uoais_uisn_gt_aabb20` | UOAIS-Net + UISN | 冻结UOAIS；复用同一UISN工作区规则，outlier=0.02m |

S1B官方推理不使用GT工作区，不能把UISN工作区扩展为所有方法的“共同官方工作区”。上表的默认规则是本方法侧合同；Diffusion-Suction原论文未公开或尚未核实的工作区字段保持`unverified`，不能据此断言其历史实现与本方法完全相同。各对照的证据状态以公共页为准。

仅两个UISN配对profile允许读取GT前景工作区：先将标签坍成`seg > 0`，按UISN原`get_workspace_mask(cloud, gt_seg, trans, outlier=0.02)`等价规则求所有前景联合3D AABB，不按GT id拆物体。**该工作区会改变合法支持和候选池**，必须在输入权限及候选身份中如实记录；完成工作区筛选后，GT前景、对象身份/类别、位姿和完整几何不得继续成为实例分组或模型特征，也不得用GT补漏检、删误检或对预测实例做匹配过滤。实际实例仍完全来自UOIS/UOAIS预测mask。外部评测端正常使用GT，不与模型输入权限混淆。

与S1B、Diffusion-Suction或MultiGraspNet比较时，本方法是“冻结实例分割前端 + B评分网络”的完整系统，不读取GT前景来裁工作区，也不读取GT实例来分组或过滤。UISN配对优先核验并使用作者发布的对应mask资产；如需重新生成，固定前端代码、checkpoint和预处理，并说明是否已验证与作者缓存一致，不能仅凭同名模型认定输入相同。

分割漏检、无合法候选和`no_pose`均保留对应原始frame；全部23040帧进入公开评价，不能从分母移除。native的GT实例评价及产线目标mask流程继续按原合同执行，不冒充该公开预测实例协议。

### 7.2 可追溯资产与信息披露

每个公开前端release保存：前端名称/代码commit/配置、checkpoint真实SHA-256、权重原始来源及额外训练数据说明、原始RGB-D/标定身份、每帧mask文件SHA-256、完整frame覆盖账本、生成成功/失败/空mask状态、实例/有效支持统计与错误原因。另按推理profile保存工作区规则/代码及参数hash、逐帧工作区mask hash、裁剪后的支持/query身份和预测SHA-256；UISN profile同时记录GT联合前景输入及变换的hash，默认profile明确`uses_gt_workspace=false`。复用预测必须同时满足前端与工作区身份一致，不能将同一UOIS前端的有GT工作区预测复用于无GT工作区比较。使用发布缓存时同时保存下载来源、归档hash及解包后的逐帧hash；任何缺帧或处理错误均须显式处置并保留日志，不静默跳过。coverage须区分“文件覆盖完整”与“预测检出完整”，后者不能由前者推断。

90场景开发与100场景正式训练的manifest、teacher release和checkpoint具有独立身份，至少绑定`profile`、`phase`、scene/frame清单hash、训练抽样计划hash和teacher合同/代码hash。teacher数学定义保持不变；即使底层相同精确姿态标签可去重存储，也必须经过身份核验，并由各自release manifest引用，禁止将90场景缓存直接改名冒充100场景完成清单。

公共吸取评分网络只使用源域监督，不混入1676/399/87条产线动作或其微调权重。冻结分割前端可能使用额外训练数据，必须单独披露；因此不能把整个系统的训练信息写成“除吸盘半径外完全相同”。对原论文未公开的历史依赖、权重或训练细节记录`unreported`，不以本次推测补齐。以上profile字段及公开缓存builder属于待实现接口，本页不提供伪装为现有功能的数据生成命令。
