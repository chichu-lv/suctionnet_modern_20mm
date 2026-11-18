# B 数据准备：共同实例清单与同姿态标签

本页将[共同数据合同](../DATA_AND_LABELS.md)落实为 B 分支的数据交付顺序。其产物供[预处理](PREPROCESSING.md)、[训练](TRAINING_PLAN.md)和[评价](EXPERIMENT_PLAN.md)使用。实例 builder、RGB-D 数据集读取与拆分 teacher 已分别实现在 `scripts/build_instance_dataset.py`、`suction20/instance_data.py`、`suction20/instance_teacher.py`；已通过合成资产接线检查，真实完整数据重算尚未运行。

## 1. 输入资产和本轮划分

源域准备 SuctionNet 的配准 RGB-D、实例掩码、K、场景/相机/对象变换、完整对象 `dense_point_clouds`、既有 R20 Seal 候选和冻结物理配置。完整对象只用于离线 teacher 和评价；网络读取可见 RGB-D。目标域准备既有双目/深度、RGB、目标 mask、标定、原规划接触姿态和人工结果。

按本轮决定，源域直接使用全部100个训练场景，从头训练；不再设置90/10开发划分或第二轮final refit。源域test用于选best，结果身份为 `test_selected`，不能作为独立盲测成绩。先按scene或既有group固定split，再展开frame/实例/动作；不得先裁图后随机分样本。

| 源域集合 | 固定范围 |
|---|---|
| train | scene 0–99全部100场景；frame 0–255 |
| test / best选择集 | scene 100–189全部90场景；每场景frame 0–255，总23040帧 |
| test子集 | seen 100–129、similar 130–159、novel 160–189 |

源manifest的 `split` 为 `train/test`；测试子类写入 `test_subset=seen|similar|novel`。源test标签仅用于评价/选模，不参与梯度更新；选择行为本身仍使用了test信息。同帧实例同split；该划分不是CAD身份互斥。目标域继续保留下面的train/val/test：

| 目标集合 | 动作总数 | 成功 / 失败 |
|---|---:|---:|
| train | 1676 | 1051 / 625 |
| val | 399 | 209 / 190 |
| test | 87 | 65 / 22 |

保持原始清单，不为配比重新拆分。同 capture 的深度版本、裁剪、相邻帧和同摆放/session 继承既有 group；共享同一观测的多个实例必须同 group、同 split。按既有sample/capture/group ID和规范化观测路径检查共同来源，不计算文件/数组校验和，也不校验旧字段；这不能证明不同路径文件的内容不重复或未被修改。目标数量和已知 group 情况以[共同数据页及其原始汇总来源](../DATA_AND_LABELS.md)为准；不推测缺少的 group 数量。

## 2. 逐阶段数据产物

以下产物由已实现的builder生成，路径相对于其manifest根目录解析。缓存保存数组与独立JSON元数据，使用 `allow_pickle=False` 读取；每次实际加载每份NPZ一次，并检查schema、shape/dtype、有限性、p/n精确对齐和物理质量公式。

| 产物 | 必须包含 | 交给谁 |
|---|---|---|
| 原始观测包 | RGB、米制轴向 depth、实例 mask、K、gravity、有效性、标定身份 | 共用预处理 |
| 实例 manifest | 一行一个实例观测；见下表 | builder、训练/评价 sampler |
| normal/query 缓存 | v3 normal/spec、完整最多256候选、原像素 ID、精确 p/n、格号/轮次/顺序、有效标记 | A/B 共用 |
| source label 缓存 | 每个精确 query 的 S/W/self/SW/INST；物理v2 schema 与身份 | 源预训练、源回放、实例评价 |
| target attempt 表 | 一行一次原始规划动作及结果、支持与排除原因 | 目标 BCE 与同动作评价 |
| 准备报告 | split/group/语义ID与来源路径统计、支持/法向/候选有效率、排除原因、teacher 成本和校验结果 | 训练准入与实验追溯 |

源manifest固定 `schema_version=instance_observation_v2`、`manifest_version=suction20.instance_manifest.v4`；保存sample/instance/capture/group身份、scene/frame/instance_row/camera、split/test_subset、观测路径与可读定义、query_identity原像素ID序列、label路径/teacher_spec/physics、normal/sampler/precision规范、源资产路径与实际标定参数、support/query数量及trainable/status。完整候选数组存于观测NPZ，标签NPZ只包含本次实际求值前缀。`--without-labels` 时 `label_cache=null`，数据读取器不能将其当作有监督训练缓存。

## 3. 实例、观测与 query 对齐

保持相机 x 向右、y 向下、z 向前；深度为 z 轴米制距离。统一 RGB/深度/mask 的校正相机网格和时间，`T_a_b` 表示 b→a。原像素身份固定为 `source_point_id=v*W+u`。相机系 gravity 保存完整3分量及来源；现有目标域为传送带平面法向的近似来源，不能写成已测惯性方向。[坐标依据](../DATA_AND_LABELS.md)

源训练原始有效支持低于当前B默认100点门槛的实例标记 `insufficient_support`，不进入常规采样；验证/测试保留实例身份并纳入失败分母。`mu_vis` 为全部有效可见支持的均值，在候选或图像缩放前计算；它不是物理 COM。

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

attempt 表保存 `attempt_id/sample_id/group_id`、`point_cam_m/normal_out_cam`、`outcome/label_valid`、`pose_source=planned_pipeline_contact`、`actual_execution_verified=false`、`pose_support_valid/exclusion_reason`、原始p/n及已提供的实际坐标变换。

历史 p/n 从原记录以 float64 读取，保留有向符号。使用同实例原始XYZ中 `distance <= 0.005 + BOUNDARY_ROUNDOFF_M` 的最近点验证中心支持，`BOUNDARY_ROUNDOFF_M=1e-12 m` 与A共用，并列取最小 source ID；锚点不替换原动作中心或轴。无锚点标记 `pose_geometry_invalid`，不当成吸取失败。每条动作仅提供一次直接 BCE；不注入候选池，也不向20mm圆盘扩散标签。

## 6. 缓存、成本与准入

缓存定义保存观测/实例语义ID、normal/sampler/precision、query原像素ID、质量/schema、teacher_spec与完整小型physics。直接比较query/label的实际p/n和顺序；dense/Seal资产路径只记录出处，不证明内容版本相同。读取器接受旧v3且忽略旧校验和字段；旧physics缺失时保留null，仅比较双方已知配置，不强制重算已通过实际数组校验的标签，也不在加载时重写旧NPZ。光度增强不改物理标签；改变法向、候选、p/n、物理规则或精度合同须更新身份。旧 S/W 仅在原精确姿态和对应身份不变时可复用。

builder按scene/frame清单生成缓存，已有同身份缓存可复用；训练重复instance draw不重复调用teacher。源train默认只重算前128个query，test必须覆盖完整最多256池。先用显式分片在64–128个覆盖尺寸和dense点数的真实实例测加载、Seal秒/query、包络时间、秒/instance、缓存大小/命中率，再运行完整100-train/90-test构建。分片只是成本和接线检查，不冒充完整选模集。

生产metadata_json版NPZ通过 `load_labels` / `InstanceDataset` 检查实际数组及已知配置。`geometry_contract.py check-manifest` 只核对sample/group/capture声明；`revision_contract.py check-labels`只支持其顶层schema参考demo，不能用来读取生产NPZ，也不重新执行真实teacher。用法见[共同复现命令](../TRAIN_EVAL_REPRODUCE.md)。进入训练前仍须完成真实 p/n 回读、S/W 与冻结 evaluator 对齐、两个 self 分项对齐、`C_old_total=C_self OR C_external` 的独立兼容检查、split 泄漏核验及全部排除统计。

## 7. 可执行的源数据构建

先将变量改为真实资产目录；`DATASET` 下须有官方 `scenes/` RGB、depth、label、meta、`camera_poses.npy`、`cam0_wrt_table.npy` 和完整 `dense_point_clouds`。MAT元数据必须含factor_depth、相机内参、对象身份与姿态；`R20_LABELS` 须含 `config_frozen.json`、`seal_r20/poisson/`。缺完整dense资产时明确失败，不用mesh或可见点云替代teacher的完整目标。

### 7.1 完整自动分片构建

```bash
DATASET=/data/SuctionNet
R20_LABELS=/data/suction20_r20
CACHE=/data/instance_v3
python scripts/build_instance_dataset.py --dataset-root "$DATASET" --label-root "$R20_LABELS" --output "$CACHE/train" --split train --camera realsense --workers 4
python scripts/build_instance_dataset.py --dataset-root "$DATASET" --label-root "$R20_LABELS" --output "$CACHE/test" --split test --camera realsense --workers 4
```

默认train=scene 0–99、test=100–189，均包含frame 0–255。默认深度范围沿用0.2–1.2m，可显式传 `--depth-min-m` / `--depth-max-m`；修改后会进入观测身份，必须对train与selection保持一致。可选 `--camera kinect` 使用对应相机资产，不能把两个相机的结果混成一项。若场景同类对象重复，提供 `--instance-mask-root` 中按instance row标记的真实掩码；类别ID不能充当重复实例的唯一身份。

未显式传入 `--scenes` 和 `--frames` 时，最终产物仍为 `$CACHE/train/manifest_train.jsonl`、对应 `.summary.json` 和test对应文件。`--workers 1` 在该输出目录下写 `observations/<camera>_s0000_f0000_i000.npz` 及 `labels/`；`--workers 4` 按scene分派最多4个独立进程，在最终输出目录的 `shards/part_XXXX/` 下各自再建立scene/frame专属目录，完成后自动合并到最终输出目录。多进程要求显式 `--split train` 或 `--split test`；合并清单保存到worker资产的相对路径，无需移动NPZ，也不要假定中间清单直接位于 `part_XXXX/manifest_train.jsonl`。

**只要显式传入任一 `--scenes` 或 `--frames`，就自动在 `--output` 下建立独立目录**，观测、标签、manifest和summary都放入该目录。以 `--output "$CACHE" --camera realsense --split train` 为例：

| 显式选择 | 最终清单相对于 `$CACHE` 的路径 |
|---|---|
| `--scenes 0-49`，frames省略 | `shards/realsense/train/scenes_0000-0049/frames_0000-0255/manifest_train.jsonl` |
| `--scenes 0-1 --frames 0,8` | `shards/realsense/train/scenes_0000-0001/frames_0000_0008/manifest_train.jsonl` |
| `--scenes 0-99 --frames 0-255` | `shards/realsense/train/scenes_0000-0099/frames_0000-0255/manifest_train.jsonl` |

显式列出全部ID也遵守独立目录规则；实际ID排序、去重后压缩为可读范围，长而稀疏的范围会拆为多级目录，避免单级路径过长。目录名称记录选择范围，不是文件内容摘要。显式子集结合 `--workers N` 时，表中的目录仍是最终发布目录，worker目录位于其内部；返回清单路径不受worker数量影响。

### 7.2 同一输出根的手工分片与完整合并

下面是上述完整自动构建的替代方式。四次构建使用**同一个** `--output "$CACHE"`，每次写入独立目录；可以顺序运行，也可以在不同终端并行运行。每个split的两个分片都成功完成后，再执行该split的合并命令：

```bash
python scripts/build_instance_dataset.py --dataset-root "$DATASET" --label-root "$R20_LABELS" --output "$CACHE" --split train --camera realsense --scenes 0-49
python scripts/build_instance_dataset.py --dataset-root "$DATASET" --label-root "$R20_LABELS" --output "$CACHE" --split train --camera realsense --scenes 50-99
python scripts/build_instance_dataset.py --dataset-root "$DATASET" --label-root "$R20_LABELS" --output "$CACHE" --split test --camera realsense --scenes 100-144
python scripts/build_instance_dataset.py --dataset-root "$DATASET" --label-root "$R20_LABELS" --output "$CACHE" --split test --camera realsense --scenes 145-189

python scripts/merge_instance_manifests.py --manifests "$CACHE/shards/realsense/train/scenes_0000-0049/frames_0000-0255/manifest_train.jsonl" "$CACHE/shards/realsense/train/scenes_0050-0099/frames_0000-0255/manifest_train.jsonl" --output "$CACHE/train" --split train
python scripts/merge_instance_manifests.py --manifests "$CACHE/shards/realsense/test/scenes_0100-0144/frames_0000-0255/manifest_test.jsonl" "$CACHE/shards/realsense/test/scenes_0145-0189/frames_0000-0255/manifest_test.jsonl" --output "$CACHE/test" --split test
```

合并后训练读取 `$CACHE/train/manifest_train.jsonl`，best选择读取 `$CACHE/test/manifest_test.jsonl`，分别完整覆盖100个训练场景和90个测试场景的全部256帧。builder和merger的stdout均为JSON，其 `manifest` 字段是**实际最终清单的绝对路径**；自动化脚本应读取该字段，尤其不要自行拼接稀疏范围或worker中间路径。磁盘summary中的 `manifest` 字段仍为相邻清单的文件名。

### 7.3 发布保护、小规模检查与旧缓存

不同显式范围的串行或并行构建不改写输出根已有的完整清单。每个实际发布目录使用Linux排他文件锁，覆盖该目录的构建/自动合并全过程；手工merger同样持有目标目录锁。另一进程正在写相同目录时立即失败，不排队，也不会开始写该分片的NPZ。进程退出或异常时释放锁，锁文件保留以复用同一inode；不要在任务运行期间删除锁文件。

manifest和summary分别使用唯一临时文件原子替换，清单最后发布。已有清单必须有可读的coverage summary；同名清单的相机或split不符、或者新清单没有包含已有的全部scene/frame覆盖，都会拒绝替换。不能用单个partial分片覆盖已经发布的全量清单；应合并全部分片或换一个输出目录。更换相机或不兼容的观测/teacher配置同样应使用新的缓存目录。

合并器重定位NPZ相对路径并保留全部 `empty_gt_frames`；拒绝输入分片的scene/frame重叠、重复实例、跨split/相机、缺失文件、漏帧和已知physics冲突。默认要求整个split的全部scene×256帧；summary记录实际覆盖、空GT帧、实例状态和 `complete_requested_split`。仅小规模接线可用 `--allow-partial` 合并，训练时仍须另传 `--allow-partial-manifests`，不能把partial输出或改名后的分片声明为完整数据。

小规模构建可加入 `--scenes 0-1 --frames 0,8`，其真实路径见上表；test分片须使用100–189范围内的scene。`--train-queries 64` 是独立预算配置，后续训练配置必须同步；test仍求值最多256。`--without-labels` 可以不传label-root，只生成观测缓存，适用于推理或输入检查，不能用于源监督或完整选模。

旧布局的缓存和清单**不会自动搬迁**，仍可按其原路径读取或传给合并器；新显式分片请求只查找自己的新目录，不会自动从旧输出根查找或复制NPZ。合并也只改写路径引用，不搬迁源资产，因此必须保留被合并清单引用的分片目录。所有检查继续基于schema、实际数组、姿态/标签关系、已知配置和覆盖声明，不生成或验证文件内容摘要。

builder的 `--min-support-points` 默认100，写入观测、清单与summary；A读取模型 `min_support_points`，B当前未暴露该构造参数、训练/评价helper回退100；不要在B YAML中加入未支持的构造参数。数据的 `has_supervision` 只看有效标签，构建时 `trainable` 只作诊断，不能替代运行门槛。

Teacher的可调用接口为：

```python
from suction20.instance_teacher import InstanceTeacher
teacher = InstanceTeacher(dataset_root, label_root, camera="realsense")
labels = teacher.evaluate(scene_id, frame_id, instance_row, points, normals, query_ids)
```

上面Python示例中的路径、scene/frame/instance_row和p/n须由真实观测提供；p/n必须来自对应观测的精确query；它不是将任意同位置旧标签挂接到新轴的快捷入口。

## 8. 目标域RGB-D资产接线

`InstanceAttemptDataset` 直接读取既有 `schema_version=suction20.target.v1`、`label_kind=attempt_success` JSONL。其 `observation` 必须是RGB-D来源，含真实 `rgb_path`、`depth_path`、`instance_mask_path`、`depth_scale_m`、`intrinsics`、`g_cam`，以及原动作姿态、标签和split/group。相对路径以manifest目录为根解析，记录来源路径及sample/group/观测语义ID，不扫描文件字节生成校验和。

旧pointcloud-only记录不具备B原生像素读取所需的信息，会明确报错；需补回该次观测的原RGB-D和mask，不能反向渲染点云伪造原相机纹理。当前未新增一个自动补齐缺失资产的目标builder，也未改动1676/399/87划分。可用历史 `planned_pipeline_contact` 直接评分，不能以最近canonical query替代动作。

## 9. 公开方法比较的数据协议：待定

公开评价“只参数化吸盘半径”的方案当前 **deferred**；不生成名为已匹配公开协议的release，不把R10等价门禁或分割前端下载作为本轮开训条件。论文基线数值与来源继续保存在[实验方案](EXPERIMENT_PLAN.md)和[公共比较参考](../PUBLIC_METHOD_COMPARISON.md)，B的对应值保持待测。

现有GT实例source test评价是 `test_selected` 实例物理评价，不能冒充全景预测实例AP。未来若恢复公开对比，仍需逐方法确认同一测试集、相机、指标、前端/工作区权限、完整帧分母和半径差异，并显式披露本次test选模行为；不能仅因scene范围相同就宣称除吸盘外完全一致。UISN的GT联合前景工作区不能自动扩展为S1B或Diffusion的权限，预测实例漏检、误检与无姿态也不能从完整测试分母删除。
