# A 数据准备：实例、历史动作与同姿态物理标签

本文定义进入 A 预处理和训练前应形成的数据资产。依据为[共同数据合同](../DATA_AND_LABELS.md)、[A v3 设计](../ROUTE_A.md)和[设计 JSON](../reference/route_a.json)。共同实例builder、RGB-D数据读取和同姿态teacher已由 `scripts/build_instance_dataset.py`、`suction20/instance_data.py` 与 `suction20/instance_teacher.py` 实现。合成资产接线已验证，真实完整数据重算与成本核验尚待运行。

## 1. 输入资产与本轮划分

源域输入包括 SuctionNet 配准 RGB-D、GT 实例掩码、内参、相机/场景/对象位姿、完整对象 `dense_point_clouds`、原 R20 Seal 候选及冻结物理配置。完整对象几何和原候选均值仅供 teacher/评价，模型只接收目标实例的当前可见观测。

本轮从头直接使用source全部100个训练场景、每场景256帧；不再90/10拆分或追加final refit。best仅由source test的固定小部分选择：每个test场景取8帧。必须记录 `test_selected`，选择scope是test subset，不能视为独立测试估计。先按scene划分，再展开相机视角与实例：

| 集合 | scene | frame/view |
|---|---|---|
| train | 0–99全部100场景 | 0–255全部可抽样，按固定sampler取样 |
| selection / test subset | 100–189全部90场景 | 每场景固定8帧：0、32、64、96、128、160、192、224 |
| 完整test / seen | 100–129 | 每场景256个；训练后报告 |
| 完整test / similar | 130–159 | 每场景256个；训练后报告 |
| 完整test / novel | 160–189 | 每场景256个；训练后报告 |

源manifest的 `split` 为 `train/test`，没有源val；测试另存 `test_subset=seen|similar|novel`，selection也保持 `split=test`。默认选择集是90×8=720帧，而完整source test是23040帧。前者用于训练过程中的best更新，后者仅在训练后对固定best作报告，不重新选模。两者的标签都不进入梯度更新，但选择行为本身已使用test信息，因此全量报告也不是独立测试估计。同帧实例不跨split，这也不等于CAD身份互斥划分。源训练有效原始支持低于模型min_support_points（默认100）的实例记录 `insufficient_support`；在所选评价帧和完整测试中均保留其身份与无输出结果，计入各自指标分母。

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
| 实例 manifest | `schema_version=instance_observation_v2`、`manifest_version=suction20.instance_manifest.v4`、sample/instance/capture/group ID、split、source_kind |
| 观测资产 | RGB、米制轴向 depth、实例 mask、K、完整 gravity_cam 及有效性；RGB/深度在同一校正网格 |
| 身份 | 观测/标定来源路径与实际参数、normal/sampler/precision显式规范、有序query原像素ID、teacher_spec与小型physics字典 |
| query 索引 | `query_cache`；source ID、原像素、格号、轮次、query_id、精确 p/n 与 query_valid |
| label 索引 | 源域 `label_cache`；目标历史动作没有物理真值时为空，不能补零 |
| 历史动作 | attempt/sample/group ID、原相机系 p/n、outcome、label_valid、pose_source、actual_execution_verified |
| 支持和变换 | `pose_support_valid`、排除原因、原始p/n及已提供的实际变换；与outcome有效性分开 |

相机坐标采用 x向右、y向下、z向前；深度为相机 z 轴距离。使用更新后的 K 反投影，长度单位为米。历史姿态用原标定变换到相机系，以 float64 读取并保留有向轴；最近点只可验证支持，不能吸附记录中心。当前重力是负传送带平面法向近似，来源标记随样本保留。

源builder在最终发布目录写manifest/summary，单进程观测与完整query数组在该目录的 `observations/`，实际求值的标签前缀在 `labels/`；显式scene/frame选择会自动创建独立发布目录，多进程由合并清单引用worker资产。目标动作沿用既有JSONL。路径相对manifest根解析，NPZ为数组和JSON元数据、`allow_pickle=False`加载，每次实际加载每份NPZ一次，核对结构、p/n与标签公式；产物路径与CLI见第7节。读取器同时接受旧v3，忽略旧校验和字段，不计算文件/数组校验和，不作内容篡改检测保证。

## 3. 按实际需要构建缓存

1. 核对源资产及目标原始记录，固定split/group、语义样本ID、标定与规范化来源路径；输出接受与排除清单。
2. 按[预处理方案](PREPROCESSING.md)生成 v3 法向和完整最多256 query序列。源训练/回放取前128；验证、测试使用固定完整池；历史动作另存精确 p/n。
3. builder按scene/frame清单生成可复用缓存。train默认对前128 query求值，selection与完整test均为每实例完整最多256；重复draw不重算teacher。先核验小规模成本，再准备100个train场景全部帧和90个test场景各固定8帧的正式选择清单；完整90场景×256帧test缓存用于训练后的全量报告。
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

这是 `InstanceTeacher.evaluate` 已实现的运算顺序；p_object/n_object由精确相机系p/n按已校验变换得到，raw_seal须经冻结归一化后才成为S20标签。旧 `evaluate_poses`仍计算混合碰撞，可核对 S/W，但不能直接返回新 self分项。冻结约定包括：

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

缓存定义保存观测/实例语义ID、normal/sampler/precision、query原像素ID、质量/schema、teacher_spec与完整小型physics。加载时直接比较实际query/label的p/n和顺序；dense/Seal文件路径只记录出处，不证明文件内容相同。旧缓存缺physics时保留null，仅比较双方都已知的配置，不要求重算已通过真实数组校验的标签；加载不会重写旧NPZ。几何正确性基线从原深度/内参构造float64，之后才转网络FP32；已经量化的FP32 XYZ升精度不能恢复原数值。光度增强不改变标签，任何p/n变更必须重新匹配缓存身份。

## 6. 可运行检查与完成条件

生产缓存直接用真实加载器检查，替换下面manifest路径：

```python
from suction20.instance_data import InstanceDataset
source = InstanceDataset('/data/instance_v3/train/manifest_train.jsonl',
                         split='train', training_queries=128)
item = source[0]  # 真实数组、p/n顺序、标签公式与已知配置；无校验和读取
```

`geometry_contract.py check-manifest` 只检查其sample/group/capture声明；`revision_contract.py check-labels` 只读取其顶层schema参考demo，不能用于生产metadata_json版NPZ。生产标签用 `load_labels` / `InstanceDataset`。合成检查不替代真实完整dense资产上的同姿态核验；builder/teacher命令见下节。

完成本阶段需有：固定清单及排除账本；scene/group/sample/规范化观测路径不跨split；标定与原姿态回读报告；真实同p/n的S/W对齐；self两分项与原目标部分对齐；独立外部项与self合并可重建旧总C；错误法向/姿态/teacher身份的缓存被拒绝；正式训练所需缓存预算与实测成本报告。

## 7. 可执行的源数据构建

先将变量改为真实资产目录；`DATASET` 下须有官方 `scenes/` RGB、depth、label、meta、`camera_poses.npy`、`cam0_wrt_table.npy` 和完整 `dense_point_clouds`。MAT元数据必须含factor_depth、相机内参、对象身份与姿态；`R20_LABELS` 须含 `config_frozen.json`、`seal_r20/poisson/`。缺完整dense资产时明确失败，不用mesh或可见点云替代teacher的完整目标。

### 7.1 开训前：全量train与固定8帧selection

先构建较小的selection缓存并读取stdout中的实际清单路径，再准备完整训练清单：

```bash
set -euo pipefail
DATASET=/data/SuctionNet
R20_LABELS=/data/suction20_r20
CACHE=/data/instance_v3
SELECTION_FRAMES=0,32,64,96,128,160,192,224
mkdir -p "$CACHE"
python scripts/build_instance_dataset.py --dataset-root "$DATASET" --label-root "$R20_LABELS" --output "$CACHE/selection" --split test --camera realsense --frames "$SELECTION_FRAMES" --workers 4 > "$CACHE/selection_build.json"
SELECTION_MANIFEST=$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["manifest"])' "$CACHE/selection_build.json")
python scripts/build_instance_dataset.py --dataset-root "$DATASET" --label-root "$R20_LABELS" --output "$CACHE/train" --split train --camera realsense --workers 4
```

训练命令使用 `--source-manifest "$CACHE/train/manifest_train.jsonl" --selection-manifest "$SELECTION_MANIFEST"`。显式 `--frames` 会生成独立目录，不能把selection路径假定为 `$CACHE/selection/manifest_test.jsonl`；保存的stdout JSON用于下一次shell会话重新读取真实路径。

默认 `pretrain.selection_frames_per_scene=8`：对0–255帧按8位bit-reversal顺序取固定前缀，再按帧ID排序得到 `0,32,64,96,128,160,192,224`。每个test场景都用这8帧，共90×8=720帧；16帧配置包含这8帧，256帧配置才是全量。候选查询预算不变：所选帧中的**全部可见实例**均保留，每实例仍用完整最多256个query；空GT帧以覆盖声明计入，不能漏掉，也不能删去无输出实例。

默认8750步预训练每 `pretrain.evaluate_every=875` 步在同一固定子集评价一次，共10次；只有这套子集指标更新 `best.pt`。source仍使用0–99全部100场景、每场景256帧。selection可传完整test manifest由训练入口自动筛选，也可只构建上述720帧的选择清单；只要覆盖全部90场景的固定所选帧，两者都不需要 `--allow-partial-manifests`。只含8帧的summary可标 `complete_requested_split=false`，这表示没有覆盖整个test split，并不使已完整覆盖约定帧的正式selection退化为smoke。 新版配置显式设置8；旧配置若缺少 `selection_frames_per_scene`，为兼容旧运行仍按256处理，使用本轮预算时须保留该字段。

默认深度范围沿用0.2–1.2m，可显式传 `--depth-min-m` / `--depth-max-m`；修改后会进入观测身份，必须对train与selection保持一致。可选 `--camera kinect` 使用对应相机资产，不能把两个相机的结果混成一项。若场景同类对象重复，提供 `--instance-mask-root` 中按instance row标记的真实掩码；类别ID不能充当重复实例的唯一身份。

### 7.2 训练后：完整test构建与报告

完整test资产可以在需要最终报告时再构建，开训前不必完成90场景×256帧的teacher重算：

```bash
python scripts/build_instance_dataset.py --dataset-root "$DATASET" --label-root "$R20_LABELS" --output "$CACHE/test" --split test --camera realsense --workers 4
```

训练结束后固定已由小selection选出的 `best.pt`，将 `$CACHE/test/manifest_test.jsonl` 交给 `scripts/evaluate_instance.py --domain source --split test` 生成全量23040帧报告；具体checkpoint和output参数见[运行说明](../IMPLEMENTATION.md)。该独立CLI不更新best、不再比较候选checkpoint。因为选择子集来自test，完整test报告仍保留 `test_selected`，不能称为独立测试估计。

### 7.3 输出目录与手工完整分片

未显式传入 `--scenes` 和 `--frames` 时，最终产物仍为 `$CACHE/train/manifest_train.jsonl`、对应 `.summary.json` 和test对应文件。`--workers 1` 在该输出目录下写 `observations/<camera>_s0000_f0000_i000.npz` 及 `labels/`；`--workers 4` 按scene分派最多4个独立进程，在最终输出目录的 `shards/part_XXXX/` 下各自再建立scene/frame专属目录，完成后自动合并到最终输出目录。多进程要求显式 `--split train` 或 `--split test`；合并清单保存到worker资产的相对路径，无需移动NPZ，也不要假定中间清单直接位于 `part_XXXX/manifest_train.jsonl`。

**只要显式传入任一 `--scenes` 或 `--frames`，就自动在 `--output` 下建立独立目录**，观测、标签、manifest和summary都放入该目录。以 `--output "$CACHE" --camera realsense --split train` 为例：

| 显式选择 | 最终清单相对于 `$CACHE` 的路径 |
|---|---|
| `--scenes 0-49`，frames省略 | `shards/realsense/train/scenes_0000-0049/frames_0000-0255/manifest_train.jsonl` |
| `--scenes 0-1 --frames 0,8` | `shards/realsense/train/scenes_0000-0001/frames_0000_0008/manifest_train.jsonl` |
| `--scenes 0-99 --frames 0-255` | `shards/realsense/train/scenes_0000-0099/frames_0000-0255/manifest_train.jsonl` |

显式列出全部ID也遵守独立目录规则；实际ID排序、去重后压缩为可读范围，长而稀疏的范围会拆为多级目录，避免单级路径过长。目录名称记录选择范围，不是文件内容摘要。显式子集结合 `--workers N` 时，表中的目录仍是最终发布目录，worker目录位于其内部；返回清单路径不受worker数量影响。

#### 同一输出根的手工分片与完整合并

下面是完整train与完整test自动构建的替代方式，完整test部分可留到训练后执行；固定8帧selection仍按第7.1节单独准备。四次构建使用**同一个** `--output "$CACHE"`，每次写入独立目录；可以顺序运行，也可以在不同终端并行运行。每个split的两个分片都成功完成后，再执行该split的合并命令：

```bash
python scripts/build_instance_dataset.py --dataset-root "$DATASET" --label-root "$R20_LABELS" --output "$CACHE" --split train --camera realsense --scenes 0-49
python scripts/build_instance_dataset.py --dataset-root "$DATASET" --label-root "$R20_LABELS" --output "$CACHE" --split train --camera realsense --scenes 50-99
python scripts/build_instance_dataset.py --dataset-root "$DATASET" --label-root "$R20_LABELS" --output "$CACHE" --split test --camera realsense --scenes 100-144
python scripts/build_instance_dataset.py --dataset-root "$DATASET" --label-root "$R20_LABELS" --output "$CACHE" --split test --camera realsense --scenes 145-189

python scripts/merge_instance_manifests.py --manifests "$CACHE/shards/realsense/train/scenes_0000-0049/frames_0000-0255/manifest_train.jsonl" "$CACHE/shards/realsense/train/scenes_0050-0099/frames_0000-0255/manifest_train.jsonl" --output "$CACHE/train" --split train
python scripts/merge_instance_manifests.py --manifests "$CACHE/shards/realsense/test/scenes_0100-0144/frames_0000-0255/manifest_test.jsonl" "$CACHE/shards/realsense/test/scenes_0145-0189/frames_0000-0255/manifest_test.jsonl" --output "$CACHE/test" --split test
```

合并后训练读取 `$CACHE/train/manifest_train.jsonl`，训练后全量报告读取 `$CACHE/test/manifest_test.jsonl`，分别完整覆盖100个训练场景和90个测试场景的全部256帧。若把完整test清单传给训练的selection入口，仍会按配置筛出固定8帧；不会因清单较大自动改为全量选模。builder和merger的stdout均为JSON，其 `manifest` 字段是**实际最终清单的绝对路径**；自动化脚本应读取该字段，尤其不要自行拼接稀疏范围或worker中间路径。磁盘summary中的 `manifest` 字段仍为相邻清单的文件名。

### 7.4 发布保护、小规模检查与旧缓存

不同显式范围的串行或并行构建不改写输出根已有的完整清单。每个实际发布目录使用Linux排他文件锁，覆盖该目录的构建/自动合并全过程；手工merger同样持有目标目录锁。另一进程正在写相同目录时立即失败，不排队，也不会开始写该分片的NPZ。进程退出或异常时释放锁，锁文件保留以复用同一inode；不要在任务运行期间删除锁文件。

manifest和summary分别使用唯一临时文件原子替换，清单最后发布。已有清单必须有可读的coverage summary；同名清单的相机或split不符、或者新清单没有包含已有的全部scene/frame覆盖，都会拒绝替换。不能用单个partial分片覆盖已经发布的全量清单；应合并全部分片或换一个输出目录。更换相机或不兼容的观测/teacher配置同样应使用新的缓存目录。

合并器重定位NPZ相对路径并保留全部 `empty_gt_frames`；拒绝输入分片的scene/frame重叠、重复实例、跨split/相机、缺失文件、漏帧和已知physics冲突。默认要求整个split的全部scene×256帧；summary记录实际覆盖、空GT帧、实例状态和 `complete_requested_split`。通用merger的 `--allow-partial` 可合并显式成本/接线分片，也可合并只包含固定8帧的selection分片；是否可供正式选模，要再按全部90场景及配置要求的帧集合检查。正式selection覆盖齐全时训练不加 `--allow-partial-manifests`；真正缺少所需scene/frame的接线任务才需该标志，不能把它登记为正式选择集或完整test报告。

小规模构建可加入 `--scenes 0-1 --frames 0,8`，其真实路径见上表；test分片须使用100–189范围内的scene。`--train-queries 64` 是独立预算配置，后续训练配置必须同步；test仍求值最多256。`--without-labels` 可以不传label-root，只生成观测缓存，适用于推理或输入检查，不能用于源监督或正式选模。

旧布局的缓存和清单**不会自动搬迁**，仍可按其原路径读取或传给合并器；新显式分片请求只查找自己的新目录，不会自动从旧输出根查找或复制NPZ。合并也只改写路径引用，不搬迁源资产，因此必须保留被合并清单引用的分片目录。所有检查继续基于schema、实际数组、姿态/标签关系、已知配置和覆盖声明，不生成或验证文件内容摘要。

builder的 `--min-support-points` 默认100，写入观测、清单与summary；训练/评价实际读取模型的 `model.min_support_points`，默认同为100。数据的 `has_supervision` 只看有效标签，构建时 `trainable` 只作诊断；变更阈值应同时记录两者，不能用缓存布尔值阻止模型使用新的阈值。

Teacher的可调用接口为：

```python
from suction20.instance_teacher import InstanceTeacher
teacher = InstanceTeacher(dataset_root, label_root, camera="realsense")
labels = teacher.evaluate(scene_id, frame_id, instance_row, points, normals, query_ids)
```

上面Python示例中的路径、scene/frame/instance_row和p/n须由真实观测提供；p/n必须来自对应观测的精确query；它不是将任意同位置旧标签挂接到新轴的快捷入口。

## 8. 目标域RGB-D资产接线

`InstanceAttemptDataset` 直接读取既有 `schema_version=suction20.target.v1`、`label_kind=attempt_success` JSONL。其 `observation` 必须是RGB-D来源，含真实 `rgb_path`、`depth_path`、`instance_mask_path`、`depth_scale_m`、`intrinsics`、`g_cam`，以及原动作姿态、标签和split/group。相对路径以manifest目录为根解析，记录来源路径及sample/group/观测语义ID，不扫描文件字节生成校验和。

旧pointcloud-only记录不具备共同organized法向、候选及RGB接线所需的信息，会明确报错；需补回该次观测的原RGB-D和mask，不能反向渲染点云伪造原相机纹理。当前未新增一个自动补齐缺失资产的目标builder，也未改动1676/399/87划分。可用历史 `planned_pipeline_contact` 直接评分，不能以最近canonical query替代动作。

## 9. 公开方法比较的数据协议：待定

公开评价“只参数化吸盘半径”的方案当前 **deferred**；不生成名为已匹配公开协议的release，不把R10等价门禁或分割前端下载作为本轮开训条件。论文基线数值与来源继续保存在[实验方案](EXPERIMENT_PLAN.md)和[公共比较参考](../PUBLIC_METHOD_COMPARISON.md)，A的对应值保持待测。

现有GT实例source test评价是 `test_selected` 实例物理评价，不能冒充全景预测实例AP。未来若恢复公开对比，仍需逐方法确认同一测试集、相机、指标、前端/工作区权限、完整帧分母和半径差异，并显式披露本次test选模行为；不能仅因scene范围相同就宣称除吸盘外完全一致。UISN的GT联合前景工作区不能自动扩展为S1B或Diffusion的权限，预测实例漏检、误检与无姿态也不能从完整测试分母删除。
