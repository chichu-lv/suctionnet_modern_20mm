# 公开数据适配器与批数据接口

> 状态：拟实施规格。本文规定后续 `suction20/datasets.py` 的行为；当前仓库已经实现数据生成，尚未实现本文的数据加载器、体素批处理器及模型接口。源码核查覆盖旧数据方法 `dc4ad6b00535c89ece1feef2751ba86006e5f97e` 和当前仓库 `b2680fb47fe6f6a99fcb310e3428c77de1f55406`。

本阶段使用已经生成的公开 SuctionNet R20 标签进行预训练。后续少量自有场景由 `suction20/target_dataset.py` 接入相同的观测输入接口；公开物理代理分数和真机成败标签保留各自语义，详见 [预训练策略](TRAINING.md) 与 [自有场景微调](FINETUNING.md)。

**默认复用用户已经用旧 `dc4ad6b` 方法完整制作的数据。** `source_format=legacy_dc4ad6b` 时，适配器保留现有 S/W/C、GT-A/GT-B，只在读取时恢复原始候选轴并计算缺失的观测法向。缺少 `r20-v2.1` 新字段不是重新制作数据的理由，也不是训练入口的硬错误。具体派生公式、字段决策和可选缓存见 [已有数据调整方案](LEGACY_DATA_ADJUSTMENT.md)。

配置同时写 `compatibility: auto` 时，其含义是：遇到明确标记 `schema_version=r20-v2.1` 的源文件，选新格式语义；没有这个新 schema 的既有文件，使用 `source_format=legacy_dc4ad6b` 作为默认来源说明。`source_format` 在此模式下不是强制覆盖明确 schema 的开关。派生 sidecar 的 `adapter_schema` 只说明派生缓存，不能把原始旧文件误判为新生成器输出。两条路径最终输出下文同一接口。

本适配层只读取、组织和增强已有数据，不修改 Seal/Wrench/Collision 的公式、候选索引或 G0-G2 数据生产流程。数据生成字段的简表见 [DATA_SCHEMA.md](../DATA_SCHEMA.md)，运行方法见 [DATA_PIPELINE.md](../DATA_PIPELINE.md)。

## 1. 与上一版数据接口的实际差异

当前生成器新写出的逐帧文件使用 `schema_version=r20-v2.1`；用户已有旧文件仍可能保存 `config_version` 和 `seal_generation_hash` 而没有新 schema 字段。两种来源进入同一个训练接口，不能因为字段同名就认定主法向语义相同。

| 项目 | 当前实现 | 本适配器默认做法 |
|---|---|---|
| 主轴标签 | `gt_normal` 已是生成质量标签时使用的原始候选外法向 | 直接读取作为 `normal_label` |
| PCA 法向 | 独立为 `gt_normal_pca` / `normal_pca_valid` | 主模型不使用；留给独立几何对比 |
| 观测法向 | 已保存 `normals_raw` / `normals_raw_valid` | 直接读取作为 `normal_obs` / `normal_obs_valid` |
| 主法向存储掩码 | `normal_valid` 仍按“无碰撞且归一化 Seal > 0.2”筛选 | 不当作纯几何有效性；训练另派生 `axis_valid` |
| 版本字段 | 保存 `schema_version` | 不要求旧版 `config_version` 或 `seal_generation_hash` 存在 |
| 场景元信息 | `scene_meta_r20` 保存物体行号、物体 ID、数量、统计及几何配置 | 不假定其中保存了物体变换矩阵 |

上述实现见 [`generate_view()`](../../suction20/full_data.py) 和 [`compute_scene()`](../../suction20/full_data.py)。其中 `gt_normal` 在所有成功匹配点上写入，不受 `normal_valid` 的高质量筛选限制。因此零 Seal 或碰撞候选也有可用的原始几何轴。

对于默认旧数据入口，上表中新增加的 `normals_raw` 等字段由适配器派生，不要求源 NPZ 存在；旧 `gt_normal` 可能是 PCA 方向，因此始终按候选索引恢复 `normal_label`。旧源字段留作原始记录，不原位改写。

## 2. 文件读取与元信息来源

适配器接收两个根目录和一个明确的相机名：

```text
dataset_root/                         # 官方 RGB-D、相机内外参、物体模型
  scenes/scene_SSSS/<camera>/
    depth/FFFF.png
    label/FFFF.png
    meta/FFFF.mat
    camera_poses.npy
    cam0_wrt_table.npy

label_root/                           # 本仓库已经生成的 R20 输出
  config_frozen.json
  scene_meta_r20/SSSS_<camera>.json
  seal_r20/poisson/OOO_labels.npz
  scenes/scene_SSSS/<camera>/graspness_r20_v2/FFFF.npz
```

`<camera>` 默认 `realsense`，保持与既有数据制作配置一致。使用新格式或已有派生缓存时，一次样本读取只需逐帧 NPZ、已缓存的相机变换及图像尺寸；无增强时无需再次估计法向。默认旧数据入口还读取原始深度与内参以获得观测法向，读取物体候选数组以恢复标签轴；这些读取可按场景/物体缓存，不触发物理重标注。

`label_root` 指向用户已经完成的数据目录，其 `config_frozen.json` 是半径、包络、深度范围、归一化参数的权威来源。使用普通 JSON 读取并按需要提取字段；不能把旧冻结配置交给当前 `config_from_file()` 强制校验，后者已移除旧 `wrench_threshold_factor` 键，会把合法旧配置误报为未知字段。模型配置与数据配置分开保存。

当前场景 JSON 的 `objects` 实际结构为：

```python
{
    "scene_id": 0,
    "camera": "realsense",
    "objects": [
        {"row": 0, "obj_id": 3, "n": 1234,
         "collision_ratio": ..., "wrench_eval_mean": ...,
         "wrench_annot_mean": ...},
        # 后续实例按相同场景行号组织
    ],
    "schema_version": "r20-v2.1",
    "radius_m": 0.020,
    "envelope_segments_m": [...],
    "wrench_schema": "annot_eval_dual_v1",
    # 其余为公式和坐标来源的说明
}
```

上例中的数值只说明结构，不代表实际运行结果。`objects` 的行号字段是 `row`，并没有 `T_world_obj`。需要物体变换时，调用现有 `scene_frame0(dataset_root, scene_id, camera)`；该函数由原始 `meta/0000.mat`、`camera_poses.npy` 和 `cam0_wrt_table.npy` 计算，返回带 `instance_row`、`obj_id`、`T_world_obj` 的记录。不要混用这两种记录。

## 3. 逐点字段、单位和索引

令一帧保留的观测点数为 `N`。默认保留文件中的完整观测点云；背景及邻物也进入场景主干。下表的浮点输入统一转为 `float32`，张量索引在进入 PyTorch 后使用 `int64`，布尔掩码使用 `bool`。

| 原字段 | 形状 | 适配后用途 | 语义 |
|---|---|---|---|
| `xyz` | `[N,3]` | `xyz_cam_m` | 相机坐标，单位 m；不按物体大小归一化 |
| `normals_raw`（新格式；旧数据派生） | `[N,3]` | `normal_obs` | 来自观测深度；朝向相机；无效行输入置零 |
| `normals_raw_valid`（新格式；旧数据派生） | `[N]` | `normal_obs_valid` | 当前观测法向是否可用；增强后重新计算 |
| `instance_id` | `[N]` | 默认 `instance_id_input`，另保留 `instance_id_gt` | 场景物体列表的一基行号；0 是背景 |
| `objectness` | `[N]` | 默认 `foreground_input` 的核对项 | 原始 `class_id > 0` |
| `class_id` | `[N]` | 元信息/物体文件定位 | 官方物体 ID 加 1；不作为类别 embedding 输入 |
| `source_indices` | `[N]` | `source_indices` | 原图行优先展平索引，增强/采样后仍指向原像素 |
| `candidate_row` | `[N]` | 监督侧索引 | 当前物体 Seal 数组零基下标；未匹配为 -1 |
| `candidate_id` | `[N]` | 监督侧追溯信息 | 原 mesh 顶点 ID，不能代替 `candidate_row` |
| `match_distance` | `[N]` | 监督侧诊断 | m；未匹配为 `inf`；不输入网络 |
| `gt_valid` | `[N]` | 质量监督掩码来源 | 匹配成功才有已评价标签；False 不代表失败 |
| `gt_seal_norm` | `[N]` | `seal_target` | 当前冻结归一化后的 `S` |
| `gt_wrench_eval` | `[N]` | `wrench_target` | 当前主 Wrench；`gt_wrench` 只是同值别名 |
| `collision_flag` | `[N]` | `collision_free_target` | True 是碰撞，故目标 `C = ~collision_flag` |
| `gt_A` / `gt_B` | `[N]` | `gt_A` / `gt_B` | `A = S*C`；`B = S*W*C` |
| `gt_normal` | `[N,3]` | 新格式直接作为 `normal_label`；旧数据按候选重建 | 旧文件可能是 PCA，不能直接假定是原始轴 |
| `normal_valid` | `[N]` | 保留原字段，不用于默认轴损失 | 新格式为成功匹配、无碰撞、`S > 0.2`；旧格式还依赖 PCA extras |

`gt_seal_raw`、`gt_wrench_annot`、`gt_orient`、`gt_seal_custom` 和 PCA 两字段可留在监督/诊断侧，不属于主网络输入。主 `W` 使用默认 1 kg 和物体全部候选点的几何均值形成的抗力矩代理，并不是已测量的物体重量、质心或真机成功概率。

索引必须按以下关系使用：

```python
scene_row = instance_id_gt[i] - 1         # 场景实例行，定位物体
seal_row = candidate_row[i]            # 该物体候选数组行
mesh_vertex_id = candidate_id[i]       # 只作追溯

object_record = scene_records[scene_row]
seal = load_seal_for_object(object_record["obj_id"])
normal_object = seal["normals"][seal_row]
```

`candidate_row` 不能用于访问 `scene_records`；`instance_id` 不能直接当物体类别 ID；`candidate_id` 不能直接访问 Seal 数组。候选身份的完整键是 `(scene_id, instance_id_gt, candidate_row)`，逐帧观测身份还包含 `camera`、`frame_id` 和 `source_indices`。

## 4. 四类掩码必须分开

```python
foreground_gt = instance_id_gt > 0
foreground_input = instance_id_input > 0

quality_valid = gt_valid & foreground_gt

axis_label_valid = (
    quality_valid
    & isfinite(normal_label).all(axis=-1)
    & (norm(normal_label, axis=-1) > 1e-6)
)
axis_valid = axis_label_valid & normal_obs_valid

candidate_eligible = foreground_input & normal_obs_valid
```

`normal_obs_valid` 还应包含观测向量有限且非零的判断，并基于**当前增强后的观测**计算。`axis_valid` 不额外要求 `S > 0.2`、无碰撞或原 `normal_valid=True`。原 `normal_valid` 字段保持原样；新掩码使用新名称，不覆盖它。

| 掩码 | 控制对象 | 能否参与推理筛选 |
|---|---|---|
| `quality_valid` | q0/q、S/W/C 的监督点 | 不能；推理没有这个 GT |
| `axis_valid` | 法向残差损失 | 不能；包含标签有效性 |
| `candidate_eligible` | 候选中心 | 可以；只来自观测和前置实例 |
| `normal_pca_valid` | 可选 PCA 对比 | 主模型不使用 |

未匹配点的质量数值为 0、`collision_flag=False` 只是存储占位。所有质量头，包括 Collision BCE，都用 `quality_valid`。有效的零质量点必须保留，不能因为 `gt_B == 0` 而删掉。观测法向无效的点可继续作为场景上下文；其法向输入置零，并保留有效性通道。

默认公开数据预训练用 GT 实例掩码作为前置分割输入，因此 `instance_id_input == instance_id_gt`。评估已有前置模型时，两者必须分开：预测实例编号供池化和候选生成，GT 编号供损失与评价。预测 ID 不要求与 GT ID 数值一致，不能直接覆盖 GT 归属。预测漏分的 GT 点可以失去候选资格，但不从评价目标实例集合中消失。

## 5. 主轴与重力：直接读取和等价重建

### 5.1 按明确的数据来源选择读取路径

```python
if source_format == "legacy_dc4ad6b":      # 用户既有数据，默认
    normal_label = recover_candidate_axis(frame, scene_records, seal_cache, T_cw)
    normal_obs, normal_obs_valid = read_or_derive_observed_normals(frame, key)
elif source_format == "r20_v2_1":
    normal_label = np.asarray(frame["gt_normal"], np.float32)
    normal_obs = np.asarray(frame["normals_raw"], np.float32)
    normal_obs_valid = np.asarray(frame["normals_raw_valid"], bool)
```

这里的 `source_format` 是经过上文 `compatibility:auto` 解析后的有效来源。新格式若没有持久化某个观测派生字段，也可使用相同 RGB-D 派生路径；不能只因缺少观测缓存就要求重跑物理标签。

有效的标签轴归一化为单位向量，无效占位保持零。不要朝相机重新翻转标签轴来掩盖符号问题；其向外方向已与物理标签一致。观测法向也使用当前生成器的朝相机约定。预测外法向记为 `n_hat`，朝表面运动的接近方向为 `-n_hat`。

### 5.2 原始候选轴的等价恢复公式

下面是默认旧数据入口采用的读取公式，也是当前新格式主轴标签的来源。物体候选和场景位姿在 worker 内缓存，计算只涉及索引与旋转。新 `r20-v2.1` 文件不需要重复执行该恢复。

```python
from suction20.full_data import scene_frame0, load_seal, to_unit

cam_root = dataset_root / "scenes" / f"scene_{scene_id:04d}" / camera
T_wc = (
    np.load(cam_root / "cam0_wrt_table.npy")
    @ np.load(cam_root / "camera_poses.npy")[frame_id]
)
T_cw = np.linalg.inv(T_wc)
records = scene_frame0(dataset_root, scene_id, camera)

normal_label = np.zeros_like(frame["xyz"], dtype=np.float32)
for rec in records:
    instance = rec["instance_row"] + 1
    pix = np.flatnonzero(frame["gt_valid"] & (frame["instance_id"] == instance))
    row = frame["candidate_row"][pix]
    seal = load_seal(seal_root / f'{rec["obj_id"]:03d}_labels.npz')
    T_co = T_cw @ rec["T_world_obj"]
    normal_label[pix] = to_unit(seal["normals"][row] @ T_co[:3, :3].T)
```

这里的 `rec` 来自 `scene_frame0()`，不是直接读取 `scene_meta_r20["objects"]` 得到的项。该函数内部采用 `T_world_cam0 = cam0_wrt_table @ camera_poses[0]`，再乘 frame-0 的物体相机位姿。上式与 `generate_view()` 写入 `gt_normal` 的计算相同。

旧格式若曾把 PCA 写入 `gt_normal`，不能靠字段同名推断轴语义。本轮直接选用上述恢复路径复用已完成的数据，不重新计算 Seal、Wrench 或 Collision。恢复结果可以只存在于训练样本内，也可以保存为独立派生缓存。

### 5.3 重力与坐标增强

世界/桌面系单位重力方向取 `[0,0,-1]`。相机系向量为：

```python
g_cam = T_wc[:3, :3].T @ np.array([0., 0., -1.])
```

向量只乘旋转，不加平移。`g_cam` 按帧存为 `[3]`，batch 中为 `[B,3]`。若做坐标系刚体增强 `p_new = p @ R.T + t`，则 `normal_obs`、`normal_label` 和 `g_cam` 同时乘 `R.T`；标量质量标签保持不变。不做随机尺度缩放，20 mm 半径和现有物理标签的尺度保持固定。

## 6. 像素对齐、观测法向与增强顺序

`source_indices` 遵循原深度图的行优先布局：

```python
H, W = depth_image.shape
v = source_indices // W
u = source_indices % W
assert np.all((0 <= source_indices) & (source_indices < H * W))

instance_id_input = predicted_instance_image.reshape(-1)[source_indices]
# 若实验使用 RGB：
rgb_points = rgb_image.reshape(H * W, 3)[source_indices]
```

评价归属用源 NPZ 中 `source_indices → instance_id` 的对应关系；原官方 `label/*.png` 存的是 `class_id`，不能把它的像素值直接当场景实例行号。评价器需要零基实例行时，使用 `gt_instance_row = instance_id_gt - 1`，背景 0 自然对应 -1。

当前 NPZ 未保存 `image_shape`、相机内参或深度尺度。`H,W` 从对应原图读取；`intrinsic_matrix` 和 `factor_depth` 从同一帧 `meta/FFFF.mat` 读取，不固定假定 720×1280。当前模型不需要 RGB，只有实际开启 RGB 实验时才加载颜色。

已有 `read_frame()` 的观测法向来自组织化深度差分：对邻域左右/上下点作差，求叉积并归一化，排除无效深度、退化叉积及超过 10 mm 的深度跳变，最后使法向朝向相机。它返回：

```python
xyz, normals, normal_ok, class_ids, source_indices, image_shape
```

不做几何增强时，新格式直接复用 `normals_raw`；旧格式使用同一 `read_frame()` 规则计算并按 `source_indices` 对齐，或读取已经生成的派生缓存。若加入轻量深度噪声或点丢弃，按以下顺序处理：

1. 在观测深度/组织化点图上施加本轮增强；利用 `source_indices` 保持像素联系。
2. 对当前观测重算差分法向及有效掩码。复用 `read_frame()` 的数学规则；不要再次调用它读取未增强的原始深度后当作新法向。
3. 删除观测点时，对所有逐点输入、标签和索引应用同一个选择数组。保留的 `source_indices` 仍是原像素索引，不改为样本中的新行号。
4. 若再施加刚体坐标增强，同转点、观测法向、标签法向和重力。
5. 更新候选及损失掩码，重新计算可见实例中心与尺寸，然后体素化。

深度噪声只改变观测，不把 `normal_label` 改成当前噪声法向。GT 轴仅随坐标变换旋转。模型要从观测估计标签轴；输入中不能出现 `gt_normal`、`gt_normal_pca`、`gt_valid` 或质量标签。

## 7. 划分与冻结归一化

公开数据开发阶段按场景划分，默认相机为 RealSense：

| 集合 | 场景 |
|---|---|
| 预训练训练子集 | 0000-0099 中 `scene_id % 10 != 9` 的 90 个场景 |
| 开发验证子集 | 0009、0019、0029、0039、0049、0059、0069、0079、0089、0099 |
| Seen 测试 | 0100-0129 |
| Similar 测试 | 0130-0159 |
| Novel 测试 | 0160-0189 |

同一场景的 256 个视角不能跨训练/验证集合。具体每轮取视角数、训练预算和验证频率由 [TRAINING.md](TRAINING.md) 统一规定，适配器接受显式的 `(scene_id, camera, frame_id)` 列表与 epoch 种子，不在每次 `__getitem__` 中自行随机划分集合。

`full_cli.fit_normalization()` 默认用官方训练池 0000-0099 中出现的物体，将其候选 Seal 汇总后计算第 1/99 百分位并写入 `config_frozen.json`。归一化定义为：

```python
S = clip((seal_raw - lo) / (hi - lo), 0.0, 1.0)
A = S * collision_free
B = A * wrench_eval
```

配置模板里的 `seal_norm_min=0`、`seal_norm_max=1` 只是默认值，不能代替本次输出的冻结值。模型直接消费 `gt_seal_norm`、`gt_A`、`gt_B`；模型训练、所有消融、验证、测试及项目 R20 姿态评测使用同一套实际 `lo/hi`。不逐帧或逐实例重新 min-max，不用测试场景重新拟合。

既有归一化使用整个官方训练池，本轮内部验证用于模型选择，沿用这套已固定的标签定义。无需为划分 90/10 场景重做已完成的数据。后续自有数据若能生成同定义的物理代理，也沿用预训练 `lo/hi`；若只有稀疏真机成败结果，使用独立的成功标签字段，不能直接把 0/1 改名为 `gt_B`。

## 8. 单帧 Sample 与 batch 合约

接口将“推理能看到的输入”“训练/评价标签”“可追溯元信息”分开。

```python
Sample = {
    "inputs": {
        "xyz_cam_m": Float[N, 3],           # m，相机系或一致增强后的坐标
        "normal_obs": Float[N, 3],
        "normal_obs_valid": Bool[N],
        "instance_id_input": Long[N],
        "g_cam": Float[3],
        "source_indices": Long[N],         # 只读追踪索引，不是输入特征
    },
    "targets": {
        "gt_A": Float[N], "gt_B": Float[N],
        "seal_target": Float[N], "wrench_target": Float[N],
        "collision_free_target": Float[N],
        "normal_label": Float[N, 3],
        "quality_valid": Bool[N], "axis_valid": Bool[N],
        "instance_id_gt": Long[N],
        "gt_valid": Bool[N],               # 可保留作诊断
    },
    "meta": {
        "sample_uid": (scene_id, camera, frame_id),
        "source_indices": Long[N],
        "candidate_row": Long[N], "candidate_id": Long[N],
        "image_shape": (H, W),
        "T_world_cam": Float[4, 4],
        "source_format": "legacy_dc4ad6b",   # 或明确选择r20_v2_1
        "source_schema_version": None,      # 新源文件可填其实际schema字符串
        "target_kind": "r20_proxy",
    },
}
```

`Float[N,3]` 等是规格记法，不是已经存在的 Python 类型。`meta` 保留在 CPU；增强后保留源相机变换及必要的增强变换，评价时先将输出恢复到原相机坐标，避免拿增强坐标与原场景几何比较。正式验证和测试不开随机增强。

采用 packed batch，点数为 `P = sum(N_b)`，不把整帧点云补成统一长度：

| batch 字段 | 形状 | 约定 |
|---|---|---|
| `xyz_cam_m`、`normal_obs` | `[P,3]` | 各样本按顺序拼接 |
| 点级输入掩码、实例 ID | `[P]` | 同样拼接 |
| `batch_index` | `[P]` | 点属于哪个 batch 样本，0-based |
| `point_offsets` | `[B+1]` | 各帧在 packed 点数组中的起止位置 |
| `g_cam` | `[B,3]` | 按 `batch_index` 广播，或在实例/候选处索引 |
| `voxel_coords` | `[V,4]` int32 | 固定顺序 `(batch,z,y,x)` |
| `voxel_features` | `[V,8]` float32 | 仅由观测输入构造 |
| `inverse_map` | `[P]` int64 | 原始观测点到本 batch 体素行的映射 |
| `source_indices` | `[P]` int64 | 原像素索引，仅用于结果追踪；不是特征 |
| `voxel_origins` | `[B,3]` | 各帧整数网格平移量，仅用于坐标重建 |
| packed targets | 与点数组一致 | 保持独立字典，不拼进特征 |

实例聚合键用 `(batch_index, instance_id_input)`，损失中的实例均衡键用 `(batch_index, instance_id_gt)`；两个键都排除背景 0。不能把两个 batch 样本中的 `instance_id=1` 聚成同一实例。每项损失先在实例内对有效点求均值，再在每帧内对该项实际有监督的实例求均值，最后在 batch 内对该项有监督的帧求均值。不能跳过帧层级而将全 batch 的实例直接平均；空集合不进入分母，整项为空时返回与预测计算图相连的零。

自有场景复用上面的 `inputs` 和 batch 格式，标签由 `target_dataset.py` 按可用监督提供。没有 Seal、Wrench 或轴标签时，保持对应监督掩码关闭；不能填零并当作已知失败。是否使用成功分类/排序头及其字段，以 [FINETUNING.md](FINETUNING.md) 为准。

## 9. 体素化与 inverse map

模型体素大小初值为 **0.002 m**。这是模型配置，和数据生成配置中的 `voxel_m=0.005`、碰撞障碍点下采样 `obstacle_voxel_m=0.003` 是不同用途；不要为调整模型分辨率改动标签生成配置。

单帧 8 维点特征为：

```python
point_features = concat(
    (xyz_cam_m - scene_visible_center) / 0.1,  # 3维；固定数值尺度，非物体尺寸归一化
    where(normal_obs_valid[:, None], normal_obs, 0.0),  # 3维
    (instance_id_input > 0)[:, None].float(),  # 1维
    normal_obs_valid[:, None].float(),    # 1维
)  # [N, 8]
```

`scene_visible_center` 为当前保留观测点的均值。真实物理坐标仍保留在 `inputs.xyz_cam_m`，读取自源文件的 `xyz`；局部分区以这些坐标和预测轴计算，不能用数值缩放后的特征坐标或体素中心代替接触几何。

体素化伪代码如下：

```python
grid_xyz = floor(xyz_cam_m / voxel_size_m).astype(int32)
origin_xyz = grid_xyz.min(axis=0)
grid_xyz = grid_xyz - origin_xyz        # spconv空间坐标保持非负

unique_xyz, inverse_local = unique_rows(grid_xyz, return_inverse=True)
voxel_features = scatter_mean(point_features, inverse_local)
voxel_coords = concat(
    full((len(unique_xyz), 1), batch_id),
    unique_xyz[:, [2, 1, 0]],            # z,y,x
)
inverse_map_for_sample = inverse_local + preceding_voxel_count
```

全场景先建体素/空间索引，再做候选与物理区采样。不要先随机抽少量全场景点，导致 18-22 mm 环带在进入模型之前丢失。

主干包装器约定输出 `voxel_embeddings[V,D]` 与输入 `voxel_coords` 的体素行对应；若稀疏库输出行序不同，包装器先按坐标重新对应。逐点特征逻辑上为：

```python
point_embeddings = voxel_embeddings[inverse_map]
```

实现时可以按实例或候选邻域按需 gather，避免无必要地同时存储完整 `[P,D]` 副本。无论采用哪种实现，实例归属、候选中心和局部分组继续使用原观测点的 ID/坐标；不把 GT 在体素内求均值后当成新的点级标签。

## 10. Dataset、collate 与 forward 的伪代码

以下定义的是待实现的函数职责；示例中的 helper 名称不是当前仓库已经提供的 API。

```python
class PublicSuction20Dataset(Dataset):
    def __init__(self, dataset_root, label_root, frame_keys,
                 camera="realsense", augment=None, instance_source="gt",
                 source_format="legacy_dc4ad6b", compatibility="auto",
                 adapter_cache=None):
        self.dataset_root, self.label_root = Path(dataset_root), Path(label_root)
        self.keys = list(frame_keys)             # 已按场景划分
        self.data_cfg = read_json(self.label_root / "config_frozen.json")
        self.augment = augment
        self.instance_source = instance_source
        self.source_format, self.adapter_cache = source_format, adapter_cache
        self.compatibility = compatibility
        # 按场景缓存相机位姿数组和图像尺寸；无需逐样本重复读全部元信息。

    def __getitem__(self, index):
        key = self.keys[index]
        frame = read_npz_copy(view_npz_path(key)) # allow_pickle=False
        effective_format = resolve_source_format(
            frame, fallback=self.source_format, compatibility=self.compatibility
        )
        derived = adapt_source_view(
            frame, key, effective_format, self.data_cfg, self.adapter_cache
        )  # 旧数据恢复轴/观测法向；新数据读取已存字段；均保留现有质量分
        geometry = {
            "xyz_cam_m": frame["xyz"],
            "normal_obs": derived["normal_obs"],
            "normal_obs_valid": derived["normal_obs_valid"],
            "g_cam": gravity_in_camera(key),
        }
        instance_id_gt = frame["instance_id"]
        instance_id_input = read_input_instances(
            key, frame["source_indices"], self.instance_source, instance_id_gt
        )
        labels = copy_public_targets(frame)      # 保留现有S/W/C/A/B、gt_valid、instance_id_gt
        labels["normal_label"] = derived["normal_label"]  # 不加入geometry
        meta = make_frame_meta(key, frame)

        geometry, instance_id_input, labels, meta = augment_and_select_aligned(
            geometry, instance_id_input, labels, meta, self.augment
        )
        inputs = build_observed_inputs(geometry, instance_id_input)
        inputs["source_indices"] = meta["source_indices"]  # 追踪，不能拼入特征
        targets = derive_loss_masks_and_targets(labels, inputs)
        return {"inputs": inputs, "targets": targets, "meta": meta}


def collate_public(samples, voxel_size_m=0.002):
    inputs = pack_points_and_voxelize(samples, voxel_size_m)
    targets = pack_targets_in_same_point_order(samples)
    meta = [s["meta"] for s in samples]
    return {"inputs": inputs, "targets": targets, "meta": meta}


for batch in loader:
    inputs, targets = batch["inputs"], batch["targets"]
    # encode_scene只读取inputs；内部训练采样器明确允许读取targets。
    # GT不拼入特征、不交给主干或质量头；q0部分仍由当前模型提议。
    pred = model(inputs, sampler_mode="train", targets=targets)
    loss = loss_fn(pred, targets)
```

推理调用 `model(inputs, sampler_mode="inference")`，不传 `targets`；候选只使用前置实例、观测法向有效性和预测粗分。`encode_scene(inputs)` 在训练和推理都只接收观测输入。训练 `forward` 将 `targets` 传给内部训练采样器属于明确的监督采样策略，不等同于把 GT 分数拼到网络输入。`loss_fn` 根据配置选择 `gt_A` 或 `gt_B`，各个消融共享同一加载、增强、索引和采样规则。

公开数据 q/S/W/C 标签的范围均是 `[0,1]`；文件中的旧 `normal_valid` 不决定新模型是否能看到某个点，`quality_valid` 也不决定主干输入范围。背景、未知点和被遮挡区域附近的可见点依然能为场景特征提供上下文。

## 11. 实现时的最小验收

现有数据代码的 `pytest` 继续照仓库开发约定运行；它验证的是已有合成输入路径，不代表完整真实 SuctionNet 数据已经验收。适配器实现后只增加以下直接覆盖接口风险的检查，不为本次文档提交新造测试框架。

1. **索引与主轴对应。** 构造 `instance_id`、`candidate_row`、`candidate_id` 互不相等的小例子，确认读取的是正确物体的正确候选行；在一个已有帧核对直接 `gt_normal` 与本节等价恢复公式一致。
2. **未知/负例与输入隔离。** 修改一个 `gt_valid=False` 点的占位分数不应改变质量损失；修改有效零分应改变对应监督。固定训练采样索引后，只改 GT 不能改变模型输入特征。
3. **像素及选择对齐。** 用小尺寸图验证 `source_indices = v*W+u`；执行一次点丢弃后，点、实例、GT、源像素仍一一对应，法向掩码使用增强后的观测。
4. **batch 与 inverse map。** 两个样本都含实例 1，池化/损失仍分别统计；同体素多个点映回同一体素特征，标签保持逐点身份。
5. **坐标与方向一致。** 对一个普通刚体旋转，同转 `xyz_cam_m/normal_obs/normal_label/g_cam`；法向和重力的内积不变、质量标量不变，恢复到源相机坐标后的输出一致。确认零分/碰撞点只要原始轴和当前观测法向有效，就可参与轴监督。

这些检查通过后即可接入预训练，不额外要求全量数据重新标注、逐物体实测质量/质心或新的异常恢复工程。

## 12. 源码依据

- [`suction20/full_data.py`](../../suction20/full_data.py)：`scene_frame0`、`read_frame`、`compute_scene`、`generate_view`、`wrench_scores_dual`。
- [`suction20/full_cli.py`](../../suction20/full_cli.py)：`fit_normalization` 和 `run-all` 的执行顺序。
- [`configs/suction20.json`](../../configs/suction20.json)：半径、包络、深度范围和默认参数；运行时归一化以生成的 `config_frozen.json` 为准。
- [固定核查版本](https://github.com/chichu-lv/suctionnet_modern_20mm/tree/b2680fb47fe6f6a99fcb310e3428c77de1f55406)：上述“当前实现”描述的版本依据。
- [用户既有数据方法](https://github.com/chichu-lv/codex-profile/tree/dc4ad6b00535c89ece1feef2751ba86006e5f97e/suction20_data_starter)：默认旧数据入口的字段和计算依据，详见 [调整方案](LEGACY_DATA_ADJUSTMENT.md)。
