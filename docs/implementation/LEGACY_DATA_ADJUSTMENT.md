# 已完成旧数据的训练适配方案

> 状态：默认只读适配已实现，可选 sidecar 转换与读取支持尚未交付。用户已按修改前的方法完整制作数据；本文件的源码差异对照以 `codex-profile@dc4ad6b00535c89ece1feef2751ba86006e5f97e` 和 `suctionnet_modern_20mm@b2680fb47fe6f6a99fcb310e3428c77de1f55406` 为基准。当前加载器及真实数据有界训练已完成检查，范围见 [验证记录](../MODEL_VALIDATION.md)；实际命令见 [运行指南](../MODEL_USAGE.md)。

**结论：复用已经完成的数据。Seal、Wrench、Collision、GT-A、GT-B 均原样保留；需要补的是训练接口中的原始候选轴和观测法向。** 默认由 DataLoader 在读取时派生，之后如果确有重复计算开销，再做独立的派生缓存。无需把整套数据改写成新生成器格式才能预训练。

本文件只负责已有数据如何进入后续模型。模型侧统一输入/目标格式见 [DATA_ADAPTER.md](DATA_ADAPTER.md)，公开数据训练和自有场景微调分别见 [TRAINING.md](TRAINING.md)、[FINETUNING.md](FINETUNING.md)。

## 1. 对比范围与明确结论

旧源码位于 `suction20_data_starter/suction20/`，新源码位于 `suction20/`。对照结果如下：

| 项目 | 旧 `dc4ad6b` | 新 `b2680fb` | 已有数据处理 |
|---|---|---|---|
| Seal 半径和环带 | 20 mm 半径；72 扇区；18-22 mm 环形搜索 | 相同；新增对上游 72 扇区的显式检查 | 保留已有 `seal_r20` |
| Wrench 主评分 | 切向力矩向量 L2 范数，阈值 `k*r*pi` | 相同 | 保留 `gt_wrench_eval` / `gt_wrench` |
| 历史 Wrench | 同一力矩，阈值乘 `sqrt(2)` | 相同 | 保留 `gt_wrench_annot`，不改作主训练目标 |
| Collision | 22/28 mm 三段几何包络，轴向到 150 mm | 相同 | 保留全部碰撞标签 |
| 观测到候选匹配 | 同实例，5 mm，30°法向门控，最近几何候选 | 相同 | 保留索引、距离、`gt_valid` |
| Seal 归一化 | 训练池候选第 1/99 百分位，写冻结配置 | 相同 | 沿用已有冻结上下界与已有分数 |
| GT-A/GT-B | `A=S*C`，`B=S*W_eval*C` | 相同 | 不重新生成 |
| `gt_normal` | extras 存在时写 PCA 法向；否则原始候选轴 | 始终写原始候选轴 | 旧文件不能直接当主轴标签，按候选恢复 |
| 观测法向 | 匹配时计算，但不保存 | 保存 `normals_raw` / `normals_raw_valid` | 旧文件从原始 RGB-D 派生 |
| 法向掩码 | PCA 有效、无碰撞且 `S>0.2`；无 extras 时全 False | 无碰撞且 `S>0.2` | 两者都不等于原始轴几何有效性；训练独立派生 `axis_valid` |
| 逐帧版本记录 | `config_version`、`seal_generation_hash` | `schema_version=r20-v2.1` | 保留原始记录，不要求旧文件伪装为新格式 |

源码中 `scene_frame0`、`read_frame`、`match_instance`、`wrench_scores_dual`、`collision_mask_for_instance`、`resolve_instance_ids`、`add_pca_extras`、`fit_normalization` 的函数 AST 在这两个提交间相同。主标量标签的算法并未因为此次仓库迁移发生变化。

尤其要区分两件事：旧官方 API 的力矩坐标分量实现与本项目 L2 定义之间的差别，和这次旧数据到新模型接口的差别。**用户这份旧 `dc4ad6b` 数据本来就使用 L2，没有发生“旧数据是另一种范数、新数据才改 L2”的迁移。** 也没有从固定 `[0,1]` 原始 Seal 切换到新归一化的变化：旧流水线 `run-all` 已调用同一 `fit_normalization()`。这些都不构成重算现有标签的理由。

## 2. 现有目录与配置优先级

适配器指向用户已经完成的输出目录，不新跑 `run-all`：

```text
$DATASET/                              # 原始公开数据，继续使用原来的版本
  scenes/scene_SSSS/<camera>/
    depth/FFFF.png
    label/FFFF.png
    meta/FFFF.mat
    camera_poses.npy
    cam0_wrt_table.npy

$OLD_OUT/                              # 已经制作完成，训练只读
  config_frozen.json
  seal_r20/poisson/OOO_labels.npz
  seal_extras_r20/OOO.npz
  wrench_label_r20/SSSS_<camera>_wrench_scores.npz
  collision_label_r20/SSSS_<camera>_collision.npz
  scene_meta_r20/SSSS_<camera>.json
  scenes/scene_SSSS/<camera>/graspness_r20_v2/FFFF.npz
```

使用原数据制作时的相机、原始深度和候选文件。恢复轴时读取 `$OLD_OUT/seal_r20/poisson`，不是重新采样一份物体候选；这样原 `candidate_row` 和 `candidate_id` 的对应关系不变。

参数优先级固定为：

1. **数据含义：** `$OLD_OUT/config_frozen.json` 和已生成字段；包括实际 `lo/hi`、接触半径、碰撞包络、深度范围。
2. **模型设置：** 新的训练配置；例如 2 mm 网络体素、512 个种子、学习率与 batch size。
3. **仓库模板：** `configs/suction20.json` 只用于描述默认生产方法，不覆盖既有数据配置。

旧冻结配置中含 `wrench_threshold_factor=1`，当前生成器已移除该配置键，并会在 `config_from_file()` 中拒绝未知键。适配器应使用普通 `json.load()` 读取原冻结配置，再访问所需字段；不能用当前生成器的配置校验器把这份合法旧配置判成错误。这个键在旧实现中也没有把主 Wrench 改为历史 `sqrt(2)` 版本。

模型网络体素设成 `0.002 m` 不改变旧标签生成配置中的 `voxel_m=0.005`。两者承担不同职责；不能因此重制 5 mm 采样的物体标签或逐帧 GT。

## 3. 字段决策表

### 3.1 完整保留的字段

| 已有字段 | 适配策略 | 原因 |
|---|---|---|
| `xyz`、`source_indices` | 原样作为输入位置和原像素对应；训练增强另行处理 | 逐帧观测已经对齐 |
| `class_id`、`instance_id`、`objectness` | 保留 GT 身份；按任务构造输入实例掩码 | 不改场景实例顺序 |
| `candidate_row`、`candidate_id`、`match_distance` | 保留 | 供精确恢复与追溯 |
| `gt_valid` | 保留 | 未匹配仍是未知；不是负样本 |
| `gt_seal_raw`、`gt_seal_norm` | 保留 | 不重新拟合 `lo/hi` |
| `collision_flag` | 保留 | True 仍表示碰撞 |
| `gt_wrench_eval`、`gt_wrench` | 保留，主训练用 eval | 与新模型目标定义一致 |
| `gt_wrench_annot` | 保留但不作为主 Wrench | 历史口径独立存在 |
| `gt_A`、`gt_B` | 保留；模型直接读取 | 不重复乘分、不替换标签 |
| `gt_orient`、`gt_seal_custom` | 保留为辅助信息 | 默认主模型不使用 |
| `config_version`、`seal_generation_hash` | 保留为来源记录 | 不从不存在的新版本字段推断无效 |

所谓“保留”，是直接使用已生成数值，而不是在转换时顺手重算一遍。模型的 `gt_A`/`gt_B` 可以是这些数组的同值读取。

### 3.2 重新解释或派生的字段

| 训练接口字段 | 旧数据来源 | 操作 |
|---|---|---|
| `normal_label` | 原 `candidate_row` + 对应物体 `seal["normals"]` + 物体到相机旋转 | 索引和旋转；不用旧 `gt_normal` 替代 |
| `normal_obs` | 原 RGB-D 的局部差分法向 | 调用既有 `read_frame` 数学规则，按 `source_indices` 对齐 |
| `normal_obs_valid` | 该观测法向的有效性 | 保留无效状态，输入法向无效处置零 |
| `g_cam` | 原相机到桌面变换 | 只旋转单位重力向量 |
| `image_shape` | 原帧深度 PNG | 从实际图像读取 `H,W` |
| `quality_valid` | `gt_valid & (instance_id_gt>0)` | 训练掩码，不改源文件 |
| `axis_valid` | `quality_valid`、原始轴有效性、当前观测法向有效性 | 独立训练掩码，不继承旧 PCA 掩码 |

旧 `gt_normal` 和 `normal_valid` 留在原文件中供原始记录使用。默认不将它们改名为 `gt_normal_pca` / `normal_pca_valid`：旧 `gt_normal` 是否为 PCA 取决于当时是否生成 extras，旧 `normal_valid` 还叠加了碰撞/Seal 条件，和新文件的纯 PCA 有效性并不完全同义。主训练不需要迁移这组可选比较标签。

## 4. 原始候选轴：准确恢复，不重新评价

### 4.1 场景实例与候选行号

旧逐帧代码写入：

```python
instance_id = scene_instance_row + 1
candidate_row = matched_seal_array_row
candidate_id = seal["indices"][candidate_row]
```

因此恢复使用 `candidate_row`，不是 `candidate_id`。背景的 `instance_id=0` 和未匹配点的 `candidate_row=-1` 不进入索引操作。

旧 `scene_meta_r20` 中的 `objects` 各项是 `{row, obj_id, n, ...统计}`，并没有 `T_world_obj`。它另外保存 `configuration_hash`、`seal_generation_hash` 和公式说明。不能把这个 JSON 当作已经包含完整姿态的对象缓存。

现有 `scene_frame0()` 从原始 frame-0 元数据返回下列记录，才适合恢复：

```python
{
    "instance_row": row,                # 0-based场景行号
    "obj_id": object_id,                # 0-based官方物体文件编号
    "T_world_obj": matrix_4x4,
}
```

其计算与旧版本相同：

```python
T_world_cam0 = cam0_wrt_table @ camera_poses[0]
T_world_obj = T_world_cam0 @ T_cam0_obj_from_meta_0000
T_world_cam = cam0_wrt_table @ camera_poses[frame_id]
T_cam_obj = inverse(T_world_cam) @ T_world_obj
```

### 4.2 可直接实现的派生函数

以下代码是算法伪代码；实际只读入口为 `PublicSuction20Dataset`，所引用的 `scene_frame0`、`load_seal`、`read_frame` 已存在于当前源码。

```python
def recover_original_axis(old, records, seal_cache, T_world_cam):
    n_label = np.zeros_like(old["xyz"], dtype=np.float32)
    T_cam_world = np.linalg.inv(T_world_cam)

    for rec in records:
        pix = np.flatnonzero(
            old["gt_valid"]
            & (old["instance_id"] == rec["instance_row"] + 1)
        )
        if len(pix) == 0:
            continue

        seal = seal_cache[rec["obj_id"]]  # 来自原$OLD_OUT/seal_r20/poisson
        row = old["candidate_row"][pix]
        T_cam_obj = T_cam_world @ rec["T_world_obj"]
        n = seal["normals"][row] @ T_cam_obj[:3, :3].T
        n_label[pix] = n / np.linalg.norm(n, axis=-1, keepdims=True)

    return n_label
```

这个函数没有重新计算 Seal、力矩、碰撞或点匹配，只把已有候选方向旋转到原相机坐标系。它也不会把表面外法向改成相反的接近方向；模型执行定义仍为接近运动沿 `-n_hat`。

## 5. 从原深度派生观测法向并对齐

旧数据生成时已经调用过 `read_frame()` 以做法向匹配，只是没有把观测法向写入 NPZ。新旧函数的计算相同，因此这里重复的是轻量观测处理，不是重标注。

```python
def derive_observed_normals(old, dataset_root, scene_id, frame_id, frozen_cfg):
    xyz_raw, n_raw, ok_raw, _, src_raw, image_shape = read_frame(
        dataset_root, scene_id, frame_id, frozen_cfg
    )
    H, W = image_shape

    # 通过原像素索引对应，既不重排旧点，也不假定新读出的数组行号相同。
    row_for_pixel = np.full(H * W, -1, dtype=np.int64)
    row_for_pixel[src_raw] = np.arange(len(src_raw))
    rows = row_for_pixel[old["source_indices"]]
    present = rows >= 0

    normal_obs = np.zeros_like(old["xyz"], dtype=np.float32)
    obs_valid = np.zeros(len(old["xyz"]), dtype=bool)
    normal_obs[present] = n_raw[rows[present]]
    obs_valid[present] = ok_raw[rows[present]]
    normal_obs[~obs_valid] = 0.0
    return normal_obs, obs_valid, image_shape
```

`frozen_cfg` 直接来自旧 `config_frozen.json`，保留原深度范围和相机选择。模型输入位置继续使用旧 `xyz`；上例中的 `xyz_raw` 不是新的位置标签。图像宽高从原帧获取，不固定为某个分辨率。

相机重力为：

```python
g_cam = T_world_cam[:3, :3].T @ np.array([0., 0., -1.])
```

若训练增强改变了深度或丢弃点，随后根据当前观测重算法向和观测有效性；若仅做刚体坐标增强，同时旋转观测法向、标签轴和重力。不能在深度增强之后继续把缓存的原始观测法向当成增强后的观测。

## 6. 默认方案：训练时适配

后续 `suction20/datasets.py` 中实现一个来源分支即可，不增加独立物理数据生产阶段：

```python
class LegacyViewAdapter:
    def __init__(self, dataset_root, label_root):
        self.dataset_root = Path(dataset_root)
        self.label_root = Path(label_root)
        self.frozen_cfg = json.loads(
            (self.label_root / "config_frozen.json").read_text()
        )
        self.seal_root = self.label_root / "seal_r20" / "poisson"
        self.scene_cache = {}      # scene_frame0记录、相机位姿
        self.object_cache = {}     # 按需读取原Seal候选数组

    def read(self, scene_id, frame_id):
        old = read_old_view_npz(scene_id, frame_id, allow_pickle=False)
        records, T_world_cam = self.get_scene_geometry(scene_id, frame_id)
        n_label = recover_original_axis(
            old, records, self.get_object_arrays(records), T_world_cam
        )
        n_obs, obs_valid, image_shape = derive_observed_normals(
            old, self.dataset_root, scene_id, frame_id, self.frozen_cfg
        )

        m_q = old["gt_valid"] & (old["instance_id"] > 0)
        axis_label_ok = (
            np.isfinite(n_label).all(-1)
            & (np.linalg.norm(n_label, axis=-1) > 1e-6)
        )
        m_axis = m_q & axis_label_ok & obs_valid

        return {
            "inputs": {
                "xyz_cam_m": old["xyz"],
                "normal_obs": n_obs,
                "normal_obs_valid": obs_valid,
                "instance_id_input": old["instance_id"],  # 默认GT-mask预训练
                "g_cam": T_world_cam[:3, :3].T @ np.array([0., 0., -1.]),
                "source_indices": old["source_indices"], # 追踪用途，不是特征
            },
            "targets": {
                "instance_id_gt": old["instance_id"],
                "gt_A": old["gt_A"], "gt_B": old["gt_B"],
                "seal_target": old["gt_seal_norm"],
                "wrench_target": old["gt_wrench_eval"],
                "collision_free_target": (~old["collision_flag"]).astype(np.float32),
                "normal_label": n_label,
                "quality_valid": m_q, "axis_valid": m_axis,
                "gt_valid": old["gt_valid"],
            },
            "meta": {
                "sample_uid": (scene_id, self.frozen_cfg["camera"], frame_id),
                "source_indices": old["source_indices"],
                "candidate_row": old["candidate_row"],
                "candidate_id": old["candidate_id"],
                "image_shape": image_shape,
                "T_world_cam": T_world_cam,
                "source_format": "legacy_dc4ad6b",
                "target_kind": "r20_proxy",
            },
        }
```

示例中的 `read_old_view_npz`、`get_scene_geometry`、`get_object_arrays` 是说明用 helper 名称，实际入口见 `suction20/datasets.py`；示例返回的是未增强的统一 Sample，后续增强必须更新观测法向及掩码。物体/场景数组按需缓存到 DataLoader worker，避免每帧重复打开同一批文件。预测分割实验只替换 `inputs.instance_id_input`，不覆盖 `targets.instance_id_gt`。读取输出进入 [统一 Sample 接口](DATA_ADAPTER.md#8-单帧-sample-与-batch-合约) 后，模型无需知道该样本来自旧文件还是新文件。

与模型统一的调用为 `model(inputs, sampler_mode="train", targets=targets)`；其中 `encode_scene` 始终只读 `inputs`，只有内部训练采样器可以读取 `targets`。推理使用 `model(inputs, sampler_mode="inference")`。各项损失按“实例内有效点均值 → 帧内有监督实例均值 → batch 内有监督帧均值”归约，不能将全 batch 的实例直接混成一层平均。

训练配置中的来源约定为：

```yaml
# 已实现训练入口的配置片段；不是full_cli的数据生产参数文件
data:
  source_format: legacy_dc4ad6b
  compatibility: auto
  source_root: /data/SuctionNet
  label_root: /data/suction20_output        # 既有完成目录
  frozen_label_config: /data/suction20_output/config_frozen.json
  adapter_cache: null                      # 默认按需派生

model:
  voxel_size_m: 0.002                      # 只控制网络
```

半径、碰撞包络与 Seal 归一化不在模型配置里重新指定一套值。质量头直接读取旧 `gt_B`，不把恢复法向理解成需要给每个点重新生成另一套质量目标。

`compatibility:auto` 遇到明确的 `schema_version=r20-v2.1` 时读取新格式语义；对用户这批没有新 schema 的已有文件，`source_format:legacy_dc4ad6b` 指定其默认来源。它不要求旧文件新增 schema，也不把 sidecar 的 `adapter_schema` 误认为原文件已升级。未来 sidecar 读取应与当前直接派生提供相同输出字段。

## 7. 可选方案：一次性派生缓存

### 7.1 什么时候使用

当前默认使用已实现的训练时适配。若 CPU 法向计算/原深度读取成为反复训练的主要开销，可以按本节方案开发第 4-5 节派生结果的缓存。缓存只减少重复读取与计算，不改变监督含义，也不是开始预训练的前置工作。

建议保存独立 sidecar，而不是覆盖旧 NPZ 或复制所有 S/W/C 数组：

```text
$ADAPTER_CACHE/
  adapter_manifest.json
  source_config_frozen.json                # 旧冻结配置的原样副本
  scenes/scene_SSSS/<camera>/FFFF.npz
```

每个 sidecar 保存：

| 字段 | 形状 | 内容 |
|---|---|---|
| `adapter_schema` | 标量字符串 | `legacy-dc4ad6b-derived-v1`，不冒充生成器 schema |
| `source_indices` | `[N]` | 原样复制，用于读取时对应原点行 |
| `normal_label` | `[N,3]` | 恢复的原始候选轴 |
| `normal_obs` | `[N,3]` | 原始观测法向；无效处零 |
| `normal_obs_valid` | `[N]` | 原始观测法向有效性 |
| `g_cam` | `[3]` | 原相机系重力 |
| `image_shape` | `[2]` | 原图 `H,W` |

不保存 `quality_valid`/`axis_valid` 的最终增强结果；它们在运行时结合当前点选择和法向有效性派生。原质量标签继续从 `$OLD_OUT` 读取。`adapter_manifest.json` 记录来源格式、源目录、两个源码提交、所用相机、候选路径和已缓存的 frame keys。无需另做全数据哈希扫描或再次跑 G0-G2。

### 7.2 转换过程

1. 从已有场景/帧列表逐项读取旧 NPZ。
2. 调用与训练时适配**同一个**派生函数，得到轴、观测法向、重力和图像尺寸。
3. 按原点序写入 sidecar；源数据只读。先支持所需训练/验证帧，不要求一开始缓存所有测试帧。
4. 加载时对齐该帧 `source_indices` 和点数，再组合“原 NPZ 的质量标签 + sidecar 的派生输入/轴”。未缓存的帧继续按需计算。
5. 几何增强仍在加载之后执行；不能把未增强缓存中的法向直接用于增强后的点图。

同一个源帧的内存派生结果和 sidecar 读取结果应相同。换训练网络宽度、学习率或 2 mm 体素设置不使这个缓存失效，因为这些设置没有改变原始观测或标签轴。

### 7.3 待实现的命令接口约定

当前仓库**没有**本节的转换脚本；下面是便于实现者开发的命令接口草案，不是现在即可执行的命令，也不表示本次已完成文件转换。

```text
python scripts/build_adapter_cache.py
  --source-format legacy_dc4ad6b
  --dataset-root /data/SuctionNet
  --label-root /data/suction20_output
  --output-root /data/suction20_adapter_cache
  --scenes 0:100
  --frames all
```

约定 `--scenes`/`--frames` 复用现有 `range_spec` 的区间语义，`0:100` 为 0-99。脚本只构建本节 sidecar；不调用 `seal_run`、`compute_scene`、`generate_view` 或 `fit_normalization`，不对原数据传入 `--overwrite`。后续同时实现转换脚本和 sidecar 读取支持后，训练配置才可将 `adapter_cache` 指向输出目录；当前保持 `null`。

## 8. 接口检查与后续实验

前两项已有实现与检查覆盖，记录见 [验证记录](../MODEL_VALIDATION.md)；第三项仅在后续选择实现 sidecar 时适用：

1. 取一个普通帧，检查恢复轴使用的是 `candidate_row` 与正确实例姿态；轴为单位向量、方向约定一致。检查观测法向按 `source_indices` 与原点对应。
2. 确认进入训练的 `gt_A`、`gt_B`、S/W/C 与旧 NPZ 数值相同；保留有效零分，忽略未知点。旧 `normal_valid=False` 不妨碍几何有效的原始候选轴进入新 `axis_valid`。
3. 若实现 sidecar，比较同帧直接派生与缓存读取结果，再使用同一训练入口跑一个 batch。模型输入不能包含旧 `gt_normal`、新恢复的 `normal_label` 或质量 GT；标签只进入损失和明确的监督采样。

旧数据 adapter 和有界训练链路已完成；后续为正式公开预训练与模型对照、选定公开权重、少量自有场景微调。派生缓存仅在确有加载开销时按需考虑，没有因为新版新增字段而重跑整套物理标签的步骤。

## 9. 可追溯源码

- [旧 `full_data.py`](https://github.com/chichu-lv/codex-profile/blob/dc4ad6b00535c89ece1feef2751ba86006e5f97e/suction20_data_starter/suction20/full_data.py)：旧逐帧字段、PCA 替换、原始候选轴来源、Wrench 与 Collision。
- [旧 `full_cli.py`](https://github.com/chichu-lv/codex-profile/blob/dc4ad6b00535c89ece1feef2751ba86006e5f97e/suction20_data_starter/suction20/full_cli.py)：既有 `fit_normalization` 及冻结配置。
- [旧 `patch_official.py`](https://github.com/chichu-lv/codex-profile/blob/dc4ad6b00535c89ece1feef2751ba86006e5f97e/suction20_data_starter/suction20/patch_official.py)：20 mm、18-22 mm 环带补丁。
- [新 `full_data.py`](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/b2680fb47fe6f6a99fcb310e3428c77de1f55406/suction20/full_data.py)：主轴/观测法向的新保存格式及同一物理标签公式。
- [新 `full_cli.py`](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/b2680fb47fe6f6a99fcb310e3428c77de1f55406/suction20/full_cli.py)：同一归一化方法。
