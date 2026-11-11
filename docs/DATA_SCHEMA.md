# 训练数据字段说明

## 文件层级

| 文件（`$OUT` 相对路径） | 数据粒度 | 主要字段 |
|---|---|---|
| `seal_r10/poisson/XXX_labels.npz` | 单物体候选点 | 原 10 mm Seal |
| `seal_r20/poisson/XXX_labels.npz` | 单物体候选点 | `points`, `normals`, `indices`, `scores`, `collision` |
| `seal_extras_r20/XXX.npz` | 单物体候选点 | `pca_normals`, `normal_valid`, `seal_custom` |
| `wrench_label_r20/SSSS_realsense_wrench_scores.npz` | 场景中的各实例候选 | `wrench_annot_i`, `wrench_eval_i`, `arr_i`, `physics_i`, `orient_i` |
| `collision_label_r20/SSSS_realsense_collision.npz` | 场景中的各实例候选 | `arr_i`，True=碰撞 |
| `scene_meta_r20/SSSS_realsense.json` | 场景 | `objects`, `radius_m`, `wrench_schema` |
| `scenes/scene_SSSS/realsense/graspness_r20_v2/FFFF.npz` | 每个相机点 | 下述逐点 GT |

场景 Wrench 和 Collision 的后缀 `i` 是**该场景物体列表中的 0-based 行号**，不是全局物体类别编号。所有物理评价均使用与 Seal 候选点对应的顺序。

## 相机点级文件（`N` 是该帧实际保留的观测点数）

| 字段 | 形状 | 来源/语义 |
|---|---|---|
| `xyz` | `[N,3]` | 深度图 + 内参反投影 |
| `source_indices` | `[N]` | 原图像展平索引，用于追踪原像素 |
| `class_id` | `[N]` | 官方类别 Mask，0=背景 |
| `instance_id` | `[N]` | 当前场景 1-based 实例行号，0=背景 |
| `objectness` | `[N]` | `class_id > 0` |
| `candidate_row` | `[N]` | 在该物体 Seal 候选数组中的 0-based 行号 |
| `candidate_id` | `[N]` | 原 Mesh 顶点索引，用于追溯；**不作为 Seal 行号** |
| `match_distance` | `[N]` | 相机观测点到最近合格候选的距离（m） |
| `gt_valid` | `[N]` | 该点是否有可用的物理 GT；False≠负样本 |
| `gt_seal_raw` | `[N]` | 该候选的 20 mm Seal 原始分数 |
| `gt_seal_norm` | `[N]` | 训练集固定上下界归一化 Seal |
| `gt_seal_custom` | `[N]` | 可选 PCA 平整度评分，与主 Seal 分开 |
| `gt_wrench_annot` | `[N]` | `sqrt(2)*rkπ` 阈值对应 Wrench |
| `gt_wrench_eval` | `[N]` | `rkπ` 阈值对应的主 Wrench |
| `gt_wrench` | `[N]` | `gt_wrench_eval` 的别名 |
| `gt_orient` | `[N]` | 接近方向与重力关系的独立分数 |
| `collision_flag` | `[N]` | 已匹配候选是否与场景几何碰撞 |
| `gt_A` | `[N]` | `seal_norm * (~collision_flag)` |
| `gt_B` | `[N]` | `gt_A * gt_wrench_eval` |
| `gt_normal` | `[N,3]` | **原始候选法向旋转至相机坐标系**，与质量评分姿态相同 |
| `normal_valid` | `[N]` | 是否参与主法向监督 |
| `gt_normal_pca` | `[N,3]` | 可选 PCA 法向，不能覆盖主 GT |
| `normal_pca_valid` | `[N]` | PCA 法向有效性 |
| `normals_raw` | `[N,3]` | 深度图局部差分得到的相机观测法向 |
| `normals_raw_valid` | `[N]` | 观测法向是否可用 |
| `schema_version` | 标量字符串 | 文件结构版本 |

**索引示例：** `instance_id=2` 表示场景第 2 个实例，`candidate_row=125` 表示该实例物体 Seal 数组第 126 行，`candidate_id=2476` 表示原 Mesh 顶点 2476。需要读取 `seal["normals"][125]`，不可用 2476 索引候选 Seal 数组。

## 网络输入与监督

- 模型输入可以仅用 `xyz`，如使用观测法向可再取 `normals_raw`，需要考虑 `normals_raw_valid`。
- GT-A/GT-B、Seal、Wrench、`gt_normal` 是标签而非推理输入，**不应泄漏到输入特征**。
- 点级质量损失只使用 `gt_valid=True` 的点，未匹配的点即使数值为 0 也不能当作负样本。
- 对于圆形吸盘，若模型预测的是**接近方向**而不是**物体外法向**，应按统一坐标定义处理符号。
