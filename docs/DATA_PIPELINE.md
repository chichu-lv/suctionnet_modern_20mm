# 20 mm 吸盘训练数据制作流程

范围：官方 10 mm 校验 → 20 mm Seal → 场景 Wrench/Collision → 单视角 Suction Graspness GT → 数据验收。**尚未在完整真实数据集上验收。**

## 1. 准备输入

从 [SuctionNet](https://graspnet.net/) 下载数据，配置目录 `$DATASET`：

```text
models/000/nontextured.ply ... models/087/nontextured.ply
dense_point_clouds/000.npz ... 087.npz
seal_label/000_seal.npz ... 087_seal.npz
wrench_label/0000_wrench.npz ... 0189_wrench.npz
scenes/scene_XXXX/object_id_list.txt
scenes/scene_XXXX/realsense/{depth,label,meta}/FFFF.{png,png,mat}
scenes/scene_XXXX/realsense/{camera_poses.npy,cam0_wrt_table.npy}
```

当官方稠密点云由 `create_dense_pcd.py` 输出为每物体 `points.npy / normals.npy` 时：

```bash
python scripts/pack_dense.py --input-dir "$DATASET/dense_raw/000" \
    --output "$DATASET/dense_point_clouds/000.npz"
```

原始 `suction_annotation` 从官方仓库单独克隆：

```bash
git clone https://github.com/graspnet/suction_annotation.git /workspace/suction_annotation
pip install -e ".[dev]"
```

## 2. 推荐先跑一个小规模场景

以下假设 `$DATASET`、`$UPSTREAM`、`$OUT` 已设置为实际绝对路径。

```bash
# G0+G1：先检查 0、1、2 物体的官方 10mm Seal 回归以及 20mm Seal
python -m suction20.full_cli seal --dataset "$DATASET" --upstream "$UPSTREAM" \
  --output "$OUT" --mode poisson --objects 0,1,2

# G0 历史 Wrench：重放官方 10mm 标注，默认逐点零误差校验
python -m suction20.full_cli wrench-g0 --dataset "$DATASET" --upstream "$UPSTREAM" \
  --output "$OUT" --mode poisson --g0-wrench-scenes 0,1,2

# G1：PCA 法向为可选补充，不替换物理评分使用的候选法向
python -m suction20.full_cli extras --dataset "$DATASET" \
  --output "$OUT" --seal-root "$OUT/seal_r20/poisson" \
  --config configs/suction20.json --objects 0,1,2
```

在做任何 `scene` / `views` 前，需先完成**该场景包含的所有物体**的 Seal 标注。随后：

```bash
python -m suction20.full_cli fit-norm --dataset "$DATASET" \
  --output "$OUT" --seal-root "$OUT/seal_r20/poisson" \
  --config configs/suction20.json

python -m suction20.full_cli scene --dataset "$DATASET" \
  --output "$OUT" --seal-root "$OUT/seal_r20/poisson" \
  --config "$OUT/config_frozen.json" --scenes 0

python -m suction20.full_cli views --dataset "$DATASET" \
  --output "$OUT" --seal-root "$OUT/seal_r20/poisson" \
  --config "$OUT/config_frozen.json" --scenes 0 --frames 0,1

python -m suction20.full_cli stats --dataset "$DATASET" \
  --output "$OUT" --config "$OUT/config_frozen.json" --scenes 0 --frames 0,1
```

完整数据制作（88 物体、190 场景、RealSense 每场景 256 帧）：

```bash
export DATASET=/path/to/SuctionNet
export UPSTREAM=/path/to/suction_annotation
export OUT=/path/to/suction20_output
bash scripts/run_data_pipeline.sh
```

如果已存在旧运行结果、修改过参数或改过源代码，请使用对应阶段 `--overwrite`，或指定新的 `$OUT` 目录。这里只保留直接可读的参数文件与结果，不做哈希指纹校验。

## 3. 数据来源和公式

1. **Seal**：原始 10 mm 环带为 9–11 mm；20 mm 改为 18–22 mm；仍用官方 72 个角度区间及表面拟合和变形评分。候选点和 `indices` 保持不变，不在主 Seal 分支混入 Mesh 射线求交。
2. **Wrench**：保存 `wrench_annot_N`（`sqrt(2)*r*k*pi` 阈值）及 `wrench_eval_N`（`r*k*pi` 阈值），另存 `physics_N`。GT-B 只使用 `wrench_eval_N`。
3. **Collision**：同一场景所有物体和桌面点云，使用分段包络检查 22 mm 杯体及 28 mm 安装/接近区域。
4. **视角匹配**：RGB-D 反投影获得相机 XYZ；通过同实例、5 mm 距离与 30° 法向约束选择最近候选。不先删除负样本。匹配失败为 `gt_valid=False`。
5. **监督目标**：`gt_A = seal_norm * (~collision)`；`gt_B = gt_A * wrench_eval`。归一化由训练场景的物体标签估计并冻结。

**重要校验点：** 官方 `suctionnetAPI` 的 Wrench 计算方式不只是阈值不同，实际分量聚合方式也应与本工程进行基准比对。当前代码继承原项目的切向力矩 L2 形式，不能仅凭字段名就认定其与在线评测器数值完全一致。该项已经列入项目待办。

## 4. 查看标签热力图

```bash
python -m suction20.qa_vis seal \
  --r10 "$OUT/seal_r10/poisson/000_labels.npz" \
  --r20 "$OUT/seal_r20/poisson/000_labels.npz" \
  --output "$OUT/reports/seal_compare_000.png"

python -m suction20.qa_vis view \
  --gt "$OUT/scenes/scene_0000/realsense/graspness_r20_v2/0000.npz" \
  --height 720 --width 1280 --output "$OUT/reports/gt_0000.png"
```

注意：默认脚本只是生成标签，不负责下载第三方数据，也不自动完成实物吸盘标定。
