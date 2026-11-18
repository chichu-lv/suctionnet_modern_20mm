# 路线 A 单物体稀疏点云与 RGB

## 1. 目标和实现状态

路线 A 面向已由上游分割出的单个目标物体，以 RGB、可见深度几何和重力方向预测接触姿态的 Seal、Wrench、纯物理质量及历史动作成败评分。共享数据与标签定义见 [DATA_AND_LABELS.md](DATA_AND_LABELS.md)，训练预算、验证和复现顺序见 [TRAIN_EVAL_REPRODUCE.md](TRAIN_EVAL_REPRODUCE.md)。

本文固定实现基线为 [`87f29d29`](https://github.com/chichu-lv/suctionnet_modern_20mm/commit/87f29d29bea8fa5153d6df3f8253d9ced9e4b441)。以下网络和超参数是**待实现的新设计合同**；现有仓库提供可复用的稀疏骨干、几何算子和任意姿态评价器，尚无路线 A 的完整训练或部署入口。本文中的 teacher 示例调用真实已有 API，训练接口则明确列为新增工作。

新模型命名建议为 `InstanceSuction3D`，使用独立 schema 和 checkpoint。已有 E5 保留用于结果追溯；新输入、新物理目标与新区域结构不能通过旧 E5 的 YAML 开关获得。

## 2. 相对原提案的确定修改

| 原提案 | 本设计 |
|---|---|
| 2048 点承担所有几何 | 2048 个主干点与完整可见实例支撑点分开 |
| 可见点均值作为物理质心 | 均值仅用于平移归一化；Wrench teacher 延续冻结协议 |
| 输入只提供重力 `g_z` | 输入完整三维单位重力向量 |
| 固定法向后直接使用旧标签 | 对同一查询点和同一查询轴重算 S、W |
| 原五区分组删掉外围区域即可 | 新建三径向区域模块，重新定义特征维度与参数组 |
| 将一次成败监督扩散到 20 mm 邻域 | 该邻域提供动作特征；标签只监督记录的动作 |
| 默认乘 `q^0.5` 作为成功概率 | 源域按 q 排名；目标域默认按 task 评分，乘积是另行验证的排序规则 |

去掉全部碰撞项后，质量名称统一为 `Q_SW20`。原 `collision_flag` 同时含目标自碰和场景包络碰撞，因此不能将这一改变描述成“仅删除外部环境碰撞”。纯 SW 不提供场景路径可行性；模型输出与上游/下游既有执行接口保持清晰分工。[原碰撞实现](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/full_data.py)

## 3. 输入与两个点集

### 3.1 单物体样本

一个源域样本对应 `scene_id + frame_id + instance_id`。先依据同一时刻、同一相机的实例掩码选择有效 RGB-D 像素，再反投影为相机系点云。训练/验证/测试划分继承场景或采集组划分，不能在裁出实例后重新随机拆分。

`P_support=[M,3]` 保存全部有效原始可见实例点；RGB、像素索引和法向等数组与其逐行对齐。默认至少保留 100 个有效原始点，低于此数记录 `insufficient_support`；训练不进入常规采样，验证/测试保留身份并将无输出计入主指标失败分母。这一初始阈值不代表已经具备足够唇口覆盖。

中心固定为抽样前全部有效原始观测点的均值：

```text
mu_vis_cam_m = mean(P_support, axis=0)
delta_p = p_cam_m - mu_vis_cam_m
```

`mu_vis` 是随视角变化的可见几何中心，不是质量中心，也不是 teacher 中使用的完整对象中心。所有物理半径仍以米计算；不能按物体大小将坐标缩放到单位球。

### 3.2 主干点

从 `P_support` 中用确定性 FPS 选择最多 2048 个不重复点，形成 `P_backbone`。候选顺序、首点和并列时的原始索引规则固定。训练中若选择随机采样，必须单独命名配置并保存 RNG，不能在复现时悄悄替换 FPS。

不足 2048 点时，固定张量可补零并提供 `point_valid`；体素构建、均值、池化和损失均排除 padding。重复点不得当作额外观测或覆盖证据。[现有 FPS 接口](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/models/geometry_ops.py)

### 3.3 12 维点特征

| 特征 | 通道 | 默认数值处理 |
|---|---:|---|
| `delta_p` | 3 | 米制相对坐标除以固定 0.1 m |
| RGB | 3 | 对齐 sRGB 转 float至[0,1]，再用共同mean/std归一化；见训练规范 |
| 观测外向法向 | 3 | 单位向量；无效行置零 |
| `g_cam` | 3 | 相机系单位重力方向，逐点广播 |

`normal_valid` 是独立旁路掩码，不计入 12 维特征。它控制默认候选资格，并进入局部几何统计；无效法向点仍可贡献颜色和位置上下文。整个模型不读取 `gt_normal` 或 GT 物体完整几何作为输入。

RGB 增强初始设置：brightness/contrast/saturation 幅度各 0.2，hue 幅度 0.05，随机灰度概率 0.2。增强随机状态进入 checkpoint。颜色抖动只能作为正则化，不能保证网络不依赖颜色。

按 2 mm 网格将主干点体素化。坐标为连续内存 `int32[V,4]`，次序 `batch,z,y,x`；特征为 `float32[V,12]`，保留 `inverse_map[N]`。法向平均只使用有效法向，平均后重新归一化；抵消或空集合置零并更新有效性。

## 4. 观测法向与姿态条件

默认在完整 `P_support` 上按[共同法向规范](DATA_AND_LABELS.md)为查询点构建同实例 15 mm 球邻域，以 PCA 最小特征值方向估计法向。要求至少 20 个独立点；将协方差特征值按 `0<=lambda0<=lambda1<=lambda2` 排序，要求有限且 `lambda1>1e-10 m²`，以排除线状/点状退化。曲率 `lambda0/(lambda0+lambda1+lambda2)`仅记录，不设0.15硬筛选。非有限结果和不足邻居均标为无效，不从 GT 补法向。

观测法向用 `dot(n, -p_cam) >= 0` 定号，使其朝向可见相机侧。法向估计必须在 2048 点抽样之前完成。当前仓库的 organized depth normals 使用有限差分及深度跳变门槛，不能将其描述为现成的 15 mm PCA 实现。[观测预处理](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/observation.py)

推理 canonical 查询使用该确定性观测法向。历史动作查询使用记录自身的 `normal_out_cam`，只做合法性检查和单位化；不得重新朝相机翻转或替换为局部观测法向。[历史动作读入](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/target_dataset.py)

网络接触分组、teacher 和评价均使用同一 `(p,n_out)`。固定法向消除了学习轴更新带来的变化，但不会修复旧缓存与新查询之间的差异；原标签轴和位置不同就必须重算。

## 5. 对齐姿态的 SW teacher

### 5.1 所需已有资产

teacher 使用原 SuctionNet 数据根目录中的场景位姿、相机位姿与完整对象 `dense_point_clouds`，以及已有 R20 标签根目录中的 `seal_r20/poisson` 候选和 `config_frozen.json`。这些资产仅用于离线标签与评价，不输入网络。

Seal 沿用冻结 20 mm 定义、18–22 mm 唇口环及 72 个角度分箱。teacher 必须使用完整对象密集几何，不能将网络 2048 点、2 mm 体素或部分可见点云替代它。Wrench 沿用原评价公式、基准质量和完整原候选点均值中心；该中心属于协议代理，不是额外测量的 COM。[任意姿态评价器](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/eval_r20.py)

`Q_SW20=S20*W20`。不能从 `gt_B` 除以碰撞门控恢复 SW；碰撞为零时被抹去的质量无法逆推。旧缓存独立 S/W 只对应其原候选姿态，不自动成为新查询标签。

### 5.2 已有 API 的执行示例

在上述基线环境中，设置 `SUCTION_DATASET_ROOT`、`SUCTION_LABEL_ROOT`、`INSTANCE_QUERY_NPZ` 和 `INSTANCE_LABEL_OUT`。查询 NPZ 由数据构建步骤产生，包含标量 `scene_id/frame_id`、`point_cam_m[K,3]`、`normal_out_cam[K,3]`、布尔 `query_valid[K]`、`query_id[K]` 和零起始 `gt_instance_row[K]`。有效查询必须属于前景；padding保持无标签，输出必须指定新的标签文件。

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import numpy as np
from suction20.eval_r20 import load_scene_eval_context, evaluate_poses

dataset_root = Path(os.environ["SUCTION_DATASET_ROOT"])
label_root = Path(os.environ["SUCTION_LABEL_ROOT"])
output = Path(os.environ["INSTANCE_LABEL_OUT"])
if output.exists():
    raise FileExistsError(f"Use a new output file: {output}")
cfg = json.loads((label_root / "config_frozen.json").read_text())
with np.load(os.environ["INSTANCE_QUERY_NPZ"], allow_pickle=False) as q:
    scene_id, frame_id = int(q["scene_id"]), int(q["frame_id"])
    p = np.asarray(q["point_cam_m"], np.float64)
    n = np.asarray(q["normal_out_cam"], np.float64)
    rows = np.asarray(q["gt_instance_row"])
    valid = np.asarray(q["query_valid"])
    query_id = np.asarray(q["query_id"])
count = len(p)
if (p.shape != (count, 3) or n.shape != p.shape
        or rows.shape != (count,) or query_id.shape != (count,)
        or valid.shape != (count,) or valid.dtype.kind != "b"):
    raise ValueError("Invalid query shapes or query_valid dtype")
if not valid.any() or np.any(rows[valid] < 0):
    raise ValueError("Need valid foreground queries")
if not np.allclose(np.linalg.norm(n[valid], axis=1), 1., atol=1e-5):
    raise ValueError("Query axes must already be unit outward vectors")
ctx = load_scene_eval_context(
    dataset_root, label_root, scene_id,
    camera="realsense", frozen_cfg=cfg,
)
ev = evaluate_poses(ctx, frame_id, p[valid], n[valid], rows[valid])
seal, wrench = np.full(count, np.nan), np.full(count, np.nan)
seal[valid], wrench[valid] = ev["S"], ev["W"]
output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    output, schema_version=np.array("suction20.instance_sw20.v1"),
    quality_contract=np.array("gt_SW20_pose_v1"), query_id=query_id,
    scene_id=scene_id, frame_id=frame_id, gt_instance_row=rows,
    points=p, normal_out=n, seal=seal, wrench=wrench,
    q_sw=seal * wrench, label_valid=valid,
)
print(output)
PY
```

输出使用[参考校验器](reference/geometry_contract.py)的字段 `points,normal_out,seal,wrench,q_sw,label_valid`；输入生产查询仍用 `point_cam_m,normal_out_cam`，两组命名通过此处显式转换。S/W同时生成，共用label_valid；padding的质量为NaN且mask=False，不能补零当负例。该示例产生数值NPZ，完整缓存构建器仍须按共同规范另存身份sidecar。

该参考函数会顺带计算原碰撞结果，但新质量不使用它。后续实例专用 teacher 可复用 `seal_at_pose` 和 `wrench_scores_dual` 消除这部分开销，先与本参考路径核对数值。

缓存身份至少绑定查询点/轴、源观测/实例、法向算法与参数、冻结物理配置和 teacher 代码版本。颜色增强不改变标签；刚性变换要一致变换 p、n、g 并记录逆变换；改变点或重新估计轴后，旧姿态标签不得继续使用。

## 6. 网络与张量尺寸

### 6.1 编码和全局上下文

新 wrapper 将固定 ResUNet14 的输入改为 12 通道，保持其 512 维输出和 512→256 投影。每个主干点得到 `F[N,256]`。当前 wrapper 的输入为硬编码 8 通道，必须新增实现，不编辑受 blob 校验约束的外部源码。[骨干 wrapper](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/models/backbones/spconv_resunet14.py)

对真实主干点做 masked mean 和 max，拼成 512 维后由 MLP 512→128→64 得到对象上下文 `G[64]`。这提供全物体可见形状统计，不调用旧 `instance_fusion`，也不把可见形状等同完整物体质量分布。

### 6.2 三径向区域接触编码

每个查询以精确 p 为中心、n 为外向轴建立局部正交坐标；网络观测支持的轴向截断统一为 `|h|<=40 mm`，与路线B及参考工具一致。这是输入几何的有限窗口，不改变teacher中无此高度截断的物理Seal公式。

| 区域 | 径向边界 | 每查询最多采样 | 区域输出 |
|---|---|---:|---:|
| core | 0≤rho<10 mm | 32 | 64 |
| inner | 10≤rho<18 mm | 32 | 64 |
| lip | 18≤rho≤22 mm | 64 | 64 |

查询只在完整同实例 `P_support` 内进行。保存截断前区域点数、sample mask、法向有效性和 lip 的 8 扇区观测计数。8 扇区是网络描述，72 分箱是物理 teacher；两者用途不同。

支撑点特征由主干点特征做同实例确定性 kNN3 逆距离插值，零距离直接取同点特征，距离相同按原索引定序。初始最大关联距离为 10 mm；超限返回无效特征标记，并仍保留真实局部几何。该插值是新增模块，不能用稀疏主干索引直接读取完整支撑点数组。

每个有效局部点用 256→64 投影特征拼接 7 维描述 `(x/r,y/r,h/40mm,rho/r,n_obs·n,feature_valid,normal_valid)`，其中r取该径向区域的外半径，再经 MLP 71→64→64。feature_valid表示10mm以内能关联主干特征，同实例条件已由查询保证。区域 masked mean/max 加 `log1p(count)` 与非空标记共 130 维，经 130→64 得到区域特征。空区域输出零并保留 mask；padding 不增加计数。新增MLP隐藏层使用ReLU、带bias、无dropout，输出层按对应表指定激活。

### 6.3 姿态特征和输出头

查询主干特征 `Fq[256]`、G[64]、三块区域特征共192、lip计数8，以及 `delta_p/0.1,n,g,n·g` 共10维，拼成530维，经 MLP 530→256→256得到 h。

| 头 | 输入/层宽 | 输出语义 |
|---|---|---|
| q | 256→128→1，sigmoid | 同姿态纯 SW 代理质量 |
| seal | 三块区域与计数共200→128→1，sigmoid | 归一化 Seal |
| wrench | h、G、delta_p/0.1、n、g 共329→128→1，sigmoid | 冻结协议的 Wrench |
| ell | 256→128→1 | Laplace 残差尺度的对数 |
| task | h、n、n·g 共260→128→1 | 成败 logit，推理时 sigmoid |

现有 `heads.py` 不包含 ell 头；上述头需要新增。ell 初始化为 `log(0.1)`，前向限定在 [-6,1]，不能称为任务成败的校准方差。[已有模型和评分头](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/models/suction20net.py)

## 7. 损失与精确历史动作查询

源域单查询目标为 `exp(-ell)*abs(q-Q_SW20)+ell+0.5*SmoothL1(seal,S20,beta=.05)+0.25*SmoothL1(wrench,W20,beta=.05)`。它使用 Laplace log-scale，不将 ell 称为 log-variance。没有碰撞损失或学习轴损失；可用标签的归约规则与预算见共享训练规范。

保留两个职责明确的新接口：

```python
encoded = model.encode_instance(observation)
scores = model.score_at_pose(encoded, p_cam_m, n_out_cam, query_batch_index)
```

`score_at_pose` 支持任意有观测支持的 p/n。历史动作按原 `planned_pipeline_contact` 转换结果直接查询；不要求注入256候选，也不将特征关联点替换成动作中心。特征无法关联时记录 `pose_feature_valid=False`，跳过该条监督，不能当作失败负例。

目标域以该动作的 task logit 计算一次 BCEWithLogits，混入0.5源域回放。前200步/后800步的参数组按新模型重新定义，必须覆盖全部可训练参数且互斥。没有邻域标签扩散，也不以单次失败推断 Seal/Wrench 必定失败。[已有精确动作微调语义](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/finetuning.py)

## 8. 推理到机器人位姿

在完整 support 的法向有效池确定性选取最多256个 canonical 候选，并以观测法向评分。源域模型按 q 排名；目标域模型默认按 task probability 排名，q、Seal、Wrench、ell同时返回用于分析。单次前向不采用 MC Dropout。

保留物理距离与角度 NMS，初始阈值为10mm、15°；阈值作用于相机系真实坐标和轴。无有效候选时输出 `status=no_pose, reason=no_candidate`。若采用 `P_task*q^alpha`，alpha必须在开发阶段固定，且该乘积只称排序分数。

恢复相机接触点 `t=delta_p+mu_vis`。统一工具+Z为接近方向，故 `z_tool=-n_out`；选与z最不平行的相机基向量b（同值按x、y、z顺序），令`x=normalize(b×z_tool)`、`y=z_tool×x`、`R=[x,y,z_tool]`，与参考工具的固定roll一致。点与法向本身仅约束5自由度，roll由上述规则或既有工具约束固定。

返回 `T_cam_contact`、外向轴、分数、实例身份与预处理版本；机器人变换为 `T_base_tcp=T_base_cam*T_cam_contact*T_contact_tcp`。接触中心不能直接当作TCP原点，历史动作与新输出必须使用同一工具标定定义。

## 9. 落地清单与验收

新增模块包括 instance dataset/collate、12通道骨干wrapper、三径向区域grouping、精确pose评分、SW loss与teacher缓存、独立参数分组及训练入口。现有 E0–E5 结构校验和 source_losses 只接受原目标；不能把未实现的路线命令写成现成可执行功能。

先通过共享几何契约示例，再核验一个真实实例的 p/n 重评、padding与空区域、历史动作保真及固定划分；之后按 [训练与复现规范](TRAIN_EVAL_REPRODUCE.md) 实施新入口。新 checkpoint 默认从新架构初始化，不通过已退役的迁移兼容路径恢复E5。

报告 batch1 的 model-only 与端到端 p50/p95，列出硬件、点数、预处理、支撑点查询、传输和NMS边界。原15–30ms保留为工程目标；在测量前不作为已实现性能或训练收敛承诺。

