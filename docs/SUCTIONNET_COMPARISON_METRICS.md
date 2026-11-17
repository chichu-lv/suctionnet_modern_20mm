# SuctionNet 方法对照与评价指标计算规范

更新：2026-11-17（北京时间）。项目依据：`cd010/from-scratch-performance`，提交 `ba3d6463386cd7287550e3464358ad4825003104`。

本文选择四项使用 SuctionNet-1Billion 的代表工作，确定其评价指标，并给出可直接用于计算的公式、输入格式和汇总方法。范围是**指标与比较口径**；本次不复现对方网络，不启动训练或补测。

## 1. 选择哪些方法

| 方法 | 年份与代表性 | SuctionNet 使用证据 | 本次对照的指标 | 来源 |
|---|---|---|---|---|
| **SuctionNet 原基线 / S1B** | 2021；基准提出者，RGB-D 稠密吸取预测 | 原论文 §III-C、表 IV | Top-50 与 Top-1 的 AP、AP₀.₄、AP₀.₈；实机 R_grasp、R_object、R_mixture | [论文][S1]、[官方评价代码][S2] |
| **UISN**：Uncertainty-Aware Suction Grasping for Cluttered Scenes | 2024；实例点云、不确定性与候选选择，与本项目实例路线相关 | 作者安装官方 API，`eval.py` 直接调用三个官方测试集评价函数 | 作者脚本返回 Top-50 AP、Top-1 AP；阈值分项由同一 API 产生 | [作者文献页][S6]、[评价脚本][S7] |
| **Diffusion-Suction** | 2025；扩散模型生成吸取评分，代表不同建模路线 | 论文 Table I 为 SuctionNet，V-A 明确沿用其协议，作者 README 链接官方评价代码 | Top-50 / Top-1 各自的 AP、AP₀.₄、AP₀.₈；实机 R_grasp、R_object；推理时间 | [论文 v2][S9]、[作者仓库][S10] |
| **MultiGraspNet-vac. / MultiGraspNet** | 2026；稀疏 3D U-Net，单吸取与双夹具联合训练 | 论文 §IV-A、Table I 的 Vacuum 部分使用对齐后的 SuctionNet RealSense 数据 | Top-50 AP、AP₀.₄、AP₀.₈；实机 R_grasp、R_object、R_mix、R_seen；参数量、模型大小、FPS、延迟 | [论文 v2][S11]、[作者项目][S12] |

**建议主对照采用这四项。** MultiGraspNet 优先取单吸取版本 `MultiGraspNet-vac.`，完整多任务版本可另列一行，不把两者混成同一配置。前三项的公开评价链路指向官方 API；MultiGraspNet 的聚合公式已由论文确认，底层物理实现尚不能由已发布源码逐项确认，具体区别见第 6 节。

如需增加历史参照，可选 **Dex-Net 3.0（SuctionNet 官方评测版本）**。SuctionNet 原论文直接评测其既有预训练模型，不能写成“Dex-Net 3.0 原论文在 SuctionNet 上训练”。FCN/Zeng 也应写成“SuctionNet 适配版”，其原工作早于此数据集。[S1]

## 2. 先统一指标名称

| 建议名称 | 实际含义 | 不能混用的名称/解释 |
|---|---|---|
| `AP (Top-50)` | 对 k=1…50 的 Precision@k 求均值，再对四个物理质量阈值求均值 | 不是 P@50；不是 PR 曲线面积 |
| `AP₀.₄ (Top-50)` | 固定物理质量阈值 0.4，平均 Precision@1…50 | 0.4 不是模型分数筛选阈值，也不是 IoU |
| `AP₀.₈ (Top-50)` | 同上，质量阈值改为 0.8 | 不是检测 AP80 |
| `AP (Top-1)` | 每帧全场景最高模型分数姿态，在四个质量阈值下的通过率均值 | 不是每实例 Top-1；不是固定阈值成功率 |
| `AP₀.₄ (Top-1)` | 每帧全场景最高分姿态的物理质量是否 ≥0.4 | 不是实际机器人执行成功率 |
| `attempt_PR_AUC` / classification AP | 目标历史动作上的阶梯式 precision–recall 平均精度 | 与上面所有 SuctionNet AP 不同 |

正文与表头统一写 **Top-50 / Top-1**，避免裸写 `AP50`；本次所选吸取基准没有默认的 Top-100 主指标。官方阈值集合为 `{0.2, 0.4, 0.6, 0.8}`，不是 COCO 的 IoU 阈值集合。[S1][S2]

## 3. 官方 SuctionNet 指标的完整计算链路

### 3.1 数据、划分和统计单位

| 集合 | 场景编号 | 每相机帧数 | 用途 |
|---|---|---:|---|
| 官方 train | 0000–0099 | 100 × 256 = 25,600 | 原始训练划分 |
| Test-Seen | 0100–0129 | 30 × 256 = 7,680 | 已见物体、新场景 |
| Test-Similar | 0130–0159 | 7,680 | 相似但未见物体 |
| Test-Novel | 0160–0189 | 7,680 | 新物体 |

Kinect 与 RealSense 分别报告；单相机全部测试是 23,040 帧。原论文以及 Diffusion-Suction 表中的 `Kinect/RealSense` 是两个相机结果，不是均值/标准差。官方每个子集的评价张量为 `[30 scenes, 256 views, 50 prefixes, 4 thresholds]`。[S1][S2][S9]

本项目首先对照 **RealSense**，与现有训练输入一致；不因整理指标额外安排另一相机的训练。

### 3.2 区分预测分数与评价质量

对一个预测吸取姿态，记：

- `p`：接触点，原始相机坐标系，单位米；
- `n`：单位外向吸取轴；
- `s_hat`：模型给出的排序分数；
- `Q_official(p,n)`：评价器用真实场景几何在该姿态上重新计算的物理质量。

**先按 `s_hat` 排序，再用 `Q_official` 判断正确与否。** 不能按重算后的物理真值排序，不能直接以网络分数是否大于 0.4 判断预测正确，也不能用附近预生成标签替代当前 `(p,n)` 的评价。[S2][S3]

### 3.3 一帧的官方处理顺序

1. 读取该帧预测的分数、方向与位置。
2. 运行全局 NMS：平移阈值 **0.02 m**，方向阈值 **181°**。在正常方向夹角范围内，后者相当于不因方向不同而保留距离很近的候选。
3. 用 GT 物体位姿构建相机系场景模型。根据最近模型点，将预测位置关联到物体；用于关联的物体模型按 **5 mm** 下采样。
4. 每个关联物体按模型分数最多保留 **10** 个候选。
5. 使用场景模型及桌面检查碰撞，按预测位置和方向在该物体稠密模型上重算 Seal、Wrench。
6. 合并各物体候选，重新按模型分数降序排列，统计前 1…50 个前缀。

碰撞候选的质量置零，**继续占据排名位置**，不删除后再用后面的好候选补位。评价中的 GT 模型用于判定质量和物体归属，不等于允许将 GT 几何作为网络输入。[S2][S3]

### 3.4 公式

设帧 `v` 经过上述处理后有 `M_v` 个候选，其顺序由模型分数决定。令 `T={0.2,0.4,0.6,0.8}`，物理质量为：

$$
Q_{v,i}=S_{\mathrm{seal},v,i}S_{\mathrm{wrench},v,i},
$$

其中碰撞或无法形成有效接触时由评价器置零。固定质量阈值 `t`：

$$
P_v(k,t)=\frac{1}{k}\sum_{i=1}^{\min(k,M_v)}\mathbf{1}[Q_{v,i}\ge t].
$$

候选不足 `k` 个时，分母仍是 `k`，缺位相当于失败。官方实现使用 `>=`；计算时遵循代码的这一边界。[S2]

对测试帧集合 `V`：

$$
AP_t^{(50)}=\frac{1}{|V|}\sum_{v\in V}\frac{1}{50}\sum_{k=1}^{50}P_v(k,t),
\qquad
AP^{(50)}=\frac{1}{4}\sum_{t\in T}AP_t^{(50)}.
$$

Top-1 则为：

$$
AP_t^{(1)}=\frac{1}{|V|}\sum_{v\in V}P_v(1,t),
\qquad
AP^{(1)}=\frac{1}{4}\sum_{t\in T}AP_t^{(1)}.
$$

三个测试子集分别汇总。完整官方子集中每场景均为 256 帧，所以“所有帧等权平均”与“场景内先平均，再对场景等权平均”一致。输出 JSON 建议保留 `[0,1]` 小数，论文表格乘 100 后标 `%`。[S2]

### 3.5 两个小例子

**例一：Top-50 AP 不等于 P@50。** 仅为演示，暂把最大前缀改成 4，某阈值下按模型排序得到 `[通过, 失败, 通过, 失败]`。四个 precision 为 `1, 1/2, 2/3, 1/2`，前缀平均为 **66.67%**，而 P@4 为 **50%**。正式指标仍使用 50 个前缀。

**例二：Top-1 AP 不等于 0.4 阈值通过率。** 若一帧最高分姿态的物理质量为 `0.65`，它通过 0.2、0.4、0.6 三个阈值，未通过 0.8，因此该帧 `AP (Top-1)=0.75`，而 `AP₀.₄ (Top-1)=1`。这两个数也都不是该动作真实执行成功概率。

### 3.6 可直接使用的聚合代码

以下是本文按公式编写的聚合示例；**输入质量必须已由正确的评价器计算，并保持模型排序**。它不实现或替代物理评价器。

```python
import numpy as np

THRESHOLDS = np.array([0.2, 0.4, 0.6, 0.8])

def precision_grid(quality_in_prediction_order):
    q = np.asarray(quality_in_prediction_order, dtype=np.float64)[:50]
    hits = np.zeros((50, 4), dtype=np.float64)
    hits[:len(q)] = q[:, None] >= THRESHOLDS[None, :]
    return hits.cumsum(axis=0) / np.arange(1, 51)[:, None]

def summarize_official_result(res):
    # 官方单个 split 的 res 形状为 [30, 256, 50, 4]。
    # res 保存的是每个前缀的 precision，而不是原始质量。
    a = np.asarray(res, dtype=np.float64)
    return {
        "AP_top50": float(a.mean()),
        "AP04_top50": float(a[..., :, 1].mean()),
        "AP08_top50": float(a[..., :, 3].mean()),
        "AP_top1": float(a[..., 0, :].mean()),
        "AP04_top1": float(a[..., 0, 1].mean()),
        "AP08_top1": float(a[..., 0, 3].mean()),
    }
```

若 `res` 已经是官方返回的 precision 张量，直接汇总即可，不能再做一次 `cumsum`。如从逐候选质量重新汇总，则对每帧调用 `precision_grid` 后按场景、视角组织结果。

## 4. 官方物理评价究竟计算什么

本文的 **Official SuctionNet** 固定指官方 API 提交：

```text
graspnet/suctionnetAPI
0dac9f7e5b2f7ea801cddf9655cb82a259681976
```

### 4.1 几何与物理参数

| 项目 | 官方发布实现 |
|---|---|
| 吸盘半径 | **10 mm**，不是直径 10 mm |
| 唇口环带 | `9 mm < rho < 11 mm` |
| 环带采样 | 72 扇区，各取沿吸取轴的最高点；缺扇区则 Seal 为零 |
| 接触内部突出 | `rho < 10 mm` 且沿外向轴高度 `>5 mm` 时失败 |
| 物体中心 | 关联物体稠密模型点的均值 |
| Wrench 参数 | `k=15.6`、`g=9.8`，阈值 `pi*r*k` |
| 外部碰撞包络 | 半径 10 mm，沿吸取轴 `10 mm < z < 100 mm` 的圆柱 |
| Seal 归一化 | 评价函数直接返回其 Seal 公式结果，不使用本项目的冻结 `lo/hi` 再归一化 |

以上参数来自 `eval_utils.py`；物体/桌面构建和相机变换来自 `suctionnet_eval.py`。[S2][S3]

### 4.2 公式解释

令理想环上的相邻顶点弦长为 `l_j`，加上各扇区实际高度后的三维边长为 `l'_j`，高度为 `z_j`。当前 API 的 Seal 可写成：

$$
D=\min_j\frac{l_j}{l'_j},\quad
F=\exp\left[-10^5\frac{1}{72}\sum_{j=1}^{72}(z_j-\bar z)^2\right],\quad
S_{\rm seal}=D F.
$$

这里长度单位是米。代码实际拟合常高度环，而不是让平面自由倾斜。切向力矩在 API 的吸盘局部坐标中记为 `tau_y,tau_z`，Wrench 为：

$$
S_{\rm wrench}=1-\min\left(1,\frac{\max(|\tau_y|,|\tau_z|)}{\pi r k}\right).
$$

这些是**发布实现的等价解释**。它与论文中形变表达及力矩 L2 范数不完全一致。代码还用含平移的点变换函数生成重力方向后归一化；单独将它“修正”为纯旋转也会改变评价版本。因此，要与公开论文表格对照，直接调用固定官方 API；本文不要求修改这些历史实现，也不建议自行按论文重写一套后仍称官方指标。[S1][S3]

## 5. 从本项目输出计算官方指标的接口

这一节是后续评价接线说明，不是对方网络的复现方案。

### 5.1 预测数组与路径

实际官方代码和 baseline 使用：

```text
每行 Nx7：
[pred_score, nx, ny, nz, px, py, pz]

文件：
dump_root/test_seen/scene_0100/realsense/suction/0000.npz
dump_root/test_similar/scene_0130/realsense/suction/0000.npz
dump_root/test_novel/scene_0160/realsense/suction/0000.npz

每场景帧编号：0000…0255
NPZ 数组键：arr_0
```

**方向在前、位置在后。** 官方 README 的文字顺序与实际读取代码不一致，应以 `suction.py` 和 baseline 导出代码为准。位置使用原始相机系、单位米；方向是单位外向轴。[S4][S5]

### 5.2 本项目的导出位置

现有 [`predict_sample`](../suction20/validation.py) 已返回 `point_cam_m`、`normal_out_cam`、`q`，且已变换回原始相机系。官方导出应接在它之后、**项目 `nms_prediction` 之前**，让官方评价器执行自己的 NMS 与物体上限。

```python
from pathlib import Path
import numpy as np

# prediction = predict_sample(...)，无训练增强，使用 inference 候选模式。
# 已剔除 seed padding；此处保留全部有效提议，不先取项目 Top-50。
score = prediction["q"]           # 公开预训练对照使用物理质量头
normal = prediction["normal_out_cam"]
point = prediction["point_cam_m"]
arr = np.column_stack([score, normal, point]).astype(np.float32)

path = (Path(dump_root) / split / f"scene_{scene_id:04d}"
        / camera / "suction" / f"{frame_id:04d}.npz")
path.parent.mkdir(parents=True, exist_ok=True)
np.savez(path, arr)                # 默认键为 arr_0
```

物理阈值不用于预先过滤预测；不使用 GT 质量过滤候选，也不将评价器算出的质量写进 `pred_score`。任务微调头如果另做公开数据评价，应明确写出其 checkpoint 和排序分数，不能与 pretrained `q` 结果混名。

### 5.3 官方评价调用

```python
from suctionnetAPI import SuctionNetEval

evaluator = SuctionNetEval(root=dataset_root, camera="realsense")
res_seen, _, _ = evaluator.eval_seen(dump_folder=dump_root, proc=4)
res_similar, _, _ = evaluator.eval_similar(dump_folder=dump_root, proc=4)
res_novel, _, _ = evaluator.eval_novel(dump_folder=dump_root, proc=4)

np.save("official_seen_precision.npy", res_seen)
np.save("official_similar_precision.npy", res_similar)
np.save("official_novel_precision.npy", res_novel)

seen_metrics = summarize_official_result(res_seen)
similar_metrics = summarize_official_result(res_similar)
novel_metrics = summarize_official_result(res_novel)
```

`proc` 只是评价进程数，不改变公式。后续接入只需保存返回的 precision 张量，六个主指标即可一次算全；不用为每个阈值重新推理。[S2][S7]

## 6. 每种方法具体需要注意什么

### 6.1 SuctionNet 原基线

以原论文 **Table IV** 的测试结果为主，读取 Top-50/Top-1、Seen/Similar/Novel 和 Kinect/RealSense 对应列；正文 AP 定义见 §III-C。物理公式帮助理解，数值计算采用第 4 节固定 API。不要把 RGB 消融表中的其他设置混入主模型行。[S1]

### 6.2 UISN

作者 `eval.py` 直接调用官方 `eval_seen/eval_similar/eval_novel`，并保存返回的 `res`。发布的评价 shell 脚本使用 RealSense；Python 入口也支持 Kinect。本文确认的是**作者公开脚本的计算方式**；没有据此补写未核对的论文表格数值或实机指标。[S7]

公开推理代码中，不确定性校准的 `mean_score * (1-normalized_uncertainty)` 用于实例内预选 200 点，实际提交数组首列保存的是相应点的 `mean_score`。评价器再按提交分数排序。这改变候选选择，但不改变官方物理质量、阈值与 AP 定义。[S8]

### 6.3 Diffusion-Suction

**只用 Table I 的 SuctionNet 实验对照公开基准。** 该表分别报告两个相机、三个 split，以及 Top-50/Top-1 的三类 AP。V-A 和作者 README 指向官方评价协议。[S9][S10]

Table II 是自建 Parcel-Suction-Dataset，其标签/评价额外包含 visibility，并有自定义 Wrench 处理。该表的 94.42%（Top-50 AP）和 98.00%（Top-1 AP）不是 SuctionNet 成绩；例如 SuctionNet 的 RealSense Seen Top-50 AP 是 Table I 的 31.90%。这些数字在此仅用于识别表格，不与本项目当前 R20 开发结果直接排名。[S9]

### 6.4 MultiGraspNet

使用 Table I 的 **Vacuum** 部分，优先 `MultiGraspNet-vac.`；`MultiGraspNet` 是联合两任务训练的版本。表格使用 RealSense。§IV-A 给出 `k=1…50`、吸取阈值 0.2/0.4/0.6/0.8 和均值 AP；未把 Top-1 作为该主表列。[S11]

有一处定义需保留：正文称吸取阈值为 **seal coefficient**，而官方 API 阈值化的是 Seal×Wrench。它声明沿用 SuctionNet，但作者项目读取时仍标代码待发布，因此这里只确认其**论文报告与聚合形式**，不宣称其底层 evaluator 已与官方逐项核对。我们的指标计算统一采用已公开的官方 API；如以后取得该方法原始输出，可用同一评价器重算，而不是从术语差异推断一个不存在的“已验证完全等价”。[S11][S12]

## 7. 实机指标与效率指标

### 7.1 实机指标的分子、分母

设 `N_attempt` 为实际尝试数、`N_success` 为成功次数、`N_object` 为初始物体数、`N_clear` 为最终清除数、`N_attempt_clear` 为最终清除物体上消耗的尝试数。

| 指标 | 公式 | 方向 | 本次文献中的使用 |
|---|---|---|---|
| 抓取成功率 `R_grasp` | `N_success / N_attempt` | ↑ | SuctionNet、Diffusion-Suction、MultiGraspNet |
| 清场率 `R_object` | `N_clear / N_object` | ↑ | 上述三项 |
| SuctionNet `R_mixture` | `N_clear / N_attempt_clear` | ↑ | 原 SuctionNet |
| MultiGraspNet `R_mix` | `N_attempt_clear / N_clear` | ↓ | MultiGraspNet |
| MultiGraspNet `R_seen` | `N_detected / N_object` | ↑ | 感知到的物体比例，与 test_seen 划分无关 |

**两篇论文的 mixture/mix 方向相反。** 按各自文字定义，在相同计数下互为倒数，不能作为同名百分率直接比较。MultiGraspNet 没有在该指标段详细规定检测实例跨帧合并规则，因此本文不补造其实现。[S1][S11]

这些论文的清场实验使用连续三次无法抓取物体的停止条件；实验统计必须包含实际动作结果。我们的历史规划动作 AUROC、几何 AP、模型最高预测分，都不能代入这些分子。当前任务是整理指标，**不以此安排新一轮机器人试验**。[S1][S9][S11]

### 7.2 效率指标

Diffusion-Suction 报告推理时间；MultiGraspNet Table IV 报告参数量、模型大小、FPS 和延迟。[S9][S11]

| 指标 | 计算口径 | 本项目记录建议 |
|---|---|---|
| 参数量 | `sum(p.numel() for p in model.parameters())` | 总参数与可训练参数需要时分列 |
| Latency | 完成一帧预测所需墙钟时间 | batch=1；写明预处理、实例分割、传输、NMS 是否计入 |
| FPS | 完成帧数 / 总计时秒数 | 仅在同一计时边界、串行 batch=1 时可用 `1000/平均延迟ms` 换算 |
| 模型大小 | 实际保存权重文件大小，或明确的参数存储估计 | 注明格式/精度；不是 GPU 峰值显存 |

已有训练吞吐“样本/秒”和训练显存记录不能填入上述推理表。不同论文的设备和计时范围不同，文献时间可作原文报告值，不能直接据此判断本项目加速比。

## 8. 与本项目当前指标的对应关系

### 8.1 哪些一致，哪些不同

| 项目 | 官方 SuctionNet | 本项目当前 R20 |
|---|---|---|
| 吸盘半径 | 10 mm | 20 mm |
| 接触几何 | 官方 R10 环带与碰撞圆柱 | 冻结的 R20 环带、内部碰撞及 cup/mount/approach 包络 |
| Seal | API 返回值 | 按已有冻结 `seal_norm_min/max` 归一化后裁剪 |
| Wrench 力矩范数 | 两切向分量绝对值的 max | 切向力矩 L2 |
| 评价中心 | 稠密模型点均值 | 既有 Seal 候选位置均值 |
| NMS | 全局 20 mm、181° | 每输入实例 10 mm、15°；位置与方向同时满足才抑制 |
| 每物体/实例上限 | 按评价器关联物体，每物体 10 个 | 按输入实例 ID，每实例 10 个 |
| AP 聚合 | 4 阈值 × 50 个前缀 | 同样的聚合结构 |
| Top-1 | 每帧全场景最高分，通常再平均 4 阈值 | **每可见 GT 实例**最高分、固定 Q20≥0.4；在全场景 Top-50 截断前取 |
| 当前已发布结果 | 论文通常用官方测试集 | 本项目为 10 场景 ×32 帧开发验证 |

依据：[官方 API][S2][S3]、[项目指标实现](../suction20/metrics.py)、[R20 物理评价](../suction20/eval_r20.py)、[实验规格](implementation/EXPERIMENTS.md)。

因此，**现有 `AP_R20` 可以说采用 SuctionNet 式 AP 聚合，不能直接称为官方 SuctionNet AP。** 仅把当前评价器的半径参数改成 10 mm，也不会自动还原其余规则。

例如已选 E5 best 的 **13.12%** 是开发集每实例 Top-1 Q20≥0.4；其 **10.60%** 是该模型的开发集 AP-R20。两者都不能与论文 RealSense/Seen 的官方 AP 数字直接比较大小。[已有实验报告](experiments/CD010-20261117/pretraining/REPORT.md)

### 8.2 实例输入和训练范围也要明确

本轮公开预训练验证使用 GT 实例 mask；UISN 的实例输入来自 UOIS/UOAIS，其他方法也有各自可用输入。官方评估器内部使用 GT 模型判分，并不能消除“GT mask 已经进入网络”这一输入差异。[S6][S8][已有实验资料](experiments/CD010-20261117/pretraining/REPORT.md)

若后续使用已有前置预测 mask 做公开对照，写明分割来源；若继续用 GT mask，标为 **oracle-instance input**，保留为相应条件结果。无需为此训练新的分割模型。现有权重训练使用 90 个官方训练场景、10 个场景留作开发验证，也应如实说明，不为补齐文献训练数量而重新训练。

### 8.3 两套报告各回答一个问题

**公开文献对照表：** 在相同相机、官方测试 split 和固定官方 R10 API 下，报告第 3 节的六个分数。用现有 R20 训练权重评价时标注 **R20-trained / official R10 evaluation**；模型自身的 20 mm 接触编码保持已训练语义，只把最终姿态交给 R10 基准判分。这回答跨基准表现，不自动等于我们的 20 mm 业务效果。

**项目 R20 表：** 保留当前物理评价和 AP-R20、每实例 Top-1、碰撞率；如需论文式读法，可从同一批逐候选 Q20 额外聚合 `R20 AP₀.₄/₀.₈ (Top-50)` 和 `R20 scene AP (Top-1)`。新列明确标 R20，不能填入文献 R10 列。比较其他方法的 R20 效果需要其预测姿态在同一 R20 评价器下重算，仅有论文表格无法完成换算。

这两套口径均不要求复现或重训其他方法。**本次只确定规范，不启动这两套评估。**

## 9. 建议最终保留的最小指标集合

### 9.1 公开基准主表

每种方法/配置、每个相机、每个 split 一行：

```text
method, variant, camera, split, input_mask_source,
train_scene_count, training_target, evaluator_commit,
AP_top50, AP04_top50, AP08_top50,
AP_top1, AP04_top1, AP08_top1
```

只有在原文或原始评价张量中确有相应结果时填写；例如 MultiGraspNet 主表未报 Top-1，就留空，不由 Top-50 AP 推算。

### 9.2 项目与微调结果

| 使用场景 | 建议继续保留 |
|---|---|
| R20 物理能力 | `AP_R20`、`instance_Top1_Q20_at_0_4`、`collision_rate`、`no_candidate_rate` |
| 记录动作的目标适配 | `attempt_AUROC`、`attempt_PR_AUC`；训练过的任务概率头可报 BCE、Brier |
| 实际执行记录 | 成功次数/尝试次数；有清场任务时加清除物体数/初始物体数 |

项目 `attempt_PR_AUC` 的实现为 `sum((recall_j-recall_{j-1}) * precision_j)`，按相同预测分数分组后累计，属于分类 average precision；`attempt_AUROC` 是另一类排序指标。这些不是公开论文的物理 AP，保留原名即可。[项目实现](../suction20/metrics.py)

当前发布的逐帧预训练验证记录只保留聚合指标，没有完整逐候选姿态和物理分解，不能从现有 `AP_R20` 一个标量反推出 AP₀.₄、AP₀.₈ 或整帧 Top-1。已有官方 `res` 张量可直接汇总；已有完整姿态可直接评价；只有汇总日志的部分，需要以后对选定冻结模型做一次推理/评价导出，**不需要重新训练，也不需要将六组实验全部再跑一遍**。[已有数据说明](experiments/CD010-20261117/pretraining/REPORT.md)

## 10. 来源、版本与核对范围

本文优先使用论文、作者仓库和官方评价代码。UISN 的指标定义由作者公开脚本确认；Diffusion-Suction 由论文与作者 API 链接确认；MultiGraspNet 的聚合定义来自论文，其“seal coefficient”术语差异保留在第 6 节，没有假定未公开实现的细节。

| 编号 | 一手来源 | 核对位置 |
|---|---|---|
| S1 | [SuctionNet-1Billion 论文][S1] | §III-C、§V-B、Table IV、§V-E |
| S2 | [官方 `suctionnet_eval.py`][S2] | 固定 `0dac9f7`；阈值、视角、AP、三个 split |
| S3 | [官方 `eval_utils.py`][S3] | 同提交；NMS、关联、物体上限、Seal/Wrench/碰撞 |
| S4 | [官方 `suction.py`][S4] | 同提交；数组列顺序、NPZ 读取 |
| S5 | [官方 baseline 导出][S5] | 固定 `93d604d`；score/normal/point 拼接 |
| S6 | [UISN 作者机构文献页][S6] | RA-L 2024、方法与 SuctionNet 数据使用 |
| S7 | [UISN `eval.py`][S7] | 固定 `4ad50bf`；直接调用官方 evaluator |
| S8 | [UISN `inference.py`][S8] | 同提交；候选校准、提交分数、划分与输出 |
| S9 | [Diffusion-Suction 论文 v2][S9] | Table I/II、V-A、V-E |
| S10 | [Diffusion-Suction README][S10] | 固定 `c91f0c7`；evaluation metric 链接 |
| S11 | [MultiGraspNet 论文 v2][S11] | IV-A、Table I、IV-D/E |
| S12 | [MultiGraspNet 作者项目][S12] | 论文/代码入口状态 |

[S1]: https://arxiv.org/pdf/2103.12311
[S2]: https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/suctionnet_eval.py
[S3]: https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/utils/eval_utils.py
[S4]: https://github.com/graspnet/suctionnetAPI/blob/0dac9f7e5b2f7ea801cddf9655cb82a259681976/suctionnetAPI/suction.py
[S5]: https://github.com/graspnet/suctionnet-baseline/blob/93d604dd752d2ef865c80c8ecba96218560b516f/neural_network/inference.py
[S6]: https://research.cuhk.edu.hk/en/publications/uncertainty-aware-suction-grasping-for-cluttered-scenes-2/
[S7]: https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/eval.py
[S8]: https://github.com/rcao-hk/UISN/blob/4ad50bfb4c29fc47f1e80eb8e4d95e666e8e48f2/inference.py
[S9]: https://arxiv.org/html/2502.07238v2
[S10]: https://github.com/TAO-TAO-TAO-TAO-TAO/Diffusion_Suction/blob/c91f0c76b54aedeb22c29f0ecf100efe9198c827/README.md
[S11]: https://arxiv.org/html/2602.06504v2
[S12]: https://vandal-lab.github.io/multigraspnet-project/
