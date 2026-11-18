# B 模型设计：共享图像编码与姿态条件环查询

模型 `InstanceSuction2DFastV3` 已实现在 `models/instance_route_b.py`，同时提供别名 `InstanceRouteB`。数值依据为[B v3 原规范](../ROUTE_B.md)与[机器可读设计](../reference/route_b.json)；架构全图见[方法图说明](../METHOD_ARCHITECTURE_B.md)。ResNet-18/GN、原生纹理和接触分支直接用PyTorch实现，不依赖torchvision、spconv或逐候选局部栅格。真实网络CPU前向、反向和边界测试已通过，真实数据训练、GPU性能与部署导出尚未验证。

## 1. 编码一次，多姿态共用

接口合同：

```text
encoded = model.encode_instance(observation)
scores = model.score_at_pose(
    encoded, points_cam_m, normals_out_cam, query_batch_index
)
```

encoded 保留共享上下文和纹理特征、原始 organized 几何/有效性、相机与裁图变换、可见中心和尺寸。`score_at_pose` 接受任意合法的有向 p/n；canonical 与历史规划动作调用同一接口，历史轴不能改成该像素的观测法向。默认最多256查询，按64查询分块可控制内存；分块顺序和大小不改变分数。

输出包含 q_inst、seal、wrench、ell、task_logit、query_valid 和诊断。query 的合法中心支持与各环槽的 contact_valid 是不同层次；环全空按空集合同返回特征/摘要，不自动制造失败标签或额外硬阈值。[数据和精确动作合同](../DATA_AND_LABELS.md)

## 2. 两条共享编码路径

### 上下文 encoder 和位置语义

13×128×128 输入进入 `weights=None` 的 ResNet-18：7×7/stride2/padding3、64通道 stem；3×3/stride2/padding1 max-pool；BasicBlock 数[2,2,2,2]、通道[64,128,256,512]、stage stride[1,2,2,2]。所有 BN（含残差投影）替换为 GN8，eps=1e-5、affine=True。保持标准卷积、ReLU 和 shortcut 顺序，没有 decoder。

| 读取位置 | 特征/变换 | 输出 |
|---|---|---|
| L1 | 64×32×32 → 带bias的1×1 Conv到32通道；无额外激活 | query双线性采样32维 |
| L2 | 128×16×16 → 带bias的1×1 Conv到32通道；无额外激活 | query双线性采样32维 |
| L1+L2 | 位置特征拼接 | Fq64 |
| L4 | 512×4×4；valid由输入mask adaptive max pool到4×4；masked mean/max | 1024→128 Linear + ReLU，G128 |

query 原图投影映射到 `(u_ctx,v_ctx)`，L1/L2 特征坐标分别是 `(u_ctx/4,v_ctx/4)`、`(u_ctx/8,v_ctx/8)`。对特征坐标 a、宽 Wf，`grid=2*(a+0.5)/Wf-1`；`align_corners=False`、zeros padding。固定卷积中心从输入像素坐标0开始，不能直接复用上下文图的归一化 grid。L4 无有效位置时不产生有效实例编码。

### 原生纹理 stem

5×H_pad×W_pad 输入依次经过：Conv2d 5→16，3×3/padding1；depthwise Conv2d 16→16，3×3/padding1/groups16；pointwise Conv2d 16→16，1×1。全部 stride1、bias=False，每层 GN8（eps=1e-5、affine=True）和 ReLU。F_tex 保持原生 padded ROI 大小，所有查询共用；GN 保持[预处理页](PREPROCESSING.md)的固定 padding 规则。

## 3. 姿态坐标图与理想环

使用冻结的两个 Z 轴 Rodrigues 图。默认 `a=(0,0,-1)`、`b1=(1,0,0)`、`b2=(0,-1,0)`；令 `v=a×n`、`c=a·n`：

$$R_a=I+[v]_\times+\frac{[v]_\times^2}{1+c},\qquad e_1=R_a b_1,\quad e_2=R_a b_2.$$

当 `1+c<=1e-4` 时切到 `a=(0,0,+1)`、`b1=(1,0,0)`、`b2=(0,1,0)`，记录 chart；历史 n 保持符号。此特征坐标图不是输出工具 roll，也不保证全球连续。

| 分支 | 名义半径/mm | 实际允许区域/mm |
|---|---|---|
| core | 0、5、9 | [0,10) |
| inner | 10、14、17 | [10,18) |
| lip | 18、20、22 | [18,22] |

`theta_j=2π(j+0.5)/72+phi`，`x_rj=p+r*(e1*cos(theta)+e2*sin(theta))`。验证/推理 phi=0；训练由seed、样本身份、精确p/n字节表示的稳定query身份和成功更新步确定性生成[0,2π)相位。该身份也适用于候选池之外的历史动作，不依赖当前query顺序或chunk大小；只改变特征采样，不改变p/n或teacher标签。9×72=648逻辑槽，r=0只读取一次再广播，最多577个不同模板位置；不同模板仍可能命中同一原像素，不能当577个独立观测。

## 4. 从真实深度读取，再做支持门控

用 K 将理想 x_rj 投回原图，几何采用 nearest source pixel 的真实 XYZ、法向、实例和有效性。纹理仅在四个双线性角点均在图内、同目标且有有效深度、Z范围≤10mm时双线性；否则读取同一 nearest 像素的 F_tex并记录回退。无有效 nearest 时置零并记录缺测。投影非有限、z≤0或越界先使用有限 grid 占位，再按 mask 清零。

对真实点令 `delta=x_obs-p`、`h=n·delta`、`d_t=delta-h*n`、`rho=||d_t||`，不能把理想 x_rj 的 h=0当成观测。设 `e_r=e1*cos(theta)+e2*sin(theta)`、`e_t=-e1*sin(theta)+e2*cos(theta)`，r>0的接触槽同时要求：

$$m_{contact}=m_{obs}\land[\rho\in I_{region}]\land[|\rho-r|\le .003]
\land[|e_t\cdot d_t|\le .003]\land[e_r\cdot d_t>0].$$

单位为米；m_obs只表示有限、有效、同目标的原始深度。r=0仅要求 m_obs、rho≤3mm且位于core，不按 theta 做切向/半平面门控；同一中心mask广播72槽。两个3mm门槛是冻结工程初值，不能解释为已测噪声，也不能扩大18/22mm边界来填满支持。

必须在门控前分出 m_obs 与实际 h。contact拒绝只清零接触token；凸起/凹陷高度留给原始摘要。h/.040是缩放，**不是±40mm截断**。法向无效不等于深度无效；不因无 n_obs 删除仍合法的深度/纹理支持。几何判定从原始输入使用float64，1e-12m仅用于参考规定的舍入；正径向半平面仍严格>0。[门控数值边界](../BOUNDARY_REVIEW.md)

## 5. 96维环特征与30维显式摘要

单槽为 F_tex16 加9个量：h/.040、(rho-r)/.022、e_t·delta/.022、n_obs·n、normal_valid、observed_valid、in_region、g·e_r、g·n。invalid法向只使相关法向量为0。先清理非有限值，再将**全部25维**按m_contact置零，不能让重力等字段从拒绝槽泄漏。

每区域三个半径拼为 `[Q,75,72]`。三个区域使用相同结构、独立权重：75→16/kernel1，再两层16→16/kernel3；stride1、bias=False，kernel3显式首尾循环padding1。每层固定顺序为清零→Conv→逐角度跨16通道LayerNorm→ReLU→按原mask清零。LN使用 `[Q,72,16]` 最后一维、eps=1e-5、affine/bias=True，不能跨角度统计。

原角支持 `a_j=any_r(m_contact[r,j])`，层间不膨胀、不压紧角序列，不按有效邻居数再缩放卷积。最后仅在原a_j上 masked mean16/max16；全空显式返回零，无NaN或−inf。每区32、三区96维。

| 摘要 | 精确定义 | 维数 |
|---|---|---:|
| 每半径接触覆盖 | contact_count/72 | 9 |
| 每半径最大循环缺口 | 最长连续无效槽数/72，跨71→0；全有效0、全无效1 | 9 |
| 每名义区域观测比例 | observed_count/216 | 3 |
| 每名义区域接触拒绝比例 | observed且非contact数量/observed_count；空观测0 | 3 |
| 每名义区域正/负高度极值 | max(0,max h)/.040、max(0,max(-h))/.040；空观测0，不裁剪 | 6 |

支持摘要固定按core/inner/lip及列出的半径顺序，每个半径依次拼接 `(contact_fraction, gap_fraction)`，共18维；raw摘要按每个区域依次拼接 `(observed_fraction, rejection_fraction, max_positive_height, max_negative_height)`，共12维。不能改成先拼全部9个coverage、再拼全部9个gap。

raw摘要按**名义模板区域**汇总门控前观测；实际rho越区也保留。中心72复制只占一个半径权重，中心支持摘要只能是(1,0)或(0,1)。离散覆盖不是面积覆盖，raw高度不是完整自碰认证。

## 6. 330维融合及五个独立头

`G128 + Fq64 + ring96 + support18 + raw12 + descriptor12 = 330`。descriptor为 `(p-c_vis)/.1`、n、g、可见三轴尺寸/.1。融合MLP 330→128→128，各Linear带bias、隐藏ReLU；没有dropout。

五个头均从h_pose128读取，都是128→64→1、隐藏ReLU：q_inst/Seal/Wrench经sigmoid；ell为Laplace log b，按[训练方案](TRAINING_PLAN.md)初始化/限幅；task输出原始logit。没有独立碰撞头，q不是预测Seal×Wrench的乘积，q_sw只作离线诊断。源阶段随机初始化的task头不参与排序。

实现返回 `ell` 和同义 `log_scale`；若读取兼容字段 `log_variance`，其定义是 `log(2)+2*ell`，不是ell本身。返回的 `points_cam_m` 为原始query中心，`normals_out_cam` 为保留原有向符号的单位轴；`valid/query_valid` 显式标记中心支持与轴是否合法。无支持query的数值占位输出不得解释为真实失败标签。

```python
from models.instance_route_b import InstanceSuction2DFastV3

model = InstanceSuction2DFastV3(query_chunk=64, phase_seed=42)
model.eval()
encoded = model.encode_instance(observation)  # prepare_instance或缓存读取的单实例字典
scores = model.score_at_pose(encoded, observation['query_points'], observation['query_normals'])
# 历史姿态可直接传入原始p_plan/n_plan；encode_instance只需执行一次。
```

首版仅支持单实例编码；`query_batch_index` 省略或全部为0，其他值明确报错。`parameter_groups()` 返回互斥且覆盖全部参数的encoder/decoder/contact/heads：encoder为ResNet-18；decoder包括中层/G投影和纹理stem；contact包括三个环编码器及融合MLP；heads为五个输出头。训练器通过 `set_training_context(update, seed)` 设置相位身份。

## 7. 模型交付验收

必须核验 Fq 梯度定位到L1/L2相应位置；stride4/8单像素脉冲无固定偏移；任意合法历史轴确实改变本次投影条件；共享编码不随query重跑；chunk64与完整查询一致。45°、h=-6mm的18mm理想环两侧actual rho约24.254558/12.254558mm，应拒绝lip但保留负高度。反向槽、中心相位不变、71→0缺口、全空LN/池化、GN padding、输入精度和非有限grid均须覆盖。

整数角槽循环移位的一致性可检查，但有限插值不提供任意连续SO(2)不变性；连续phase敏感性另测。拟导出的Conv/GN/LN、投影、gather/grid_sample及mask组合必须实际解析和对照，不能仅凭“普通二维算子”宣称完整TensorRT已可用。后处理与导出边界见[下一页](POSTPROCESSING.md)。

`python -m pytest -q tests/test_instance_route_b.py` 当前通过12项CPU测试，覆盖真实网络前后向、Fq位置及梯度、chunk一致性、掩码LN、循环缺口、视差反例、原生四角采样/回退、空query/实例、NaN/Inf和超大有限坐标的有限反向梯度、相位身份及参数组。另已连接共同 `prepare_instance` 做合成RGB-D输入到网络的验证。以上不是实物成功率、真实teacher重算或GPU时延证据。
