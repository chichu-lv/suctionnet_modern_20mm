# B 预处理：共同几何、候选与两张共享图像

输入为[数据准备](DATA_PREPARATION.md)交付的同网格 RGB-D、mask、K、gravity 和实例身份；输出为共同 query 清单及 B 的上下文/纹理张量。以下步骤来自[共同数据规范](../DATA_AND_LABELS.md)与[B v3 规格](../ROUTE_B.md)，顺序和门槛属于缓存及输入身份。

## 1. 先保留原始几何，再计算法向副本

在校正相机网格上，从原始深度与内参以 float64 构造 `P=Z*K^-1*[u,v,1]`。`keep` 要求目标实例、有限正深度及既有上游深度范围，不在此新增设备深度阈值。保留原始 RGB、depth、XYZ、原像素 ID、有效 mask；B 以 organized 张量存放，不建立 KD-tree 或全量无序点云索引。

在全部 keep 像素计算 `c_vis` 与可见实例三轴尺寸，在主干图像缩放或 query 抽样后不重算中心。invalid 数据先通过选择排除，不能用 `NaN*0` 参与求和。几何正确性基线从输入构造开始保留 float64；网络特征最终转 FP32，不能先量化 XYZ 再升精度冒充同一基线。[精度合同](../BOUNDARY_REVIEW.md)

## 2. 唯一默认法向

`normal-spec=organized_masked3x3_span3_v3`：

1. 仅对 keep 中心，在3×3原始 XYZ 邻域取同实例、keep 且与中心 `abs(ΔZ)<0.010m` 的等权均值；至少5个支持点（含中心）才为 `smooth_valid`。
2. 在平滑 XYZ 上用±3像素求 `dx` 和 `dy`，法向未归一化值为 `cross(dx,dy)`。
3. 中心及四个差分端点必须 smooth_valid；水平、竖直完整7像素原始路径必须同实例且全部 keep，每对相邻原始深度差严格<10mm。
4. 叉积有限且模长>1e-8 m²；归一化后仅对自动观测轴按 `dot(n,-p_raw)>=0` 定号。
5. 无效法向置零并保存 `normal_valid=false`，不退回±1、不跨洞连接端点、不用两端总深度差替代逐段门控。

平滑 XYZ 仅用于求 n，原始 p、depth、c_vis 和真实接触高度不动。法向无效像素仍可能提供 RGB/深度支持，但不提供 canonical 候选轴。记录各失效原因和支持率；±3像素不是恒定毫米邻域，也不保证真实台阶均被10mm门控隔离。

## 3. 先生成完整候选池，再截训练前缀

`sampler-spec=image_stratified_morton_refill_v3`。用完整实例 mask 的像素外边界 bbox（包含 mask 内无深度处）划分16×16单元，eligible 为 keep 且 normal_valid。source ID 唯一。

格顺序固定为 x/y 四位交错 Morton 码的8位 bit reversal 升序，不能使用普通 Morton 的局部前缀或行优先。首轮每个非空格取距格中心最近的 eligible 像素；之后按同样格顺序逐轮回填，每格取到本格已选集合最小二维距离最大的剩余像素。所有并列取最小 source ID，耗尽格跳过。

达到256或耗尽 eligible 后停止。训练128、可选64直接切同一完整序列的前缀；padding 只补 mask，不复制点。query 的 p 从原始 XYZ 读取，n 从 v3 法向读取。保存格号、轮次、query_id、eligible数、非空格数、首轮/回填数、回填轮数和停止原因。该二维覆盖不保证三维表面积均匀；实际米制间距和候选质量需在实验中报告。

历史查询走原始 `p_plan,n_plan` 和≤5mm同实例几何锚点核验，不运行“候选替换规划点”逻辑。共享候选算法和历史查询规则必须与 A 相同。[共同 query 合同](../DATA_AND_LABELS.md)

## 4. 上下文图：13×128×128

完整 mask bbox 两侧分别扩 `ceil(0.1w)`、`ceil(0.1h)` 像素，再补为同中心正方形。设边长 L、原图像素外边界左上角 `(l,t)`、`s=128/L`：

$$
u_{ctx}=s(u-l)-0.5,\qquad v_{ctx}=s(v-t)-0.5.
$$

保存仿射 A、逆变换、`K_ctx=A*K`、裁图边界和有效 mask，保证大物体完整进入上下文。它用于全局与中程语义，不把128像素等同于256mm。

| 通道 | 固定输入 |
|---|---|
| 3 | RGB，按固定 mean/std 归一化 |
| 3 | `(XYZ-c_vis)/0.1m` |
| 3 | 观测外向单位法向；无效为零 |
| 3 | 相机系单位重力，全图广播 |
| 1 | 有效目标实例 mask |

全局通道共同采用同一个原图 source pixel 的 nearest 采样，不能让 RGB 与 XYZ 来自不同像素或深度层。invalid 位置最终清零，法向有效性仍保留在旁路元数据。

## 5. 原生纹理图：5×H×W

取完整原图实例 bbox，四边扩3个原生像素，超图处填零，**不 resize**。输入为 RGB3、`(Z-Z_med)/0.1m` 一通道、有效实例 mask 一通道；Z_med 仅由目标内原始有效深度计算。深度不跨实例插值，invalid 全通道最终清零。

将 ROI 只向右、向下补零到32的最小整数倍：`H_pad=32*ceil(H_roi/32)`，宽度同理；保留原点、真实 ROI 大小与 mask。microbatch=1，梯度累积不把不同实例补成同一更大尺寸。GN 会包含 padding 位置，训练、验证、推理及导出必须使用同一 padding 合同；动态 profile 不能静默改变为固定大画布。

两张图共享同一次实例光度变换：brightness/contrast/saturation=0.2、hue=0.05，灰度概率0.2；归一化 mean=(0.485,0.456,0.406)、std=(0.229,0.224,0.225)。归一化之后再将 invalid 清零。验证/推理关闭随机光度增强；不做默认米制尺度缩放。[增强与训练依据](../TRAIN_EVAL_REPRODUCE.md)

## 6. 图像坐标与 query 读取的接线边界

预处理输出原图→上下文/ROI 的变换，模型按 query 投影位置读取 Fq 和投影环。几何 nearest 必须读取原始深度/法向/身份；纹理只在四角同目标有效且深度范围≤10mm时双线性，否则退回同一 nearest source pixel。特征图半像素和 gate 的完整算式集中在[模型设计](MODEL_DESIGN.md)。

对投影非有限、z≤0、越界或无原深度支持的槽，保留 missing mask；送入 `grid_sample` 前替换为有限占位坐标，再按 mask 清零。不给 grid 传 NaN，不使用 border/reflection 制造观察到的表面。

## 7. 实施产物和验收

输出包至少包括 `context[13,128,128]`、`texture[5,H_pad,W_pad]`、原始 organized 几何/法向/mask、K及仿射/逆变换、c_vis/尺寸、RGB/normal有效性、query清单、所有版本/hash和统计。固定几何和 query 可缓存；随机光度增强与训练环 phase 不能混入无身份的静态缓存。

必须通过：原始投反影；上下文边界/ROI原点回读；大小物体完整性；fx≠fy；不同深度层不混合；法向断层/空洞拒绝且原点不变；稀疏mask回填、无重复、64/128前缀；空实例/全invalid无NaN。法向宽stencil和纹理不足的真实比例另报，不能只凭合成参考通过即宣告预处理成本和质量达标。

可立即运行 `sampling_contract.py selftest`、`contact_boundary_contract.py self-test` 等[共同参考命令](../TRAIN_EVAL_REPRODUCE.md)。真正的图像 builder、PyTorch 输入变换与 source pixel 读取组合仍须实现后在真实观测上核验。
