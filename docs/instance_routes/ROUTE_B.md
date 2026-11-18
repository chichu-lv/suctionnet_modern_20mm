# 路线 B 单物体 2.5D 上下文与物理接触图

本文定义 B v1 的输入、几何变换、网络接口和输出协议。基线为 `suctionnet_modern_20mm@87f29d29bea8fa5153d6df3f8253d9ced9e4b441`；新的 B 模型与训练入口属于待实现内容，几何参考实现和可运行检查见[复现说明](TRAIN_EVAL_REPRODUCE.md)。数据、标签和坐标的共同定义见[数据与标签契约](DATA_AND_LABELS.md)。

## 1. 采用两级 2D 表征

B v1 使用一张完整物体上下文图和每个候选的一张局部接触图。全局图保留物体形状、RGB和重力线索；局部图以候选点和候选轴为条件，在切平面中提供固定毫米尺度。学习模块全部由普通二维卷积、池化和MLP组成；深度反投影、姿态坐标变换、栅格化和最终坐标还原单独作为几何前后处理。

这比原设想的“单张128×128图直接输出全部动作热力图”多了局部构图与查询计算。它换取明确的20 mm接触范围，以及对任意历史动作轴的正确评分接口。没有依据承诺单物体2–5 ms；速度按本文第11节实测。

| 项目 | B v1 固定默认 |
| --- | --- |
| 全局图 | 完整实例128×128等比letterbox，记录仿射与真实物理尺寸 |
| 局部图 | 64×64，切平面1 mm/格 |
| 候选预算 | 每物体最多256个；与A共用候选清单，query chunk=64 |
| 全局主干 | ResNet-18 U-Net，输出64通道特征图 |
| 局部主干 | 轻量3层2D CNN，通道16/32/64 |
| 接触区域 | 核心r<10 mm；内区10≤r<18 mm；唇口18≤r≤22 mm |
| 输出 | 每query的q、Seal、Wrench、ell和task logit |
| 训练标签 | 同一(p,n)下的S20、W20和SW；目标动作单点单轴BCE |
| 原始几何 | 保留原分辨率实例内XYZ、法向及source index |

256个候选是有限预算，并不保证连续表面全覆盖。首版不引入学习粗筛：固定表面采样后对全部有效候选评分，避免把候选召回变化混入评分能力的比较。

## 2. 修正中位深度缩放的物理含义

针孔模型为：

$$
u=f_xX/Z+c_x,\qquad v=f_yY/Z+c_y.
$$

若以参考深度Z0作图像仿射归一化，横纵缩放应分别为：

$$
\gamma_x=\frac{Z_0}{f_xs_0},\qquad
\gamma_y=\frac{Z_0}{f_ys_0}.
$$

在深度Z处，局部正对平面对应的真实横向格距是`s0·Z/Z0`。例如Z比Z0大20%，名义2 mm格距实际为2.4 mm。对倾斜表面，投影还受到切平面方向的影响，因此一个物理接触圆一般不会成为恒定半径的图像圆。[OpenCV投影模型](https://docs.opencv.org/4.10.0/d9/d0c/group__calib3d.html)

局部投影Jacobian为：

$$
J_\pi(p)=\begin{bmatrix}f_x/Z&0&-f_xX/Z^2\\0&f_y/Z&-f_yY/Z^2\end{bmatrix}.
$$

若切平面基为E=[e1,e2]，接触圆在局部线性近似下按`J_pi E`变形。该结果是投影公式的直接推导。B因此不把全局图的尺度当作Seal接触尺度，而在局部图中直接定义毫米坐标。

另一个必须修正的点是空洞卷积：2 mm格距下，3×3、dilation=10的轴向采样点距中心20 mm，角点距中心28.28 mm，并且含有中心点。它不是20 mm圆环；显式圆盘/环带由物理坐标mask定义。[PyTorch Conv2d](https://docs.pytorch.org/docs/2.15/generated/torch.nn.Conv2d.html)

## 3. 原始观测与全局图

### 3.1 先在原图计算几何

RGB、深度、实例mask必须对齐同一相机；图像已校正，K对应当前分辨率。双目深度若位于左红外相机，必须使用对应内参和RGB外参完成对齐。元数据保存相机标定版本、深度单位和配准方式。

对目标实例的有效像素反投影为米制XYZ。输入法向与canonical候选轴统一读取A/B共同的法向缓存：在原分辨率同实例支持中取15 mm三维球邻域，去重后至少20点，以PCA求法向，退化判断及有向符号按[公共法向协议](DATA_AND_LABELS.md)执行。球邻域本身不保证隔离同一物体的全部孔洞和深度断层；如增加连通性筛选，须修改共同normal-spec并重建两路线的缓存。无效法向保持零向量，并保留normal_valid供候选选择。

基线的`depth_geometry`已实现米制反投影；`organized_normals`使用XYZ差分、10 mm跳变门控及朝相机符号，且没有实例ID参数。这是现有代码的实际行为说明，不是B的新法向默认值；B不能另走该差分路径或自行添加不同深度阈值，替换上述共同PCA缓存。[基线 observation.py](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/observation.py)

### 3.2 完整实例上下文

取完整目标mask的像素包围框，加10%边距，等比缩放并letterbox到128×128。超出原图的部分填0并置valid=0；不得以名义256 mm物理视野截掉大物体两端。保存原图到上下文图的仿射A及其逆。

具体取包含端点的宽高`w=u_max−u_min+1, h=v_max−v_min+1`，两侧边距分别为`ceil(0.1w)`和`ceil(0.1h)`；扩展后以原框中心为中心补成边长L的正方形。若正方形左上像素边界为`(l,t)`，令`s=128/L`，则原图像素中心映射为`u'=s(u−l)−0.5, v'=s(v−t)−0.5`。这固定了半像素约定；不因库的resize默认值而改变。

可见中心c_vis使用原分辨率有效实例点均值。它是坐标原点与可见几何代理，不是真实质量质心。全局池化和实例尺寸用于补充局部图无法观察的物体信息；Wrench标签仍来自公共协议中冻结的完整物体物理定义。

全局图固定13通道：

| 通道 | 数量 | 内容 |
| --- | ---: | --- |
| RGB | 3 | 与深度配准后的RGB；按公共颜色归一化 |
| 相对XYZ | 3 | (X−c_vis)/0.1 m，保留物理尺寸 |
| 观测法向 | 3 | 单位法向；无效法向为0 |
| 重力 | 3 | g_cam完整单位向量，广播至有效实例位置 |
| valid | 1 | 目标实例且深度有效；与normal_valid语义分开 |

valid=1而法向全零表示深度可用、法向不可用的上下文。只有normal_valid为真的点才能直接提供canonical候选轴。normal_valid随数据保留，不因13通道输入被删除。

首版所有全局通道使用nearest采样，保留同一source index，避免把RGB、几何和mask采到不同层。高频RGB仍由原分辨率局部图提供。若以后改用带mask的RGB插值，应更新输入版本、记录有效支持比例，并对跨实例和跨深度层情况关闭插值；不作为首版未说明的默认。

图像变换后的内参可以写为K_ctx=A K；由于使用真实XYZ通道，网络无需从假定恒定mm/px反推几何。s_ctx与物体尺寸在元数据中记录，局部接触尺度不从s_ctx推导。

## 4. 候选与任意姿态查询

推理候选为`query=(p_cam_m,n_out_cam)`。canonical模式从完整实例有效表面生成固定候选清单，再读取该点观测法向；不足256个唯一候选时填充无效槽位，禁止复制点来虚增候选数量。

目标微调直接插入历史`planned_pipeline_contact`中的点和有向轴。历史轴不是可由当前观测法向替换的辅助字段：两者不同就代表不同动作。坐标变换、轴符号与数据有效性按[公共契约](DATA_AND_LABELS.md)处理。

网络对外提供统一逻辑接口：

```python
encoded = model.encode_instance(observation)
scores = model.score_at_pose(
    encoded, points_cam_m, normals_out_cam, query_batch_index
)
```

这是A/B统一的待实现接口契约，不是现有E5构造器可直接接入的新模型。批量查询用`query_batch_index[Q_total]`标明每个查询所属实例，标量输出头形状为[Q_total]；固定[B,K]的调度器可按原顺序reshape，并另保留query_valid。每个实例的global feature只计算一次；历史动作通常每实例1个查询，源域和推理每实例最多256个，以64个query分块。内部可用score_queries辅助分块，但对外合同保持encode_instance/score_at_pose。

若实现额外canonical热力图，它只能表示各像素固定观测法向的分数。任意历史轴仍必须走query接口，不能用附近像素热力图值代替。

## 5. 1 mm切平面局部图

### 5.1 构造右手表面坐标系

n_out为离开物体表面的单位向量。选择与n最不平行的相机坐标轴b，构造：

$$
e_1=\frac{b\times n}{\|b\times n\|},\qquad e_2=n\times e_1.
$$

surface frame的第三轴为n。工具+Z的接近方向另取−n，两者不可混用。对每个原始实例点x：

$$
u=e_1^T(x-p),\quad v=e_2^T(x-p),\quad h=n^T(x-p).
$$

u、v、h全为米。选取`−0.032≤u<0.032, −0.032≤v<0.032, |h|≤0.040`的观测点进入局部构图。轴向40 mm窗口与A及参考工具一致；它是网络观测支持范围，不修改teacher的物理定义。GT物体点云只用于标签teacher，不进入模型输入。b以`argmin(abs(n))`选择，同值按相机x、y、z顺序取第一个，与参考工具的确定性roll一致。

### 5.2 确定性栅格化

局部网格64×64、s=0.001 m。连续点落格：

$$
i=\lfloor u/s+32\rfloor,\qquad j=\lfloor v/s+32\rfloor.
$$

格中心位于`(i+0.5−32)s`；范围为−31.5…31.5 mm。边界是±32 mm，偶数网格中query原点位于四个中心格的交界。文档、NumPy参考、PyTorch实现和导出前处理必须共享这一约定。

同格多点按`(abs(h), source_index)`升序选择第一点。RGB、法向、u/v/h和source index全部取自同一个winner；禁止各通道独立max。该选择只定义“离查询平面最近的观测层”，不证明工具接近路径无遮挡。

同时记录同格点高度跨度、层冲突计数和唇口观测支持。存在多层时保留诊断；不能通过选最近层把其它观测层消失的事实解释为物理可行。部署的既有规划器继续处理外部环境与可达性。

无观测点的格值置0、valid=0。不做平面补洞，不把缺测自动写成Seal=0。参考实现可用NumPy稳定排序，优化实现可用原生PyTorch张量/散射操作；几何前处理不要求可微。

### 5.3 局部输入10通道

| 通道 | 数量 | 内容 |
| --- | ---: | --- |
| RGB | 3 | winner对应RGB |
| 有符号高度 | 1 | h/0.040 m，保留相对真实查询点的偏差 |
| 局部法向 | 3 | (n_obs·e1,n_obs·e2,n_obs·n)；无效时0 |
| 实际局部坐标 | 2 | winner的(u,v)/0.032 m，不用格中心冒充观测点 |
| valid | 1 | 该格存在目标实例内有效观测点 |

“1 mm/格”描述切平面参数格距。观测点仍有离散采样误差、空洞和噪声，不代表获得了1 mm精度的完整真实表面。

参考工具`geometry_contract.py`的`X`保留h、u、v的原始米制值，便于核对几何。模型输入adapter再分别将通道3除以0.040、通道7:9除以0.032；不能把参考输出直接当作已归一化的网络张量。零法向支持点仍可进入局部图，其三维法向通道为0，valid只表示几何/RGB支持存在。

### 5.4 物理区域池化

用winner的实际`r=sqrt(u²+v²)`定义核心、内区和唇口。即使点位于18/22 mm边界附近，也按实际坐标判定，不因格中心量化改变区域。

在局部特征上使用显式mask的mean/max池化，另输入各区有效支持比例。唇口可分72角度区间，保留每区是否有观测点及高度摘要，避免整体均值掩盖连续缺失的一段唇口。观测扇区统计不等于物理teacher的Seal真值。

普通多尺度卷积可用于提特征，但不替代显式区域定义。5 px图像腐蚀也不作为20 mm唇口支持的充分条件。

## 6. 固定网络与张量流

全局ResNet-18 U-Net从13通道输入提取[B,64,128,128]上下文特征，使用有效mask的平均/最大池化得到[B,128]全物体向量。首层为13→64的7×7、stride2、padding3卷积；随后按固定ResNet-18结构得到stem64@64×64、layer1的64@32×32、layer2的128@16×16、layer3的256@8×8和layer4的512@4×4。初始化weights=None。

解码器每层bilinear上采样（align_corners=False）并拼接同分辨率skip，依次为768→256@8×8、384→128@16×16、192→64@32×32、128→64@64×64；每层用两个3×3卷积、BatchNorm与ReLU。最后上采样至128×128并用两个64→64的3×3卷积恢复输出，不增加原图RGB旁路。BatchNorm采用eps=1e-5、momentum=0.1；目标域冻结规则依共同训练规范。

局部CNN接收[B×Q,10,64,64]，依次使用3×3卷积（padding1）、通道16/32/64、stride2/2/2、GroupNorm(groups=8,eps=1e-5)和ReLU，输出[B×Q,64,8,8]。valid经三次2×2 max-pool降到8×8，masked mean/max池化形成128维局部向量；原始1 mm图的区域统计作为旁路，避免8×8特征图把窄唇口完全压缩。

query描述包含`(p−c_vis)/0.1m`、n_out、切平面重力`[e1,e2,n]^T g_cam`及实例三轴尺寸/0.1m，共12维。固定几何统计为156维：三个区域各自的占据格数/4096、h均值/0.040、h总体标准差/0.040、有效法向的平均local_z（共12维），加唇口72个扇区的非空标记和各扇区h均值/0.040（共144维）。空区域/扇区统计置零并保留非空标记；法向均值只使用normal_valid点。三分区按实际u/v、valid联合计算。

全物体向量128、局部向量128、query描述12和几何统计156共424维，经MLP 424→128→128。五个头均为128→64→1；隐藏层ReLU、带bias、无dropout，输出激活见下表。源域不优化task头，其输出仅在对应目标域训练完成后参与排序。

| 输出 | 范围/语义 |
| --- | --- |
| q | sigmoid后[0,1]，该(p,n)的SW代理 |
| seal | sigmoid后[0,1]，同姿态S20 |
| wrench | sigmoid后[0,1]，同姿态W20 |
| ell | 公共损失定义中的受限Laplace残差对数尺度log_laplace_scale，不是log variance |
| task_logit | 未sigmoid的历史动作成败logit |

损失函数、ell的精确定义和初始化遵循[训练复现协议](TRAIN_EVAL_REPRODUCE.md)。不同输出的标量含义固定后才能比较A/B；不能用同一个名称同时表示log variance与Laplace log scale。

## 7. 标签和两阶段训练

### 7.1 源域

每个监督项绑定样本、实例、p、n及物理协议版本。若从原来的静态标签姿态换成观测法向，teacher必须在新姿态重算S20/W20。仅把旧标签投影到2D不能完成这一对齐。

可见性、实例归属和深度一致性检查通过后，有标签query才参与损失。无标签像素为未知，不以0填满整张图，也不在不同姿态标签之间双线性插值。loss先按实例内部有效query归一化，再按实例平均，避免大物体凭像素多占据过大权重。

### 7.2 目标域

对历史动作计算一次`model.score_at_pose(encoded,p_plan,n_plan,query_batch_index)["task_logit"]`并施加BCEWithLogits。20 mm局部感受野为模型提供动作周围的表面证据，不向周围其它位置复制这个成败标签。

不使用半径20 mm的高斯区域logit平均作为默认监督。若以后有定位误差测量，可以另行定义毫米和角度均已标定的动作不确定性积分；吸盘半径不是定位误差的估计。

200步头部训练、800步有限解冻及源域回放遵循共同协议。B应按自己的模块树明确冻结/解冻global decoder、local CNN和query head，不能照抄A的contact_grouping层名。启动时导出可训练参数清单和optimizer分组，便于检查阶段切换。

## 8. 推理与位姿输出

完整实例只编码一次；固定候选清单按64分块构图和评分；无效槽位置query_valid=false。未微调时用物理排序，微调后用共同协议冻结的task/physics组合，不给未训练task head的随机输出排序权。

返回每query的原始p、n、q/seal/wrench/ell/task、有效性和支持诊断。可视化热力图由query source index散射；未评价像素保留未知。NMS使用真实米制距离与轴角度，不能把固定11×11图像池化解释为固定毫米间距。

最终点直接使用被选query的p_cam_m。不要再用5×5深度中位数替换Z；如果确需局部表面拟合更新接触点，应把更新过程作为候选定义的一部分，并使teacher和评价使用同一更新后姿态。

圆形吸盘法向不唯一决定roll。surface frame第三轴为n，工具接近轴为−n；roll规则、相机到机器人基座外参、contact到TCP外参及右手性检查见[统一位姿输出](TRAIN_EVAL_REPRODUCE.md)。输出contact SE(3)不自动等于法兰命令。

沿用基线输出语义`point_cam_m`、`normal_out_cam`和`source_indices`，并可构造`[score,nx,ny,nz,px,py,pz]`供既有评价接口读取。[基线 infer.py](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/scripts/infer.py)

## 9. 支持不足与无位姿返回

无有效深度、无可靠候选轴、目标实例关联失败或没有任何有效query时返回`status=no_pose`，并保存原因；禁止返回全零坐标或随机argmax作为成功结果。空query batch不能触发除零或产生NaN排序。

局部图缺测率、唇口扇区支持、格内多层冲突及中心表面偏差属于独立诊断。支持阈值需由开发数据确定并冻结；不能把未经验证的阈值当物理成功率。尚未确定阈值时报告分数和unsupported状态，最终选取只在通过已冻结数据有效性与规划条件的候选中进行。

真实动作的成败仍可能包含模型未观察到的执行、外部干涉等因素。任务分支学习现有输入条件下的结果关联；外部场景未输入B，就不能要求B从单物体图完整判断外部碰撞原因。

## 10. 几何与接口验收

| 用例 | 预期 |
| --- | --- |
| fx=600、fy=500、Z=0.6m、直径40mm平面圆 | 原图横40px、纵33.333px；正确区分横纵尺度 |
| Z0=0.6m、Z=0.72m | 名义2mm仿射格实际为2.4mm，不能误报恒尺度 |
| 60度倾斜平面 | 图像中的接触圈为椭圆；局部平面h约0且物理半径保持20mm |
| d=10、2mm格距的3×3核 | 角点距中心28.284mm，证明不是20mm圆环 |
| 图像边界距离10mm | 20mm唇口仍可能越界，不以5px腐蚀通过验收 |
| 改变历史动作轴、保留p | 局部图和标签条件改变，不复用canonical像素标签 |
| 同格多点 | abs(h)最近、source index打破平局；通道来自同一winner |
| mask外、深度缺测、跨实例 | 不成为有效支持，未知标签不进入loss |
| 大于256mm的物体 | 完整进入global上下文，local仍为1mm/格 |
| 所有query无效 | 明确no_pose，输出无NaN且不伪造可执行位姿 |

坐标往返、表面frame与tool frame、固定query序列以及数据划分的检查由[复现协议](TRAIN_EVAL_REPRODUCE.md)统一执行。不同实现的局部图比较除数值外还需逐项比较格索引、winner source index、有效mask和区域归属。

## 11. 部署边界与延迟

首版只导出global encoder、local encoder和query head；几何前后处理独立运行。固定输入shape、K、chunk及模型初始化后，才开始ONNX/TensorRT数值核对。不存在“所有处理天然可导出”的前提。

如果将grid_sample加入引擎，固定align_corners=False：连续原图像素u对应归一化坐标`2(u+0.5)/W−1`。插值、padding与resize规则必须一致，不能用border/reflection给缺测区域造表面。[PyTorch grid_sample](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.grid_sample.html)和[TensorRT GridSample](https://docs.nvidia.com/deeplearning/tensorrt/11.4.1/_static/operators/GridSample.html)说明了算子语义；实际支持仍由锁定版本的导出与解析测试确认。

投影参考函数用`valid=false`和NaN标识无效坐标；送入grid_sample前必须将其mask掉并构造有限的占位采样坐标，禁止把NaN网格直接交给该算子。NaN标签也仅允许存于label_valid=false行；无效label行不参与损失。

投影、坐标变换、mask、点选择和最终位姿优先FP32；卷积可验证FP16。记录logit差异、排序top1一致率和最终姿态误差，不只检查张量shape。[TensorRT精度说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)解释了FP16范围、敏感计算及混合精度取舍。

计时拆分为配准/原图几何、global构图、local构图、网络、NMS/位姿还原，并给包含这些阶段的每物体和每帧p50/p95。记录物体数、有效点数、候选数、硬件、精度、软件版本、warmup和同步方式。分割/双目网络若不在计时内，单独注明边界。

两级B增加了256次局部query构图与局部CNN计算，未必在所有物体数量下比A快。只有同硬件、同候选、同输出协议和同计时边界的结果才支持速度比较；“2–5 ms”保留为待验证的工程目标。
