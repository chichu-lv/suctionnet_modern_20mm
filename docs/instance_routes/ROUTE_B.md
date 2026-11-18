# 路线 B v2：共享二维编码与投影环带查询

本文替换 B v1 的 U-Net 加逐候选切平面栅格方案。基线仓库为 suctionnet_modern_20mm@87f29d29bea8fa5153d6df3f8253d9ced9e4b441；旧设计留存于 [e495fd8](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/e495fd89678b5f696e1819437fef5ced5103aa57/docs/instance_routes/ROUTE_B.md)，不作为另一条同时维护的默认路线。**本文是经审查修订的工程规格，新的 B 模型、训练器和部署引擎尚未实现或测速。** 公共定义见[数据与标签](DATA_AND_LABELS.md)与[训练复现](TRAIN_EVAL_REPRODUCE.md)。

## 1. 本次结构修正及路线边界

旧版确实存在两处问题：高分辨率解码特征只做全物体池化，没有查询位置特征；第 5.4 节要求分区特征池化，第 6 节却只保留 8×8 整图池化和几何旁路。三次 stride=2 不等于数学上消灭所有细纹理，但无法兑现“在 4 mm 唇口带上显式提取纹理”的设计承诺。

B v2 改为完整实例只编码一次，所有查询共享二维语义和原始分辨率纹理特征。固定物理圆环只用于生成少量投影采样坐标；不再建立每个查询的 64×64 局部图，不做逐查询二维 CNN，不建立三维球查询索引，也不对全量观测点做 PCA。

| 项目 | B v2 固定默认 |
| --- | --- |
| 全局上下文 | 完整物体 128×128、13 通道，ResNet-18 encoder，全部 GroupNorm |
| 查询语义 Fq | 在两级中层特征按查询投影位置采样，共 64 维 |
| 全物体特征 G | 最深层 masked mean/max 后投影为 128 维 |
| 纹理 | 原生像素 ROI、5 通道，stride=1 的共享轻量 stem，16 维 |
| 接触查询 | 在给定 p、n 的平面上生成 9 个半径、72 个角度的采样位置 |
| 环带编码 | 角度方向共享 circular Conv1d，三个区域共 96 维，无角度槽位展平 |
| 候选 | 共同二维分层采样最多 256 个；源域默认前 128 个 |
| 综合质量 | Q_inst20 = S20 × W20 × (1 − C_self)，另存 SW 和各分量 |
| 训练/评分接口 | encode_instance / score_at_pose，历史动作保持自己的有向轴 |

B v2 的定位是**使用普通图像算子、共享特征和向量化投影的轻量查询模型**。它仍需要相机投影、深度反投影及姿态数学，不能宣称完全没有三维几何。它去掉的是无组织点云邻域搜索、逐候选栅格重建和逐候选图像编码。严格的单张 canonical 热力图只能评价固定观测法向；若用其替代任意历史轴查询，仍会重现姿态条件失配。

## 2. 原图输入、法向与候选

### 2.1 图像几何

RGB、深度、实例 mask 与 K 必须位于同一已校正相机坐标系。米制 XYZ 作为有组织图像的逐像素张量计算，不构建 KD-tree 或稀疏三维索引。可见点均值 c_vis 用于平移归一化，完整实例三轴尺寸用于全局几何描述；两者均不冒充真实质量质心。

法向采用共同 v2 的 organized_normals：在单实例有效点图中使用 ±1 像素对称差分叉乘，要求所需邻居有效、相对两端 Z 跳变严格小于 10 mm，叉乘模长大于 1e-8，并按相机可见侧定号。边界、缺测和退化位置法向无效。必须复用公共算法与版本，不自行增加高斯平滑或把 15 mm PCA 继续作为默认前处理。[现有实现](https://github.com/chichu-lv/suctionnet_modern_20mm/blob/87f29d29bea8fa5153d6df3f8253d9ced9e4b441/suction20/observation.py)

该法向更便宜，但可能更受深度噪声影响；这是一项需要在固定验证集测量的精度与延迟取舍。若将来改用降采样 PCA，必须另立 normal-spec、重建对应 teacher 标签，并分别报告法向失败率、评分质量和延迟。

### 2.2 公共二维分层候选

A/B 比较使用 image_stratified_16x16_bit_reversed_morton_v2。按完整实例的像素包围框分为 16×16 单元，每格从深度与法向有效的目标像素中选取距离格中心最近者；距离相等取最小原图 source ID。单元顺序为四位 x/y Morton 编码的八位 bit reversal，空格跳过，因此前 64、128、256 项是同一清单的嵌套前缀。

源域训练默认使用前 128 个查询；64 为预注册的小预算比较项。验证和推理最多 256 个，记录实际有效数量。有限图像分层覆盖不等于三维表面均匀覆盖，必须报告共同候选的质量召回。目标域历史动作直接查询 p_plan、n_plan，不要求落在候选清单中，也不把历史轴替换为观测法向。按公共规则以≤5mm最近同实例原始几何锚点验证中心支持，锚点不移动p；单个历史查询可用organized XYZ上的向量化距离归约，不需空间树。canonical查询直接使用已知source ID。

## 3. 两张共享图像：语义与纹理

### 3.1 128×128 完整物体上下文

取完整实例包围框，两侧分别加 ceil(0.1w)、ceil(0.1h) 像素边距，再围绕同一中心补成正方形。边长 L、左上像素边界 (l,t)、s=128/L，原图像素中心映射为：

$$
u_{\rm ctx}=s(u-l)-0.5,\qquad v_{\rm ctx}=s(v-t)-0.5.
$$

保存 A、A 的逆及 K_ctx=A K。完整对象不得被名义 256 mm 的视野截断。13 个通道与 v1 一样为 RGB3、(XYZ−c_vis)/0.1m 的 3 通道、观测法向3、相机系单位重力3和有效实例 mask1。RGB 按共同 mean/std 归一化；invalid 位置最终清零。全局通道采用同一原图 source pixel 的 nearest 采样，不混合不同实例或深度层。

这张图承担全局与中程语义，不承担毫米纹理分辨率。128×128 与固定物理毫米比例不存在普遍等价关系。

### 3.2 原生像素纹理 ROI

取完整实例原图包围框，并在四边扩展 3 个原生像素；超出原图部分填零。保留原图像素分辨率，不 resize。输入为归一化 RGB3、相对深度 (Z−Z_med)/0.1m、有效实例 mask1，共 5 通道。相对深度不跨实例插值；invalid 通道清零。共享纹理 stem 只运行一次：

| 层 | 固定定义 |
| --- | --- |
| T1 | Conv2d 5→16，3×3，stride1，padding1，bias=False；GN8；ReLU |
| T2 | depthwise Conv2d 16→16，3×3，groups16，stride1，padding1，bias=False；GN8；ReLU |
| T3 | pointwise Conv2d 16→16，1×1，bias=False；GN8；ReLU |

GN 的 eps=1e-5、affine=True。特征图 F_tex 保持 ROI 的 H×W 原生分辨率；不存在 64→8 的局部下采样。RGB 纹理能否解析仍受原始相机采样和焦距限制，stride1 不能恢复传感器未观察到的接缝。

例如沿某方向的 4 mm 物理带宽投影不到 2 像素时，应记录 underresolved 诊断并分析该子集；此阈值只是采样诊断，不直接等价于吸取不可行。不得为了通过诊断把低分辨率图放大后声称获得更多独立观测。首版固定只向右、向下补零至32的最小整数倍：H_pad=32*ceil(H_roi/32)、W_pad=32*ceil(W_roi/32)，不移动原像素原点，并保留真实大小mask。训练microbatch=1、验证及推理使用同一规则；梯度累积不要求把不同ROI再补成同尺寸。导出dynamic profile必须支持这些实际padded尺寸，最大尺寸由生产资产确定；不能为了某个engine静默补成更大常量。GN会把padding位置纳入统计，修改padding规则需要新输入合同和数值核验，不声称尺寸变化对分数无影响。

## 4. Encoder 与显式逐查询语义 Fq

全局 ResNet-18 使用 weights=None、13→64 的 7×7/stride2/padding3 stem，3×3/stride2/padding1 max-pool，四级 BasicBlock 数为 [2,2,2,2]，各级通道 [64,128,256,512]，stage stride [1,2,2,2]。所有 BatchNorm 替换为 GN8，eps=1e-5、affine=True；残差投影分支也使用 GN。保持普通 ResNet-18 的 ReLU、shortcut 和卷积位置。

输出层级为 stem64@64²、L1 64@32²、L2 128@16²、L3 256@8²、L4 512@4²。没有 U-Net decoder。

L1 和 L2 分别用带 bias 的 1×1 卷积投影为 32 通道，在 query 投影位置双线性采样并拼成 Fq[64]。这两个投影不额外施加激活。查询点映射为 (u_ctx,v_ctx) 后，对本固定网络的特征索引分别是 (u_ctx/4,v_ctx/4) 和 (u_ctx/8,v_ctx/8)：上述 stride/padding 的特征中心从原图中心坐标 0 起算，不能直接复用输入图归一化 grid 而漏掉半像素偏移。进入 grid_sample 的归一化坐标由对应特征索引 a 和特征宽度 Wf 计算为 2(a+0.5)/Wf−1，align_corners=False，padding_mode=zeros。

L4 的有效 mask 用输入 valid 的 adaptive_max_pool2d 得到 4×4 mask。masked mean/max 拼为 1024 维，再经 Linear 1024→128、ReLU 得到 G[128]。无有效位置时不产生有效实例编码。

原设计的 BN 在目标域本已要求冻结 running stats 与 affine，因而“微调必然漂移”不是对原规格的准确描述；实现遗漏冻结仍是风险。v2 统一 GN，避免这项状态管理。GN 在训练和评估中使用输入自身的统计量，仍需遵守模块解冻规则。[PyTorch GroupNorm](https://docs.pytorch.org/docs/2.15/generated/torch.nn.GroupNorm.html)

## 5. 稳定坐标图与投影接触采样

### 5.1 两个固定坐标图

旧 argmin(abs(n)) 会在最小分量切换面上产生 roll 突变，但并不总是 90°。例如 n=(0.5,0.5,sqrt(0.5))，由 x 轴和 y 轴生成的两组切向基夹角约 109.47°。模型必须减少对这类任意坐标选择的依赖。

v2 使用固定两图 Rodrigues 定义。默认参考轴 a=(0,0,−1)、参考切向基 b1=(1,0,0)、b2=(0,−1,0)。令 v=a×n、c=a·n：

$$
R_a=I+[v]_\times+\frac{[v]_\times^2}{1+c},\qquad
e_1=R_a b_1,\quad e_2=R_a b_2.
$$

若 1+c≤1e-4，则改用 a=(0,0,+1)、b1=(1,0,0)、b2=(0,1,0)，重新计算并记录 chart=positive_z；其余记为 negative_z。n 只做合法性检查和单位化，历史符号不变。

该定义消除了普通可见侧的 x/y 最小分量切换面，但在备用图切换处仍不可能保证全球连续。工具实际 roll 的连续选择属于机器人姿态规划，不能把特征坐标图直接当作允许机械手突然转动的指令。

### 5.2 固定采样半径与角度

| 区域分支 | 半径，mm | 目标物理范围 |
| --- | --- | --- |
| core | 0、5、9 | 0≤rho<10 mm |
| inner | 10、14、17 | 10≤rho<18 mm |
| lip | 18、20、22 | 18≤rho≤22 mm |

角度为 theta_j=2π(j+0.5)/72+phi，j=0…71。默认验证和推理 phi=0；训练时按 seed、样本 ID、query ID、成功更新步确定地抽取 phi~Uniform[0,2π)。这是接触特征采样相位增强，不改变 p、n 或 teacher 姿态标签。

每个点：

$$
x_{r,j}=p+r(e_1\cos\theta_j+e_2\sin\theta_j)
$$

通过真实 K 投影回原图与纹理 ROI。一共 9×72 个逻辑位置，r=0 只采样一次然后广播，因此每 query 最多 577 个不同采样位置；256 个 query 为 147,712 个位置。所有 query 合并为一个或若干张量批次运行，不创建 256 个局部图。首版允许按 64 个 query 分块控制峰值内存，分块不得改变分数。

3×3 dilation=10 的核既含中心又有 28.28 mm 的对角点，不能替代上述 20 mm 圆环。中位深度仿射也只在参考深度平面上给出名义比例；倾斜面和不同深度仍按针孔模型变化。[OpenCV 投影模型](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)

### 5.3 实际深度、遮挡和采样有效性

投影的 x_rj 是假设接触平面上的位置；该像素真正观察到的表面可能位于另一深度。几何通道采用 nearest 原图 source pixel，读取其真实 XYZ、法向和实例有效性；计算 delta=x_obs−p、h=n·delta、rho=||delta−hn||。不能将假设点的 h=0 当作观测表面平整证据。

纹理特征默认双线性采样。仅当四个采样角点都在图内、属于该目标实例且有效深度、角点间 Z 范围≤10 mm 时允许双线性；否则使用同一个 nearest source pixel 的 F_tex，并记录回退。nearest 像素无效时，纹理及观测几何置零，观测有效位为0。这只约束插值支持，不按 10 mm 过滤相对查询平面的真实高度 h。前方凸起、凹陷和相对查询的深度偏移都应保留为模型输入。

投影深度≤0、坐标非有限或完全越界的位置为缺测；送入 grid_sample 前用有限占位坐标并通过 mask 清零，禁止传 NaN grid。使用 zeros padding，不用 border/reflection 制造表面。

投影采样不能遍历全部三维唇口，也不能发现相机不可见的背面或完整接近柱体。它保留有限观测证据，并与含自碰项的 teacher 学习对应关系；不构成几何自碰认证。外部场景碰撞仍交由现有规划器，目标自身碰撞以公共协议单独定义和记录。

## 6. 保留唇口纹理的环带网络

### 6.1 每个角度位置的 25 维 token

每个半径/角度采样位置包含 16 维 F_tex 与以下 9 维辅助量：

| 字段 | 定义 |
| --- | --- |
| height | h/0.040m |
| radial_error | (rho−r)/0.022m |
| tangential_offset | e_t(theta)·delta/0.022m，e_t=−e1 sin(theta)+e2 cos(theta) |
| normal_alignment | n_obs·n；法向无效时为0 |
| normal_valid | 观测法向是否有效 |
| observed_valid | 是否有有效目标深度支持 |
| in_region | 实际 rho 是否落在本区域，且 observed_valid=True |
| gravity_radial | g_cam·(e1 cos(theta)+e2 sin(theta)) |
| gravity_normal | g_cam·n |

height 不因超出 ±40 mm 被删除；0.040m 只是特征缩放，不是去掉凸起障碍的门控。观测无效时前 7 项及纹理清零；后两项属于已知查询几何，可以保留。r=0 的观测与纹理只读取一次，但与角度有关的 token 分量仍按对应 theta 计算。

各分支按给定三个半径拼接为 [Q,75,72]。这定义的是查询平面上的投影采样模板；实际观测是否在区域内由 in_region 显式标识。它不是对完整三维环带中全部观测点的精确积分，也不把离散投影采样密度等同于真实面积覆盖。

### 6.2 共享角度卷积与无槽位池化

三个区域分别使用结构相同、权重独立的 Conv1d：75→16 的 kernel1，随后两层 16→16 的 kernel3。三层 stride 都为1、bias=False，均接 GN8(eps=1e-5,affine=True) 和 ReLU；kernel3 显式首尾 circular padding1。特征保留 72 个角位置到最后，不做角度方向下采样。

每个区域最后沿 72 个角度做 mean/max，得到 32 维，三者拼成 96 维。missing 与 in_region 标记在卷积前已进入 token，缺测 token 不能先被抹除成“与观测平面相同”的数据。统计池化包含固定 72 个位置，以便缺测模式仍影响网络；不得在实现时改成只保留有深度角度再压紧序列，否则会改变相邻扇区关系。

这样保留了 18、20、22 mm 三层原分辨率纹理，并用共享角度卷积描述相邻缺口，而没有将 72 个有序槽位直接展平接到 MLP。纯循环平移会等变地移动中间序列，并被最终 mean/max 消去；任意连续 roll 仍受有限采样和插值影响，不能声称严格 SO(2) 不变。训练相位增强与验证相位敏感性检查共同约束该误差。

此环编码的卷积 MAC 按规格为每 query 590,976，256 query 约 0.151 GMAC；旧局部二维三层 CNN 单独约 0.981 GMAC。这里仅比较两段指定卷积，不含共享编码、投影、采样、归一化、内存访问和 CPU/GPU 同步，不能换算为未测量的毫秒或端到端加速比。

### 6.3 融合与输出

query 描述为 (p−c_vis)/0.1m、n_cam、g_cam、实例三轴尺寸/0.1m，共 12 维，全部使用相机坐标，不把任意切向基的重力分量直接绑定 MLP 槽位。

| 融合输入 | 维度 |
| --- | ---: |
| 全物体 G | 128 |
| 查询语义 Fq | 64 |
| core / inner / lip 环特征 | 96 |
| query 描述 | 12 |
| 合计 | 300 |

融合 MLP 为 300→128→128；五个输出头均为128→64→1。Linear 带 bias，隐藏层 ReLU，无 dropout。q_inst、seal、wrench 经 sigmoid；ell 是公共协议中的 Laplace 残差 log-scale；task_logit 不预先 sigmoid。源域只监督物理分支，task 头在目标域训练完成前不参与排序。

q_inst 监督 Q_inst20=S20·W20·(1−C_self)。q_sw 只作为独立缓存物理分量与报告项，不用同一输出名隐式切换目标；S20/W20 辅助头保留连续物理结构。不要通过除旧 gt_B 恢复自碰或 SW。质量契约改为 gt_INST20_pose_v2，旧 checkpoint 不允许无标识继续训练。

## 7. 两阶段训练和稀疏历史动作

源域只对本 query 姿态的有效 teacher 标签计算损失。默认使用共同候选清单前128项，teacher 生成、缓存身份、仅自碰物理定义、样本预算与验证频率见共同文档。颜色增强在全局图和原生纹理图共享同一实例级参数；相位增强只改变环采样，不改 teacher 姿态。

目标域对历史 p_plan、n_plan 计算一次 BCEWithLogits。actual_execution_verified=false 意味着标签可能包含落点、跟踪和执行误差，不能把单点监督写成真实执行位姿已知。轻度毫米/角度微扰只作为需预注册、在验证集评估的鲁棒正则，遵循共同训练说明；不恢复半径20 mm的标签扩散。

v2 微调前200步只训练五个输出头；后800步再解冻融合 MLP、Fq 两个投影、G 投影、纹理 stem 与环带编码，低学习率按共同规范。ResNet-18 encoder 保持冻结。各 GN 的 affine 是否可训练由所在模块是否解冻决定；不存在 BatchNorm running stats。模块组名必须由新模型导出，不能照抄旧 global decoder/local CNN 的名字。训练记录成功步、有效查询数量、无效原因和相位种子。

## 8. 统一接口、排序和位姿输出

模型对外合同：

    encoded = model.encode_instance(observation)
    scores = model.score_at_pose(
        encoded, points_cam_m, normals_out_cam, query_batch_index
    )

输出是每个给定 p、n 对应的 q_inst、seal、wrench、ell、task_logit、query_valid 和诊断。源域与未微调模型按 q_inst 排序；目标域默认 task_probability，其他组合是单独冻结的验证项。共享图像编码不随查询重算，query chunk 不改变结果。

canonical 可视化图按原图 source ID 散射已评分候选；未评分像素保持未知。它不是已经完成全图稠密网络预测，也不能表示任意历史轴。

NMS 按真实米制距离和轴角度执行；最终返回选中 query 原来的 p、n，不再用 5×5 深度中位数偷偷移动接触点。工具 +Z=−n；机器人 roll 和相机到基座、contact 到 TCP 外参按共同输出协议。无有效深度、法向或候选时返回 no_pose，并保留在评价分母中。

## 9. 几何、网络和数值验收

| 验收项 | 必须检查的结果 |
| --- | --- |
| Fq 局部语义 | 同物体两个位置有独立采样路径，梯度回到对应中层特征；不退化为复制 G |
| 原生纹理保留 | 唇口18/20/22 mm分别采样；没有8 mm局部格或角度下采样 |
| 投影误差 | fx≠fy、倾斜面、不同深度和非零 query轴偏差均与针孔投影一致 |
| 特征图坐标 | 检查 stride4/8 和 half-pixel 规则；单像素脉冲不能出现固定偏移 |
| 法向切换 | 跨 abs(nx)=abs(ny) 的普通可见法向，无旧 argmin 图切换 |
| 备用图 | 靠近 +Z 时有限、正交、右手；历史轴不翻转；记录 chart |
| 角度偏移 | 整数槽循环平移后池化一致；连续 phase 的 logit/top1敏感性另报 |
| 遮挡或深度跳变 | 不跨实例插值；凸起 h 保留；不存在把投影平面当观测平面的补零 |
| 丢失扇区 | observed_valid=0 与真实 h=0 平面不同；顺序不能压紧 |
| 训练/导出 | 无 BN；GN affine 与模块冻结一致；输出与 PyTorch 对齐 |
| 大物体 | 全局图包含整个对象，纹理图保留原生分辨率并记录尺寸 |
| 空 query / 空实例 | no_pose，排序无 NaN，不伪造坐标 |

几何参考检查的通过只证明对应代数/数据合同，不代表本模型已实现、收敛、导出成功或满足产线延迟。

## 10. 部署与必须测量的边界

B v2 的学习与采样图使用 Conv2d、Conv1d（可表示为二维高1卷积）、GroupNorm、MLP、投影张量算术、gather/grid_sample 和池化。是否在锁定 ONNX/TensorRT 版本中完整导出，应通过实际解析与数值验收确定，不能因“普通算子”直接宣称零部署工作。[PyTorch grid_sample 语义](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.grid_sample.html)

若 circular padding 的导出存在限制，可用首尾切片 concat 后执行无 padding 卷积，数学语义不变；变更应通过整数循环平移检查。几何、投影、mask和最终位姿至少用 FP32；卷积 FP16 需另核对分数、top1、坐标和有效性。

计时至少拆为配准/深度图几何、法向、上下文/纹理编码、环位置生成与采样、环编码/头、NMS/位姿。分别报告 model-only 和端到端 p50/p95，记录 ROI 大小、对象数、候选数、有效观测点数、精度、硬件、软件版本及同步边界。

还需测量传感器原生纹理分辨率不足的比例、organized normals 的失败与噪声、二维候选召回以及有限投影环对自碰障碍的漏检。若这些指标不满足业务需要，应优先使用已复用三维接触几何的路线 A。B 的提速价值必须由同硬件、同候选预算和同输出任务的实测证明；2–5 ms 仍不是已取得的结果。
