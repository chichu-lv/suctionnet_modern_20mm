# B 训练方案：同姿态物理预训练与规划动作适配

本页按[共同训练合同](../TRAIN_EVAL_REPRODUCE.md)固定首轮 B 配方；参数源为[route_b.json](../reference/route_b.json)。新实例训练器尚待实现，这份设计 JSON 不是现有 E5 脚本的运行配置。本阶段不预设15–20个epoch或数小时收敛。

## 1. 进入训练的条件与环境

先完成[数据](DATA_PREPARATION.md)的真实资产/teacher核验、[预处理](PREPROCESSING.md)的坐标和缓存身份，以及[模型](MODEL_DESIGN.md)的 mask/维数/任意轴查询验收。固定同一源/目标清单与验证 query，记录被排除实例或动作及原因；不能每次验证改变有效子集以改善指标。

B 的学习网络从新架构初始化，ResNet-18 `weights=None`。生产环境沿用共同 PyTorch2.5.1，若用 torchvision 则固定0.20.1，不自动下载ImageNet权重。学习网络不需要spconv，离线teacher仍有其科学计算和数据依赖。参考脚本的Python3.12.14/NumPy2.3.5环境与既有GPU环境分开，完整环境说明见[共同复现页](../TRAIN_EVAL_REPRODUCE.md)。

每次真实运行保存代码commit、环境/硬件、配置、源/目标manifest和标签hash、normal/sampler/输入/精度合同、RNG与参数组。旧8通道E5或A权重不能通过改key或补零静默恢复为B。

## 2. 源域物理损失和正确归约

q_inst、Seal、Wrench经过sigmoid；q真值为同一p/n的 `S20*W20*(1-C_self)`，任务头在源阶段不监督。ell输出Laplace残差的log b，初始化log(0.1)，前向限制到[-6,1]：

$$L_q=\exp(-\ell)|\hat q-Q_{inst20}|+\ell,$$

$$L_{pre}=L_q+0.5\,SmoothL1_{\beta=.05}(\hat S,S_{20})
+0.25\,SmoothL1_{\beta=.05}(\hat W,W_{20}).$$

忽略Laplace常数log2；方差为 `2*exp(2*ell)`，ell不能称为log-variance或直接成功置信度。q由独立质量头回归，不用预测Seal/Wrench乘积替代。

首版共同label_valid要求S/W和两个self分项都已知且合法。每项先选择有效query，对每实例取均值，再对batch中有监督实例取均值，不能按全batch所有query直接平均而使有效候选多的物体更重。无有效监督返回显式zero/no-supervision并计数；不能用NaN乘零实现mask。真实Q=0保留为负例。

**没有有效源监督的实例不计入140000个成功受监督实例暴露。** 所有读取、排除、无监督和失败更新另计，训练器以实际参与成功更新的监督实例为预算；不能把空loss或跳过的实例当作完成预算。

## 3. native 首轮预训练默认

本节及第4–5节保留 native 90/10源域开发与产线适配配方。公共论文比较使用第7节独立的 `public20` profile，正式100场景重训不沿用本节dev选best规则。

| 项目 | 固定值 |
|---|---|
| 样本 | 一个实例的一个相机观测；共同完整候选前128项 |
| 抽样 | seed=42；均匀scene→均匀frame→该帧均匀训练有效实例 |
| 成功受监督实例预算 | 140000 |
| 有效batch / 默认执行 | 16；单GPU microbatch1、累积16 |
| 成功optimizer更新 | 8750 |
| 优化器 | AdamW；betas=(0.9,0.999)，eps=1e-8，weight_decay=.01 |
| 学习率 | peak3e-4；438次成功更新线性warmup，随后cosine至1e-5 |
| 精度 | 网络FP32；原始几何构造及新接触判定float64；teacher保持冻结算术 |
| 验证 | 成功步875、1750、…、8750，共10次 |
| 保存 | best.pt、last.pt；完整恢复状态 |

RGB jitter与灰度化遵循[预处理](PREPROCESSING.md)，同一实例的上下文/原生纹理使用同一光度参数。训练环phase由seed、样本/query ID和成功步确定抽取；仅改变观测特征采样，不改标签p/n。验证/推理phase=0。首版不缩放米制几何、不为query改轴后复制源标签。

旧E5的全场景曝光不能等价换算为单实例收敛点；140000是本方案固定工程预算。64-query只作为另命名对照；验证始终使用共同最多256池。后续AMP或microbatch变更须独立记录，成功更新与consumed_instance_draws分开统计；seed也不保证所有CUDA/grid_sample路径跨设备逐位确定。

### 源域选模

第一关键字为 `InstTop1_INST20@0.4` 最大，第二为 selected-Q宏平均最大，再并列选更早成功步。两个量都按frame内实例→scene内有效frame→scene等权聚合；no_pose对两者贡献0，空GT帧排除并单计。另报有输出条件下的Q微平均和输出数。

这只是共同协议的分母细化，不改变物理目标。存在所选姿态或Coverage等指标所需query的未知/非法teacher标签时，该评价不完整，应修复缓存；不能把未知补Q=0、删除实例分母或据此选best。详见[实验指标定义](EXPERIMENT_PLAN.md)。

## 4. 目标域1000次成功更新

每步抽2个target观测和2个source观测。target按既有train观测均匀抽样，默认不做成功/失败1:1重采样。一个观测有多个有效历史动作时先对动作损失求均值。历史动作直接以p_plan/n_plan调用同一评分接口，每动作一次 `BCEWithLogits(task_logit,outcome)`；不能先sigmoid再传该loss，也不反推S/W真值。

$$L_{ft}=\frac{L_{t1}+L_{t2}}2+0.5\frac{L_{s1}+L_{s2}}2.$$

逐条反向时target各权重0.5，source各0.25；之后**不再整体除4**。源回放仍取共同前128 query及上述物理损失。无几何关联目标动作不补失败；无有效源监督也不冒充完整2+2更新，处理/消费统计和实际有效监督必须保存。

| 成功更新 | 可训练模块 | 学习率 |
|---|---|---|
| 1–200 | q、Seal、Wrench、ell、task五个输出头 | heads3e-4 |
| 201–1000 | 五头；query融合、L1/L2投影、G投影、native纹理stem、三个环编码器 | heads1e-4；其余1e-5 |
| 全程 | ResNet-18 encoder冻结 | 不更新 |

B没有BN。GN/LN的affine跟随各自模块的冻结状态；LN继续仅对每角16通道归一化并按原mask清零。阶段切换保留已有head的Adam moments，只加入/解冻既定参数组，不重建optimizer丢失状态。参数组名和成员hash入checkpoint，不能照抄旧decoder/local-CNN名字。

每100次成功更新评价固定399条val动作；AUROC最大选best，依次以BCE更低、更早步打破平局。保留总数、有效数、正负数和排除原因；单类分组AUROC标未定义。87条test在权重、排序、数据口径冻结后一次性使用。

## 5. 执行误差正则：默认关闭

当前目标含义是给定观测和**规划命令**的结果，`actual_execution_verified=false`不能推出某个执行误差分布。若开发验证明确需要，沿用既有可选pose-consistency：lambda=.1、每动作最多2个、切平面半径2mm均匀圆盘、原有向轴周围2°均匀立体角球冠。先保留原动作唯一BCE，再加扰动概率对原姿态stop-gradient概率的均方一致性。

微扰需通过同实例/同观测组件/中心支持/断层规则，不通过丢弃，J=0正则为0；不将一次结果复制为多条BCE，不给改变p/n的源姿态复制旧S/W/self。该项默认disabled，幅度不是实测噪声，仅用固定开发集选用，test不参与。[完整定义](../TRAIN_EVAL_REPRODUCE.md)

## 6. 恢复状态和训练准入验收

checkpoint至少包含：模型类/版本、13/5通道输入、330融合及环门控/LN/summary合同、normal/sampler与精确query hash、INST/self配置、归一化/padding、精度/ranking、权重/优化器/scaler、成功步/监督实例/已消费draw、参数组、sampler/loader/RNG、源目标清单和代码身份。resume只接受同身份；初始化与resume互斥，不恢复退役迁移白名单。

计划新增 `train_instance.py --route b --stage pretrain|finetune` 等CLI需先实现parser、本地路径校验和运行配置转换；此处不是可复制执行的现成训练命令。当前可跑的是[数学参考命令](README.md)。

训练接线完成后，只做能解除具体风险的短链路验收：真实实例teacher与输入姿态一致；少量更新有有效梯度；冻结模块/affine确实不变；阶段切换moments保留；恢复后下一批样本/query/phase/LR一致；相同位置不同轴改变条件；输出仍为被评分p/n。取得这些证据后执行固定正式预算，不把短测损失下降写成模型效果结论。

## 7. 公共论文比较训练：public20

本节按[公共方法比较协议](../PUBLIC_METHOD_COMPARISON.md)准备本方法的公开比较权重，与native物理研究和产线1000步微调分别记录。它不改变B模型、损失、13/5通道输入、法向/候选规范或 `gt_INST20_pose_v2` teacher；公开指标和外部物理评价以公共页为准。

### 7.1 先开发、冻结，再从头重训

1. **90/10源域开发。** 使用[数据页](DATA_PREPARATION.md)的90个train场景及10个dev场景确定源域训练配置和推理规则；正式预算固定为下表，不从开发best所在步推导。本阶段可按第3节选择开发best。public20的选择过程不读取产线train/val/test，不用产线AUROC选择公开模型。每个公开比较单元的冻结前端、工作区身份及排序规则在正式测试前登记。
2. **冻结配置。** 保存开发结果、已选超参数、随机种子、140000成功实例预算、初始化方式、测试前端、工作区及推理配置的hash。此后不根据正式测试结果调整法向、候选、排序、阈值或训练长度。
3. **100场景正式重训。** 新建运行身份，从头初始化B，保持ResNet-18 `weights=None`，不加载开发best、旧E5或产线微调权重。训练集合为RealSense scene 0–99、每场景frame 0–255；均匀scene→均匀frame→该帧训练有效GT实例抽样。损失和teacher与第2节一致。
4. **固定终点权重。** 训练达到8750次成功optimizer更新后，使用该次的`last.pt`作为唯一正式公开权重。原dev场景已进入训练，不能再由它们选best、早停或决定训练长度；即使保留训练诊断，也不得称为独立验证。未完成预算的中断checkpoint不作为正式结果。

| public20正式重训项 | 固定执行 |
|---|---|
| 输入训练集合 | 100场景 × 256个可抽样视角，只有源域监督 |
| 成功受监督实例预算 | 140000；无监督、失败更新和被跳过项另计 |
| 有效batch / 默认执行 | 16；microbatch1、梯度累积16 |
| 成功更新数 | 8750；最后成功步即选定权重 |
| 优化器、学习率、query、增强、精度 | 沿用第3节被冻结配置，不因扩成100场景另改teacher或模型 |
| 正式checkpoint | 完整状态的`last.pt`；不得用已纳入train的dev再选`best.pt` |
| 目标域适配 | 不执行第4–5节产线训练/选模；不混产线动作、回放适配结果或ft checkpoint |

`source-only`约束指吸取评分网络的训练；冻结分割前端的额外训练数据和预训练权重来源仍须独立披露，不能因此宣称整个系统的训练信息只相差吸盘半径。架构中的task头可以存在，但未经目标监督的task输出不能参与公开候选排序；采用公共页约定的源域排序规则。

### 7.2 身份、推理和正式测试

开发与正式训练分别保存manifest、scene/frame范围、抽样计划、teacher release、checkpoint和run目录。checkpoint追加`profile=public20`、`phase=development|final_refit`、`train_manifest_sha256`、`teacher_release_sha256`、`frozen_config_sha256`、成功更新/成功实例数和`selection_rule`；正式项明确`selection_rule=last_successful_update_8750`。这些是拟实现的元数据字段，旧训练器不会自动支持。resume仅限同一phase及完整身份；不得把90场景训练中途状态恢复为100场景正式运行。

`public20`训练权重与推理profile分开登记。S1B、Diffusion-Suction和MultiGraspNet配对默认使用`uois_no_gt_workspace`：冻结UOIS，完整图像与有效depth输入，不用GT工作区，其他过滤按预先冻结的本方法规则。UISN的UOIS配对独立使用`uois_uisn_gt_aabb20`，UOAIS配对独立使用`uoais_uisn_gt_aabb20`；三者共用同一源域`last.pt`，不重新训练，但分别推理并保存预测SHA-256。只有前端与工作区身份均一致时才可复用预测，不能把默认UOIS结果直接作为UISN配对结果。

两个UISN profile将GT标签先坍成`seg > 0`，仅复用其原联合前景3D AABB工作区规则（outlier=0.02m）；该筛选确实改变支持和候选池，必须记录，不能称其不影响候选。GT不再用于实例身份、模型特征或对预测mask的匹配过滤，外部评测端正常使用GT。完整权限、逐帧工作区/mask/query hash、覆盖与错误账本依[数据页第7节](DATA_PREPARATION.md)执行；漏检帧保留在23040帧分母内。Diffusion-Suction历史工作区字段若未公开或未核实，保持`unverified`，不推断为已与本方法匹配。

只运行本方法在RealSense全部90个测试场景的23040帧推理及公共API评价，对方采用原论文报告值，不安排对方重训。测试使用完成预算后的同一源域权重，外部评测的AP聚合、半径例外与碰撞合同均引用公共页；此处不将native实例teacher当作公开benchmark evaluator。对历史依赖未固定或未公开的原论文细节保留`unreported`。

`public20` profile、100场景runtime配置、上述checkpoint字段及公开推理适配器均须先实现和校验，本页没有可直接执行的新增训练命令。
