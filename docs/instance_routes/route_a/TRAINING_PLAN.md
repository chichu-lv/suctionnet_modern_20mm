# A 训练方案：物理预训练与历史动作适配

本文固定 A 首轮训练执行计划，沿用[共同训练合同](../TRAIN_EVAL_REPRODUCE.md)及 [route_a.json](../reference/route_a.json)。`design_spec`不是现有E5训练器可直接加载的运行配置；A生产训练入口仍待接线。开始正式训练前先完成[数据](DATA_PREPARATION.md)、[模型](MODEL_DESIGN.md)及[真实短链路验收](EXPERIMENT_PLAN.md)。

## 1. 环境、初始化和运行身份

A优先沿用仓库已记录的Python3.10、PyTorch2.5.1+cu124、spconv-cu120 2.3.6组合，安装依据见[既有运行指南](../../MODEL_USAGE.md)。新12通道架构从头初始化；旧E5的8通道或跨路线checkpoint不能改键、补零后伪装为正常resume。数学参考使用独立Python3.12.14/NumPy2.3.5环境，不为执行参考工具升级GPU环境。

每次真实运行保存完整运行配置、environment.txt、代码commit、GPU/CPU线程、Torch/CUDA/cuDNN/spconv版本、真实microbatch、数据/标签/标定hash、normal/sampler/精确query身份及随机种子。初始化和同身份resume互斥，保持当前checkpoint政策。

## 2. 源域损失与有效归约

模型q、Seal、Wrench经sigmoid；ell为Laplace `log b`，初始化log(.1)、前向限制[-6,1]。同姿态真值为 `Q_inst=S20*W20*(1-C_self)`：

```text
Lq = exp(-ell) * abs(q_inst_hat - Q_inst) + ell
Lpre = Lq + 0.5*SmoothL1_beta0.05(seal_hat, S20)
          + 0.25*SmoothL1_beta0.05(wrench_hat, W20)
```

省略常数log2，ell不是log方差；对应残差方差是 `2*exp(2*ell)`。ell不是已校准的任务概率，不进入默认排序。

首版三个物理损失使用共同label_valid：S/W及两个self分项全部已知。先选有效query，再对单实例平均，最后对batch中有监督实例平均。未知标签和padding不参与计算，不能NaN乘零；Q_inst=0仍是合法负例。全false实例返回no-supervision并计数，不计为成功更新的受监督实例暴露；运行数据计划应提供足量合法项，替补/跳过与额外消费必须记录。

## 3. native 首轮预训练配方

本节及第4–5节保留 native 90/10源域开发与产线适配配方。公共论文比较使用第8节独立的 `public20` profile，正式100场景重训不沿用本节dev选best规则。

| 参数 | A默认 |
|---|---|
| 采样单位/顺序 | 先均匀选90个train scene，再均匀frame，再均匀该frame训练有效实例；seed=42 |
| query | 每实例共同完整池前128；64属于独立可选配置 |
| 成功预算 | 140000个参与成功更新的实例暴露 |
| 有效batch | 16实例；单GPU默认microbatch1、累积16 |
| 成功optimizer更新 | 8750 |
| 优化器 | AdamW，betas(.9,.999)、eps1e-8、weight_decay .01 |
| 学习率 | peak3e-4，前438个成功更新线性warmup，之后cosine至1e-5 |
| 精度 | 网络FP32；从原数据构造float64接触几何与边界判定，冻结teacher保留原算术 |
| 验证 | 每875个成功更新，875至8750共10次 |
| 选模 | scene-macro InstTop1_INST20@0.4最大，再按所选Q_inst宏平均最大，仍平局选更早步 |
| 保存 | best.pt与last.pt及完整状态 |

训练与回放的raw支持、主干点采用同一实例级RGB变换：brightness/contrast/saturation=.2、hue=.05、灰度概率.2；mean/std依预处理合同。首版不做几何缩放。验证保持固定实例/query顺序和归一化。

次级所选Q_inst宏平均采用与主指标相同的frame内实例→scene内frame→scene等权聚合，no_pose贡献0，空GT帧排除并另计；另报有输出姿态的条件Q微平均及输出数。这是共同评价口径的细化，不改变物理目标。若被选姿态或候选覆盖指标所需query的teacher标签未知/非法，该次评价标不完整并修复，不能将未知标签补0或删除实例分母后参与选模。

140000是新工程预算，不由旧E5的场景更新步数证明最优。成功步、成功实例暴露和已消费样本分别记录；后续AMP发生skip时不能消耗了数据却增加成功预算。microbatch×累积须为16，但不同microbatch会改变BN前向统计，不宣称逐位等价。PyTorch/CUDA仍需记录实际确定性限制。

## 4. 1000步目标域微调

从同身份源域best初始化新的task适配。每次成功更新抽2个target实例观测和2个source实例观测；target按train观测均匀，不默认按成败1:1重采样。每观测若有多条有效历史动作，先在观测内平均其BCE：

```text
Lt(obs) = mean(BCEWithLogits(score_at_pose(p_plan,n_plan).task_logit, outcome))
Lft = (Lt1 + Lt2)/2 + 0.5*(Ls1 + Ls2)/2
```

逐条反向时两个target各权重0.5、两个source各0.25；之后不再整体除4。任务BCE读取原始logit。历史结果不产生Seal/Wrench标签，也不向20mm圆盘扩散。source回放使用同一v3 query身份及物理v2标签，默认128 query。

| 成功更新 | A可训练模块 | 学习率 |
|---|---|---|
| 1–200 | q/Seal/Wrench/ell/task输出头 | heads3e-4 |
| 201–1000 | 上述头 + decoder、接触编码、query融合及对象池化投影 | heads1e-4，其余1e-5 |
| 全程 | encoder冻结；所有BatchNorm running stats与affine冻结 | — |

新增层的实际成员必须与参数组清单/hash一致；不能直接照搬E5组名。第201步切换保留head的Adam moments，只新增/解冻既定组，不重建优化器丢失状态。

每次 `model.train()`之后重新将所有有running stats的BN设eval并冻结affine；仅requires_grad=false不足以冻结统计。拒绝不具备tracked running stats却声称执行同一冻结策略的实现。微调前后比较running_mean、running_var、num_batches_tracked及affine保持不变；梯度累积不等于BN看到了batch16。

## 5. 开发验证与可选姿态一致性

每100个成功更新在固定399条val动作上评价。以AUROC最大选择best，再按BCE低、再按更早步。总399、有效数、正负数和排除原因同时报告；保持固定可比较的动作子集，单类group的AUROC记未定义。

默认 `pose_consistency.enabled=false`。若开发集暴露执行偏差敏感性，可按共同规范预注册一次对照：每动作最多2个合法微扰，切平面半径2mm均匀圆盘、原轴2°立体角均匀球冠；原动作保留唯一BCE，增加权重.1的概率一致性MSE，以精确姿态概率stopgrad为参照。扰动需同实例/组件和合法支持，J=0时该项为0，不复制源物理标签到新p/n。这些数值不是已测执行误差，不作为新默认。

87条test在权重和ranking模式冻结后使用一次，不用于选alpha、法向或学习率。当前目标监督估计的是给定观测与规划命令的成败；实际末端执行轨迹未核验。后处理按[固定模式](POSTPROCESSING.md)执行。

## 6. Checkpoint和恢复验收

checkpoint至少保存：架构/12通道/531融合/谐波身份，normal/sampler，精确p/n列表hash，物理self/INST配置，RGB归一化，ranking模式，模型/优化器/scaler，成功步及成功/消费暴露数，参数组成员，loader/sampler/RNG，源/目标manifest和代码身份。

短链路必须证明：物理/任务loss实际反向且相应目标可下降；保存恢复后的下一批实例/query、学习率和采样RNG一致；第201步参数组正确、head moments保留；无监督项不伪装成功曝光；BN冻结真实生效。完成后才运行正式8750/1000步预算，不用旧迁移白名单绕过身份错误。

## 7. 当前命令与待实现入口

当前可复现数学入口：

```bash
python docs/instance_routes/reference/geometry_contract.py selftest
python docs/instance_routes/reference/revision_contract.py selftest
python docs/instance_routes/reference/sampling_contract.py selftest
python docs/instance_routes/reference/contact_boundary_contract.py self-test
```

未来 `scripts/train_instance.py`应提供route/stage/config/manifest/label-manifest/run-dir/device等参数及互斥初始化/resume；这是[共同CLI合同](../TRAIN_EVAL_REPRODUCE.md)而非现有命令。实施时显式转换design_spec为已校验本地路径的runtime配置，旧 `train_pretrain.py`/`finetune.py`继续服务既有E5。正式运行结果与可复核文件要求见[实验方案](EXPERIMENT_PLAN.md)。

## 8. 公共论文比较训练：public20

本节按[公共方法比较协议](../PUBLIC_METHOD_COMPARISON.md)准备本方法的公开比较权重，与native物理研究和产线1000步微调分别记录。它不改变A模型、损失、12通道输入、法向/候选规范或 `gt_INST20_pose_v2` teacher；公开指标和外部物理评价以公共页为准。

### 8.1 先开发、冻结，再从头重训

1. **90/10源域开发。** 使用[数据页](DATA_PREPARATION.md)的90个train场景及10个dev场景确定源域训练配置和推理规则；正式预算固定为下表，不从开发best所在步推导。本阶段可按第3节选择开发best。public20的选择过程不读取产线train/val/test，不用产线AUROC选择公开模型。每个公开比较单元的冻结前端、工作区身份及排序规则在正式测试前登记。
2. **冻结配置。** 保存开发结果、已选超参数、随机种子、140000成功实例预算、初始化方式、测试前端、工作区及推理配置的hash。此后不根据正式测试结果调整法向、候选、排序、阈值或训练长度。
3. **100场景正式重训。** 新建运行身份，从头初始化A，不加载开发best、旧E5或产线微调权重。训练集合为RealSense scene 0–99、每场景frame 0–255；均匀scene→均匀frame→该帧训练有效GT实例抽样。损失和teacher与第2节一致。
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

### 8.2 身份、推理和正式测试

开发与正式训练分别保存manifest、scene/frame范围、抽样计划、teacher release、checkpoint和run目录。checkpoint追加`profile=public20`、`phase=development|final_refit`、`train_manifest_sha256`、`teacher_release_sha256`、`frozen_config_sha256`、成功更新/成功实例数和`selection_rule`；正式项明确`selection_rule=last_successful_update_8750`。这些是拟实现的元数据字段，旧训练器不会自动支持。resume仅限同一phase及完整身份；不得把90场景训练中途状态恢复为100场景正式运行。

`public20`训练权重与推理profile分开登记。S1B、Diffusion-Suction和MultiGraspNet配对默认使用`uois_no_gt_workspace`：冻结UOIS，完整图像与有效depth输入，不用GT工作区，其他过滤按预先冻结的本方法规则。UISN的UOIS配对独立使用`uois_uisn_gt_aabb20`，UOAIS配对独立使用`uoais_uisn_gt_aabb20`；三者共用同一源域`last.pt`，不重新训练，但分别推理并保存预测SHA-256。只有前端与工作区身份均一致时才可复用预测，不能把默认UOIS结果直接作为UISN配对结果。

两个UISN profile将GT标签先坍成`seg > 0`，仅复用其原联合前景3D AABB工作区规则（outlier=0.02m）；该筛选确实改变支持和候选池，必须记录，不能称其不影响候选。GT不再用于实例身份、模型特征或对预测mask的匹配过滤，外部评测端正常使用GT。完整权限、逐帧工作区/mask/query hash、覆盖与错误账本依[数据页第7节](DATA_PREPARATION.md)执行；漏检帧保留在23040帧分母内。Diffusion-Suction历史工作区字段若未公开或未核实，保持`unverified`，不推断为已与本方法匹配。

只运行本方法在RealSense全部90个测试场景的23040帧推理及公共API评价，对方采用原论文报告值，不安排对方重训。测试使用完成预算后的同一源域权重，外部评测的AP聚合、半径例外与碰撞合同均引用公共页；此处不将native实例teacher当作公开benchmark evaluator。对历史依赖未固定或未公开的原论文细节保留`unreported`。

`public20` profile、100场景runtime配置、上述checkpoint字段及公开推理适配器均须先实现和校验，本页没有可直接执行的新增训练命令。
