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

## 3. 唯一首轮预训练配方

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
