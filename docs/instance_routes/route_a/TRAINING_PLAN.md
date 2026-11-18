# A 训练方案：物理预训练与历史动作适配

A新训练器已实现在 `scripts/train_instance.py` / `suction20/instance_training.py`，运行配置为 `configs/instance_a.yaml`。[route_a.json](../reference/route_a.json) 继续作为design_spec，不直接传给旧E5脚本。当前从头直接100场景训练、完整source test选best，公开半径-only协议deferred。真实SpConv/GPU和完整训练尚未执行；先完成[数据](DATA_PREPARATION.md)和[真实短链路验收](EXPERIMENT_PLAN.md)。

## 1. 环境、初始化和运行身份

A优先沿用仓库已记录的Python3.10、PyTorch2.5.1+cu124、spconv-cu120 2.3.6组合，安装依据见[既有运行指南](../../MODEL_USAGE.md)。新12通道架构从头初始化；旧E5的8通道或跨路线checkpoint不能改键、补零后伪装为正常resume。数学参考及CPU测试使用独立Python3.12.14/NumPy2.3.5环境；13项A测试显式注入测试backbone，不代表上述真实GPU环境已验证，不为执行参考工具升级GPU环境。

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

## 3. 直接100场景源预训练

source train为scene 0–99全部100场景、frame 0–255，模型从头初始化；source test为scene 100–189完整90场景、每场景256帧，直接用于best选择。不再执行90/10开发后final refit。test不参与反向传播，但其选模用途必须进入结果身份。

| 参数 | A默认 |
|---|---|
| 采样单位/顺序 | 先均匀选100个train scene，再均匀frame，再均匀该frame训练有效实例；seed=42 |
| query | 每实例共同完整池前128；64属于独立可选配置 |
| 成功预算 | 140000个参与成功更新的实例暴露 |
| 有效batch | 16实例；单GPU默认microbatch1、累积16 |
| 成功optimizer更新 | 8750 |
| 优化器 | AdamW，betas(.9,.999)、eps1e-8、weight_decay .01 |
| 学习率 | peak3e-4，前438个成功更新线性warmup，之后cosine至1e-5 |
| 精度 | 网络FP32；从原数据构造float64接触几何与边界判定，冻结teacher保留原算术 |
| 选模评价 | 每875个成功更新，875至8750共10次；每次完整source test，不抽帧 |
| 选模 | scene-macro InstTop1_INST20@0.4最大，再按所选Q_inst宏平均最大，仍平局选更早步 |
| 保存 | test-selected best.pt，训练进度/预算终点last.pt及完整状态 |

训练与回放的raw支持、主干点采用同一实例级RGB变换：brightness/contrast/saturation=.2、hue=.05、灰度概率.2；mean/std依预处理合同。首版不做几何缩放。验证保持固定实例/query顺序和归一化。

次级所选Q_inst宏平均采用与主指标相同的frame内实例→scene内frame→scene等权聚合，no_pose贡献0，空GT帧排除并另计；另报有输出姿态的条件Q微平均及输出数。这是共同评价口径的细化，不改变物理目标。若被选姿态或候选覆盖指标所需query的teacher标签未知/非法，该次评价标不完整并修复，不能将未知标签补0或删除实例分母后参与选模。

best及其source test成绩标记 `selection_role=test_selected`、`independent_test_estimate=false`，不能作为独立盲测或填写论文全景AP。8750成功步的last是固定预算终点；提前暂停的last仅表示进度。

默认 `singleton_bn_policy=running_stats` 仅在训练时某BN每通道输入值数≤1时使用已有running mean/variance，不更新该次统计/计数；其余源训练输入保持普通BN路径，不将源预训练全部BN设eval。微调冻结规则见下一节。此策略进入架构身份，不用增加伪造活动点避免BN退化。

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

## 5. 目标验证与可选姿态一致性

每100个成功更新在固定399条val动作上评价。以AUROC最大选择best，再按BCE低、再按更早步。总399、有效数、正负数和排除原因同时报告；保持固定可比较的动作子集，单类group的AUROC记未定义。

默认 `pose_consistency.enabled=false`；当前训练CLI未实现该可选微扰正则，不能把它标成已启用。若目标399 val暴露执行偏差敏感性，须先实现并验证，随后可按共同规范预注册一次对照：每动作最多2个合法微扰，切平面半径2mm均匀圆盘、原轴2°立体角均匀球冠；原动作保留唯一BCE，增加权重.1的概率一致性MSE，以精确姿态概率stopgrad为参照。扰动需同实例/组件和合法支持，J=0时该项为0，不复制源物理标签到新p/n。这些数值不是已测执行误差，不作为新默认。

目标87条test在权重和ranking模式冻结后使用一次，不用于选alpha、法向或学习率。当前目标监督估计的是给定观测与规划命令的成败；实际末端执行轨迹未核验。后处理按[固定模式](POSTPROCESSING.md)执行。

## 6. Checkpoint和恢复验收

checkpoint至少保存：架构/12通道/531融合/谐波与singleton BN身份，normal/sampler，精确p/n列表hash，物理self/INST配置，RGB归一化，ranking模式，模型/优化器/scaler，成功步及成功/消费暴露数，参数组成员，loader/sampler/RNG，源/目标manifest和代码身份。

短链路必须证明：物理/任务loss实际反向且相应目标可下降；保存恢复后的下一批实例/query、学习率和采样RNG一致；第201步参数组正确、head moments保留；无监督项不伪装成功曝光；BN冻结真实生效。完成后才运行正式8750/1000步预算，不用旧迁移白名单绕过身份错误。

## 7. 可执行命令与运行产物

按[数据准备](DATA_PREPARATION.md)生成完整train/test缓存。真实A运行需要兼容PyTorch/CUDA/spconv环境及固定外部骨干文件；`backbone_root`不是可选的测试替代入口。已有下载/校验命令如下：

```bash
python scripts/setup_backbone.py --output /data/backbones/graspness_modern
python scripts/train_instance.py --config configs/instance_a.yaml --stage pretrain --backbone-root /data/backbones/graspness_modern --source-manifest /data/instance_v3/train/manifest_train.jsonl --selection-manifest /data/instance_v3/test/manifest_test.jsonl --run-dir /data/runs/a_source100 --device cuda
```

以上路径须替换为真实目录。也可在runtime YAML中设置 `model.backbone_root`，CLI覆盖值会进入运行配置身份。wrapper默认8通道保留旧E5，A显式传12，从头初始化；首轮不传 `--init-checkpoint` 或目标数据。CPU单元测试的注入backbone不能由CLI自动替换真实SpConv，也不能用这些单元测试生成可发布的A训练结果。

需要暂停后恢复时，使用以下两条命令代替一次性完整执行，保持相同8750步预算、配置、路径和数据身份：

```bash
python scripts/train_instance.py --config configs/instance_a.yaml --stage pretrain --backbone-root /data/backbones/graspness_modern --source-manifest /data/instance_v3/train/manifest_train.jsonl --selection-manifest /data/instance_v3/test/manifest_test.jsonl --run-dir /data/runs/a_source100 --device cuda --stop-after 10
python scripts/train_instance.py --config configs/instance_a.yaml --stage pretrain --backbone-root /data/backbones/graspness_modern --source-manifest /data/instance_v3/train/manifest_train.jsonl --selection-manifest /data/instance_v3/test/manifest_test.jsonl --run-dir /data/runs/a_source100 --device cuda --resume /data/runs/a_source100/last.pt
```

`--stop-after` 只在成功更新边界暂停，不缩短原正式预算；提前暂停的last不是预算完成。小规模source分片检查须显式 `--allow-partial-manifests`，身份记录partial，不能当作完整test选模结果。正式运行要求完整100-train/90-test、每场景256帧覆盖；未知/错姿态teacher标签不能补0后继续选best。

目标微调从同路线源best初始化，source train用于回放，目标train用于BCE，目标399 val作selection：

```bash
python scripts/train_instance.py --config configs/instance_a.yaml --stage finetune --backbone-root /data/backbones/graspness_modern --source-manifest /data/instance_v3/train/manifest_train.jsonl --target-manifest /data/attempts/train.jsonl --selection-manifest /data/attempts/val.jsonl --init-checkpoint /data/runs/a_source100/best.pt --run-dir /data/runs/a_target --device cuda
```

目标manifest须具备原RGB-D/instance mask及规划p/n；旧pointcloud-only清单需补原资产。目标87 test不传入训练器。微调best为validation-selected，仍保留源预训练test选模来源。恢复微调用 `--resume /data/runs/a_target/last.pt` 替换 `--init-checkpoint`，二者互斥。

运行目录保存配置、数据/模型身份、事件、best/last模型和优化器、成功步/实际监督计数、source/target sampler、增强及全局RNG、参数组和选模策略。默认网络FP32、接触几何CPU float64；没有从CPU单元测试推导已达到GPU吞吐。真实资产、真实SpConv前后向和正式训练仍待完成。

## 8. 公开比较暂缓

`public_evaluation.status=deferred`、`radius_only_protocol=null`。本轮不实现或运行“官方评价器只改半径”，不再创建单独public20 final-refit或指定last-only公开权重，也不把R10等价门禁作为开训条件。各论文原值继续保留，本方法公开结果待测。

未来恢复公开评价时，先冻结每方法的测试/指标/输入权限和半径口径，并披露source test选模。当前test-selected实例Top1不能填入论文全景AP列，更不能标成与对手仅吸盘尺寸不同的独立测试结果。
