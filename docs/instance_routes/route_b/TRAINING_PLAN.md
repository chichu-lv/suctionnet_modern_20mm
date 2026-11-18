# B 训练方案：同姿态物理预训练与规划动作适配

本页给出已实现训练器 `scripts/train_instance.py` / `suction20/instance_training.py` 的B运行方式，使用 `configs/instance_b.yaml`。设计依据为[route_b.json](../reference/route_b.json)，该JSON继续作为设计合同，不直接传给训练CLI。源域直接100场景从头训练，best只使用source test全部90场景各固定8帧的子集选择；公开半径评价协议待定。本阶段不预设15–20个epoch或数小时收敛。

## 1. 进入训练的条件与环境

先完成[数据](DATA_PREPARATION.md)的真实资产/teacher核验、[预处理](PREPROCESSING.md)的坐标和缓存身份，以及[模型](MODEL_DESIGN.md)的 mask/维数/任意轴查询验收。固定同一源/目标清单与验证 query，记录被排除实例或动作及原因；不能每次验证改变有效子集以改善指标。

B的学习网络从新架构初始化，ResNet-18不加载外部预训练权重；实现直接使用PyTorch 2.5.1，不依赖torchvision或spconv，也不自动下载ImageNet权重。离线teacher仍有其科学计算和数据依赖。参考脚本的Python3.12.14/NumPy2.3.5环境与既有GPU环境分开，完整环境说明见[共同复现页](../TRAIN_EVAL_REPRODUCE.md)。

每次真实运行保存代码commit、环境/硬件、配置、源/目标manifest与标签路径、实际标定和已知physics、normal/sampler/输入/精度合同、RNG与参数组。旧8通道E5或A权重不能通过改key或补零静默恢复为B。

## 2. 源域物理损失和正确归约

q_inst、Seal、Wrench经过sigmoid；q真值为同一p/n的 `S20*W20*(1-C_self)`，任务头在源阶段不监督。ell输出Laplace残差的log b，初始化log(0.1)，前向限制到[-6,1]：

$$L_q=\exp(-\ell)|\hat q-Q_{inst20}|+\ell,$$

$$L_{pre}=L_q+0.5\,SmoothL1_{\beta=.05}(\hat S,S_{20})
+0.25\,SmoothL1_{\beta=.05}(\hat W,W_{20}).$$

忽略Laplace常数log2；方差为 `2*exp(2*ell)`，ell不能称为log-variance或直接成功置信度。q由独立质量头回归，不用预测Seal/Wrench乘积替代。

首版共同label_valid要求S/W和两个self分项都已知且合法。每项先选择有效query，对每实例取均值，再对batch中有监督实例取均值，不能按全batch所有query直接平均而使有效候选多的物体更重。无有效监督返回显式zero/no-supervision并计数；不能用NaN乘零实现mask。真实Q=0保留为负例。

**没有有效源监督的实例不计入140000个成功受监督实例暴露。** 所有读取、排除、无监督和失败更新另计，训练器以实际参与成功更新的监督实例为预算；不能把空loss或跳过的实例当作完成预算。

## 3. 直接100场景源预训练

训练集合固定scene 0–99全部100场景、每场景frame 0–255；模型从头初始化。选模集合固定source test scene 100–189全部90场景、每场景8帧，共720帧。完整90场景×256帧test留作训练后固定best的报告。test子集用于选best的决定明确进入运行身份，不作为独立泛化估计；不再执行90/10开发后final refit。

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
| 选模评价 | `pretrain.evaluate_every: 875`；8750成功步内共10次，每次固定90场景×8帧、每实例完整最多256-query |
| 选择范围 | `pretrain.selection_frames_per_scene: 8`；`selection_scope=fixed_test_subset` |
| 保存 | test-selected best.pt；`pretrain.checkpoint_every: 875`，每875成功步保存last.pt，最终步记录预算完成；完整恢复状态 |

子集评价与恢复checkpoint保存使用独立配置，默认间隔都为875步；保存last本身不触发额外评价。每次源评价均使用相同的90场景×8帧，共10次，原预算终点也在这个子集上评价，不在末步自动切换到全量test。微调仍每100成功步评价并保存。若另设源评价节点，`evaluation_steps`须为预算内严格递增的正整数列表，不能与`evaluate_every`同时配置；训练器始终将原预算终点纳入评价。帧数、节点和保存间隔进入运行配置，严格resume时保持一致。

`pretrain.selection_frames_per_scene`默认8，固定帧ID为 `0,32,64,96,128,160,192,224`，按8位bit-reversal前缀取样并排序；16帧配置包含这些帧，256帧配置才使用完整test。旧配置未声明该字段时仍保持256帧语义。全部90个场景的所需帧及其中全部可见实例保留，每实例仍使用完整最多256个query；GT空帧以显式声明计入覆盖，无法向、少支持或无输出实例仍进入相应分母。完整test manifest会自动筛选，也可直接使用只含固定所选帧的正式selection清单，均不需`--allow-partial-manifests`。

固定小子集减少周期评价成本，但较少观测可能使相近checkpoint的排名波动；重复使用同一帧集合只能消除每次随机换样本的差异，不能保证选出的best在完整test上最优。训练结束后由独立CLI对已固定best生成完整test报告，不用该报告重新选择checkpoint。

RGB jitter与灰度化遵循[预处理](PREPROCESSING.md)，同一实例的上下文/原生纹理使用同一光度参数。训练环phase由seed、样本身份、精确p/n字节的query身份和成功步确定抽取，与query顺序或chunk无关；仅改变观测特征采样，不改标签p/n。验证/推理phase=0。首版不缩放米制几何、不为query改轴后复制源标签。

旧E5的全场景曝光不能等价换算为单实例收敛点；140000是本方案固定工程预算。64-query只作为另命名对照；验证始终使用共同最多256池。后续AMP或microbatch变更须独立记录，成功更新与consumed_instance_draws分开统计；seed也不保证所有CUDA/grid_sample路径跨设备逐位确定。

### 源域选模与结果身份

源test固定子集只评价，不参与反向传播，但会用于checkpoint选择。`best.pt`必须标记`selection_role=test_selected`、`selection_scope=fixed_test_subset`、`independent_test_estimate=false`；由该best计算的子集成绩与训练后完整test报告都不能作为独立盲测，也不能填写“公平官方测试”结果行。`last.pt`在8750成功更新完成时是固定预算终点，提前暂停的last只是进度checkpoint。

第一关键字为 `InstTop1_INST20@0.4` 最大，第二为 selected-Q宏平均最大，再并列选更早成功步。两个量都按frame内实例→scene内有效frame→scene等权聚合；no_pose对两者贡献0，空GT帧排除并单计。另报有输出条件下的Q微平均和输出数。

这只是共同协议的分母细化，不改变物理目标。存在所选姿态或Coverage等指标所需query的未知/非法teacher标签时，该评价不完整，应修复缓存；不能把未知补Q=0、删除实例分母或据此选best。详见[实验指标定义](EXPERIMENT_PLAN.md)。

周期源评价只遍历全部90场景的固定8帧、共720帧对应实例，`evaluation_stepXXXXXX.json`保存该子集指标/选择键、selection范围和`record_details`。默认 `evaluation_records=best_and_final` 将逐实例/动作明细流式写入临时 `.records.jsonl.gz`，只在当次更新best或最终步保留，其余删除；最终步明细仍是固定子集，不代表已完成完整test。可选 `all` 保留各节点、`none` 完全不写明细；不内嵌records数组，也不把全部明细常驻内存。该策略减少长期存储，仍有一次临时压缩流写入。

共享训练器读取模型的 `min_support_points`；当前B构造器没有该参数，因此回退默认100，不能在B YAML中加入该未支持选项。无有效标签和支持不足分别记账，不把任一跳过项计入140000成功实例预算；builder的准备阈值另存metadata，不以缓存trainable代替运行门槛。

## 4. 目标域1000次成功更新

每步抽2个target观测和2个source观测。target按既有train观测均匀抽样，默认不做成功/失败1:1重采样。一个观测有多个有效历史动作时先对动作损失求均值。历史动作直接以p_plan/n_plan调用同一评分接口，每动作一次 `BCEWithLogits(task_logit,outcome)`；不能先sigmoid再传该loss，也不反推S/W真值。

$$L_{ft}=\frac{L_{t1}+L_{t2}}2+0.5\frac{L_{s1}+L_{s2}}2.$$

逐条反向时target各权重0.5，source各0.25；之后**不再整体除4**。源回放仍取共同前128 query及上述物理损失。无几何关联目标动作不补失败；无有效源监督也不冒充完整2+2更新，处理/消费统计和实际有效监督必须保存。

| 成功更新 | 可训练模块 | 学习率 |
|---|---|---|
| 1–200 | q、Seal、Wrench、ell、task五个输出头 | heads3e-4 |
| 201–1000 | 五头；query融合、L1/L2投影、G投影、native纹理stem、三个环编码器 | heads1e-4；其余1e-5 |
| 全程 | ResNet-18 encoder冻结 | 不更新 |

B没有BN。GN/LN的affine跟随各自模块的冻结状态；LN继续仅对每角16通道归一化并按原mask清零。阶段切换保留已有head的Adam moments，只加入/解冻既定参数组，不重建optimizer丢失状态。参数组名和完整成员列表写入checkpoint，不能照抄旧decoder/local-CNN名字。

共同的源评价、目标动作评价与`infer_instance`入口逐模块保存`.training`，正常返回或异常退出均直接恢复每层原状态。即使根模块处于train、部分子模块处于eval，评价之后也保持原有组合；A冻结的BN不会被根模块`train(True)`意外重新启用，B同样保持各模块的原始模式。

每100次成功更新评价固定399条val动作；AUROC最大选best，依次以BCE更低、更早步打破平局。保留总数、有效数、正负数和排除原因；单类分组AUROC标未定义。87条test在权重、排序、数据口径冻结后一次性使用。

## 5. 执行误差正则：默认关闭

当前目标含义是给定观测和**规划命令**的结果，`actual_execution_verified=false`不能推出某个执行误差分布。若开发验证明确需要，沿用既有可选pose-consistency：lambda=.1、每动作最多2个、切平面半径2mm均匀圆盘、原有向轴周围2°均匀立体角球冠。先保留原动作唯一BCE，再加扰动概率对原姿态stop-gradient概率的均方一致性。

微扰需通过同实例/同观测组件/中心支持/断层规则，不通过丢弃，J=0正则为0；不将一次结果复制为多条BCE，不给改变p/n的源姿态复制旧S/W/self。该项默认disabled，当前训练CLI未实现该可选微扰正则；不得在运行配置中标为已启用。幅度不是实测噪声；后续实现和预注册后只能在目标399 val上决定选用，目标87 test不参与。[完整定义](../TRAIN_EVAL_REPRODUCE.md)

## 6. 恢复状态和训练准入验收

checkpoint至少包含：模型类/版本、13/5通道输入、330融合及环门控/LN/summary合同、normal/sampler/precision与query/label精确ID、p/n对齐合同、INST/self配置、归一化/padding、精度/ranking、权重/优化器/scaler、成功步/监督实例/已消费draw、参数组、sampler/loader/RNG、源目标清单路径与采样ID序列，以及完整config_definition和显式runtime_contract。初始化与resume互斥；按模型/配置/已知physics/预处理和状态结构检查兼容性。旧校验和字段忽略，未知physics保留null，不计算文件或数组校验和，也不要求为迁移metadata重算全部teacher。

`--init-checkpoint`用于开始新的微调运行：先检查源checkpoint配置与已有`identity.config_definition`自洽，再比较模型语义。B的`model`配置仅允许执行分块`query_chunk`不同；`phase_seed`、`training_phase`及其他结构/几何参数仍严格匹配，路线、质量、数据/teacher合同不放宽，权重使用`strict=True`加载。路线A的执行例外是`backbone_root`和`query_chunk_size`，其点采样、几何和BN策略仍严格匹配；B不接受`backbone_root`，也不能用A权重初始化。不能先覆盖执行字段再忽略源配置矛盾。

微调初始化在`model_compatibility`中记录`semantic_model`、`source_execution`及`target_execution`。普通`--resume`继续要求完整配置相等，包含chunk、phase、评价节点及保存间隔，并恢复优化器、采样和RNG；新微调允许的chunk变化不扩展到同一次运行的resume。已省略的模型参数按实际构造器默认值参与语义比对，未知新增参数也保留在比较中。

运行配置校验 `instance_runtime_v1`、route=b、source scene范围和 `public_evaluation.status=deferred`。CLI支持严格resume和finetune初始化，参数见第7节；不通过旧E5参数或替换文件名绕过模型/数据身份。

已有模型CPU合成前后向验证；真实资产具备后，只做能解除具体风险的短链路验收：真实实例teacher与输入姿态一致；少量更新有有效梯度；冻结模块/affine确实不变；阶段切换moments保留；恢复后下一批样本/query/phase/LR一致；相同位置不同轴改变条件；输出仍为被评分p/n。取得这些证据后执行固定正式预算，不把短测损失下降写成模型效果结论。

## 7. 可执行命令与运行产物

先按[数据准备](DATA_PREPARATION.md)生成全部100场景的完整train缓存和正式固定帧selection缓存。以下示例沿用已有完整test manifest，训练入口仍按配置自动筛为每场景8帧；也可将各条源训练命令的`--selection-manifest`替换为数据准备页得到的`"$SELECTION_MANIFEST"`，无需先构建完整test。两种来源都不加`--allow-partial-manifests`，同一次运行的resume继续使用相同数据路径和定义。以下路径替换为真实路径，在仓库根目录执行；选择CPU可检查接线，正式GPU运行需已安装匹配环境。

```bash
python scripts/train_instance.py --config configs/instance_b.yaml --stage pretrain --source-manifest /data/instance_v3/train/manifest_train.jsonl --selection-manifest /data/instance_v3/test/manifest_test.jsonl --run-dir /data/runs/b_source100 --device cuda
```

首轮不传 `--init-checkpoint`，也不传目标域数据。`--device cpu` 使用相同模型和几何合同，不能由CPU短测推定GPU时延。完整运行默认8750成功更新；140000是实际有监督实例暴露预算，不是100场景全帧的epoch计数。

`train_instance.py`、`infer_instance.py`、`evaluate_instance.py`统一支持正整数`--query-chunk-size`；对B映射为`model.query_chunk`，默认64，例如附加`--query-chunk-size 32`可减小查询分块。新微调初始化或独立推理/评价可改变这一执行选项，候选池、监督query数、phase规则及权重结构不随之改变。训练覆盖值写入新运行配置，后续`--resume`必须保持相同实际chunk；不据此声称不同设备或分块具有逐位相同的浮点结果。B无需也不接受`--backbone-root`；A训练则要求该参数或配置中的实际路径，`null`在读取manifest前明确报错。

暂停和严格恢复使用同一运行目录、同一配置和数据身份：

```bash
python scripts/train_instance.py --config configs/instance_b.yaml --stage pretrain --source-manifest /data/instance_v3/train/manifest_train.jsonl --selection-manifest /data/instance_v3/test/manifest_test.jsonl --run-dir /data/runs/b_source100 --device cuda --stop-after 10
python scripts/train_instance.py --config configs/instance_b.yaml --stage pretrain --source-manifest /data/instance_v3/train/manifest_train.jsonl --selection-manifest /data/instance_v3/test/manifest_test.jsonl --run-dir /data/runs/b_source100 --device cuda --resume /data/runs/b_source100/last.pt
```

`--stop-after`保留原8750步预算身份；提前中止不生成“预算完成”结论。正式运行要求100个train场景各256帧，以及90个test场景各配置所需的固定8帧；完整覆盖这套selection不属于partial运行。真正缺少所需scene/frame的接线任务才显式使用`--allow-partial-manifests`并记录partial身份；不要同时修改训练步数来伪装恢复同一预算。

训练结束后的全量报告使用已经固定的源best和完整test清单，命令见[后处理](POSTPROCESSING.md)。`evaluate_instance.py`直接评价完整manifest，不继承训练的子集view，也不改写`best.pt`。报告中`evaluation_scope=complete_manifest`描述本次评价覆盖，`source_pretrain_selection_scope=fixed_test_subset`描述源checkpoint的选模来源；两个字段用途不同。全量报告包含曾用于选模的帧，继续保留`test_selected`和`independent_test_estimate=false`。

微调使用源train作回放，目标train动作作BCE，目标399 val作selection；目标87 test不传给训练器：

```bash
python scripts/train_instance.py --config configs/instance_b.yaml --stage finetune --source-manifest /data/instance_v3/train/manifest_train.jsonl --target-manifest /data/attempts/train.jsonl --selection-manifest /data/attempts/val.jsonl --init-checkpoint /data/runs/b_source100/best.pt --run-dir /data/runs/b_target --device cuda
```

目标manifest必须具备真实RGB-D/instance-mask及原规划p/n；旧pointcloud-only清单需补原资产。finetune的 `best.pt` 由399 val选择，身份为validation-selected，不抹除其源预训练test选模来源。恢复finetune时用 `--resume /data/runs/b_target/last.pt` 替换 `--init-checkpoint`；二者互斥。

运行目录保存配置/数据/模型身份、事件记录和best/last完整状态；checkpoint包含优化器、抽样进度、增强RNG、全局RNG、成功步与有效监督计数、参数组、选模策略，以及固定selection的帧定义和有序实例身份。被评价query的源test标签缺失、非法或与精确姿态不一致时明确报错，不能默认为0后继续选best。当前代码实现不等同于已得到真实checkpoint：真实完整资产、正式训练及GPU运行仍未执行。

## 8. 公开比较暂缓

`public_evaluation.status=deferred`、`radius_only_protocol=null`。本轮不实现或运行“官方评价器只改半径”，不再创建单独public20 final-refit或指定last-only公开权重，也不把公开R10等价检查当作开训门槛。论文数值继续保留，本方法公开结果待测。

未来恢复公开对比须先冻结评价/测试输入权限，逐项核对测试集和指标，并披露source test选模。当前实例test-selected结果不能填入论文全景AP列，更不能标成与对手仅吸盘尺寸不同的独立测试结果。
