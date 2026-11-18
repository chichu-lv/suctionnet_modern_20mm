# B 训练方案：同姿态物理预训练与规划动作适配

本页给出已实现训练器 `scripts/train_instance.py` / `suction20/instance_training.py` 的B运行方式，使用 `configs/instance_b.yaml`。设计依据为[route_b.json](../reference/route_b.json)，该JSON继续作为设计合同，不直接传给训练CLI。源域直接100场景从头训练，完整source test选best；公开半径评价协议待定。本阶段不预设15–20个epoch或数小时收敛。

## 1. 进入训练的条件与环境

先完成[数据](DATA_PREPARATION.md)的真实资产/teacher核验、[预处理](PREPROCESSING.md)的坐标和缓存身份，以及[模型](MODEL_DESIGN.md)的 mask/维数/任意轴查询验收。固定同一源/目标清单与验证 query，记录被排除实例或动作及原因；不能每次验证改变有效子集以改善指标。

B的学习网络从新架构初始化，ResNet-18不加载外部预训练权重；实现直接使用PyTorch 2.5.1，不依赖torchvision或spconv，也不自动下载ImageNet权重。离线teacher仍有其科学计算和数据依赖。参考脚本的Python3.12.14/NumPy2.3.5环境与既有GPU环境分开，完整环境说明见[共同复现页](../TRAIN_EVAL_REPRODUCE.md)。

每次真实运行保存代码commit、环境/硬件、配置、源/目标manifest和标签hash、normal/sampler/输入/精度合同、RNG与参数组。旧8通道E5或A权重不能通过改key或补零静默恢复为B。

## 2. 源域物理损失和正确归约

q_inst、Seal、Wrench经过sigmoid；q真值为同一p/n的 `S20*W20*(1-C_self)`，任务头在源阶段不监督。ell输出Laplace残差的log b，初始化log(0.1)，前向限制到[-6,1]：

$$L_q=\exp(-\ell)|\hat q-Q_{inst20}|+\ell,$$

$$L_{pre}=L_q+0.5\,SmoothL1_{\beta=.05}(\hat S,S_{20})
+0.25\,SmoothL1_{\beta=.05}(\hat W,W_{20}).$$

忽略Laplace常数log2；方差为 `2*exp(2*ell)`，ell不能称为log-variance或直接成功置信度。q由独立质量头回归，不用预测Seal/Wrench乘积替代。

首版共同label_valid要求S/W和两个self分项都已知且合法。每项先选择有效query，对每实例取均值，再对batch中有监督实例取均值，不能按全batch所有query直接平均而使有效候选多的物体更重。无有效监督返回显式zero/no-supervision并计数；不能用NaN乘零实现mask。真实Q=0保留为负例。

**没有有效源监督的实例不计入140000个成功受监督实例暴露。** 所有读取、排除、无监督和失败更新另计，训练器以实际参与成功更新的监督实例为预算；不能把空loss或跳过的实例当作完成预算。

## 3. 直接100场景源预训练

训练集合固定scene 0–99全部100场景、每场景frame 0–255；模型从头初始化。选模集合固定source test scene 100–189、每场景全部256帧。test用于选best的决定明确进入运行身份，不作为独立泛化估计；不再执行90/10开发后final refit。

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
| 选模评价 | 成功步875、1750、…、8750，共10次；每次完整source test，不抽帧 |
| 保存 | test-selected best.pt，以及训练进度/预算终点last.pt；完整恢复状态 |

RGB jitter与灰度化遵循[预处理](PREPROCESSING.md)，同一实例的上下文/原生纹理使用同一光度参数。训练环phase由seed、样本身份、精确p/n字节的query身份和成功步确定抽取，与query顺序或chunk无关；仅改变观测特征采样，不改标签p/n。验证/推理phase=0。首版不缩放米制几何、不为query改轴后复制源标签。

旧E5的全场景曝光不能等价换算为单实例收敛点；140000是本方案固定工程预算。64-query只作为另命名对照；验证始终使用共同最多256池。后续AMP或microbatch变更须独立记录，成功更新与consumed_instance_draws分开统计；seed也不保证所有CUDA/grid_sample路径跨设备逐位确定。

### 源域选模与结果身份

源test只评价，不参与反向传播，但会用于checkpoint选择。`best.pt`必须标记 `selection_role=test_selected`、`independent_test_estimate=false`；由该best计算的相同test成绩不能作为独立盲测、也不能填写“公平官方测试”结果行。`last.pt`在8750成功更新完成时是固定预算终点，提前暂停的last只是进度checkpoint。

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

微扰需通过同实例/同观测组件/中心支持/断层规则，不通过丢弃，J=0正则为0；不将一次结果复制为多条BCE，不给改变p/n的源姿态复制旧S/W/self。该项默认disabled，当前训练CLI未实现该可选微扰正则；不得在运行配置中标为已启用。幅度不是实测噪声；后续实现和预注册后只能在目标399 val上决定选用，目标87 test不参与。[完整定义](../TRAIN_EVAL_REPRODUCE.md)

## 6. 恢复状态和训练准入验收

checkpoint至少包含：模型类/版本、13/5通道输入、330融合及环门控/LN/summary合同、normal/sampler与精确query hash、INST/self配置、归一化/padding、精度/ranking、权重/优化器/scaler、成功步/监督实例/已消费draw、参数组、sampler/loader/RNG、源目标清单和代码身份。resume只接受同身份；初始化与resume互斥，不恢复退役迁移白名单。

运行配置校验 `instance_runtime_v1`、route=b、source scene范围和 `public_evaluation.status=deferred`。CLI支持严格resume和finetune初始化，参数见第7节；不通过旧E5参数或替换文件名绕过模型/数据身份。

已有模型CPU合成前后向验证；真实资产具备后，只做能解除具体风险的短链路验收：真实实例teacher与输入姿态一致；少量更新有有效梯度；冻结模块/affine确实不变；阶段切换moments保留；恢复后下一批样本/query/phase/LR一致；相同位置不同轴改变条件；输出仍为被评分p/n。取得这些证据后执行固定正式预算，不把短测损失下降写成模型效果结论。

## 7. 可执行命令与运行产物

先按[数据准备](DATA_PREPARATION.md)生成完整train/test缓存。以下路径替换为真实路径，在仓库根目录执行；选择CPU可检查接线，正式GPU运行需已安装匹配环境。

```bash
python scripts/train_instance.py --config configs/instance_b.yaml --stage pretrain --source-manifest /data/instance_v3/train/manifest_train.jsonl --selection-manifest /data/instance_v3/test/manifest_test.jsonl --run-dir /data/runs/b_source100 --device cuda
```

首轮不传 `--init-checkpoint`，也不传目标域数据。`--device cpu` 使用相同模型和几何合同，不能由CPU短测推定GPU时延。完整运行默认8750成功更新；140000是实际有监督实例暴露预算，不是100场景全帧的epoch计数。

暂停和严格恢复使用同一运行目录、同一配置和数据身份：

```bash
python scripts/train_instance.py --config configs/instance_b.yaml --stage pretrain --source-manifest /data/instance_v3/train/manifest_train.jsonl --selection-manifest /data/instance_v3/test/manifest_test.jsonl --run-dir /data/runs/b_source100 --device cuda --stop-after 10
python scripts/train_instance.py --config configs/instance_b.yaml --stage pretrain --source-manifest /data/instance_v3/train/manifest_train.jsonl --selection-manifest /data/instance_v3/test/manifest_test.jsonl --run-dir /data/runs/b_source100 --device cuda --resume /data/runs/b_source100/last.pt
```

`--stop-after` 保留原8750步预算身份；提前中止不生成“预算完成”结论。若只使用显式scene/frame分片进行接线，可加入 `--allow-partial-manifests`，运行身份会记录partial；该标志不应进入正式完整测试选模运行。不要同时修改训练步数来伪装恢复同一预算。

微调使用源train作回放，目标train动作作BCE，目标399 val作selection；目标87 test不传给训练器：

```bash
python scripts/train_instance.py --config configs/instance_b.yaml --stage finetune --source-manifest /data/instance_v3/train/manifest_train.jsonl --target-manifest /data/attempts/train.jsonl --selection-manifest /data/attempts/val.jsonl --init-checkpoint /data/runs/b_source100/best.pt --run-dir /data/runs/b_target --device cuda
```

目标manifest必须具备真实RGB-D/instance-mask及原规划p/n；旧pointcloud-only清单需补原资产。finetune的 `best.pt` 由399 val选择，身份为validation-selected，不抹除其源预训练test选模来源。恢复finetune时用 `--resume /data/runs/b_target/last.pt` 替换 `--init-checkpoint`；二者互斥。

运行目录保存配置/数据/模型身份、事件记录和best/last完整状态；checkpoint包含优化器、抽样进度、增强RNG、全局RNG、成功步与有效监督计数、参数组、选模策略。完整源test标签缺失、非法或与精确query不一致时评价报错，不能默认为0后继续选best。当前代码实现不等同于已得到真实checkpoint：完整数据构建、正式训练及GPU运行仍未执行。

## 8. 公开比较暂缓

`public_evaluation.status=deferred`、`radius_only_protocol=null`。本轮不实现或运行“官方评价器只改半径”，不再创建单独public20 final-refit或指定last-only公开权重，也不把公开R10等价检查当作开训门槛。论文数值继续保留，本方法公开结果待测。

未来恢复公开对比须先冻结评价/测试输入权限，逐项核对测试集和指标，并披露source test选模。当前实例test-selected结果不能填入论文全景AP列，更不能标成与对手仅吸盘尺寸不同的独立测试结果。
