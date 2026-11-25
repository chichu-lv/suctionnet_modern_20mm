# 训练评价与复现执行说明

> **2026-11-19：H20 从头训练方案**已新增独立配置 `configs/instance_a_h20.yaml` / `configs/instance_b_h20.yaml`（各在对应分支），采用真实微批 32、140000 个有效实例、4375 次更新、BF16 密集网络与持久 spawn 预取。启动、压测、精度边界与样本预算见 [H20_FROM_SCRATCH.md](H20_FROM_SCRATCH.md)。本页原 `instance_a.yaml` / `instance_b.yaml` 的单实例配方仍用于既有运行；新配置不直接续训旧检查点。

本页固定 v3 设计配方（物理质量仍为v2）、数据顺序、损失归约、选模、输出和实现验收。A/B设计见[总览](README.md)，坐标和标签见[数据合同](DATA_AND_LABELS.md)。初始配方只保留一套默认，避免把未实施的超参数搜索写成必要实验矩阵。

**当前执行政策：官方100个训练场景直接预训练，90个source test场景各固定8帧选 `best.pt`，标记 `test_selected`、`fixed_test_subset`；同时保留 `last.pt`。** 默认每875步评价一次，训练后单独对固定best运行完整test报告，不重新选模。取消新路线的90/10开发→refit流程。test参与选择后的同集成绩不作为独立测试估计。目标域1676/399/87划分不变。[公开仅半径评价](PUBLIC_METHOD_COMPARISON.md)设为待定，既不实现也不作为训练前置。实际运行命令见[IMPLEMENTATION.md](IMPLEMENTATION.md)。

## 1 当前可执行范围

| 项目 | 本次状态 | 执行条件 |
|---|---|---|
| 几何/schema参考工具 | 已实现、合成验证通过 | Python与NumPy；不需要GPU、数据集或模型 |
| 既有E5训练/评价入口 | 基线仓库已实现 | 按原README准备真实数据与匹配checkpoint |
| A/B实例数据builder、训练器、推理/评价 | 已实现，CPU自动测试及B合成官方格式链路验证 | 新 `scripts/*instance*.py`；旧E5入口不接收新合同 |
| 新S/W/self/INST标签缓存 | `InstanceTeacher` 已实现并核对合成资产 | 完整真实数据需要本地RGB-D、dense模型和冻结R20标签资产 |
| A/B模型效果、GPU速度、TensorRT导出 | 完整训练/性能/导出仍待执行 | B实际Torch CPU链路已测；A CPU测试注入backbone，真实SpConv仍需GPU验收 |

`reference/route_a.json` 和 `route_b.json` 是设计快照，`artifact_kind=design_spec`。它们不是当前 `scripts/train_pretrain.py` 或 `finetune.py` 的兼容配置，不能仅改后缀交给旧入口。

## 2 立即复现几何参考

在包含本文档的仓库分支根目录执行。参考工具实际验证环境为 Python 3.12.14、NumPy 2.3.5；生产GPU环境单独安装，避免为了参考脚本升级现有训练环境。

```bash
python3.12 -m venv .venv-instance-reference
. .venv-instance-reference/bin/activate
python -m pip install numpy==2.3.5

python docs/instance_routes/reference/geometry_contract.py selftest
python docs/instance_routes/reference/revision_contract.py selftest
python docs/instance_routes/reference/sampling_contract.py selftest
python docs/instance_routes/reference/contact_boundary_contract.py self-test
python docs/instance_routes/reference/sampling_contract.py demo \
  --output-dir /tmp/suction-instance-v3-sampling-demo
python docs/instance_routes/reference/revision_contract.py demo \
  --output-dir /tmp/suction-instance-v2-demo
python docs/instance_routes/reference/revision_contract.py check-labels \
  /tmp/suction-instance-v2-demo/synthetic_instance_labels.npz
```

使用新的空输出目录，工具不会覆盖已存在的demo文件。旧参考的29项检查仍验证投反影、姿态、旧SW文件和切平面栅格数学；**旧栅格已不属于B默认网络路径**。v2参考26项检查保留，覆盖旧基反例、固定坐标图、理想投影环、旧organized法向/首轮采样与物理v2标签。默认v3法向和回填必须调用 `sampling_contract.py`（26项检查）；A谐波、B接触门控/缺测摘要和逐角归一化调用 `contact_boundary_contract.py`（48项检查）。四组共129项合成检查；旧法向/一格一点函数不再是生产默认。两份v3模块使用float64参考算术，不代表生产CNN已经实现。

v3采样demo生成固定噪声平面与细长掩码的NPZ、JSON；64/128前缀取自同一256池。v2 demo生成 `[9,72,2]` 投影坐标；半径0可只读取一次，生产实现每query有577个不同采样槽位。它不执行神经网络或读取真实表面。标签示例3行，2行有效、1行masked NaN，其中一个有效点SW=0.72但存在self而Q_inst=0。

已有清单可继续运行 `python docs/instance_routes/reference/geometry_contract.py check-manifest <manifest.jsonl>`。manifest参考检查核对声明的sample_id/group_id/capture_uid/split关系，不读取文件内容，也不证明无相邻帧泄漏。`revision_contract.py check-labels`只校验其参考demo的顶层schema NPZ；生产metadata_json版标签应由 `load_labels` / `InstanceDataset`校验，旧SW检查也不能验证self语义。当次完整结果见[VALIDATION.json](reference/VALIDATION.json)；其中旧源码校验和仅保留为历史档案，不参与当前加载或验证。

## 3 生产环境和版本固定

当前仓库[运行指南](../MODEL_USAGE.md)记录的既有GPU组合是 Python3.10、PyTorch2.5.1+cu124、spconv-cu120 2.3.6；A优先沿用。B的学习网络不要求spconv，但若运行同一个物理teacher，仍需其数据与科学计算依赖。首次实现禁止把所有依赖替换成“latest”。

既有安装命令确实存在：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[model,gpu]'
python scripts/setup_backbone.py --output external/graspness_modern
python -m pip install pytest
```

B已在 `models/instance_route_b.py` 自行实现ResNet18/GN8，使用原生Torch，不需要torchvision，也不自动下载ImageNet权重。A/B首版均从新的架构初始化，RGB输入不通过静默填零/改键“恢复”旧8通道E5模型。

每个真实运行保存 `environment.txt`、完整配置、代码commit（出处记录）、数据/标签清单路径与样本ID、显式normal/sampler/physics/运行合同、随机种子、GPU型号、CUDA/cuDNN/spconv版本、CPU线程数及实际microbatch。数据、标签与checkpoint流程不计算SHA-256或替代内容校验和；旧校验和metadata直接忽略。固定外部骨干原有的Git blob SHA-1源码锁定机制保持不变。正常resume核对可用的配置/模型/运行/采样定义并恢复状态，路径和文件大小不能证明内容没有改变。[checkpoint政策](../CHECKPOINT_POLICY.md)

## 4 预训练损失与输出含义

q_inst、Seal、Wrench分别经sigmoid映射到[0,1]。v2默认 `Q_inst20=S20*W20*(1-C_self)`，保留target-self并排除外部碰撞；纯SW另存诊断，不能使用旧质量身份。新增不确定性头输出 `ell=log b`，初始化 `log(0.1)`，前向限定到[-6,1]；b是Laplace残差尺度：

$$
L_q=e^{-\ell}|\hat q-Q_{\mathrm{inst20}}|+\ell,\qquad
\mathrm{Var}=2e^{2\ell}.
$$

忽略常数log2，源域损失为
$$
L_{\mathrm{pre}}=L_q+
0.5\,\mathrm{SmoothL1}_{\beta=0.05}(\hat s,S_{20})+
0.25\,\mathrm{SmoothL1}_{\beta=0.05}(\hat w,W_{20}).
$$

不能把ell称为log方差，也不能把 `exp(-c)*SmoothL1+c` 不加定义地称为Gaussian NLL。若未来改用Gaussian，必须另设loss/config身份并改成 `0.5*(exp(-c)*(q-Q)^2+c)`。单次前向ell描述源域残差尺度，不等于MC dropout得到的模型不确定性，也不自动提供校准的抓取成功率。[Kendall与Gal](https://arxiv.org/abs/1703.04977)

每一项先只取本项valid query，再对单实例求均值，最后对batch中有监督实例求均值。不能用NaN乘零实现mask；无有效query的项返回显式zero/no-supervision并计数。源query质量为零是有效负例，不能被valid过滤。v2首版使用共同label_valid，要求S/W和两个self分量均已求值；分项缺失仅保留诊断，不伪造完整负例。

不确定性首版用于源回归损失与诊断，不额外加入 `q-lambda*sigma` 排名，也不增加多次随机推理。

## 5 唯一首轮预训练配方

| 参数 | 两路线首版默认 |
|---|---|
| 样本单位 | 一个实例在一个相机视角的观测；训练/回放取共同候选序列前128个；64为另命名可选配置，验证/测试/推理最多256 |
| sampler | 先均匀选100个训练scene，再均匀frame，再均匀该frame中训练有效实例；seed=42 |
| 预算 | 140000个参与成功更新的实例暴露 |
| 每次有效batch | 16实例 |
| 默认执行 | 单GPU，microbatch=1，累积16；仅为内存起点 |
| 成功optimizer更新 | 8750 |
| optimizer | AdamW，betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01 |
| 学习率 | peak=3e-4，warmup438次成功更新，之后cosine至1e-5 |
| 首版训练精度 | 网络FP32；新几何正确性基线从原始深度/内参以float64构造XYZ/法向/canonical query/理想环并判边界；冻结teacher保留原算术 |
| source test选模节点 | `evaluate_every: 875`，共10次；`selection_frames_per_scene: 8`，同一90×8帧子集 |
| 随机性 | Python/NumPy/Torch/CUDA和数据loader种子均固定并保存RNG |
| 源域选模 | test上的 `InstTop1_INST20@0.4` 最大；再按selected-Q宏平均最大；仍并列则更早步；保存 `test_selected` 身份 |
| 数据增强 | 每实例一致RGB jitter；灰度化概率0.2；首版不缩放米制几何 |
| checkpoint | 每875成功步保存last.pt；best.pt按上述10次固定子集结果更新，记录完整模型/优化器/采样器/RNG及定义 |
| 评价明细 | `evaluation_records=best_and_final`；逐实例流式压缩为jsonl.gz，仅当次更新best或最终步保留 |

RGB jitter固定 brightness/contrast/saturation=0.2、hue=0.05；RGB归一化固定 mean=(0.485,0.456,0.406)、std=(0.229,0.224,0.225)，invalid位置在归一化后再置零。A的raw支持与backbone点、B的全局与原分辨率纹理ROI共享同一个实例级光度变换。颜色增强不是对真实纹理特征有效性的先验保证。

默认FP32先确定可重复的训练协议。后续如选择AMP，需保存实际dtype和新执行配置；只在optimizer成功时增加有效预算，AMP skip产生的已读样本另计 `consumed_instance_draws`，不能隐瞒额外曝光或冒充成功步。microbatch×累积必须为16，但改变microbatch可能影响BatchNorm统计，不自动视作逐位等价。

140000 的预算仅计参与有效源监督且所在 optimizer 更新成功的实例暴露。没有任何有效源 query 的实例可以保留清单身份，但须记为 `has_supervision=false`，不充当有效 batch 的受监督实例，也不计入该预算；读取、过滤及跳过数量另存。累积单元按 16 个有监督实例组织，不能把无监督零损失或重复计数用于凑满成功更新。

A/B支持点数的训练与评价门槛均来自 `model.min_support_points`，默认100；B模型内部按实例内有限正深度原始像素计数并拒绝小碎片，直接评分和目标动作评价也受该门控约束。`has_supervision` 只表示存在有效teacher标签，不替代可配置支持门槛。builder另保存 `--min-support-points` 和当时 `trainable/status` 作为准备诊断，修改模型阈值时不能继续硬编码100。目标动作5mm中心支持统一使用 `BOUNDARY_ROUNDOFF_M=1e-12 m` 的闭边界。

选择帧采用8位bit-reversal固定前缀排序，8帧为 `0,32,64,96,128,160,192,224`；16帧包含8帧集合，256恢复全量。每次覆盖同一90场景的约定帧，所选帧全部可见实例与每实例完整候选池保持。子集排序有抽样波动，不保证选出全test最优checkpoint。旧配置缺少 `selection_frames_per_scene` 时仍为256。正式选择允许只构建指定帧，但要求90场景所选帧覆盖齐全；GT-empty声明保留，无输出实例留在分母，未选帧不加载。

`evaluation_steps` 与 `evaluate_every` 二选一；最终预算步使用同一子集必评，提前 `--stop-after` 不新增评价。缩短steps进行接线测试时同步调整显式节点。`evaluation_stepXXXXXX.json` 保存指标、选择键、节点/范围、`timing` 和 `record_details`；耗时覆盖选择集前向、指标和明细处理，不含之后的checkpoint保存。明细流式写入 `.records.jsonl.gz`，`best_and_final` 保留当次best或最终步；`all` 保留所有节点，`none` 不写明细。

完整test报告由训练后的独立 `evaluate_instance.py` 读取90×256帧清单生成，不重新比较或替换best。报告区分 `evaluation_scope=complete_manifest` 和源 `selection_scope=fixed_test_subset`；checkpoint及微调来源保存选择配置、精确有序实例ID，普通resume拒绝子集变化。每轮8帧/scene为全量帧预算的1/32，10轮共7200帧；这不是实测墙钟加速比。

旧E5的 `2880×48=138240` 是全场景有效观测，不是138240个单物体。新预算取140000是明确的新工程上限，不承诺为最优样本量。若有M个可训练实例观测，140000/M仅是曝光次数近似，不是shuffle-without-replacement意义的epoch；B也不承诺15–20epoch或数小时收敛。

所有首次训练使用相同的数据清单与验证query；A/B几何及网络不同，不要求其数值逐位一致。PyTorch官方指出部分CUDA操作存在非确定性；若引入对 `grid_sample` 的反向，需要记录此事实，不能仅设置seed就宣称跨设备逐位复现。[官方说明](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.grid_sample.html)

## 6 目标域1000步微调

每次成功更新抽2个target实例观测和2个source实例观测，使用两个域各自的均值：

$$
L_{\mathrm{ft}}=\frac{L_{t1}+L_{t2}}{2}
+0.5\frac{L_{s1}+L_{s2}}{2}.
$$

若逐条反向累积，两个target各权重0.5，两个source各权重0.25；之后不再整体除4。一个观测有多次历史动作，先取其有效动作平均，避免多动作观测因记录数量而额外加权。target按既有train的观测均匀采样，不做默认成功/失败1:1重采样；group用于划分约束与分组报告。

每条target动作：`logit=score_at_pose(p_plan,n_plan).task_logit`，`BCEWithLogits(logit,outcome)`。使用logit版本的BCE，不能先sigmoid后又传入BCEWithLogits。Seal、Wrench真值不由产线成败反推。source回放读取A/B共同的物理v2标签缓存（由v3法向/采样构造的新精确p/n身份生成）、默认128 query和同一source loss。

| 成功更新 | 可训练参数 | 学习率 |
|---|---|---|
| 1–200 | q/Seal/Wrench/ell/task各输出头 | heads 3e-4 |
| 201–1000 | 上述头；A的decoder/接触模块，B的Fq投影/共享纹理stem/环向编码；两者query融合与对象池化投影 | heads 1e-4，其余1e-5 |
| 全程 | encoder冻结；BatchNorm running stats与affine冻结 | — |

阶段切换保留已训练head的Adam moments；只新增或解冻既定参数组，不重建整个optimizer后丢失历史。B v3全局encoder与共享纹理stem采用GroupNorm(groups=8,eps=1e-5)；环向模块按 `[Q,72,16]` 的最后16通道做LayerNorm(16,eps=1e-5)，无效角在卷积前和归一化/激活后按原mask清零。从新架构初始化；没有可静默复用的旧BN权重。GroupNorm/LayerNorm不是BatchNorm，affine参数按所在模块是否解冻处理。参数组名称与完整成员列表写入checkpoint；旧E5的组名不能不经检查直接套A/B。

### 6.1 BN冻结是已有要求，执行时必须核验

原v1已写明微调期冻结BN，本次不将其描述为原规范遗漏。对A及任何保留BN的实现，`requires_grad=False`不会冻结running stats；每次`model.train()`后重新施加：

```python
import torch.nn as nn
model.train()
for module in model.modules():
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        if not module.track_running_stats:
            raise ValueError("Frozen running-stat policy requires tracked BatchNorm")
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
```

验收比较微调前后running_mean、running_var、num_batches_tracked及affine均不变。梯度累积16次不等于BN见到了batch16；它每次前向仍见microbatch大小。[PyTorch BatchNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)、[Group Normalization原论文](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.pdf)。

`evaluate_instance_source` 与 `evaluate_instance_attempts` 统一使用 `evaluation_mode`：进入前保存每个子模块的 `.training`，退出或抛错后按原值直接恢复。恢复时不调用根模块 `model.train(previous)`，也不依赖“下一训练步再次冻结BN”；返回调用方时混合train/eval状态已经保持正确。独立回归覆盖两个入口的正常及异常路径。

### 6.2 可选小幅姿态一致性，默认关闭

单次BCE学习 `P(y=1|observation,planned_command)`，并不宣称规划点是已测执行点。当前运行配置不实现该可选正则，首版只执行精确历史姿态BCE。未来实现并单独验证时先保留原动作的一次BCE，额外对每动作最多2个合法微扰计算弱一致性项：

$$
L_t=\mathrm{BCEWithLogits}(\ell_0,y)+\lambda\frac1J\sum_j
[\sigma(\ell_j)-\operatorname{stopgrad}(\sigma(\ell_0))]^2.
$$

试验初值 `lambda=0.1`，位置在半径2mm的局部切平面圆盘内均匀采样，轴在原有向轴周围2°球冠内按立体角均匀采样；J最多2，seed随checkpoint保存。它们是未验证正则超参数，不是已测执行误差，也不是新默认训练配方。

对每个扰动保留原轴符号，并按同实例、原观测连通表面、中心支持和深度断层规则核验；不通过则丢弃该扰动，J=0时一致性项为0。不将原结果复制成J条独立BCE，不扩大动作计数。不把源域S/W/self标签复制给改变后的姿态；源域仍查询固定缓存。

启用与强度只能用固定399条开发动作及支持率判断，87条test保持封存。该正则在薄接缝处也可能抹平真实差别，未经验证不默认启用。推理继续输出精确规划/候选姿态的单次分数。

若未来改为执行误差概率模型，应明确定义 `p_bar=sum(w_j*sigmoid(logit_j))` 并计算一次 `BCE(p_bar,y)`，推理同样积分或另行蒸馏；`sigmoid(mean(logit))`不是同一概率。误差核优先来自测量，不能从`actual_execution_verified=false`推得2mm/2°分布。

### 6.3 固定开发与测试

每100次成功更新在固定399条val动作评价一次；以AUROC最大选best，再按BCE低、再按更早步打破平局。始终报告总399、有效条数和跳过原因；不能每次选择不同的有效子集而比较分数。87条test在权重和排序规则冻结后使用一次，不用于挑alpha、normal半径或学习率。若某个分组只有单类，AUROC应标未定义而不是补零。

原记录的 `planned_pipeline_contact` 及 `actual_execution_verified=false` 随预测保存。几何无关联记录不当失败；报告缺失覆盖，并在相同有效动作交集上比较A/B。既有验证AUROC 0.6215→0.7121只属于旧E5，不填写到新路线结果行。

## 7 排序与统一姿态输出

源域未训练task头时按q_inst排序；微调后默认按sigmoid(task_logit)排序，alpha=0。保留用户提出的可选模式：

$$
s_{\mathrm{rank}}=P_{\mathrm{task}}\max(q_{\mathrm{inst}},10^{-6})^\alpha,\qquad
\alpha=0.5.
$$

该乘积是排序启发式，不能作为联合成功概率计算BCE/Brier。只有预先选定的开发验证对照可以改变alpha；历史动作AUROC改善也不能证明新候选策略最优。始终单独输出task概率、q和ranking score。没有task训练身份的checkpoint不能使用随机初始化task分数。

候选NMS在真实3D中心和有向轴上执行，默认中心距<10 mm且夹角<15°视为重复，按score降序、query_id稳定破同分。每实例保留Top5供下游选择，Top1为默认推荐。支持不足时返回 `no_pose` 及原因，不强行对全invalid热力图argmax。NMS与支持过滤均计入候选覆盖和耗时。

最终输出中心就是实际被评分的query中心；不再用5×5深度中位数二次修改它。若需要深度稳健修正，必须在生成query时完成；或修正后重建局部几何并重新评分。

相机系外向轴为n，工具+Z定义为接近方向时 `z_tool=-n`。选与z最不平行的固定相机基向量b，同值按x/y/z顺序，令 `x=normalize(b×z)`、`y=z×x`，构造右手 `R=[x,y,z]`；这与参考程序的固定roll完全一致。有固定工具roll偏好时用明确的切向参考代替并记录。点与轴本身没有估计roll，默认补全只为确定一个合法SE(3)，具有参考轴切换不连续性；它不再进入学习特征。轨迹执行要由固定工具roll偏好或上一姿态连续运输处理，不把静态补全等同连续运动。

$$
T_{\mathrm{base,TCP}}=
T_{\mathrm{base,cam}}\,
T_{\mathrm{cam,contact}}\,
T_{\mathrm{contact,TCP}}.
$$

contact→TCP来自工具标定，不能将唇口中心直接当法兰原点。参考工具只生成相机系contact姿态，不执行机器人动作。

当前 `infer_instance` 返回每实例字典：`schema="instance_suction_prediction_v2"`、`status/reason`、`poses[]`、ranking、合同/输入身份、NMS/过滤计数及计时。每个 `poses[i]` 包含原始评分p/n、`T_cam_contact`、`pose_id`、q/Seal/Wrench/ell、task logit/概率、ranking score和 `support` 诊断；没有顶层扁平query，也没有统一的顶层 `coverage_lip` 字段。

下面是从单实例结果中截取的字段示意，省略身份与计时等字段，数值不代表实测：

```json
{
  "schema": "instance_suction_prediction_v2",
  "instance_uid": "example_instance",
  "status": "ok",
  "reason": "ok",
  "stage": "pretrain",
  "ranking": "quality",
  "ranking_alpha": 0.0,
  "poses": [{
    "query_id": 42,
    "source_point_id": 12345,
    "point_cam_m": [0.12, -0.03, 0.70],
    "normal_out_cam": [0.0, 0.0, -1.0],
    "tool_z_cam": [0.0, 0.0, 1.0],
    "q_inst": 0.62,
    "seal": 0.81,
    "wrench": 0.77,
    "ell": -2.3,
    "task_probability": null,
    "task_logit": null,
    "ranking_score": 0.62,
    "pose_support_valid": true
  }]
}
```

CLI还添加 `schema="instance_suction_inference_v1"` 的报告外层，包含checkpoint及源初始化来源、选择政策、输入说明和逐实例结果。source checkpoint的task头未训练，输出概率/logit为null；微调checkpoint才输出任务概率。`support`是观测几何诊断，不是碰撞认证或密封成功概率。完整字段和真实序列化由 `suction20/instance_inference.py` 与 `scripts/infer_instance.py` 定义。

## 8 最少且足够的评价集合

| 层面 | 主指标 | 必须保留的解释字段 |
|---|---|---|
| 单实例含目标自碰质量 | `InstTop1_INST20@0.4` | 原始实例总数、无输出率、所选Q_inst均值；另报SW及self比例 |
| 候选生成/选择 | `Coverage_INST20@256` 与有好候选时Top1选中率 | eligible像素、首轮/回填数、回填轮次、实际K、支持过滤前后、NMS前后；不以无输出率替代质量覆盖 |
| 公共方法比较 | [公开方案待定](PUBLIC_METHOD_COMPARISON.md)，保留论文六指标参考 | 不把本轮test选模的native INST值直接填进论文AP |
| 历史动作成败 | AUROC、AP、BCE、Brier | 总动作数、有效数、正负数、逐动作分数；BCE/Brier只针对task概率 |
| 部署成本 | model-only及instance端到端p50/p95 | 硬件、dtype、原始点数、候选数、拷贝、预处理和NMS范围 |

INST主指标固定为：每frame内对可见GT实例平均，再对同scene的有效frame平均，再对scene等权平均。低于当前路线支持门槛（A/B均读取min_support_points，默认100）、无有效法向或无输出实例按不命中计入；没有任何GT实例的frame另计并不进入平均。另报微平均，避免不同聚合权重静默改变排名。

**两分支首次实现统一细化次级选模口径：** “selected-Q 宏平均（no_pose=0）”对每个可见 GT 实例取被选姿态的有效 `Q_inst`，无输出实例取 0，再采用与主指标相同的 frame→scene→scene 三层等权聚合；空 GT 帧仍另计并排除。另报“仅有输出时的 Q 均值”为所有有效输出姿态的微平均，并附输出数量与无输出率，不用它替换次级选模统计。该细化只明确统计分母，不改变物理质量、标签 schema 或原有选模优先级。

Teacher 未知不等于质量 0。对所选姿态或 Coverage 等指标所需的任何 query，若物理标签未知、非法或未完成求值，则相应评价标记为不完整并修复，不能按低质量/未命中处理，也不能删去实例以缩小分母。无输出实例没有所选姿态，不要求为它虚构物理标签。

候选覆盖定义为一个实例的固定256候选池中是否至少有一项 `Q_inst>=0.4`；“有好候选时选中率”分母仅为存在好候选的实例。覆盖与选择率以同一池统计，不把NMS后池与原始池混算。

比较原E5与新路线时，应对其实际输出姿态按同一个v2 instance teacher重新评价，或同时列出各自原协议；不能用旧13.12%的含碰撞开发指标当新INST主指标基线。首轮不要求重训第三方模型。

延迟每物体batch1，先50次预热，再至少200个固定验证实例，输出p50/p95与峰值显存；记录K实际有效数量。GPU模型时间用CUDA events并同步，端到端用墙钟同步边界。A/B必须计入3×3法向平滑、±3差分与路径门控、候选格内回填；A另计raw lip谐波，B另计共享纹理编码、投影门控/摘要与环向查询和数据拷贝；整帧多个实例另报真实汇总。上游相机和实例分割是否包含在计时内明确列出，不混称完整机器人FPS。15–30 ms与2–5 ms均不能在实现前标为测得结果。

## 9 既有基线的真实命令

当前基线CPU测试入口为：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[model]'
python -m pip install pytest
python -m pytest -q
```

以下微调命令的参数来自真实 `scripts/finetune.py`；它仅适用于匹配的旧E5/gt_B checkpoint，不能训练A/B。先按既有指南准备 `configs/local_attempt_success.yaml`，并把变量设为已有真实资产路径：

```bash
python scripts/finetune.py \
  --config configs/local_attempt_success.yaml \
  --run-dir "$FINETUNE_RUN" \
  --target-manifest "$TARGET_MANIFEST" \
  --init-checkpoint "$PRETRAIN_BEST" \
  --source-root "$SOURCE_RELEASE" \
  --label-root "$LABEL_RELEASE" \
  --backbone-root external/graspness_modern \
  --device cuda
```

`--init-checkpoint` 是微调初始化；`--resume` 是同身份恢复，二者互斥。本次文档编制未执行上述训练命令，也不需要为读取已有结果重新训练E5。

## 10 已实现入口和验收边界

| 模块 | 实际文件 | 已实现行为 |
|---|---|---|
| 数据/标签 | `scripts/build_instance_dataset.py`、`scripts/merge_instance_manifests.py`、`suction20/instance_data.py`、`instance_teacher.py`、`instance_geometry.py` | 官方RGB-D读取、v3法向/回填、同姿态S/W/self、每进程独立分片与安全合并、实际数组/schema检查 |
| A模型 | `models/instance_route_a.py`、`instance_route_a_geometry.py` | 12通道稀疏骨干、Fq、raw lip谐波、531融合、精确姿态评分 |
| B模型 | `models/instance_route_b.py` | 共享GN18/Fq/纹理、577投影槽位、96维环编码、330融合 |
| 训练 | `scripts/train_instance.py`、`suction20/instance_training.py` | source100、test选best、200+800微调、域均值损失、严格恢复 |
| 推理/评价 | `scripts/infer_instance.py`、`scripts/evaluate_instance.py`、`suction20/instance_inference.py` | 无标签推理、SE(3)/Top5/no_pose、native实例与历史动作评价 |
| 部署导出 | 尚未提供ONNX/TensorRT exporter | 后续须分别验证子图、组合数值和GPU性能 |

每条分支只包含本路线模型及运行YAML，共同模块保持相同版本。设计JSON继续保留 `artifact_kind=design_spec`，实际训练读取 `configs/instance_a.yaml` 或 `configs/instance_b.yaml`；parser拒绝把设计文件当运行配置。完整命令、数据字段和输出见[代码运行说明](IMPLEMENTATION.md)。

checkpoint保存完整 `config_definition`、显式 `runtime_contract`、route/stage、模型与优化器、成功预算、Python/NumPy/Torch/CUDA RNG、实际采样ID序列及RGB增强RNG，并保留manifest路径/数量和teacher的结构化定义。不生成或校验文件/数组校验和。source与selection的normal/sampler/precision/quality一致；双方physics都已知时直接比较，旧来源未知时写null而不伪称一致。

`--resume`核对当前运行配置、路线/阶段、环境类型、可用数据定义与模型状态形状，恢复全部调度和优化器状态。旧checkpoint的校验和字段忽略，但仍检查可得的配置、样本数量、预处理、teacher_spec和已知physics；不会仅因缺少新字段强制重算teacher。旧调度器只有不透明身份而缺少实际语义序列时，不能证明同长度清单没有重排；新版保存真实ID序列并检查顺序。`--init-checkpoint`只允许同路线、同模型配置及兼容物理/预处理的source checkpoint启动新微调阶段，记录路径/大小、原step和选模角色；不会将旧E5或跨路线权重改键载入。

B当前新运行使用 `feature_frame=two_fixed_z_charts_rodrigues_z05_v1`，并保存模型实际的 `min_support_points` 与 `identity.model_architecture`。历史配置未声明图版本时解释为旧 `two_fixed_z_charts_rodrigues_v2`，不会在读取权重时静默换图；其推理现在同样执行默认100点的模型门控。坐标图和支撑门槛均参与微调语义匹配，已有结构元数据也须一致；query_chunk仍只是执行参数。缺少新B结构身份的旧checkpoint不能精确resume，可按旧图推理，或显式指定旧图后初始化新的微调。此特征采样变更不改变p/n和teacher物理定义，不要求仅因此重新生成标签。

## 11 本次验证与后续记录

本次增加了实际模型、teacher、损失/冻结、暂停恢复、推理/评价测试；B用合成官方RGB-D/MAT/pose及完整对象资产执行真实CNN更新、完整256-query选择、保存/恢复和CLI预测。A的CPU模型测试使用显式注入backbone，真实SpConv/CUDA未在本环境执行。当前没有真实数据完整训练、GPU时延或机器人成功率结果。精确测试范围及CI记录见[验证说明](../MODEL_VALIDATION.md)。旧129项NumPy参考仍是独立数学/schema证据，不累加为新模型准确率。

完整数据执行需另记录法向有效率、候选覆盖、B径向/角槽拒绝与缺测比例、A主干关联距离和真实teacher构建耗时。未知teacher标签不能变成0；窄物体法向失效不能暗中退回旧算法；从FP32升至float64不能恢复原始精度。数据和权重保存在外部数据/运行目录，仓库只提交代码、配置、测试和可复核汇总。

