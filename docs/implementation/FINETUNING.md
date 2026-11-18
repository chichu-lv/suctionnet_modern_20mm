# 少量自有场景微调：实现规格

公开数据训练的目的是获得可迁移的预训练模型。目标域阶段默认复用选中的 `gt_B` checkpoint，用已有的少量自有场景训练 **1000 次 optimizer update**。本文件不要求重新制作公开物理标签、不新增整套目标域稠密标注工程，也不要求把六个公开数据消融都搬到目标域。源域 replay 同样沿用 [旧数据适配说明](LEGACY_DATA_ADJUSTMENT.md) 的只读入口。

**实现状态：** target Dataset、两种损失、分阶段微调入口和固定查询验证已实现；通过合成数据初始化/恢复检查。本次没有真实目标域训练结果。

## 1. 先按已有标签选模式

| 已有自有场景内容 | 支持模式 | 可训练内容 |
|---|---|---|
| 与源域物理定义一致的点级代理质量标签，可稠密也可仅部分点有效 | `dense_proxy` | 原 `q0/q_B` 头；实际存在的分量/候选轴标签可继续监督 |
| 已经执行过的吸取点、实际执行轴、成功/失败结果 | `attempt_success` | 新增 `task_success_head`；源域 replay 保持原物理头语义 |
| 只有未标注 RGB-D/分割结果 | 暂不进行监督微调 | 可运行预训练模型；不能把未标注点赋成失败或零质量 |

两种模式是不同监督接口，不能在同一个 `gt_B` 数组里混用物理代理分和真实 0/1 成败。若目前只有第二类数据，直接使用第二类；不要求为了迁就模型补齐 dense GT。

`dense_proxy` 的“同定义”包括半径、包络、Seal 计算、Wrench、碰撞以及源域冻结归一化。目标标签若已经保留 raw Seal，可只按源常量重新派生归一化值；不得对目标数据重新拟合 p1/p99。若只有来自另一套定义的最终质量分，不默认为相同监督。

## 2. 共用数据接口与分组划分

### 2.1 文件组织

实现 `suction20/target_dataset.py`，复用源 Dataset 的观测预处理和 `ModelInputs`，独立返回目标监督对象。canonical inputs 为 `xyz_cam_m/normal_obs/normal_obs_valid/instance_id_input/g_cam`，元信息使用 `meta`；dense targets 使用 `gt_B/normal_label/quality_valid/axis_valid` 等与源域一致的接口，attempt targets 保留独立对象。每行 JSONL 表示一帧带物理标签的观测，或一帧中的一次真实尝试。相同观测可以被多次尝试引用，Dataset 可在当前进程缓存其只读预处理结果。

manifest 中的相对路径以 manifest 所在目录解析；`sample_id` 在一个 manifest 内唯一。示例路径仅说明结构。

共同字段：

| 字段 | 类型 | 必需 | 说明 |
|---|---|---:|---|
| `schema_version` | string | 是 | 固定 `suction20.target.v1` |
| `sample_id` | string | 是 | 训练记录唯一标识 |
| `group_id` | string | 是 | 采集 session/同一物理摆放的分组 ID |
| `split` | string | 是 | `train`、`val`、`test` 之一 |
| `observation` | object | 是 | 下述 RGB-D 或点云入口 |
| `label_kind` | string | 是 | `dense_proxy` 或 `attempt_success` |
| `labels` / `attempt` | object | 按模式 | 分别见 §3、§4 |

`observation` 支持两种前处理入口，二选一：

```json
{"format":"pointcloud_npz","path":"observations/frame_0001.npz","g_cam":[0.0,0.0,1.0]}
```

点云 NPZ 的输入字段为 `xyz [N,3]`、`instance_id [N]`，可带 `source_indices`、`normals_raw`、`normals_raw_valid`。这里的实例 ID 是前置分割输入身份，不是物体类别。

```json
{"format":"rgbd","depth_path":"depth/frame_0001.png","instance_mask_path":"masks/frame_0001.png","intrinsics":[[615.0,0.0,319.5],[0.0,615.0,239.5],[0.0,0.0,1.0]],"depth_scale_m":0.001,"g_cam":[0.0,0.0,1.0]}
```

RGB-D 路径在线反投影并计算观测法向，不要求用户先生成另一份训练文件。`depth_scale_m` 表示深度整数值乘该系数得到米。RGB 不是当前几何模型的必需字段。

点云入口如果没有观测法向，应同时提供对应的原始深度入口信息，或由同一观测点云估计法向；不能借用标签轴代替观测轴。整个自有数据集保持同一种法向预处理配置。

### 2.2 按场景/session 划分

在开始微调前把已有数据的 `group_id` 分为 train/val/test；同一摆放的相邻帧、同一帧的所有 attempts 必须在同一集合。推荐按现有组数约 70%/15%/15% 划分，保证每个用于报告的集合有独立组，不按图像随机切分。

若现有场景仅够 train/val 两组，可做开发性对比并说明没有独立目标测试组，不为凑比例默认要求继续采集。场景少时报告每组结果及总样本量，不把数千相邻帧当成数千独立场景。

训练样本循环采样：先均匀选 `group_id`，再选该组中的观测/attempt，防止一段较长录像支配训练。验证和测试不做随机增强。目标 val 用于选阶段与 checkpoint；test 只在选择完成后评一次。

## 3. `dense_proxy`：继续学习同一物理目标

### 3.1 JSONL 示例与标签数组

```json
{"schema_version":"suction20.target.v1","sample_id":"s01_f001","group_id":"session_01/layout_01","split":"train","observation":{"format":"pointcloud_npz","path":"observations/s01_f001.npz","g_cam":[0.0,0.0,1.0]},"label_kind":"dense_proxy","labels":{"path":"labels/s01_f001.npz","source_label_config_path":"source_config_frozen.json","normal_target_semantics":"original_candidate_axis"}}
```

标签 NPZ 与该 observation 点序一一对应，或显式提供 `source_indices` 进行确定性对齐。允许只有部分点有标签。

| 字段 | 必需性 | 用途 |
|---|---|---|
| `gt_valid [N]` | 必需 | 质量标签是否存在；False 不表示负例 |
| `gt_B [N]` | 必需，或可由实际存在的分量派生 | 质量回归目标 |
| `instance_id_gt [N]` | 有真实标签实例身份时提供 | loss 的实例归一化；无独立 GT 实例时使用标签附带的实例映射并记录来源 |
| `gt_seal_raw/gt_seal_norm` | 有则使用 | 仅启用实际存在的 Seal 辅助监督 |
| `gt_wrench_eval` | 有则使用 | Wrench 辅助监督 |
| `collision_flag` | 有则使用 | collision-free 辅助监督 |
| `gt_normal [N,3]` | 有同定义候选轴标签时使用 | 法向监督；不是观测法向 |
| `axis_target_valid [N]` | 有轴标签时提供/按字段有效性派生 | 独立轴标签掩码，不从成功阈值推断 |

分量标签若只覆盖部分点，可分别提供 `seal_valid/wrench_valid/collision_valid`；没有单独掩码时，只在该分量数组实际存在、数值有限且 `gt_valid=True` 的点监督。整个字段缺失时该项 mask 全 False。标签轴进入 batch 后命名为 `normal_label`，与输入 `normal_obs` 分开。

`source_label_config_path` 指向这些标签实际使用的冻结物理定义，适配器与 source checkpoint 中的定义比较。若保留 raw 分量，归一化直接读取 checkpoint 的 `s_lo/s_hi`。不调用新数据生产器重新拟合统计，也不把 target min/max 替换进去。

### 3.2 目标损失

目标有完整分量时，使用 TRAINING 中相同权重的物理损失；只有 `gt_B` 时只启用 `L_q0+L_q`。没有的辅助标签及轴标签，其 mask 全 False，不填零伪造标签。

```python
quality_valid_t = label_exists & target.gt_valid & target_foreground_gt
axis_valid_t = axis_label_exists & target.axis_target_valid & inputs.normal_obs_valid

loss_target = L_q0_t + L_q_t
loss_target += 2.0 * masked_axis_loss(axis_valid_t)
loss_target += enabled_and_available_aux_losses()
```

目标域 dense 采样用 50% 当前模型提议与 50% 有效标注点均匀采样，不要求再生成 `gt_A` 供正负分池。没有标签的模型提议点只进入特征计算，相应 loss 被忽略。

输入、当前预测轴分组、输出 `q_B` 与源域同义。默认继续用 `q_B` 排序，不新增 task success 头。

## 4. `attempt_success`：监督真实执行姿态的任务成功头

### 4.1 最小真实尝试记录

```json
{"schema_version":"suction20.target.v1","sample_id":"s01_f001_a01","group_id":"session_01/layout_01","split":"train","observation":{"format":"rgbd","depth_path":"depth/s01_f001.png","instance_mask_path":"masks/s01_f001.png","intrinsics":[[615.0,0.0,319.5],[0.0,615.0,239.5],[0.0,0.0,1.0]],"depth_scale_m":0.001,"g_cam":[0.0,0.0,1.0]},"label_kind":"attempt_success","attempt":{"point_cam_m":[0.01,-0.02,0.45],"normal_out_cam":[0.0,0.0,-1.0],"instance_id":2,"source_index":153620,"success":1,"label_valid":true}}
```

| `attempt` 字段 | 类型/单位 | 语义 |
|---|---|---|
| `point_cam_m` | 3 个 float，m | 该观测时刻相机坐标中的实际执行接触中心 |
| `normal_out_cam` | 3 个 float，单位向量 | 对应真实执行的吸盘外向轴；不是重新估计的表面法向 |
| `instance_id` | 正整数 | 该次目标在输入实例 mask 中的 ID |
| `source_index` | 可选整数 | 原深度图展平像素索引，便于精确取得观测特征 |
| `success` | 整数 0/1 | 已有实际尝试结果 |
| `label_valid` | bool | 此结果是否可用于监督；False 时忽略该项 |

接近运动方向若指向物体，应在整理已有记录时转换成这里的外向轴，不能直接混用符号。每个数据批次固定成功定义，例如“完成吸取并搬运到指定位置”，保留它的文字定义；不将真空接通但随后掉落混写成相同成功条件。

### 4.2 必须对 `p_exec,n_exec` 打分

```python
encoded = model.encode_scene(inputs)
pose_out = model.score_at_pose(
    encoded,
    point_cam_m=attempt.point_cam_m,
    normal_out_cam=attempt.normal_out_cam,
    instance_id=attempt.instance_id,
    proposal_q0=None,
    pose_batch_index=attempt.batch_index,
)
```

该接口不能用网络当前预测的轴替换 `n_exec`，也不能用当前 Top-1 点替换 `p_exec`。否则成功标签与被评分动作不再对应。此处显式轴是**已执行动作的条件输入**，不作为法向头应学习的最佳轴。

collate将attempt展平为 `[A,3]` 的点/轴和 `[A]` 的实例/结果，并生成每条attempt的 `batch_index`；不同帧的同名实例ID不能混合。取得 anchor feature 时，优先使用 `(batch_index,source_index)` 找packed点；否则在同帧、同一输入实例中用 5 mm 内最近观测点取得特征。接触分组的中心和轴始终是实际 `p_exec,n_exec`，不因 feature 来自附近观测点就改变动作。若该条记录无法映射，`pose_feature_valid=False`，该条监督跳过并记数，不把映射失败变成抓取失败。

任务头输入与结构固定为：

```text
[pose_features(256), n_exec_cam(3), dot(n_exec_cam,g_cam)(1)]
                   -> Linear(260,128) -> LayerNorm(128) -> ReLU -> Linear(128,1)
```

显式轴附加项在未启用 contact grouping 的模型中仍保留，使任务头能区分执行方向。`task_success_head` 输出 logit，推理时取 sigmoid 得到任务成功排序分。它与原物理 `q_B` 是两个输出。

### 4.3 Loss mask 与梯度去向

```python
attempt_valid = (
    attempt.label_valid
    & pose_out.pose_feature_valid
    & finite_pose(attempt.point_cam_m, attempt.normal_out_cam)
)
loss_target = BCEWithLogitsLoss(
    task_success_logit[attempt_valid],
    attempt.success[attempt_valid].float(),
)
```

同一帧有多次尝试时，先在有效 attempts 内平均，再按目标 batch 的帧平均。第一版采用未加类别权重的 BCE，避免无必要地改变得分含义。

| 目标标签/损失 | Target branch 是否启用 |
|---|---:|
| `task_success` BCE | 是，仅真实尝试点 |
| `q0/q_B` 物理质量回归 | 否，二值结果不等于物理代理分 |
| 法向/轴损失 | 否，成功或失败不能推出最佳法向 |
| S/W/C 辅助损失 | 否，失败原因不能由一个二值结果拆出 |

目标 branch 的未尝试点全部没有成功标签；不能把周围点、低分点或未选择点记为 0。目标 branch 不通过 BCE 更新 axis head。当前 pose 特征不拼接 `proposal_q0`，它只作可选追踪信息，因此 q0 头也只接受 source 物理监督。阶段二可以更新供姿态打分使用的 decoder 特征，源域 replay 同时维持物理头与法向头。

### 4.4 任务头初始化与部署

从没有 `task_success_head` 的公开 checkpoint 开始时，仅该头随机初始化，bias 初始化为 0；backbone、fusion、grouping、q/axis/aux 参数完整继承。若是继续已有目标任务训练，`--resume` 必须载入原任务头及 optimizer 状态，不重新初始化；若用 `--init-checkpoint` 导入已有目标模型，任务头权重可继承，但 optimizer 仍重新创建。

部署时先按原 `q0` 生成候选并预测正常吸取轴，再对这些**将被执行的姿态**调用 `score_at_pose`，使用 `task_success_score` 排序。`q_B` 继续作为物理代理输出保留；默认不再乘一次 `q_B` 或硬设一个新质量阈值。

少量尝试训练出的 sigmoid 分数是任务成功评分，未经验证不宣称其数值已是准确概率。

## 5. 默认低负担训练：200 + 800 update

### 5.1 参数分组

模型实现需要显式提供不重叠的参数组；下面是语义，不依赖上游 spconv 模块的偶然变量名：

| 组 | 包含内容 | 微调处理 |
|---|---|---|
| `encoder` | 稀疏 U-Net 下采样与最深编码部分 | 全程冻结 |
| `decoder` | U-Net 上采样、512→256点投影、实例融合、区域点/池化投影及590→256→256 pose融合等全部特征网络 | 前 200 步冻结，后 800 步训练 |
| `heads` | `q0`、轴残差、最终 q、启用的 S/W/C，以及可选 task success 头 | 全程可训练，但各 branch 只用其有效 loss |
| BatchNorm 参数/统计 | 所有阶段中的 BN | 全程 `eval`；affine 参数也冻结 |

若启用了非参数池化，其本身无可更新参数，沿用上述 decoder 边界即可。encoder 冻结不意味着丢弃它的特征；仍需正常前向。包装模型应提供 `parameter_groups()`，确保每个可学习参数恰属encoder/decoder/heads一组；不能漏掉pose融合MLP而让其意外在前200步更新。此时attempt模式的step200确实使用冻结的姿态特征，只训练成功评分头和接受source监督的物理输出头。

### 5.2 学习率与 optimizer 状态

| 成功 optimizer step | 可训练组 | heads LR | decoder LR |
|---|---|---:|---:|
| `0..199` | heads | `3e-4` | 0 |
| `200..999` | decoder + heads | `1e-4` | `1e-5` |

AdamW、weight decay `1e-4`、betas `(0.9,0.999)`、eps `1e-8`；bias/归一化参数不做 weight decay。总预算固定为 1000 update，不另加长 warmup/cosine。

从公开模型进入微调时创建 **fresh optimizer、fresh AMP state、global_step=0**；不加载源 optimizer 的动量或源学习率进度。可以在新 optimizer 内从一开始登记 heads 与 decoder 两组，前 200 步将 decoder `requires_grad=False`；到 step 200 解冻并调整 LR，保持 heads 已积累的 optimizer 状态。encoder 与被冻结的 BN 参数不加入 optimizer。

每次 `model.train()` 后再次执行 `set_bn_eval_and_freeze_affine(model)`，防止外层模式切换恢复 BN 统计更新。microbatch 很小时不重新估计目标 BN 均值/方差；默认继承源域统计。

### 5.3 1:1 source replay

每次 optimizer update 使用 **2 个目标观测 + 2 个源观测**，有效 batch 为 4。单 GPU 默认 microbatch 1，依次累计四个前向/反向；目标数据少时允许循环采样已有样本。源数据直接复用已完成的旧数据与同一个只读兼容适配器，不制作新 replay 数据。

源 branch 使用该 checkpoint 对应的源物理损失；目标 branch 使用本文件选定模式的损失。两域先各自归一化，再组合：

\[
L_{FT}=L_{target}+0.5 L_{source}.
\]

不能把所有源点与目标 attempts 混在一个大平均里，否则密集源点会淹没少量真实尝试。

单帧 microbatch 的累计权重为：每个 target 帧 `0.5 × L_target_frame`，每个 source 帧 `0.25 × L_source_frame`。四帧加起来等于上式；**此后不能再统一除以 4**。如果实现改用域内批量前向，则每域先求均值，再按 1.0/0.5 加权。

source replay 不为任务成功头制造二值标签，target attempts 也不为物理头制造代理标签。两套 mask 在 `suction20/losses.py` 中分开维护。

### 5.4 伪代码

```python
while global_step < 1000:
    apply_phase(global_step)  # < 200: heads; else: decoder + heads
    model.train()
    set_bn_eval_and_freeze_affine(model)
    optimizer.zero_grad(set_to_none=True)
    target_frames = target_sampler.next_frames(2)
    source_frames = source_sampler.next_frames(2)
    for batch in target_frames:
        # attempt_success 在此只调用真实执行姿态评分和 BCE。
        scaler.scale(0.5 * target_loss(model, batch, mode)).backward()
    for batch in source_frames:
        scaler.scale(0.25 * source_loss(model, batch)).backward()
    updated = unscale_clip_and_amp_step(optimizer, scaler, max_norm=10.0)
    if updated:
        global_step += 1
    maybe_validate_and_save(global_step)
```

实际循环以成功 update 计数；AMP 跳过时不得跨越 200 步解冻边界或消耗学习率 phase。`--resume` 从保存的 phase/step 和 sampler 状态继续。

## 6. 目标数据增强与源标签继承

增强仍保持米制尺度，禁止 scale、单位球归一化。默认绕重力轴旋转和平面内平移；对 attempt 记录同时变换 `p_exec,n_exec`，对 `g_cam` 使用同一坐标变换。

目标域第一版关闭随机深度扰动和点删除，保留实际相机已有噪声，减少在稀少尝试点附近改变可见性的额外因素。源 replay 也采用相同轻量刚性增强，便于实现和比较。未标注 RGB-D 不产生自训练伪标签。

所有 target/source 分量读取 checkpoint 保存的 source label 定义。旧冻结 JSON 中存在新生产器 parser 不接受的字段时，只读适配器直接读取原 JSON 并提取所需字段；不通过新 parser 重新构造默认配置，不改变 `s_lo/s_hi`。

公开域 residual uncertainty 在微调改变特征与预测后默认关闭；即使保留其权重，也不能把它当作 target success uncertainty。默认 `uncertainty_lambda=0`。只有已有目标验证标签支持重新拟合与比较时再启用，不为此额外要求采集。

## 7. 验证与最少比较

### 7.1 固定验证频率

在 step 0（源模型）、step 200（heads-only 阶段结束）及之后每 100 次 update 验证一次；step 1000 必须验证。保留 step 200 checkpoint，最终从目标 val 选择 best，而不是默认最后一步更好。

默认只比较同一选定架构的两种状态：

1. `public_pretrained_only`：未使用自有监督的源模型。
2. `target_finetuned`：以上 1000-step 流程选择出的 best。

step 200 与后续 best 的对比可直接判断解冻 decoder 是否值得；无需再从头训练另一套架构。目标域不重跑六个源域配置。

### 7.2 `dense_proxy` 指标的边界

如果已有目标评价器可以对最终预测 `(p,n)` 重算同定义物理质量，主指标沿用 `instance_Top1_Q20_at_0_4`，并列时用 `AP_R20`。

若只有固定观测点/固定轴上的代理标签，就报告这些**已标注姿态**上的 masked MAE 与质量排序结果；不能把它们称为对任意预测姿态重算后的 AP-R20。轴标签存在时，验证 `score_at_pose(p_label,n_label)` 的质量；没有轴标签时，只将固定点质量误差作为开发诊断，最终吸取效果由已有真实评价决定。此时默认选模指标为固定监督集合上的 `masked_q_mae`（越低越好），并明确指标范围。

实现该指标时，固定使用验证集所有 `quality_valid & normal_obs_valid` 点作为查询中心，按每块512点调用 `score_at_pose`，不使用随checkpoint变化的高分seed集合；有标签轴就用该轴，没有时用预测轴并注明只是条件性诊断。按实例→帧→组平均MAE，同时记录有效查询点数。这样改善不会仅来自模型换了一批更容易的验证点。

没有的源式 evaluator 不作为开始小样本微调的阻碍，也不要求为它新建完整目标物体 mesh 管线。

### 7.3 `attempt_success` 指标的边界

零样本源模型与微调模型都在**相同的已执行姿态集合**评分：前者读取 `q_B(p_exec,n_exec)`，后者读取 `task_success_score(p_exec,n_exec)`。

- 两者都可报告 AUROC、PR-AUC，衡量与真实成功的排序关联；这里的 PR-AUC 与 AP-R20 不是一个指标。实现统一采用按 distinct score 分组的阶梯式 average precision：`sum((recall_j-recall_{j-1})*precision_j)`，不在不同实验中切换梯形积分定义。
- 目标 val 默认以 `attempt_AUROC` 选 checkpoint，并报告正负样本数与场景数。
- 不对零样本 `q_B` 报 Brier 或称其为已校准成功概率。
- 不把未执行候选设为失败，也不从尝试日志推断新策略选择点的真实抓取成功率。
- 某个小验证集合只有一种结果时，AUROC 未定义；如实报告，使用可计算的任务 loss 观察训练，不把缺失指标填成 0。

这些日志指标回答“对已有尝试的评分是否改善”，不是新策略实机成功率。如果后续进入实机验证，只需比较同一架构的 pretrained-only 与 fine-tuned，保持感知、动作与成功定义一致。可使用各 100 次正常混堆抓取的有界对比；这不是本次监督微调的前置采集要求。

## 8. Checkpoint 与运行示例

在 TRAINING checkpoint 基础上增加：

```python
{
    "stage": "target_finetune",
    "target_mode": "dense_proxy",          # 或 attempt_success
    "source_checkpoint_ref": source_ref,
    "source_label_config": original_frozen_config,
    "source_label_definition": original_physical_config,
    "quality_target": "gt_B",             # attempt 模式也保持原义
    "target_manifest_definition": actual_manifest_records,
    "target_split_groups": split_groups,
    "success_definition": success_definition_or_none,
    "task_success_head_present": mode == "attempt_success",
    "parameter_phase": "decoder_and_heads",
    "global_step": global_step,
    "bn_mode": "eval_frozen_affine",
    "best_target_metric": best_metric,
    "uncertainty_lambda": 0.0,
}
```

同时保存完整 model state、fresh optimizer 的当前 state、AMP、RNG、target/source sampler 状态及解析后的配置；从 source checkpoint 继承的 frozen config 不被目标字段覆盖。

以下是已实现的运行方式；数据与预算需按实际任务配置：

```bash
python scripts/finetune.py \
  --config docs/implementation/examples/finetune_dense_proxy.yaml \
  --init-checkpoint runs/public/selected/best.pt \
  --run-dir runs/target/dense_proxy

python scripts/finetune.py \
  --config docs/implementation/examples/finetune_attempt_success.yaml \
  --init-checkpoint runs/public/selected/best.pt \
  --run-dir runs/target/attempt_success

python scripts/finetune.py \
  --config docs/implementation/examples/finetune_attempt_success.yaml \
  --resume runs/target/attempt_success/last.pt \
  --run-dir runs/target/attempt_success
```

`--init-checkpoint` 与 `--resume` 互斥。前者是迁移初始化，重置 optimizer/step；后者恢复同一次目标训练。每次输出需要标明 source 权重、目标标签类型、分组划分、训练阶段和选模指标，避免把“源域预训练有效”与“目标域微调有效”混成一个结论。
