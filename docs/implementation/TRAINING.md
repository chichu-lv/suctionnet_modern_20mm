# 公开数据预训练：实现规格

本文规定 **Suction20Net 在公开 SuctionNet 场景上的预训练**。这一步获得可迁移的几何、接触质量与姿态表示，随后使用少量自有场景微调；公开数据上的模型选择实验不替代目标场景验证。

**实现状态：** 本文和 `examples/*.yaml` 是待实现接口规格。当前仓库已有数据制作代码；本文中的 `scripts/train_pretrain.py`、模型、Dataset 与训练入口不表示已经实现或运行过。实现者应按本文件落地，不能把命令示例当作已有功能。

相关文档：[现有数据字段](../DATA_SCHEMA.md)、[小样本微调规格](FINETUNING.md)。

## 1. 预训练范围与数据定义

### 1.1 固定继承已有数据结果

用户按旧方法制作的数据已经完成。预训练以**复用这批数据**为起点：通过只读、识别 schema/producer 版本的适配器读取 `graspness_r20_v2` 逐点文件，不要求先统一重写成新 schema，不重新执行 Seal、Wrench 或 Collision 制作。物理参数与 Seal 归一化来自该批数据的 `config_frozen.json`，而不是模型配置中的另一套半径默认值。

`data.source_format: legacy_dc4ad6b` 指定用户现有文件的默认来源，`data.compatibility: auto` 在文件带有明确新 schema 时识别新格式，其余按这个旧来源适配。具体规则见 [旧数据适配说明](LEGACY_DATA_ADJUSTMENT.md)：旧文件若把 PCA 法向放在 `gt_normal`，按 `candidate_row` 和原场景变换恢复与质量标签一致的候选轴；缺少 `normals_raw` 时，从原始深度及 `source_indices` 重建观测法向。上述调整发生在适配器，不原地覆盖现有文件，不重拟合已冻结的 Seal 归一化。

配置的 `data.source_root` 传给Dataset的 `dataset_root` 参数；`data.label_root` 指已有输出，`data.frozen_label_config` 指其冻结JSON，`data.adapter_cache` 可为空或指向可选派生缓存。这些配置名与示例文件保持一致。

主质量目标：

\[
S=\operatorname{clip}\left(\frac{S_{raw}-s_{lo}}{s_{hi}-s_{lo}},0,1\right),\quad
A=S(1-C_{collision}),\quad B=A W_{eval}.
\]

当前 `run-all` 会根据官方训练池中出现物体的 Seal 候选拟合第 1/99 百分位归一化上下界。`s_lo/s_hi` 必须使用**实际冻结文件中的数值**；配置默认的 0/1 不是已执行结果。预训练、验证、测试、目标域 `dense_proxy` 微调及 AP-R20 评价复用同一组常量。

`gt_A/gt_B` 是解析物理代理质量，未经真机标定时不解释成成功概率。`attempt_success` 微调另外训练任务成功头，详见 FINETUNING。

### 1.2 输入与标签隔离

`suction20/datasets.py` 将源数据适配成下列两部分：

```python
batch = {
    "inputs": ModelInputs(...),
    "targets": SourceTargets(...),
    "meta": FrameMetadata(...),
}
encoded = model.encode_scene(batch["inputs"])
# encode_scene 只接收 inputs。
# 训练 forward 可把 targets 传给明确的训练采样器，不能拼入输入特征。
# model(inputs, sampler_mode="train", targets=targets)
# model(inputs, sampler_mode="inference")  # 推理不传 targets。
```

关键字段约定：

| 字段 | 用途 | 是否能作为推理输入 |
|---|---|---|
| `xyz → xyz_cam_m`、`normals_raw → normal_obs`、`normals_raw_valid → normal_obs_valid` | 存储字段到当前观测输入的映射 | 是 |
| `instance_id_input` | 池化、前景候选和实例均衡采样 | 是，部署时来自已有前置分割 |
| `g_cam` | 相机坐标系中的单位重力方向 | 是，由外参/工作台坐标关系获得 |
| `instance_id_gt` | 监督身份和实例内损失归一化 | 只进入监督与评价 |
| `gt_valid`、`gt_A/gt_B`、Seal/Wrench/Collision | 物理监督 | 否 |
| `gt_normal` / 旧候选恢复值 → `normal_label` | 质量标签对应的原始候选外法向 | 否 |
| `gt_normal_pca` | 独立保存的拟合法向 | 本版不作为主轴目标或输入 |

默认的六组架构实验使用同一份 GT 实例输入以控制变量，此时两份实例 ID 相同。使用已有前置模型的预测 mask 时，只替换 `instance_id_input`，不覆盖 `instance_id_gt`。

新 schema 已保存 `normals_raw/normals_raw_valid` 时直接读取；已完成的旧文件缺少这些字段时走上述原始深度重建路径。深度噪声、点删除等增强改变了观测几何时，应相应更新观测法向和有效性。仅刚性旋转时可直接旋转观测法向。不能用 `gt_normal` 补齐缺失的观测法向。

适配后 targets 的正式字段为 `gt_A`、`gt_B`、`seal_target`（来自 `gt_seal_norm`）、`wrench_target`（来自 `gt_wrench_eval`）、`collision_free_target`（来自 `1-collision_flag`）、`normal_label`、`quality_valid`、`axis_valid`、`instance_id_gt`；可另保留原始 `gt_valid` 供追溯。以下公式中出现的存储标签名均指这一确定映射。

## 2. 六个独立实验与最终预训练权重

| 配置 | `q0/q` 目标 | 实例特征融合 | 接触分组 | S/W/C 辅助 |
|---|---|---:|---:|---:|
| E0 | `gt_A` | 关 | 关 | 关 |
| E1 | `gt_B` | 关 | 关 | 关 |
| E2 | `gt_B` | 开 | 关 | 关 |
| E3 | `gt_B` | 关 | 开 | 关 |
| E4 | `gt_B` | 开 | 开 | 关 |
| E5 | `gt_B` | 开 | 开 | 开 |

六组共有：spconv ResUNet14、逐点 256 维、同一法向头、`g_cam` 输入、相同实例均衡 seeds、相同 `q0/q` 损失、相同正常数据增强、相同优化步数和后处理。未启用接触分组时，最终 `q` 仍然是独立头，读取 seed 特征；不能同时取消第二个质量损失。

E0→E1 回答目标定义的作用；E1/E2/E3/E4 构成实例融合与接触分组的四组对照；E4→E5 回答联合辅助监督的作用。这里的“关闭实例融合”不表示移除了共有的实例均衡采样。

每组**独立从头训练**，初始化 seed 默认 42，使用相同视角抽样序列。E0→E5 不串联续训。后续目标域默认采用验证集选出的 E1–E5 中最佳 `gt_B` 模型；E0 是目标定义对照，不能把它的输出重新命名为 `q_B`。

本文件的“预训练”是指先在公开场景学习后迁移到自有场景。首轮不额外引入夹爪预训练权重、Sonata 或外部自监督 checkpoint，这些不是完成本轮工作的前提。

## 3. 场景划分、视角抽样与步数

### 3.1 固定划分

只使用官方训练池 `scene_0000`–`scene_0099` 做公开数据预训练开发：

- 验证：`scene_id % 10 == 9`，共 10 个场景。
- 训练：其余 90 个场景。
- 官方测试：seen `0100–0129`、similar `0130–0159`、novel `0160–0189`，不参与配置选择。

同一场景的全部相机视角归入同一集合。沿用已制作数据的归一化常量，不因内部 90/10 场景划分重做物理标签；该内部验证应表述为训练池内模型选择，不能称为完全独立的物体留出测试。

首轮只使用与实际部署一致的一种相机，例子为 `realsense`。不要求再训练另一种相机。

### 3.2 预算必须按 optimizer update 解释

| 参数 | 默认值 |
|---|---:|
| 每训练场景每轮视角数 | 64/256，轮内无放回 |
| 训练轮数 | 40 |
| 每轮帧数 | `90 × 64 = 5760` |
| 有效 batch | 4 个完整场景观测帧 |
| 每轮 optimizer update | 1440 |
| 总 optimizer update | **57600** |
| 总训练帧呈现量 | 230400，约 10 个全视角 epoch |
| warmup | 前 2880 次 update，即前 2 轮 |

`effective_batch_size = world_size × microbatch_size × gradient_accumulation_steps`。单 GPU 默认 `microbatch_size=1`、累积 4 次；若设备允许，可以用 microbatch 2、累积 2 次。六组实验应使用相同 microbatch，避免同时改变 BatchNorm 行为。

视角采样器使用与模型 RNG 分开的固定 seed。每轮在每个场景抽 64 个视角，轮间重新抽；六组读取同一 `(scene, frame)` 序列。不要把 40 轮误当成每轮遍历 256 视角的 40 个完整 epoch。

`global_step` 表示**成功执行的 optimizer update 数**，不是 DataLoader iteration 或 backward 次数。AMP 因非有限梯度跳过 optimizer update 时，scheduler 和 `global_step` 均不推进；累计批次计数正常清零。学习率调度、验证周期、checkpoint 均按 optimizer step 管理。停止条件以 `max_optimizer_steps=57600` 为准；40 轮与 230400 帧是无跳过更新时的名义预算，实际帧数与步数分别记录。

### 3.3 验证开销

每个验证场景固定取 `frame_id = 0,8,16,...,248`，共 32 视角；每 5760 update（4 轮）和最后一次训练后验证。使用确定性推理、相同候选数和后处理，按 `instance_Top1_Q20_at_0_4` 选择 `best`，并列时按 `AP_R20` 择优。训练损失只用于诊断，不替代最终姿态质量评价。

六组比较完成前不反复查看官方 test 结果。训练结束时若验证质量仍明显提升，才统一延长对比预算；不预先增加训练长度的参数搜索。

## 4. 观测增强与坐标约定

保留米制绝对尺度。**不做随机 scale、归一到单位球或按物体大小缩放点云**，否则半径 20 mm、标签包络和局部几何失去对应关系。

默认只做下列普通增强：

| 增强 | 默认 | 必须同步更新 |
|---|---|---|
| 绕重力轴旋转 | `[-180°,180°]` | `xyz`、观测法向、GT 轴；`g_cam`按同一坐标变换更新 |
| 工作台平面内平移 | 每轴 `[-0.02,0.02] m` | `xyz`及所有显式姿态点 |
| 轻量深度扰动 | 深度方向标准差 `0.0005 m`，截断至 `±0.0015 m` | 当前点云和观测法向；物理标签作为干净几何监督保留 |
| 轻量随机点删除 | 0–5% | 索引、实例输入、法向有效性及各标签数组一起索引 |

平移/旋转是坐标系或重力保持变换，不改变物理评分；若采用任意 3D 旋转作为坐标变换，必须同时旋转 `g_cam`，不能保留原重力却沿用旧 Wrench 标签。首轮不需要额外俯仰翻转或非刚性变形。

观测点扰动后不重算 expensive 物理标签；扰动只模拟轻量传感噪声。若实现尚不支持可靠地更新噪声后的观测法向，可以先关闭深度扰动，并对所有实验统一关闭，不能改用 GT 轴作为输入。

下采样和增强后，重新计算输入实例的可见中心、尺寸与局部空间索引。接触分组使用观测点的原始米制位置，不用体素中心替代接触点。切平面绕轴基的随机旋转只用于局部分组，不改变吸盘物理姿态。

## 5. 候选采样与训练/推理一致性

全场景输入保留目标之外的邻物和工作台上下文。候选中心满足：`instance_id_input > 0` 且当前观测法向有效；`gt_valid` 不是推理过滤条件。

默认每帧最多 512 个 seeds，按输入实例分配近似相等配额，少量点实例不重复补成虚假独立监督。所有 E0–E5 使用同一算法：

- 第 1–2 轮：每实例 50% 已知高 `gt_A`、50% 已知低 `gt_A` 的标签覆盖采样；不足部分从该实例可观测候选补齐。
- 第 3–40 轮：50% 当前模型提议、25% 已知高 `gt_A`、25% 已知低 `gt_A`。
- 高/低分界固定为 `gt_A >= 0.2` / `< 0.2`，只在 `gt_valid=True` 的监督采样池中划分。
- 模型提议由 `q0` 排序与空间覆盖选取，可包含 `gt_valid=False` 的点；这些点可进入网络计算，但相应物理损失被 mask。

GT 参与的是训练采样器，而不是输入特征。部署和验证只有模型提议路径，不能保留训练期的 GT 候选分配。

法向残差头末层零初始化；第 1 轮起输出 `n_pred`，接触分组在训练和推理都围绕 `stop_gradient(n_pred)` 构建。不能训练用 GT 轴建组、推理换预测轴。这是恢复与原候选质量对应的去噪轴，不是通过质量梯度搜索任意新姿态：缓存质量仍对应原候选轴，预测轴分组是对其近似；最终收益由重算实际输出 `(p,n)` 的评价确认。

## 6. 损失掩码与归一化

### 6.1 不把未匹配点当作负例

```python
quality_valid = targets.gt_valid & (targets.instance_id_gt > 0)

axis_target_valid = (
    targets.gt_valid
    & (targets.instance_id_gt > 0)
    & is_finite_nonzero(targets.normal_label)
)
axis_valid = axis_target_valid & inputs.normal_obs_valid

# 文件中的 normal_valid 当前是 (~collision_flag) & (gt_seal_norm > 0.2)。
# 它是数据生成器的质量筛选，不是本版主轴损失的几何有效性掩码。
```

这里的 `targets.normal_label` 是**适配后的候选轴**：新文件可直接使用正确的 `gt_normal`；旧文件由适配器按候选行恢复。不能仅凭字段名字假定所有旧文件已经采用新语义。`axis_valid` 独立于质量阈值、是否碰撞和 PCA 有效性：有匹配轴的已知低 Seal / 碰撞候选仍可提供法向监督。`gt_valid=False` 的零值不能进入质量或轴损失。

只有输入轴有效的点参与本版法向残差监督。残差基于观测轴建立，`n_obs` 与 `n_label` 均单位化且遵守外法向符号；损失不取绝对内积。

### 6.2 损失定义

令 `SL1` 为 `beta=0.05` 的 Smooth L1：

\[
L=L_q+L_{q0}+2L_n+0.50L_S+0.25L_W+0.10L_C.
\]

| 项 | 预测/目标 | mask |
|---|---|---|
| `L_q0` | 全点粗质量 `q0` 对 `gt_A` 或 `gt_B`，SL1 | `quality_valid` |
| `L_q` | seeds 最终质量 `q` 对同目标，SL1 | `seed_valid & quality_valid[seed]` |
| `L_n` | `1-dot(n_pred,n_label)` | `axis_valid` |
| `L_S` | seeds 的 sigmoid Seal 对 `gt_seal_norm`，SL1 | `seed_valid & quality_valid[seed]` |
| `L_W` | seeds 的 sigmoid Wrench 对 `gt_wrench_eval`，SL1 | `seed_valid & quality_valid[seed]` |
| `L_C` | collision-free logit 对 `1-collision_flag`，BCEWithLogits | `seed_valid & quality_valid[seed]` |

E0 质量目标为 A；E1–E5 为 B。仅 E5 启用最后三项。碰撞样本的 `q=0` 是有效质量负例，但 Seal/Wrench 辅助项仍使用其真实分量，不因总分为零把两个分量强制清零。

主排序使用 `q`；不能再次乘 `C`、`S` 或 `W`。S/W/C 只作为辅助监督与诊断输出。

每个损失项依次做：同一 `(batch_index, instance_id_gt)` 内平均 → 每帧有监督实例平均 → batch 内有监督帧平均。空监督项返回与对应预测图相连的零值。不同项分别按自己的 mask 归一化，不共用一个全体点数分母。

## 7. 优化器、BatchNorm 与训练循环

### 7.1 默认优化

- AdamW：`lr_peak=1e-3`、`weight_decay=1e-4`、`betas=(0.9,0.999)`、`eps=1e-8`。
- bias 和归一化层参数不做 weight decay；其余可训练权重使用上述 decay。
- AMP 开启；法向单位化、几何分组判定和 loss 归约使用 float32。
- 如使用梯度裁剪，统一 `max_norm=10.0`，并在 `scaler.unscale_(optimizer)` 后执行。

调度器以成功 update 索引 `k=0..57599` 计算本次学习率。前 2880 次从 `1e-5` 线性升至 `1e-3`；随后 cosine 降至 `1e-5`，最后一次 update 达到终值。warmup 与 cosine 只在 optimizer 成功更新后前进一步，不能每个 microbatch 调一次。

### 7.2 BN 规则

公开预训练时 backbone 与所有新模块处于 train 模式，BatchNorm 使用当前 microbatch 的 active voxels 更新统计。梯度累积不等于把 BN 的统计 batch 扩大成 4，报告中保留 microbatch 设置。

验证统一 `model.eval()`。目标域小样本微调时所有 BN 固定为 eval，详见 FINETUNING。不要因为预训练 microbatch=1 就默认改全套网络归一化层，首轮保持同一骨干实现。

### 7.3 循环伪代码

```python
epoch = start_epoch
while global_step < 57600:
    sampler.set_epoch(epoch)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for micro_idx, batch in enumerate(loader):
        inputs, targets = batch["inputs"], batch["targets"]
        outputs = model(inputs, sampler_mode="train", targets=targets)
        loss_dict = source_losses(outputs, targets)
        loss = loss_dict["total"] / accumulation_steps
        scaler.scale(loss).backward()
        if accumulation_boundary(micro_idx):
            scaler.unscale_(optimizer)
            clip_grad_norm_(trainable_parameters, 10.0)
            updated = amp_optimizer_step(scaler, optimizer)
            optimizer.zero_grad(set_to_none=True)
            if updated:
                global_step += 1
                scheduler.step()
            maybe_validate_and_save(global_step)
            if global_step >= 57600:
                break
    epoch += 1
```

`amp_optimizer_step` 是训练实现的建议内部辅助函数，签名允许实现时合并；`forward` 内部仅训练采样器读取 targets，`encode_scene` 与特征计算不读取标签。采样阶段随本轮训练进度更新。验证后恢复 train 模式，微调时另行重新固定 BN eval。

## 8. Checkpoint 与迁移语义

建议保存 `last.pt`（续训）和 `best.pt`（验证最佳）。只在 optimizer update 边界保存，避免为了断点重现另存未提交梯度。

```python
checkpoint = {
    "format_version": "suction20.checkpoint.v1",
    "stage": "public_pretrain",             # public_pretrain / target_finetune
    "model_state_dict": model.state_dict(),
    "model_config": resolved_model_config,
    "quality_target": "gt_B",               # E0 为 gt_A，不能混淆
    "source_label_config": frozen_label_config,
    "source_label_config_sha256": config_digest,
    "experiment": "E5",
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
    "global_step": global_step,
    "epoch": epoch,
    "next_microbatch_in_epoch": next_microbatch_in_epoch,
    "sampler_state": sampler.state_dict(),
    "rng_state": capture_python_numpy_torch_cuda_rng(),
    "split_definition": split_definition,
    "best_metric": {"name": "instance_Top1_Q20_at_0_4", "value": best_value},
    "best_tiebreak_metric": {"name": "AP_R20", "value": best_ap},
    "training_config": resolved_training_config,
    "code_revision": current_git_commit,
}
```

`source_label_config_sha256` 对冻结配置的稳定序列化内容计算，不依赖本地绝对路径。完整物理和归一化常量同时保存在 checkpoint，迁移时仍能找回原标签定义。数据目录路径可由运行参数覆盖，标签含义不能被路径覆盖。旧冻结 JSON 中可能保留 `wrench_threshold_factor` 等新生产器 parser 不接受的键，适配器应直接读取原 JSON 并提取所需字段，不借新 parser 重建默认物理配置。

两种加载语义必须分开：

| 参数 | 加载内容 | 适用情况 |
|---|---|---|
| `--resume PATH` | 权重、optimizer、scheduler、AMP、step、sampler/RNG、最佳指标 | 中断后继续同一个实验 |
| `--init-checkpoint PATH` | 兼容模型权重、源标签定义和模型结构；重建 optimizer/scheduler，step=0 | 目标域迁移或明确的权重初始化 |

从公开预训练进入目标域微调时始终用第二种。optimizer 的动量、公开训练的 cosine 位置和 `global_step=57600` 不带入微调。

`state_dict` 加载应明确列出允许新增的 `task_success_head` 参数；不要通过无限制 `strict=False` 掩盖模型结构不匹配。没有任务头的源 checkpoint 初始化目标任务头时，其他共有参数必须全部成功匹配。

## 9. 可选残差不确定性头

主模型选定后，若需要检验误差感知排序，可冻结整个预训练模型，在其最终 pose 特征后增加残差尺度头。该项称 **predictive residual uncertainty**，不宣称分离了 aleatoric 与 epistemic uncertainty。

冻结 `q` 后，令 `v=1e-4+softplus(a)`，优化：

\[
L_{var}=\tfrac12\operatorname{mean}\left[(q_{GT}-\operatorname{stopgrad}(q))^2/v+\log v\right].
\]

该项只读有物理标签的自然推理候选，不继续正负过采样。使用MODEL §10的3轮/4320步、固定LR `1e-3` 小头拟合预算。`score=q-lambda*sqrt(v)`，验证候选 `lambda={0,0.25,0.5}` 包含关闭选项；按同一 `instance_Top1_Q20_at_0_4` 主指标和AP-R20并列参考决定是否启用。训练集内拟合残差不能证明外部误差方差已校准。

它不计入六个核心架构配置，也不是开始目标域微调的前置工作。公开域拟合的尺度不会自动在目标域有效；默认目标域部署先将 `lambda=0`，只在既有目标验证数据上有证据时恢复。

## 10. 未来命令与实现完成条件

以下命令是**待实现入口的约定**：

```bash
python scripts/train_pretrain.py \
  --config docs/implementation/examples/pretrain_e5.yaml \
  --run-dir runs/public/E5_seed42

python scripts/train_pretrain.py \
  --config docs/implementation/examples/pretrain_e5.yaml \
  --run-dir runs/public/E5_seed42 \
  --resume runs/public/E5_seed42/last.pt
```

切换 E0–E4 应修改实验 flags/目标和 run-dir；训练框架必须解析为独立实验，不从前一组自动找权重。

一次可交付的公开预训练结果应包含：冻结标签配置、split、解析后的模型/训练配置、`best.pt/last.pt`、每次验证的每实例Top-1 Q20@.4与AP-R20和实际 optimizer step、硬件与训练耗时记录。报告实测结果前不填写预期指标数字。后续目标域仅微调选出的模型，不把六组源域消融全量重跑到自有数据上。
